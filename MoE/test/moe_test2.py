import os
import time
import torch
import timm
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


MODEL_NAME = "efficientnet_b0"
INPUT_SIZE = 224
NUM_EPOCHS = 300
tDATA_DIR = "../../../../../dataset/5%_RSCD/test"
MODEL_PATH = f"efficientnet_b0_300e_224x224_final.pth"
RESULT_SAVE_DIR = f"results/{MODEL_NAME}_2"

BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_CLASSES = 27
NUM_EXPERTS = 5
TOP_K = 2
PRETRAINED = False
DEVICE = torch.device("cpu")

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


os.makedirs(RESULT_SAVE_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ======================
# MoE component definitions (consistent with training script v0)
# ======================
class SharedFeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv_stem = base_model.conv_stem
        self.bn1 = base_model.bn1
        self.act1 = base_model.act1 if hasattr(base_model, 'act1') else nn.SiLU(inplace=True)
        self.blocks = nn.Sequential(*base_model.blocks[:6])

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x


class Router(nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, num_experts, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.15)
        self.layernorm = nn.LayerNorm(num_experts)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        logits = self.gap(x).squeeze(-1).squeeze(-1)
        logits = self.layernorm(logits)
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
        return logits


class ExpertNetwork(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.blocks = nn.Sequential(*base_model.blocks[6:])
        self.conv_head = base_model.conv_head
        self.bn2 = base_model.bn2
        self.act2 = base_model.act2 if hasattr(base_model, 'act2') else nn.SiLU(inplace=True)
        self.global_pool = base_model.global_pool
        self.classifier = nn.Linear(base_model.num_features, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class MoEModel(nn.Module):
    def __init__(self, num_experts, num_classes, top_k, pretrained=True):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.top_k = top_k

        base_model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.shared_extractor = SharedFeatureExtractor(base_model)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            shared_out = self.shared_extractor(dummy_input)
            router_in_channels = shared_out.shape[1]

        self.router = Router(router_in_channels, num_experts)

        self.experts = nn.ModuleList([
            ExpertNetwork(
                timm.create_model('efficientnet_b0', pretrained=pretrained),
                num_classes
            ) for _ in range(num_experts)
        ])

        self._initialize_experts_with_perturbation()

    def _initialize_experts_with_perturbation(self):
        if PRETRAINED:
            for expert in self.experts:
                if hasattr(expert.classifier, 'reset_parameters'):
                    expert.classifier.reset_parameters()
                else:
                    nn.init.normal_(expert.classifier.weight, mean=0, std=0.01)
                    if expert.classifier.bias is not None:
                        nn.init.constant_(expert.classifier.bias, 0)
        else:
            for i, expert in enumerate(self.experts):
                torch.manual_seed(42 + i)
                for m in expert.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, training=True):
        shared_features = self.shared_extractor(x)

        router_logits = self.router(shared_features)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, self.num_classes,
                                   device=x.device, dtype=shared_features.dtype)
        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = topk_indices[i, j]
                weight = topk_weights[i, j]
                expert_out = self.experts[expert_idx](shared_features[i:i + 1])
                final_output[i] += weight * expert_out.squeeze(0)

        return final_output, torch.tensor(0.0), topk_indices

    def _compute_load_balance_loss(self, router_probs):
        load = router_probs.mean(dim=0)
        ideal_load = 1.0 / self.num_experts
        return ((load - ideal_load) ** 2).mean()


# ======================
# Data preprocessing
# ======================
ttransforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ======================
# Test function
# ======================
def tmodel(model, tloader, class_names):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_expert_indices = []
    expert_count = [0] * NUM_EXPERTS

    batch_times = []
    total_samples = 0

    print("\nCPU warm-up in progress...")
    dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input, training=False)

    print("\nStarting Test...")
    inference_start = time.time()

    with torch.no_grad():
        for inputs, labels in tqdm(tloader, desc='Test progress'):
            batch_start = time.time()

            outputs, _, topk_indices = model(inputs, training=False)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            batch_times.append(time.time() - batch_start)
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_expert_indices.extend(topk_indices.cpu().numpy())

            for i in range(topk_indices.size(0)):
                for j in range(TOP_K):
                    expert_count[topk_indices[i, j].item()] += 1

    total_inference_time = time.time() - inference_start

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = np.mean(all_preds == all_labels)
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    avg_sample_time = total_inference_time / total_samples
    throughput = total_samples / total_inference_time

    print(f"\n{'=' * 60}")
    print(f"Test results statistics")
    print(f"{'=' * 60}")
    print(f"Overall test accuracy: {accuracy:.4f}")
    print(f"\nInference time statistics:")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Total samples: {total_samples}")
    print(f"  Average time per batch: {avg_batch_time * 1000:.2f}ms ± {std_batch_time * 1000:.2f}ms")
    print(f"  Average time per sample: {avg_sample_time * 1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} samples/s")
    print(f"{'=' * 60}")

    print("\nClassification report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    with open(f"{RESULT_SAVE_DIR}/classification_report.txt", 'w', encoding='utf-8') as f:
        f.write(f"Overall test accuracy: {accuracy:.4f}\n")
        f.write(f"\nInference time statistics:\n")
        f.write(f"  Total inference time: {total_inference_time:.2f}s\n")
        f.write(f"  Total samples: {total_samples}\n")
        f.write(f"  Average time per batch: {avg_batch_time * 1000:.2f}ms ± {std_batch_time * 1000:.2f}ms\n")
        f.write(f"  Average time per sample: {avg_sample_time * 1000:.2f}ms\n")
        f.write(f"  Throughput: {throughput:.2f} samples/s\n")
        f.write(f"\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    total_activations = sum(expert_count)
    expert_usage = [count / total_activations * 100 for count in expert_count]
    print("\nExpert usage statistics:")
    for i, usage in enumerate(expert_usage):
        print(f"  Expert {i}: {usage:.2f}% ({expert_count[i]}/{total_activations})")

    results = {
        "overall_accuracy": float(accuracy),
        "inference_time": {
            "total_time_seconds": float(total_inference_time),
            "total_samples": int(total_samples),
            "avg_batch_time_ms": float(avg_batch_time * 1000),
            "std_batch_time_ms": float(std_batch_time * 1000),
            "avg_sample_time_ms": float(avg_sample_time * 1000),
            "throughput_samples_per_sec": float(throughput),
        },
        "expert_usage": {
            f"expert_{i}": {
                "usage_percentage": usage,
                "activation_count": expert_count[i]
            } for i, usage in enumerate(expert_usage)
        },
        "num_samples": len(all_labels)
    }

    with open(f"{RESULT_SAVE_DIR}/tresults.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nTest results saved to {RESULT_SAVE_DIR}/")
    return accuracy, all_preds, all_labels, all_probs


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{RESULT_SAVE_DIR}/confusion_matrix.png", dpi=300)
    print(f"Confusion matrix saved to {RESULT_SAVE_DIR}/confusion_matrix.png")
    plt.close()


# ======================
# Main flow
# ======================
if __name__ == "__main__":
    print(f"{'=' * 60}")
    print(f"MoE model Test script")
    print(f"{'=' * 60}")
    print(f"Device: {DEVICE}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Test data path: {tDATA_DIR}")
    print(f"Number of experts: {NUM_EXPERTS}, TopK: {TOP_K}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"{'=' * 60}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file does not exist: {MODEL_PATH}")

    print("\nLoading model...")
    model = MoEModel(num_experts=NUM_EXPERTS,
                     num_classes=NUM_CLASSES,
                     top_k=TOP_K,
                     pretrained=False).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', [f"Class_{i}" for i in range(NUM_CLASSES)])
        best_val_acc = checkpoint.get('best_val_acc', None)
        print(f"Model loaded successfully (best validation accuracy in training: {best_val_acc:.4f})")
    else:
        model.load_state_dict(checkpoint)
        class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
        print("Model loaded successfully (legacy format)")

    print("\nLoading test data...")
    tdataset = datasets.ImageFolder(tDATA_DIR, ttransforms)
    tloader = DataLoader(tdataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS,
                         pin_memory=False)

    print(f"Test samples: {len(tdataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    accuracy, preds, labels, probs = tmodel(model, tloader, class_names)

    print("\nTest complete!")
    print(f"{'=' * 60}")