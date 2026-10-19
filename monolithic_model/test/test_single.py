import os
import time
import torch
import timm
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json

# ======================
# Configuration - Raspberry Pi 4B 4GB real hardware simulation
# Hardware specs:
#   CPU : Broadcom BCM2711, Cortex-A72 (ARMv8) 64-bit @ 1.5GHz, 4 cores
#   RAM : 4GB LPDDR4-3200
#   No GPU acceleration, CPU-only inference
# ======================
MODEL_NAME = "efficientnet_b0"
INPUT_SIZE = 224
NUM_CLASSES = 27
BATCH_SIZE = 4          # Suited to RPi4B memory bandwidth
MODEL_PATH = f"efficientnet_b0_300e_224x224_final.pth"
TEST_DATA_DIR = "../../../../../dataset/5%_RSCD/test"
RESULT_DIR = f"test_results/{MODEL_NAME}_0"
DEVICE = torch.device("cpu")
NUM_WORKERS = 4         # Matches the 4 cores of BCM2711

# Limit PyTorch threads to the number of physical cores to avoid hyper-threading contention
torch.set_num_threads(4)           # Cortex-A72 4 cores
torch.set_num_interop_threads(1)   # No inter-op parallel scheduling needed for inference

# Raspberry Pi 4B hardware description string (for logs/reports)
RPI4B_DESC = "Raspberry Pi 4B | BCM2711 Cortex-A72 @ 1.5GHz x4 | 4GB LPDDR4-3200 | CPU-only"

os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ======================
# Data preprocessing
# ======================
test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model(model_path, device):
    model = timm.create_model(MODEL_NAME, pretrained=False)
    model.reset_classifier(NUM_CLASSES)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', None)
        best_val_acc = checkpoint.get('best_val_acc', None)
        print(f"Model loaded successfully, best validation accuracy during training: {best_val_acc:.4f}" if best_val_acc else "Model loaded successfully")
    else:
        model.load_state_dict(checkpoint)
        class_names = None
        print("Model loaded successfully (legacy format)")

    model = model.to(device)
    model.eval()
    print(f"Monolithic model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, class_names


def test_model(model, test_loader, device, class_names):
    all_preds = []
    all_labels = []
    all_probs = []
    batch_times = []
    total_samples = 0

    # Raspberry Pi 4B CPU warm-up: ARM Cortex-A72 cold start is slow, so use more warm-up iterations
    print("\nCPU warm-up (Cortex-A72 @ 1.5GHz)...")
    dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        for _ in range(5):  # RPi4B ARM cores need more warm-up iterations
            model(dummy_input)

    print("\nStarting test...")
    inference_start = time.time()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', ncols=100)
        for inputs, labels in pbar:
            batch_start = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            batch_times.append(time.time() - batch_start)
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_inference_time = time.time() - inference_start

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    avg_sample_time = total_inference_time / total_samples
    throughput = total_samples / total_inference_time

    print(f"\n{'=' * 60}")
    print(f"Test results summary")
    print(f"{'=' * 60}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"\nInference time statistics:")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Total samples: {total_samples}")
    print(f"  Avg time per batch: {avg_batch_time * 1000:.2f}ms ± {std_batch_time * 1000:.2f}ms")
    print(f"  Avg time per sample: {avg_sample_time * 1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} samples/s")
    print(f"  Processor: {RPI4B_DESC}")
    print(f"{'=' * 60}")

    names = class_names if class_names else [f"Class_{i}" for i in range(NUM_CLASSES)]
    print("\nClassification report:")
    report = classification_report(all_labels, all_preds, target_names=names, digits=4)
    print(report)

    with open(os.path.join(RESULT_DIR, "classification_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
        f.write(f"Inference time statistics:\n")
        f.write(f"  Total inference time: {total_inference_time:.2f}s\n")
        f.write(f"  Total samples: {total_samples}\n")
        f.write(f"  Avg time per batch: {avg_batch_time * 1000:.2f}ms ± {std_batch_time * 1000:.2f}ms\n")
        f.write(f"  Avg time per sample: {avg_sample_time * 1000:.2f}ms\n")
        f.write(f"  Throughput: {throughput:.2f} samples/s\n")
        f.write(f"  Processor: {RPI4B_DESC}\n\n")
        f.write(report)

    results_json = {
        "overall_accuracy": float(accuracy),
        "hardware": {
            "device": "Raspberry Pi 4B",
            "soc": "Broadcom BCM2711",
            "cpu": "Cortex-A72 (ARMv8) 64-bit @ 1.5GHz",
            "cpu_cores": 4,
            "ram": "4GB LPDDR4-3200",
            "inference_backend": "PyTorch CPU (no GPU)",
            "torch_threads": torch.get_num_threads(),
        },
        "inference_time": {
            "total_time_seconds": float(total_inference_time),
            "total_samples": int(total_samples),
            "avg_batch_time_ms": float(avg_batch_time * 1000),
            "std_batch_time_ms": float(std_batch_time * 1000),
            "avg_sample_time_ms": float(avg_sample_time * 1000),
            "throughput_samples_per_sec": float(throughput),
        },
        "num_samples": len(all_labels)
    }
    with open(os.path.join(RESULT_DIR, "test_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    return all_preds, all_labels, all_probs, accuracy


def plot_confusion_matrix(labels, preds, class_names, save_path):
    names = class_names if class_names else [f"Class_{i}" for i in range(NUM_CLASSES)]
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def save_predictions(image_paths, preds, labels, probs, class_names, save_path):
    import pandas as pd

    results = []
    for img_path, pred, label, prob in zip(image_paths, preds, labels, probs):
        results.append({
            'image_path': img_path,
            'true_label': class_names[label] if class_names else label,
            'pred_label': class_names[pred] if class_names else pred,
            'correct': pred == label,
            'confidence': prob[pred],
            **{f'prob_{class_names[j] if class_names else j}': prob[j] for j in range(len(prob))}
        })

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Detailed predictions saved to: {save_path}")


# ======================
# Main
# ======================
if __name__ == "__main__":
    print(f"{'=' * 60}")
    print(f"EfficientNet-B0 test script - Raspberry Pi 4B 4GB real hardware simulation")
    print(f"{'=' * 60}")
    print(f"Device: {DEVICE}")
    print(f"Hardware specs: {RPI4B_DESC}")
    print(f"PyTorch threads: {torch.get_num_threads()} (matches 4-core Cortex-A72)")
    print(f"Model path: {MODEL_PATH}")
    print(f"Test data path: {TEST_DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS} (matches 4 physical cores)")
    print(f"{'=' * 60}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(TEST_DATA_DIR):
        raise FileNotFoundError(f"Test data directory not found: {TEST_DATA_DIR}")

    model, class_names = load_model(MODEL_PATH, DEVICE)

    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, test_transform)
    if class_names is None:
        class_names = test_dataset.classes

    print(f"Classes: {class_names}")
    print(f"Test set samples: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False   # RPi4B has no GPU, pin_memory is pointless
    )

    preds, labels, probs, accuracy = test_model(model, test_loader, DEVICE, class_names)

    confusion_matrix_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, class_names, confusion_matrix_path)

    image_paths = [path for path, _ in test_dataset.samples]
    predictions_path = os.path.join(RESULT_DIR, "predictions.csv")
    save_predictions(image_paths, preds, labels, probs, class_names, predictions_path)

    errors = np.where(preds != labels)[0]
    print(f"\nMisclassified samples: {len(errors)} / {len(labels)} ({len(errors) / len(labels) * 100:.4f}%)")

    if len(errors) > 0:
        import pandas as pd

        error_samples = [{
            'image_path': image_paths[idx],
            'true_label': class_names[labels[idx]],
            'pred_label': class_names[preds[idx]],
            'confidence': probs[idx][preds[idx]]
        } for idx in errors]

        error_df = pd.DataFrame(error_samples)
        error_path = os.path.join(RESULT_DIR, "error_samples.csv")
        error_df.to_csv(error_path, index=False, encoding='utf-8-sig')
        print(f"Misclassified sample details saved to: {error_path}")

    print(f"\nAll test results saved to directory: {RESULT_DIR}")