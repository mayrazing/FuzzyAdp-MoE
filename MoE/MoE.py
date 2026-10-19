import copy
import os
import time
import logging
import random
import numpy as np
import timm
from torch import nn, cuda, autocast, GradScaler, torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

MODEL_NAME = "efficientnet_b0"
INPUT_SIZE = 224
NUM_EPOCHS = 300
DATA_DIR = "../../../dataset/5%_RSCD"
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_DELTA = 0.001

BATCH_SIZE = 64
NUM_CLASSES = 27
NUM_EXPERTS = 5
TOP_K = 2
AUX_LOSS_WEIGHT = 1.5

SAVE_PATH = f"pth/{MODEL_NAME}/{MODEL_NAME}_{NUM_EPOCHS}e_{INPUT_SIZE}x{INPUT_SIZE}.pth"
CSV_PATH = Path(f"save_csv_log/{MODEL_NAME}/{MODEL_NAME}_train_{NUM_EPOCHS}e_{INPUT_SIZE}x{INPUT_SIZE}.csv")
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
STEP_SIZE = 5
GAMMA = 0.1
MIN_LR = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(14, os.cpu_count())
PRETRAINED = True

epochs = []
train_losses, val_losses = [], []
train_accuracy, val_accuracy = [], []
expert_usage_history = []

log_file_path = f'save_csv_log/{MODEL_NAME}/{MODEL_NAME}_{NUM_EPOCHS}e_{INPUT_SIZE}x{INPUT_SIZE}_train_time_log.txt'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

amp_scaler = GradScaler()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


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
        """Diversified expert initialization - improved version"""
        if PRETRAINED:
            for expert in self.experts:
                if hasattr(expert.classifier, 'reset_parameters'):
                    expert.classifier.reset_parameters()
                else:
                    # Manual reset
                    nn.init.normal_(expert.classifier.weight, mean=0, std=0.01)
                    if expert.classifier.bias is not None:
                        nn.init.constant_(expert.classifier.bias, 0)
        else:
            for i, expert in enumerate(self.experts):
                torch.manual_seed(42 + i)  # Different seed
                for m in expert.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, training=True):
        shared_features = self.shared_extractor(x)  # (B, C, H, W)

        router_logits = self.router(shared_features)  # (B, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, num_experts)

        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if training:
            sparse_mask = torch.zeros_like(router_probs).scatter(1, topk_indices, topk_weights)

            batch_size = shared_features.size(0)
            expert_outputs_per_expert = [
                shared_features.new_zeros(batch_size, self.num_classes)
                for _ in range(self.num_experts)
            ]
            used_experts = torch.unique(topk_indices)
            for expert_id in used_experts.tolist():
                idx = torch.where((topk_indices == expert_id).any(dim=1))[0]
                if idx.numel() == 0:
                    continue
                out = self.experts[expert_id](shared_features[idx])
                expert_outputs_per_expert[expert_id] = \
                    expert_outputs_per_expert[expert_id].index_add(0, idx, out)

            expert_outputs = torch.stack(expert_outputs_per_expert, dim=1)  # (B, num_experts, num_classes)
            final_output = torch.einsum('be,bec->bc', sparse_mask, expert_outputs)

            aux_loss = self._compute_load_balance_loss(router_probs, topk_indices)

            return final_output, aux_loss, topk_indices
        else:
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

    def _compute_load_balance_loss(self, router_probs, topk_indices):
        num_tokens = router_probs.size(0)

        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts)  # (B, top_k, num_experts)
        expert_mask = expert_mask.sum(dim=1).float()  # (B, num_experts), each row may contain multiple 1s
        f = expert_mask.sum(dim=0) / (num_tokens * self.top_k)  # (num_experts,), normalized

        p = router_probs.mean(dim=0)

        aux_loss = self.num_experts * (f.detach() * p).sum()
        return aux_loss


class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f'Early stopping: initial val accuracy: {score:.4f}')
                logging.info(f'Early stopping: initial val accuracy: {score:.4f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter}/{self.patience}')
                logging.info(f'Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'Early stopping: val accuracy improved to {score:.4f}')
                logging.info(f'Early stopping: val accuracy improved to {score:.4f}')


def dump_history_to_csv(csv_path, epochs, train_losses, train_accuracy,
                        val_losses, val_accuracy, expert_usage):
    df_dict = {
        "epoch": epochs,
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "val_loss": val_losses,
        "val_acc": val_accuracy,
    }

    num_experts = len(expert_usage[0]) if expert_usage else NUM_EXPERTS
    for i in range(num_experts):
        df_dict[f"expert_{i}_usage"] = [e[i] for e in expert_usage]

    df = pd.DataFrame(df_dict)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)


def log_expert_stats(topk_indices, labels, epoch, num_classes=None, class_names=None):
    if num_classes is None:
        num_classes = NUM_CLASSES
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    expert_count = [0] * NUM_EXPERTS
    expert_class_count = [[0] * num_classes for _ in range(NUM_EXPERTS)]

    for i in range(topk_indices.size(0)):
        label = labels[i].item()
        for j in range(TOP_K):
            expert_idx = topk_indices[i, j].item()
            expert_count[expert_idx] += 1
            expert_class_count[expert_idx][label] += 1

    total = sum(expert_count)
    expert_usage = [count / total * 100 if total > 0 else 0
                    for count in expert_count]

    print(f"\nExpert usage (Epoch {epoch}):")
    logging.info(f"\nExpert usage (Epoch {epoch}):")

    for i in range(NUM_EXPERTS):
        class_ratios = []
        for cls_idx in range(num_classes):
            ratio = expert_class_count[i][cls_idx] / max(expert_count[i], 1) * 100
            class_ratios.append(f"{class_names[cls_idx]}: {ratio:.1f}%")

        class_dist = ", ".join(class_ratios)
        print(f"  Expert {i}: {expert_usage[i]:.1f}% ({class_dist})")
        logging.info(f"  Expert {i}: {expert_usage[i]:.1f}% ({class_dist})")

    return expert_usage


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def train_model(model_, criterion_, optimizer_, scheduler_,
                num_epochs=NUM_EPOCHS,
                initial_best_val_acc_=None,
                save_path=SAVE_PATH):
    since = time.time()
    best_model_wts = copy.deepcopy(model_.state_dict())
    best_val_acc = float(initial_best_val_acc_) if initial_best_val_acc_ is not None else 0.0

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,
                                   delta=EARLY_STOPPING_DELTA,
                                   verbose=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logging.info("-" * 60)

        metrics = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                model_.train()
            else:
                model_.eval()

            running_cls_loss = 0.0
            running_aux_loss = 0.0
            running_weighted_aux_loss = 0.0
            running_corrects = 0
            all_topk_indices = []
            all_labels = []

            pbar = tqdm(dataloaders[phase], desc=f'{phase} processing', ncols=100)

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer_.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast("cuda" if cuda.is_available() else "cpu"):
                        outputs, aux_loss, topk_indices = model_(
                            inputs,
                            training=(phase == 'train')
                        )
                        cls_loss = criterion_(outputs, labels)

                        if phase == 'train':
                            current_aux_weight = AUX_LOSS_WEIGHT
                            if epoch < 20:
                                current_aux_weight = AUX_LOSS_WEIGHT * 0.1  # Apply only a small constraint early on

                            total_loss = cls_loss + current_aux_weight * aux_loss
                        else:
                            total_loss = cls_loss

                    if phase == 'train':
                        amp_scaler.scale(total_loss).backward()
                        amp_scaler.step(optimizer_)
                        amp_scaler.update()

                running_cls_loss += cls_loss.item() * inputs.size(0)
                running_aux_loss += aux_loss.item() * inputs.size(0)
                running_weighted_aux_loss += (aux_loss * current_aux_weight).item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                all_topk_indices.append(topk_indices.cpu())
                all_labels.append(labels.cpu())

                pbar.set_postfix({'cls': f'{cls_loss.item():.4f}',
                                  'aux': f'{aux_loss.item():.4f}',
                                  'weighted_aux': f'{(aux_loss * current_aux_weight).item():.4f}'
                                  })

            epoch_cls_loss = running_cls_loss / dataset_sizes[phase]
            epoch_aux_loss = running_aux_loss / dataset_sizes[phase]
            epoch_weighted_aux_loss = running_weighted_aux_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            metrics[phase] = epoch_acc

            if phase == 'train':
                train_losses.append(epoch_cls_loss)
                train_accuracy.append(epoch_acc)
            else:
                val_losses.append(epoch_cls_loss)
                val_accuracy.append(epoch_acc)
                scheduler_.step(epoch_acc)

            print(f"{phase} CLS Loss: {epoch_cls_loss:.4f} Aux Loss: {epoch_aux_loss:.6f} Weighted Aux Loss: {epoch_weighted_aux_loss:.6f} Acc: {epoch_acc:.4f}")
            logging.info(f"{phase} CLS Loss: {epoch_cls_loss:.4f} Aux Loss: {epoch_aux_loss:.6f} Weighted Aux Loss: {epoch_weighted_aux_loss:.6f} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                all_topk_indices = torch.cat(all_topk_indices, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                expert_usage = log_expert_stats(all_topk_indices, all_labels, epoch + 1, NUM_CLASSES, class_names)
                expert_usage_history.append(expert_usage)

        epochs.append(epoch + 1)

        if metrics['val'] > best_val_acc + 1e-6:
            best_val_acc = metrics['val']
            best_model_wts = copy.deepcopy(model_.state_dict())
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model_.state_dict(),
                'optimizer_state_dict': optimizer_.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names
            }, save_path)
            print(f"✓ val accuracy improved, model saved to {save_path}")
            logging.info(f"✓ val accuracy improved, model saved to {save_path}")

        print(f"Current learning rate: {[g['lr'] for g in optimizer_.param_groups]}")
        logging.info(f"Current learning rate: {[g['lr'] for g in optimizer_.param_groups]}")

        early_stopping(metrics['val'])
        if early_stopping.early_stop:
            print("Early stopping triggered")
            logging.info("Early stopping triggered")
            break

    clear_output(wait=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(epochs, train_losses, label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and validation loss')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, train_accuracy, label='Train Acc')
    axes[0, 1].plot(epochs, val_accuracy, label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and validation accuracy')
    axes[0, 1].legend()

    expert_usage_arr = np.array(expert_usage_history)
    for i in range(NUM_EXPERTS):
        axes[1, 0].plot(epochs, expert_usage_arr[:, i], label=f'Expert {i}')
    ideal_usage = 100.0 / NUM_EXPERTS
    axes[1, 0].axhline(y=ideal_usage, color='r', linestyle='--', label='Ideal mean')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Usage (%)')
    axes[1, 0].set_title('Expert activation frequency')
    axes[1, 0].legend()

    axes[1, 1].text(0.5, 0.5, 'Router discriminability analysis\n(must be recorded in the training loop)',
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Router discriminability')

    plt.tight_layout()
    plt.savefig(f'save_csv_log/{MODEL_NAME}/{MODEL_NAME}_{NUM_EPOCHS}e_history.png')

    dump_history_to_csv(CSV_PATH, epochs, train_losses, train_accuracy,
                        val_losses, val_accuracy, expert_usage_history)

    time_elapsed = time.time() - since
    logging.info(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    logging.info(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logging.info(f"Best val accuracy: {best_val_acc:.4f}")

    model_.load_state_dict(best_model_wts)
    return model_, best_val_acc


if __name__ == "__main__":
    final_save_path = SAVE_PATH.replace('.pth', f'_final.pth')

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=(x == 'train'),
                                                  num_workers=NUM_WORKERS,
                                                  pin_memory=True if torch.cuda.is_available() else False)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Classes: {class_names}")
    print(f"Dataset size - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
    print(f"Device: {DEVICE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Experts: {NUM_EXPERTS}, TopK: {TOP_K}, λ: {AUX_LOSS_WEIGHT}")
    logging.info(f"Classes: {class_names}")
    logging.info(f"Dataset size - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Workers: {NUM_WORKERS}")
    logging.info(f"Experts: {NUM_EXPERTS}, TopK: {TOP_K}, λ: {AUX_LOSS_WEIGHT}")

    model = MoEModel(num_experts=NUM_EXPERTS,
                     num_classes=NUM_CLASSES,
                     top_k=TOP_K,
                     pretrained=PRETRAINED).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ReduceLROnPlateau_scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                                    patience=STEP_SIZE,
                                                    factor=GAMMA,
                                                    min_lr=MIN_LR,
                                                    threshold=EARLY_STOPPING_DELTA,
                                                    threshold_mode='abs')

    initial_best_val_acc = None
    if os.path.exists(final_save_path):
        load_path = final_save_path
    elif os.path.exists(SAVE_PATH):
        load_path = SAVE_PATH
    else:
        load_path = None

    if load_path:
        try:
            checkpoint = torch.load(load_path, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                initial_best_val_acc = checkpoint.get('best_val_acc', None)
                print(f"Loaded model weights: {load_path}")
                print(f"Previous best val accuracy: {initial_best_val_acc if initial_best_val_acc is not None else 'N/A'}")
                logging.info(f"Loaded model weights: {load_path}")
                logging.info(f"Previous best val accuracy: {initial_best_val_acc if initial_best_val_acc is not None else 'N/A'}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights: {load_path} (legacy format)")
                logging.info(f"Loaded model weights: {load_path} (legacy format)")
        except Exception as e:
            print(f"Failed to load ({load_path}): {e}, using random initialization")
            logging.info(f"Failed to load ({load_path}): {e}, using random initialization")

    model, best_val_acc = train_model(model, criterion, optimizer,
                                      ReduceLROnPlateau_scheduler,
                                      num_epochs=NUM_EPOCHS,
                                      initial_best_val_acc_=initial_best_val_acc,
                                      save_path=SAVE_PATH)

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'best_val_acc': best_val_acc
    }, final_save_path)

    print(f"\n✓ Final model saved to {final_save_path}")
    logging.info(f"\n✓ Final model saved to {final_save_path}")