import copy
import os
import time
import logging
import random

import numpy as np
import timm
from torch import nn, cuda, autocast, GradScaler, torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pathlib import Path
import pandas as pd
from tqdm import tqdm  # Added import

# ======================
# Parameter configuration
# ======================
MODEL_NAME = "efficientnet_b0"
INPUT_SIZE = 224
NUM_EPOCHS = 300
DATA_DIR = "../../../../dataset/5%_RSCD"
# Early stopping parameters
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_DELTA = 0.001

BATCH_SIZE = 64
NUM_CLASSES = 27
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
                print(f'Early stopping: Initial validation accuracy: {score:.4f}')
                logging.info(f'Early stopping: Initial validation accuracy: {score:.4f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter} out of {self.patience}')
                logging.info(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'Early stopping: Validation accuracy improved to {score:.4f}')
                logging.info(f'Early stopping: Validation accuracy improved to {score:.4f}')


def dump_history_to_csv(csv_path, epochs, train_losses, train_accuracy, val_losses, val_accuracy):
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "val_loss": val_losses,
        "val_acc": val_accuracy,
    })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.10),  # Add color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Add translation and scaling
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ToTensor(),  # Convert to Tensor
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet standard values
    ]),
    'val': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED)

model.reset_classifier(NUM_CLASSES)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
ReduceLROnPlateau_scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                                patience=STEP_SIZE,
                                                factor=GAMMA,
                                                min_lr=MIN_LR,
                                                threshold=EARLY_STOPPING_DELTA,
                                                threshold_mode='abs')
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
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

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(dataloaders[phase], desc=f'{phase} processing', ncols=100)

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer_.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast("cuda" if cuda.is_available() else "cpu"):
                        outputs = model_(inputs)
                        loss = criterion_(outputs, labels)

                    if phase == 'train':
                        amp_scaler.scale(loss).backward()  # Backward pass with scaled gradients
                        amp_scaler.step(optimizer)  # Optimizer updates parameters
                        amp_scaler.update()  # Update the scaler

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            epoch_loss = (running_loss / dataset_sizes[phase])
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            metrics[phase] = epoch_acc

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                train_losses.append(epoch_loss)
                train_accuracy.append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                scheduler_.step(epoch_acc)
                val_losses.append(epoch_loss)
                val_accuracy.append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

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
            print(f"✓ Validation improved. Saved model to {save_path}")
            logging.info(f"✓ Validation improved. Saved model to {save_path}")

        print("Current learning rates:", [g['lr'] for g in optimizer_.param_groups])
        logging.info(f"Current learning rates: {[g['lr'] for g in optimizer_.param_groups]}")

        early_stopping(metrics['val'])
        if early_stopping.early_stop:
            print("Early stopping triggered")
            logging.info("Early stopping triggered")
            break

    clear_output(wait=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Train Acc')
    plt.plot(epochs, val_accuracy, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'save_csv_log/{MODEL_NAME}/{MODEL_NAME}_{NUM_EPOCHS}e_history.png')
    dump_history_to_csv(CSV_PATH, epochs, train_losses, train_accuracy, val_losses, val_accuracy)

    time_elapsed = time.time() - since
    logging.info(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_val_acc:.4f}")
    logging.info(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logging.info(f"Best val Acc: {best_val_acc:.4f}")

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
    print(
        f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
    print(f"Device: {DEVICE}")
    print(f"Number of workers: {NUM_WORKERS}")
    logging.info(f"Classes: {class_names}")
    logging.info(
        f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Number of workers: {NUM_WORKERS}")

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
                print(f"Loaded model weights from {load_path}")
                logging.info(f"Loaded model weights from {load_path}")
                print(f"Previous best validation accuracy: {initial_best_val_acc if initial_best_val_acc is not None else 'N/A'}")
                logging.info(f"Previous best validation accuracy: {initial_best_val_acc if initial_best_val_acc is not None else 'N/A'}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights from {load_path} (legacy format)")
                logging.info(f"Loaded model weights from {load_path} (legacy format)")
        except Exception as e:
            print(f"load error ({load_path}):{e},random init.")
            logging.info(f"load error ({load_path}):{e},random init.")
        else:
            print(f"Loaded weights from {load_path}")
            logging.info(f"Loaded weights from {load_path}")

    model, best_val_acc = train_model(model, criterion, optimizer, ReduceLROnPlateau_scheduler,
                                      num_epochs=NUM_EPOCHS,
                                      initial_best_val_acc_=initial_best_val_acc,
                                      save_path=SAVE_PATH)

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'best_val_acc': best_val_acc
    }, final_save_path)

    print(f"\n✓ Saved final model to {final_save_path}")
    logging.info(f"\n✓ Saved final model to {final_save_path}")