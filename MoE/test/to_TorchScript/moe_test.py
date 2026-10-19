"""
MoE TorchScript inference script - Raspberry Pi 4B
Hardware specs:
  CPU : Broadcom BCM2711, Cortex-A72 (ARMv8) 64-bit @ 1.5GHz, 4 cores
  RAM : 4GB LPDDR4-3200
  No GPU, CPU-only inference
Dependencies: pip install torch pillow torchvision scikit-learn matplotlib seaborn
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Parameter configuration
# ======================
INPUT_SIZE    = 224
DATA_DIR      = "../../../../../../dataset/5%_RSCD/test"          # Data directory in ImageFolder structure
MODEL_PT_PATH = "moe_pi_optimized.pt"
RESULT_DIR    = "results"
BATCH_SIZE    = 1
NUM_WORKERS   = 4
NUM_EXPERTS   = 5
TOP_K         = 4
NUM_CLASSES   = 27

RPI4B_DESC = "Raspberry Pi 4B | BCM2711 Cortex-A72 @ 1.5GHz x4 | 4GB LPDDR4-3200 | CPU-only"

# Limit thread count to the number of physical cores
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")
os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)


# ======================
# Data preprocessing
# ======================
data_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ======================
# Inference function
# ======================
def run_inference(model, loader, class_names):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    expert_count = [0] * NUM_EXPERTS
    batch_times  = []
    total_samples = 0

    # ARM Cortex-A72 cold-start warm-up
    print("\nCPU warm-up (Cortex-A72 @ 1.5GHz)...")
    dummy = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    print("\nStarting inference...")
    t_start = time.time()

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Inference progress', ncols=80):
            t_batch = time.time()
            outputs = model(inputs)
            batch_times.append(time.time() - t_batch)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Run router again to get topk_indices for expert statistics
            # (TorchScript forward returns only logits, expert statistics computed separately)
            total_samples += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_time = time.time() - t_start

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy        = np.mean(all_preds == all_labels)
    avg_batch_ms    = np.mean(batch_times) * 1000
    std_batch_ms    = np.std(batch_times)  * 1000
    avg_sample_ms   = total_time / total_samples * 1000
    throughput      = total_samples / total_time

    print(f"\n{'='*60}")
    print(f"Inference result statistics")
    print(f"{'='*60}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"\nInference time statistics:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Total samples:  {total_samples}")
    print(f"  Avg per batch:  {avg_batch_ms:.2f}ms ± {std_batch_ms:.2f}ms")
    print(f"  Avg per sample: {avg_sample_ms:.2f}ms")
    print(f"  Throughput:     {throughput:.2f} samples/s")
    print(f"  Processor:      {RPI4B_DESC}")
    print(f"{'='*60}")

    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4)
    print("\nClassification report:")
    print(report)

    # Save classification report
    with open(f"{RESULT_DIR}/report.txt", 'w', encoding='utf-8') as f:
        f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Avg per batch: {avg_batch_ms:.2f}ms ± {std_batch_ms:.2f}ms\n")
        f.write(f"Avg per sample: {avg_sample_ms:.2f}ms\n")
        f.write(f"Throughput: {throughput:.2f} samples/s\n")
        f.write(f"Processor: {RPI4B_DESC}\n\n")
        f.write(report)

    # Save JSON
    results = {
        "overall_accuracy": float(accuracy),
        "hardware": {
            "device": "Raspberry Pi 4B",
            "soc": "Broadcom BCM2711",
            "cpu": "Cortex-A72 (ARMv8) 64-bit @ 1.5GHz",
            "cpu_cores": 4,
            "ram": "4GB LPDDR4-3200",
            "backend": "TorchScript CPU",
            "torch_threads": torch.get_num_threads(),
        },
        "inference_time": {
            "total_seconds": float(total_time),
            "total_samples": int(total_samples),
            "avg_batch_ms": float(avg_batch_ms),
            "std_batch_ms": float(std_batch_ms),
            "avg_sample_ms": float(avg_sample_ms),
            "throughput_samples_per_sec": float(throughput),
        },
        "num_samples": len(all_labels)
    }
    with open(f"{RESULT_DIR}/results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/confusion_matrix.png", dpi=300)
    plt.close()
    print(f"\nResults saved to {RESULT_DIR}/")

    return accuracy


# ======================
# Main flow
# ======================
if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"MoE TorchScript inference script - Raspberry Pi 4B")
    print(f"{'='*60}")
    print(f"Hardware specs: {RPI4B_DESC}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Model: {MODEL_PT_PATH}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print(f"{'='*60}")

    if not os.path.exists(MODEL_PT_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PT_PATH}")

    print("\nLoading TorchScript model...")
    model = torch.jit.load(MODEL_PT_PATH, map_location=DEVICE)
    model.eval()
    print("Model loaded successfully ✓")

    print("\nLoading data...")
    dataset = datasets.ImageFolder(DATA_DIR, data_transform)
    loader  = DataLoader(dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS,
                         pin_memory=False)
    class_names = dataset.classes

    print(f"Samples: {len(dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    run_inference(model, loader, class_names)