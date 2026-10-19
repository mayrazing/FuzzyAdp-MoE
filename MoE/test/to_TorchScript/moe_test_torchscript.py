import os
import time
import torch
import torch.nn.functional as F
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

# ======================
# Parameter configuration - Raspberry Pi 4B 4GB real hardware simulation
# Hardware specs:
#   CPU : Broadcom BCM2711, Cortex-A72 (ARMv8) 64-bit @ 1.5GHz, 4 cores
#   RAM : 4GB LPDDR4-3200
#   No GPU acceleration, CPU-only inference
# ======================
INPUT_SIZE    = 224
NUM_CLASSES   = 27
NUM_EXPERTS   = 5
TOP_K         = 2
TEST_DATA_DIR = "../../../../../../dataset/5%_RSCD/test"
MODEL_PATH    = "moe_pi_optimized.pt"
RESULT_SAVE_DIR = "test_results/moe_rpi4b"

BATCH_SIZE  = 1   # Exported model forward hardcodes BS=1, must be 1
NUM_WORKERS = 0   # On RPi4B multiple workers compete with inference threads for cores; serial is more stable
DEVICE = torch.device("cpu")

# Limit PyTorch threads to the number of physical cores to avoid hyper-threading contention
torch.set_num_threads(4)          # Cortex-A72, 4 cores
torch.set_num_interop_threads(1)  # No inter-op parallelism needed during inference

# Raspberry Pi 4B hardware description string (for logs/reports)
RPI4B_DESC = "Raspberry Pi 4B | BCM2711 Cortex-A72 @ 1.5GHz x4 | 4GB LPDDR4-3200 | CPU-only"

os.makedirs(RESULT_SAVE_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


# ======================
# Data preprocessing
# ======================
test_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ======================
# Test function
# ======================
def test_model(model, test_loader, class_names):
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    batch_times   = []
    total_samples = 0

    # Raspberry Pi 4B CPU warm-up: ARM Cortex-A72 cold start is slow, so increase warm-up iterations
    print("\nCPU warm-up (Cortex-A72 @ 1.5GHz)...")
    dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        for _ in range(5):  # RPi4B ARM cores need more warm-up iterations
            model(dummy_input)

    print("\nStarting test...")
    inference_start = time.time()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Test progress'):
            batch_start = time.time()

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            batch_times.append(time.time() - batch_start)
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_inference_time = time.time() - inference_start

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    accuracy        = np.mean(all_preds == all_labels)
    avg_batch_time  = np.mean(batch_times)
    std_batch_time  = np.std(batch_times)
    avg_sample_time = total_inference_time / total_samples
    throughput      = total_samples / total_inference_time

    print(f"\n{'=' * 60}")
    print(f"Test result statistics")
    print(f"{'=' * 60}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"\nInference time statistics:")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Total samples: {total_samples}")
    print(f"  Average time per batch: {avg_batch_time * 1000:.2f}ms ± {std_batch_time * 1000:.2f}ms")
    print(f"  Average time per sample: {avg_sample_time * 1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} samples/s")
    print(f"  Processor: {RPI4B_DESC}")
    print(f"{'=' * 60}")

    print("\nClassification report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    with open(f"{RESULT_SAVE_DIR}/classification_report.txt", 'w', encoding='utf-8') as f:
        f.write(f"Overall accuracy: {accuracy:.4f}\n")
        f.write(f"\nInference time statistics:\n")
        f.write(f"  Total inference time: {total_inference_time:.2f}s\n")
        f.write(f"  Total samples: {total_samples}\n")
        f.write(f"  Average time per batch: {avg_batch_time * 1000:.2f}ms ± {std_batch_time * 1000:.2f}ms\n")
        f.write(f"  Average time per sample: {avg_sample_time * 1000:.2f}ms\n")
        f.write(f"  Throughput: {throughput:.2f} samples/s\n")
        f.write(f"  Processor: {RPI4B_DESC}\n")
        f.write(f"\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    results = {
        "overall_accuracy": float(accuracy),
        "hardware": {
            "device": "Raspberry Pi 4B",
            "soc": "Broadcom BCM2711",
            "cpu": "Cortex-A72 (ARMv8) 64-bit @ 1.5GHz",
            "cpu_cores": 4,
            "ram": "4GB LPDDR4-3200",
            "inference_backend": "TorchScript CPU",
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

    with open(f"{RESULT_SAVE_DIR}/test_results.json", 'w', encoding='utf-8') as f:
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
    print(f"MoE TorchScript test script - Raspberry Pi 4B 4GB real hardware simulation")
    print(f"{'=' * 60}")
    print(f"Device: {DEVICE}")
    print(f"Hardware specs: {RPI4B_DESC}")
    print(f"PyTorch threads: {torch.get_num_threads()} (corresponds to 4-core Cortex-A72)")
    print(f"Model path: {MODEL_PATH}")
    print(f"Test data path: {TEST_DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS} (serial loading, avoids competing with inference threads for cores)")
    print(f"{'=' * 60}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print("\nLoading TorchScript model...")
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print("Model loaded successfully ✓")

    print("\nLoading test data...")
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, test_transforms)
    test_loader  = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=False)  # RPi4B has no GPU, pin_memory is pointless
    class_names = test_dataset.classes

    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    accuracy, preds, labels, probs = test_model(model, test_loader, class_names)

    print("\nTest complete!")
    print(f"{'=' * 60}")