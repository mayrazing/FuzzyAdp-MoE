import csv
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]

SOURCE_TEST_SCRIPT = RUN_DIR / "test" / "moe_test2.py"
MODEL_PATH = SCRIPT_DIR / "efficientnet_b0_300e_224x224_final.pth"
TEST_DATA_DIR = REPO_ROOT / "dataset" / "5%_RSCD" / "test"
OUTPUT_DIR = SCRIPT_DIR / "test_results" / "per_class_expert_usage"

BATCH_SIZE = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)


def load_source_module():
    # The source script imports seaborn only for confusion-matrix plotting.
    # This analysis script does not call that function, so a stub avoids adding
    # a dependency just to reuse the model definition.
    sys.modules.setdefault("seaborn", types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location("moe_test2_source", SOURCE_TEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(module, device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

    model = module.MoEModel(
        num_experts=module.NUM_EXPERTS,
        num_classes=module.NUM_CLASSES,
        top_k=module.TOP_K,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        class_names = checkpoint.get("class_names")
        best_val_acc = checkpoint.get("best_val_acc")
    else:
        model.load_state_dict(checkpoint)
        class_names = None
        best_val_acc = None

    model.eval()
    return model, class_names, best_val_acc


def collect_usage(module, model, data_loader, device):
    num_classes = module.NUM_CLASSES
    num_experts = module.NUM_EXPERTS
    top_k = module.TOP_K

    expert_counts = np.zeros((num_classes, num_experts), dtype=np.int64)
    sample_counts = np.zeros(num_classes, dtype=np.int64)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Collecting expert usage"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, _, topk_indices = model(inputs, training=False)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.numel()

            labels_np = labels.cpu().numpy()
            topk_np = topk_indices.cpu().numpy()

            for label, routed_experts in zip(labels_np, topk_np):
                sample_counts[label] += 1
                for expert_id in routed_experts[:top_k]:
                    expert_counts[label, expert_id] += 1

    denom = np.maximum(sample_counts[:, None] * top_k, 1)
    usage_percent = expert_counts / denom * 100.0
    accuracy = correct / total if total else 0.0

    return expert_counts, sample_counts, usage_percent, accuracy, total


def save_csv(path, class_names, sample_counts, matrix):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "sample_count"] + [f"expert_{i}" for i in range(matrix.shape[1])])
        for class_name, sample_count, row in zip(class_names, sample_counts, matrix):
            writer.writerow([class_name, int(sample_count)] + [f"{value:.6f}" for value in row])


def plot_heatmap(path, class_names, usage_percent):
    fig, ax = plt.subplots(figsize=(8, 12))
    image = ax.imshow(usage_percent, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Usage (%)", labelpad=-5)

    ax.set_xticks(np.arange(usage_percent.shape[1]))
    ax.set_xticklabels([f"Expert {i}" for i in range(usage_percent.shape[1])])
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)

    for row in range(usage_percent.shape[0]):
        for col in range(usage_percent.shape[1]):
            value = usage_percent[row, col]
            text_color = "white" if value >= 50 else "black"
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title("")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Class")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    module = load_source_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    dataset = datasets.ImageFolder(TEST_DATA_DIR, module.test_transforms)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model, checkpoint_class_names, best_val_acc = load_model(module, device)
    class_names = checkpoint_class_names or dataset.classes

    if list(class_names) != list(dataset.classes):
        raise ValueError("Checkpoint class_names do not match dataset class order.")

    started = time.time()
    expert_counts, sample_counts, usage_percent, accuracy, total = collect_usage(
        module, model, data_loader, device
    )
    elapsed = time.time() - started

    csv_path = OUTPUT_DIR / "per_class_expert_usage_percent.csv"
    counts_csv_path = OUTPUT_DIR / "per_class_expert_usage_counts.csv"
    json_path = OUTPUT_DIR / "per_class_expert_usage.json"
    heatmap_path = OUTPUT_DIR / "per_class_expert_usage_heatmap.png"

    save_csv(csv_path, class_names, sample_counts, usage_percent)
    save_csv(counts_csv_path, class_names, sample_counts, expert_counts)
    plot_heatmap(heatmap_path, class_names, usage_percent)

    overall_counts = expert_counts.sum(axis=0)
    overall_usage = overall_counts / max(overall_counts.sum(), 1) * 100.0
    row_sums = usage_percent.sum(axis=1)

    results = {
        "split": "test",
        "dataset": str(TEST_DATA_DIR),
        "model_path": str(MODEL_PATH),
        "num_samples": int(total),
        "accuracy": float(accuracy),
        "best_val_acc_from_checkpoint": None if best_val_acc is None else float(best_val_acc),
        "num_experts": int(module.NUM_EXPERTS),
        "top_k": int(module.TOP_K),
        "usage_definition": "Hard Top-K activations per true class; each row normalized by class sample count * top_k.",
        "elapsed_seconds": float(elapsed),
        "overall_expert_usage_percent": {
            f"expert_{i}": float(value) for i, value in enumerate(overall_usage)
        },
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "outputs": {
            "heatmap_png": str(heatmap_path),
            "usage_percent_csv": str(csv_path),
            "usage_counts_csv": str(counts_csv_path),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Samples: {total}")
    print(f"Elapsed: {elapsed:.2f}s")
    print("Overall expert usage:")
    for i, value in enumerate(overall_usage):
        print(f"  Expert {i}: {value:.2f}% ({overall_counts[i]}/{overall_counts.sum()})")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
