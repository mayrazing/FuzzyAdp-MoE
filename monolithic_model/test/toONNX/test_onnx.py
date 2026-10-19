import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json

# ======================
# Configuration - Raspberry Pi 4B 4GB
# ======================
INPUT_SIZE = 224
NUM_CLASSES = 27
BATCH_SIZE = 4
ONNX_PATH = "efficientnet_b0.onnx"
TEST_DATA_DIR = "../../../../../../dataset/5%_RSCD/test"
RESULT_DIR = "test_results/efficientnet_b0_onnx_rpi4b"
NUM_WORKERS = 0

RPI4B_DESC = "Raspberry Pi 4B | BCM2711 Cortex-A72 @ 1.5GHz x4 | 4GB LPDDR4-3200 | ONNX Runtime CPU"

os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


# ======================
# Data preprocessing
# ======================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)  # HWC → CHW

def load_dataset(root):
    root = Path(root)
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples = []
    for cls in class_names:
        for img_path in sorted((root / cls).glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append((str(img_path), class_to_idx[cls]))
    return samples, class_names


def load_model(onnx_path):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_path,
        sess_options,
        providers=["CPUExecutionProvider"]
    )
    print(f"ONNX model loaded successfully: {onnx_path}")
    return session


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_model(session, test_loader, class_names):
    all_preds = []
    all_labels = []
    all_probs = []
    batch_times = []
    total_samples = 0

    input_name = session.get_inputs()[0].name

    # Warm-up
    print("\nONNX Runtime warm-up...")
    dummy = np.random.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    for _ in range(5):
        session.run(None, {input_name: dummy})

    print("\nStarting test...")
    inference_start = time.time()

    pbar = tqdm(range(0, len(samples), BATCH_SIZE), desc='Testing', ncols=100)
    for i in pbar:
        batch = samples[i:i + BATCH_SIZE]
        inputs_np = np.stack([preprocess(p) for p, _ in batch])
        labels_np = np.array([l for _, l in batch])

        batch_start = time.time()
        outputs = session.run(None, {input_name: inputs_np})[0]
        batch_times.append(time.time() - batch_start)

        probs = softmax(outputs)
        preds = outputs.argmax(axis=1)

        total_samples += len(labels_np)
        all_preds.extend(preds)
        all_labels.extend(labels_np)
        all_probs.extend(probs)

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

    # Save text report
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

    # Save JSON
    results_json = {
        "overall_accuracy": float(accuracy),
        "hardware": {
            "device": "Raspberry Pi 4B",
            "soc": "Broadcom BCM2711",
            "cpu": "Cortex-A72 (ARMv8) 64-bit @ 1.5GHz",
            "cpu_cores": 4,
            "ram": "4GB LPDDR4-3200",
            "inference_backend": "ONNX Runtime CPU",
            "ort_version": ort.__version__,
            "intra_op_threads": 4,
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
    print(f"EfficientNet-B0 ONNX test script - Raspberry Pi 4B")
    print(f"{'=' * 60}")
    print(f"Hardware specs: {RPI4B_DESC}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Model path: {ONNX_PATH}")
    print(f"Test data path: {TEST_DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'=' * 60}")

    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")
    if not os.path.exists(TEST_DATA_DIR):
        raise FileNotFoundError(f"Test data directory not found: {TEST_DATA_DIR}")

    session = load_model(ONNX_PATH)

    samples, class_names = load_dataset(TEST_DATA_DIR)
    print(f"Classes: {class_names}")
    print(f"Test set samples: {len(samples)}")

    preds, labels, probs, accuracy = test_model(session, samples, class_names)

    confusion_matrix_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, class_names, confusion_matrix_path)

    image_paths = [p for p, _ in samples]
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