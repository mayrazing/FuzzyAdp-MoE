import os
import subprocess
import re
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from datetime import datetime

# ======================
# Configuration - power consumption test (Jetson boards)
# Purpose: run continuous inference on Jetson, sample power in real time via tegrastats, and compute average power
# ======================
INPUT_SIZE    = 224
TEST_DATA_DIR = "5%_RSCD/test"
MODEL_PATH    = "moe_pi_optimized.pt"

# Only change this line when switching boards; the log file name follows automatically
# Options: Jetson_Orin_Nano / Jetson_AGX_Orin / Jetson_Thor / Jetson_Xavier_NX
DEVICE_NAME = "Jetson_AGX_Orin"
LOG_PATH    = f"power_consumption_log_{DEVICE_NAME}.txt"

BATCH_SIZE  = 1   # Exported model forward hardcodes BS=1, must be 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEARTBEAT_INTERVAL_SEC = 10  # Write a heartbeat log every 10 seconds
MAX_ROUNDS  = 10  # Stop automatically and print a summary after this many rounds; None loops forever


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


test_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def log(msg):
    """Print to terminal and append to the log file (fsync immediately each time to avoid loss on power failure)"""
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jetson_power_mw():
    """
    Read the current total Jetson power (in mW) via tegrastats --once.
    Works on Orin Nano / AGX Orin / Thor / Xavier NX, no per-board path maintenance needed.
    """
    result = subprocess.run(
        "tegrastats | head -n 1",
        shell=True, capture_output=True, text=True, timeout=5
    )
    matches = re.findall(r'[A-Z0-9_]+ (\d+)mW/\d+mW', result.stdout)
    if not matches:
        return 0
    return sum(int(v) for v in matches)


# ======================
# Main flow
# ======================
if __name__ == "__main__":
    log(f"{'=' * 60}")
    log(f"MoE TorchScript power consumption test (Jetson edition)")
    log(f"{'=' * 60}")
    log(f"Hardware: {DEVICE_NAME}")
    log(f"Inference device: {DEVICE}")
    log(f"Model path: {MODEL_PATH}")
    log(f"Test data path: {TEST_DATA_DIR}")
    log(f"Log path: {LOG_PATH}")
    log(f"Heartbeat interval: {HEARTBEAT_INTERVAL_SEC}s")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'=' * 60}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    log("Loading TorchScript model...")
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    log("Model loaded successfully")

    log("Loading test data...")
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, test_transforms)
    test_loader  = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=(DEVICE.type == "cuda"))
    log(f"Test samples: {len(test_dataset)}")

    log("\nGPU warm-up in progress...")
    dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input)
    log("GPU warm-up done, starting continuous inference...\n")

    global_start = time.time()
    last_heartbeat = global_start
    round_num = 0
    total_images = 0
    total_infer_time = 0.0  # Accumulated pure model inference time (s), excluding data loading
    power_readings = []

    with torch.no_grad():
        while MAX_ROUNDS is None or round_num < MAX_ROUNDS:
            round_num += 1
            round_start = time.time()

            for inputs, _ in test_loader:
                inputs = inputs.to(DEVICE)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(inputs)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                total_infer_time += time.perf_counter() - t0
                total_images += inputs.size(0)

                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
                    elapsed = now - global_start
                    elapsed_h = elapsed / 3600
                    pw = read_jetson_power_mw()
                    power_readings.append(pw)
                    avg_pw = sum(power_readings) / len(power_readings)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log(f"[{timestamp}] heartbeat | Round {round_num:04d} | "
                        f"Elapsed {elapsed:.0f}s ({elapsed_h:.2f}h) | "
                        f"Images processed {total_images} | "
                        f"Current power {pw}mW ({pw/1000:.2f}W) | "
                        f"Running mean {avg_pw:.1f}mW ({avg_pw/1000:.2f}W)")
                    last_heartbeat = now

            round_time = time.time() - round_start
            elapsed    = time.time() - global_start
            elapsed_h  = elapsed / 3600
            timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log(f"[{timestamp}] Round {round_num:04d} | Round duration {round_time:.1f}s | "
                f"Elapsed {elapsed:.0f}s ({elapsed_h:.2f}h)")

    # Summary
    total_elapsed = time.time() - global_start
    avg_pw = sum(power_readings) / len(power_readings) if power_readings else 0
    avg_infer_ms = total_infer_time / total_images * 1000 if total_images else 0
    system_energy_per_image_mj = avg_pw * (total_elapsed / total_images) if total_images else 0
    log(f"\n{'=' * 60}")
    log(f"Test complete")
    log(f"Total rounds:         {round_num}")
    log(f"Total images:         {total_images}")
    log(f"Total runtime:        {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    log(f"Average power:        {avg_pw:.1f}mW ({avg_pw/1000:.2f}W)")
    log(f"Per-image infer time: {avg_infer_ms:.2f}ms")
    log(f"Per-image energy:     {system_energy_per_image_mj:.3f}mJ")
    log(f"{'=' * 60}")
