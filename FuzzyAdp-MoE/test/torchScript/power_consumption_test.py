import os
import subprocess
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from datetime import datetime

# ======================
# Configuration - power consumption endurance test
# Purpose: run continuous inference on the Raspberry Pi until the power bank is drained, and estimate power from the total runtime recorded in the log
# ======================
INPUT_SIZE    = 224
TEST_DATA_DIR = "../dataset/5%_RSCD/test"
MODEL_PATH    = "../moe_pi_optimized.pt"

# Power bank parameters (fill in according to the actual power bank used)
# This power bank has a cell capacity of 20000mAh / 74Wh and a rated 5V USB output capacity of 12000mAh.
# Power estimation uses the 5V output-side energy: average power (W) = 5V output energy (Wh) / actual runtime (h).
# To compute the final average power after power loss, read "Elapsed Xs" from the last heartbeat in the log:
# Final runtime (h) = X / 3600
# Final average power (W) = 5V output energy (Wh) / final runtime (h)
# Equivalent form: final average power (W) = 5V output energy (Wh) * 3600 / X
POWER_BANK_CELL_MAH = 20000
POWER_BANK_CELL_ENERGY_WH = 74.0
POWER_BANK_RATED_OUTPUT_MAH = 12000
POWER_BANK_OUTPUT_VOLTAGE = 5.0
POWER_BANK_OUTPUT_ENERGY_WH = (
    POWER_BANK_RATED_OUTPUT_MAH / 1000 * POWER_BANK_OUTPUT_VOLTAGE
)
POWER_BANK_MIN_CONVERSION_EFFICIENCY = 0.75

BATCH_SIZE  = 1   # Exported model forward hardcodes BS=1, must be 1
NUM_WORKERS = 0   # On RPi4B multiple workers compete with inference threads for cores; serial is more stable
DEVICE = torch.device("cpu")
HEARTBEAT_INTERVAL_SEC = 10  # Write a heartbeat log every 10 seconds to reduce timing error from power loss

# Limit PyTorch threads to the number of physical cores to avoid hyper-threading contention
torch.set_num_threads(4)          # Cortex-A72, 4 cores
torch.set_num_interop_threads(1)  # No inter-op parallelism needed during inference

# Switching devices: use "RPI4B 4GB" on Raspberry Pi 4, "RPI5 16GB" on Raspberry Pi 5
DEVICE_NAME = "RPI4B 4GB"
LOG_PATH    = f"power_consumption_log_{DEVICE_NAME.replace(' ', '_')}.txt"


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


# ======================
# Main flow
# ======================
if __name__ == "__main__":
    log(f"{'=' * 60}")
    log(f"MoE TorchScript power consumption endurance test")
    log(f"{'=' * 60}")
    log(f"Hardware: {DEVICE_NAME}")
    log(f"Model path: {MODEL_PATH}")
    log(f"Test data path: {TEST_DATA_DIR}")
    log(f"Log path: {LOG_PATH}")
    log(f"Power bank cell capacity: {POWER_BANK_CELL_MAH} mAh / {POWER_BANK_CELL_ENERGY_WH:.2f} Wh")
    log(f"Power bank rated output capacity: {POWER_BANK_RATED_OUTPUT_MAH} mAh @ {POWER_BANK_OUTPUT_VOLTAGE:.1f}V")
    log(f"5V output energy used for power estimation: {POWER_BANK_OUTPUT_ENERGY_WH:.2f} Wh")
    log(f"Nominal conversion efficiency: >= {POWER_BANK_MIN_CONVERSION_EFFICIENCY * 100:.0f}%")
    log("Final average power calculation: using Elapsed Xs from the last heartbeat, "
        f"power = {POWER_BANK_OUTPUT_ENERGY_WH:.2f} * 3600 / X W")
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
                              pin_memory=False)  # RPi4B has no GPU, pin_memory is pointless
    log(f"Test samples: {len(test_dataset)}")

    # CPU warm-up (runs only once at startup)
    log("\nCPU warm-up in progress...")
    dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input)
    log("CPU warm-up done, starting continuous inference...\n")

    global_start = time.time()
    last_heartbeat = global_start
    round_num = 0
    total_images = 0

    with torch.no_grad():
        while True:
            round_num += 1
            # Drop the OS file cache before each round to simulate a camera delivering new images each time
            subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                           check=True)
            round_start = time.time()

            for inputs, _ in test_loader:
                model(inputs)
                total_images += inputs.size(0)

                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
                    elapsed = now - global_start
                    elapsed_h = elapsed / 3600
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log(f"[{timestamp}] heartbeat | Round {round_num:04d} | "
                        f"Elapsed {elapsed:.0f}s ({elapsed_h:.2f}h) | "
                        f"Images processed {total_images}")
                    last_heartbeat = now

            round_time  = time.time() - round_start
            elapsed     = time.time() - global_start
            elapsed_h   = elapsed / 3600
            timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log(f"[{timestamp}] Round {round_num:04d} | Round duration {round_time:.1f}s | "
                f"Elapsed {elapsed:.0f}s ({elapsed_h:.2f}h)")
