import os
import subprocess
import re
import time
from glob import glob
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
TEST_DATA_DIR = "../5%_RSCD/test"
MODEL_PATH    = "../moe_pi_optimized.pt"

# Only change this line when switching boards; the log file name follows automatically
# Options: Jetson_Orin_Nano / Jetson_AGX_Orin / Jetson_Thor / Jetson_Xavier_NX
DEVICE_NAME = "Jetson_Xavier_NX"
LOG_PATH    = f"power_consumption_log_{DEVICE_NAME}.txt"

BATCH_SIZE  = 1   # Exported model forward hardcodes BS=1, must be 1
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEARTBEAT_INTERVAL_SEC = 10  # Write a heartbeat log every 10 seconds
MAX_ROUNDS  = 3  # Stop automatically and print a summary after this many rounds; None loops forever

TOTAL_POWER_RAILS = (
    "VDD_IN",
    "POM_5V_IN",
)
_power_warning_logged = False
_power_source_logged = False

SYSFS_POWER_PATTERNS = (
    "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input",
    "/sys/bus/i2c/devices/*/iio:device*/in_power*_input",
    "/sys/bus/iio/devices/iio:device*/in_power*_input",
    "/sys/devices/platform/*/iio:device*/in_power*_input",
    "/sys/class/hwmon/hwmon*/power*_input",
)


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


def _read_int_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(float(f.read().strip()))
    except (OSError, ValueError):
        return None


def _read_label_for_power_file(path):
    label_candidates = []

    match = re.search(r"(in_power\d+)_input$", path)
    if match:
        prefix = match.group(1)
        label_candidates.extend((
            path.replace(f"{prefix}_input", f"{prefix}_label"),
            path.replace(f"{prefix}_input", f"{prefix}_name"),
        ))

    match = re.search(r"(power\d+)_input$", path)
    if match:
        prefix = match.group(1)
        label_candidates.extend((
            path.replace(f"{prefix}_input", f"{prefix}_label"),
            path.replace(f"{prefix}_input", f"{prefix}_name"),
        ))

    for label_path in label_candidates:
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                label = f.read().strip()
                if label:
                    return label
        except OSError:
            pass

    return ""


def _normalize_power_label(label):
    return re.sub(r"[^A-Z0-9_]+", "_", label.upper()).strip("_")


def _read_sysfs_power_mw():
    """
    Read power from Jetson INA3221/sysfs. Linux power*_input/in_power*_input values
    are usually in microwatts (uW); converted to mW here.
    """
    rails = {}
    unlabeled_values_mw = []

    for pattern in SYSFS_POWER_PATTERNS:
        for path in glob(pattern):
            raw_uw = _read_int_file(path)
            if raw_uw is None or raw_uw <= 0:
                continue

            value_mw = int(round(raw_uw / 1000.0))
            label = _normalize_power_label(_read_label_for_power_file(path))
            if label:
                rails[label] = value_mw
            else:
                unlabeled_values_mw.append(value_mw)

    for rail_name in TOTAL_POWER_RAILS:
        if rail_name in rails:
            return rails[rail_name], f"sysfs:{rail_name}"

    if rails:
        return sum(rails.values()), "sysfs:sum"

    # On Xavier NX without labels, the whole-board input power is usually the largest channel.
    if unlabeled_values_mw:
        return max(unlabeled_values_mw), "sysfs:max-unlabeled"

    return None, ""


def _parse_tegrastats_power_mw(output):
    rails = {}
    for name, current, _avg in re.findall(
        r'\b([A-Z][A-Z0-9_]*)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:mW)?/([0-9]+(?:\.[0-9]+)?)\s*(?:mW)?\b',
        output
    ):
        if name.startswith(("VDD", "POM")):
            rails[name] = int(float(current))

    for rail_name in TOTAL_POWER_RAILS:
        if rail_name in rails:
            return rails[rail_name]

    if rails:
        return sum(rails.values())

    return None


def _read_tegrastats_line(command):
    proc = None
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return proc.stdout.readline().strip() if proc.stdout else ""
    except (subprocess.SubprocessError, OSError):
        return ""
    finally:
        if proc is not None:
            try:
                proc.terminate()
            except PermissionError:
                return ""
            try:
                proc.wait(timeout=2)
            except subprocess.SubprocessError:
                try:
                    proc.kill()
                except PermissionError:
                    pass


def _read_sudo_tegrastats_line():
    """
    tegrastats started via sudo may belong to root, so plain Python cannot kill it.
    Use timeout so the sudo side exits on its own, avoiding PermissionError.
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "timeout", "2", "tegrastats"],
            capture_output=True,
            text=True,
            timeout=4
        )
    except (subprocess.SubprocessError, OSError):
        return ""

    lines = (result.stdout or "").splitlines()
    return lines[0].strip() if lines else ""


def read_jetson_power_mw():
    """
    Read the current total Jetson power (in mW).
    Prefer INA3221/sysfs, since tegrastats does not output power rails without sudo.
    Fall back to tegrastats, preferring the total input power rail.
    """
    global _power_warning_logged, _power_source_logged

    sysfs_power_mw, source = _read_sysfs_power_mw()
    if sysfs_power_mw is not None:
        if not _power_source_logged:
            log(f"[INFO] Power reading source: {source}")
            _power_source_logged = True
        return sysfs_power_mw

    last_output = ""
    for command, source in (
        (["tegrastats"], "tegrastats"),
    ):
        output = _read_tegrastats_line(command)
        if output:
            last_output = output

        tegrastats_power_mw = _parse_tegrastats_power_mw(output)
        if tegrastats_power_mw is not None:
            if not _power_source_logged:
                log(f"[INFO] Power reading source: {source}")
                _power_source_logged = True
            return tegrastats_power_mw

    output = _read_sudo_tegrastats_line()
    if output:
        last_output = output

    tegrastats_power_mw = _parse_tegrastats_power_mw(output)
    if tegrastats_power_mw is not None:
        if not _power_source_logged:
            log("[INFO] Power reading source: sudo tegrastats")
            _power_source_logged = True
        return tegrastats_power_mw

    if not _power_warning_logged:
        log(f"[WARN] Failed to parse power from tegrastats, raw output: {last_output or '<empty>'}")
        _power_warning_logged = True
    return None


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
                    if pw is not None:
                        power_readings.append(pw)
                    avg_pw = sum(power_readings) / len(power_readings) if power_readings else 0
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pw_text = f"{pw}mW ({pw/1000:.2f}W)" if pw is not None else "N/A"
                    log(f"[{timestamp}] heartbeat | Round {round_num:04d} | "
                        f"Elapsed {elapsed:.0f}s ({elapsed_h:.2f}h) | "
                        f"Images processed {total_images} | "
                        f"Current power {pw_text} | "
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
