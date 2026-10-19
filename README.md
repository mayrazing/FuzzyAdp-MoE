# FuzzyAdp-MoE

A Mixture-of-Experts model with fuzzy-logic adaptive control (Fuzzy Adaptive Mixture-of-Experts) for road surface condition image classification (RSCD, 27 classes), with deployment, inference, and power-consumption tests on a Raspberry Pi 4B and several Jetson edge devices.

The model uses EfficientNet-B0 as the backbone: the first 6 stages serve as a shared feature extractor, and the remaining stages plus the classification head are replicated into 5 experts, with a convolutional Router performing Top-2 sparse routing. On top of this, FuzzyAdp-MoE runs a fuzzy inference system before each epoch that uses expert load balance and validation-accuracy trends to dynamically adjust the Router softmax temperature τ, the load-balancing auxiliary loss weight λ, and the Router noise scale.

## Repository structure

```
.
├── datasets/
│   ├── 5%_RSCD.zip                    # 5% per-class subset of RSCD (download from Releases)
│   └── slipt_data_by_percentage.py    # Samples from full RSCD and splits 80/10/10
├── monolithic_model/                  # Baseline 1: monolithic EfficientNet-B0
│   ├── single.py                      # Training script
│   ├── pth/                           # Trained weights
│   ├── save_csv_log/                  # Training logs, curve plotting script
│   └── test/                          # PyTorch / TorchScript / ONNX testing and export
├── MoE/                               # Baseline 2: MoE with fixed hyperparameters
│   ├── MoE.py
│   ├── pth/  save_csv_log/
│   └── test/                          # Includes to_TorchScript/, to_onnx/
└── FuzzyAdp-MoE/                      # Proposed method
    ├── fuzzy_adaptive_MoE.py          # Training script
    ├── pth/  save_csv_log/
    ├── test/
    │   ├── moe_test2.py               # PyTorch CPU test (Raspberry Pi 4B)
    │   └── torchScript/               # TorchScript export, inference test, per-device power tests
    └── test2/                         # Per-class expert usage heatmap
```

## Dataset

`5%_RSCD.zip` (available on the [Releases](https://github.com/mayrazing/FuzzyAdp-MoE/releases) page, as it exceeds GitHub's file size limit) is a subset obtained by randomly sampling 5% of each class from the full RSCD dataset (`random.seed(42)`) and splitting it 80/10/10:

| Split | Images |
| --- | --- |
| train | 41111 |
| val | 5128 |
| test | 5165 |

The 27 classes combine surface wetness, material, and unevenness:

- `dry` / `wet` / `water` × `asphalt` / `concrete` × `smooth` / `slight` / `severe` (18 classes)
- `dry` / `wet` / `water` × `gravel` / `mud` (6 classes)
- `fresh_snow`, `melted_snow`, `ice` (3 classes)

To regenerate the subset from the full RSCD, edit `SOURCE_DIR` and `TARGET_DIR` in [slipt_data_by_percentage.py](datasets/slipt_data_by_percentage.py) and run it.

## Method

### MoE architecture

| Component | Description |
| --- | --- |
| `SharedFeatureExtractor` | EfficientNet-B0 `conv_stem` + `bn1` + `blocks[:6]`, shared by all samples |
| `Router` | Two 3×3 convs + one 1×1 conv → GAP → LayerNorm; Gaussian noise is added to the logits during training |
| `ExpertNetwork` × 5 | Each expert owns `blocks[6:]` + `conv_head` + a new linear classifier, all initialized from ImageNet-pretrained weights |
| Routing | Top-2; the two selected weights are renormalized and the expert outputs are summed with these weights |
| Load-balancing loss | `aux = E · Σ_i f_i · p_i`, where `f_i` is the fraction of samples routed to expert i and `p_i` is its mean routing probability |

The training loss is `CE + λ · aux`; λ is multiplied by 0.1 during the first 20 epochs as a warm-up.

### Fuzzy adaptive controller

`adaptive_tau_controller` in [fuzzy_adaptive_MoE.py](FuzzyAdp-MoE/fuzzy_adaptive_MoE.py) runs once before each epoch, using the validation results of the previous epoch:

**Inputs**

- `B`: system balance, computed from the Gini coefficient and the minimum of the expert usage rates:
  `S_usage = 5 · min_usage`, `S_gini = 200 · (0.5 − Gini)`,
  `B = min(S_usage, S_gini) − (1/β) · ln(1 + exp(−β · |S_usage − S_gini|))`, β = 0.8.
  Fuzzy sets: Unstable / Normal / Perfect.
- `T_fast`, `T_slow`: short-term (window 3) and long-term (window 5) trends of validation accuracy, `T = (1/W) · Σ_{i=1..W} (acc_t − acc_{t−i})`.
  Fuzzy sets: Degrading / Stable / Improving.

**Outputs**

- `Δτ`: temperature increment; final `τ = clip(τ_prev + Δτ, 0.85, 1.02)`, initial τ = 0.90.
- `λ`: auxiliary loss weight, range 0.5–2.5.
- `noise`: standard deviation of Router noise, range 0–0.15.

**Rules** (6 in total)

| Condition | Δτ | λ | noise |
| --- | --- | --- | --- |
| B is Unstable or T_fast is Degrading | Increase | High | High |
| B is Normal and T_fast is Stable | Keep | Medium | High |
| B is Perfect and T_fast is Improving | Decrease | Low | Low |
| T_slow is Stable | Keep | Medium | High |
| B is Normal and T_fast is Improving | Decrease | Medium | High |
| B is Perfect and T_fast is Stable | Keep | Low | High |

τ only affects the routing softmax during training; no temperature is applied at inference.

## Requirements

```
torch  torchvision  timm  numpy  pandas  matplotlib  tqdm  ipython
scikit-fuzzy           # FuzzyAdp-MoE training
scikit-learn  seaborn  # Test scripts (classification report, confusion matrix)
onnx  onnxruntime      # ONNX export and testing
```

Training uses AMP (`torch.autocast` + `GradScaler`) and uses the GPU automatically when CUDA is available.

## Usage

### 1. Prepare the data

Download `5%_RSCD.zip` from [Releases](https://github.com/mayrazing/FuzzyAdp-MoE/releases) into `datasets/`, then:

```bash
cd datasets
unzip 5%_RSCD.zip
```

The data paths in the scripts are relative and not consistent with each other (e.g. the training scripts use `DATA_DIR = "../../../dataset/5%_RSCD"`, the test scripts use `TEST_DATA_DIR = ".../dataset/5%_RSCD/test"`). Before running a script, set `DATA_DIR` / `TEST_DATA_DIR` at the top of it to the actual extracted path.

### 2. Training

Run each script from its own directory; weights and logs are written to that directory's `pth/` and `save_csv_log/`:

```bash
cd monolithic_model && python single.py              # Monolithic EfficientNet-B0
cd MoE && python MoE.py                              # MoE with fixed hyperparameters
cd FuzzyAdp-MoE && python fuzzy_adaptive_MoE.py      # Fuzzy adaptive MoE
```

Main hyperparameters (identical across the three scripts): 224×224 input, batch size 64, AdamW (lr 1e-3, weight decay 1e-4), `ReduceLROnPlateau` (patience 5, factor 0.1), up to 300 epochs, early stopping patience 8. The MoE uses 5 experts, Top-2, base λ = 1.5.

If a weight file already exists under `pth/`, the script loads it and resumes training. After training it saves:

- `pth/efficientnet_b0/efficientnet_b0_300e_224x224.pth`: checkpoint with the best validation accuracy (includes optimizer state)
- `pth/efficientnet_b0/efficientnet_b0_300e_224x224_final.pth`: final model
- `save_csv_log/efficientnet_b0/`: training log, per-epoch metrics CSV, curves

### 3. Testing

The test scripts load `efficientnet_b0_300e_224x224_final.pth` from the current directory and report overall accuracy, a classification report, a confusion matrix, and inference time:

```bash
cd FuzzyAdp-MoE/test && python moe_test2.py
cd MoE/test && python moe_test2.py
cd monolithic_model/test && python test_single.py
```

Deployment scripts:

| Script | Purpose |
| --- | --- |
| `FuzzyAdp-MoE/test/torchScript/moe_pth_to_trochscript_g.py` | Export FuzzyAdp-MoE to TorchScript (`moe_pi_optimized.pt`) |
| `FuzzyAdp-MoE/test/torchScript/moe_test_torchscript.py` | TorchScript inference test |
| `FuzzyAdp-MoE/test/torchScript/<device>/power_consumption_test_*.py` | Continuous inference on Jetson with power sampled via `tegrastats` |
| `FuzzyAdp-MoE/test2/generate_per_class_expert_heatmap.py` | Per-class expert usage statistics and heatmap |
| `MoE/test/to_onnx/`, `MoE/test/to_TorchScript/` | ONNX / TorchScript export and testing for the MoE baseline |
| `monolithic_model/test/toONNX/`, `monolithic_model/test/torchscript/` | ONNX / TorchScript export and testing for the monolithic model |

## Results

### Test accuracy

The test split contains 5165 images; all results use PyTorch CPU inference:

| Model | Test accuracy | Result file |
| --- | --- | --- |
| EfficientNet-B0 (monolithic) | 86.49% | [test_results.json](monolithic_model/test/test_results/efficientnet_b0_rpi4b/test_results.json) |
| MoE (5 experts, Top-2) | 87.57% | [tresults.json](MoE/test/results0/efficientnet_b0_3/tresults.json) |
| **FuzzyAdp-MoE** | **88.02%** | [test_results.json](FuzzyAdp-MoE/test/test_results/efficientnet_b0_rpi4b/test_results.json) |

On the test set, FuzzyAdp-MoE's expert usage is 20.1% / 20.2% / 19.3% / 20.6% / 19.8%, close to the ideal mean of 20%.

Each result directory also contains a `classification_report.txt` and inference-time statistics.

### Edge-device power consumption (FuzzyAdp-MoE, TorchScript)

| Device | Inference device | Average power |
| --- | --- | --- |
| Jetson AGX Orin | CPU | 5.62 W |
| Jetson Orin Nano | CPU | 5.21 W |
| Jetson Orin Nano | GPU | 5.19 W |
| Jetson Xavier NX | CPU | 2.43 W |
| Jetson Xavier NX | GPU | 3.03 W |

Raw logs are in the per-device directories under [FuzzyAdp-MoE/test/torchScript/](FuzzyAdp-MoE/test/torchScript/).

## License

[MIT](LICENSE) © 2026 Marazing
