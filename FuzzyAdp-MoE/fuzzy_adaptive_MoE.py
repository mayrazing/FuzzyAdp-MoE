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
import skfuzzy as fuzz
from skfuzzy import control as ctrl


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
_noise_scale_history = []
_tau_history = []
_fuzzy_stats = {
    'B': [], 'T_fast': [], 'T_slow': [],
    'tau': [], 'aux_weight': [], 'noise': []
}

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
g = torch.Generator()
g.manual_seed(42)

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

def init_fuzzy_system():
    B = ctrl.Antecedent(np.arange(70, 101, 1), 'B')
    T_fast = ctrl.Antecedent(np.arange(-0.02, 0.031, 0.001), 'T_fast')
    T_slow = ctrl.Antecedent(np.arange(-0.02, 0.051, 0.001), 'T_slow')

    delta_tau = ctrl.Consequent(np.arange(-0.04, 0.041, 0.001), 'delta_tau')
    lambda_out = ctrl.Consequent(np.arange(0.5, 2.51, 0.01), 'lambda_out')
    noise = ctrl.Consequent(np.arange(0.0, 0.151, 0.001), 'noise')

    B['Unstable'] = fuzz.trapmf(B.universe, [70, 70, 78, 84])
    B['Normal']   = fuzz.trimf(B.universe,  [80, 88, 94])
    B['Perfect']  = fuzz.trapmf(B.universe, [91, 95, 100, 100])

    T_fast['Degrading']  = fuzz.trapmf(T_fast.universe, [-0.02, -0.02, -0.008, 0.0])
    T_fast['Stable']     = fuzz.trimf(T_fast.universe,  [-0.005, 0.003, 0.012])
    T_fast['Improving']  = fuzz.trapmf(T_fast.universe, [0.008, 0.015, 0.03, 0.03])

    T_slow['Degrading']  = fuzz.trapmf(T_slow.universe, [-0.02, -0.02, -0.005, 0.002])
    T_slow['Stable']     = fuzz.trimf(T_slow.universe,  [-0.003, 0.003, 0.012])
    T_slow['Improving']  = fuzz.trapmf(T_slow.universe, [0.005, 0.018, 0.05, 0.05])

    delta_tau['Decrease'] = fuzz.trimf(delta_tau.universe, [-0.04, -0.025, 0.0])
    delta_tau['Keep']     = fuzz.trimf(delta_tau.universe, [-0.01, 0.0, 0.01])
    delta_tau['Increase'] = fuzz.trimf(delta_tau.universe, [0.0, 0.025, 0.04])

    lambda_out['Low']    = fuzz.trapmf(lambda_out.universe, [0.5, 0.5, 0.8, 1.2])
    lambda_out['Medium'] = fuzz.trimf(lambda_out.universe,  [0.8, 1.2, 1.8])
    lambda_out['High']   = fuzz.trapmf(lambda_out.universe, [1.5, 2.0, 2.5, 2.5])

    noise['Low']    = fuzz.trapmf(noise.universe, [0.0, 0.0, 0.015, 0.04])
    noise['Medium'] = fuzz.trimf(noise.universe,  [0.025, 0.06, 0.10])
    noise['High']   = fuzz.trapmf(noise.universe, [0.08, 0.12, 0.15, 0.15])

    rules = [
        ctrl.Rule(B['Unstable'] | T_fast['Degrading'],
                  (delta_tau['Increase'], lambda_out['High'], noise['High'])),

        ctrl.Rule(B['Normal'] & T_fast['Stable'],
                  (delta_tau['Keep'], lambda_out['Medium'], noise['High'])),

        ctrl.Rule(B['Perfect'] & T_fast['Improving'],
                  (delta_tau['Decrease'], lambda_out['Low'], noise['Low'])),

        ctrl.Rule(T_slow['Stable'],
                  (delta_tau['Keep'], lambda_out['Medium'], noise['High'])),

        ctrl.Rule(B['Normal'] & T_fast['Improving'],
                  (delta_tau['Decrease'], lambda_out['Medium'], noise['High'])),

        ctrl.Rule(B['Perfect'] & T_fast['Stable'],
                  (delta_tau['Keep'], lambda_out['Low'], noise['High'])),
    ]

    fuzzy_ctrl = ctrl.ControlSystem(rules)
    fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
    return fuzzy_sim


_fuzzy_system = None
def get_fuzzy_system():
    global _fuzzy_system
    if _fuzzy_system is None:
        _fuzzy_system = init_fuzzy_system()
    return _fuzzy_system


def calculate_expert_health_metrics(expert_usage):
    if not expert_usage:
        return 0.0, 0.0
    sorted_usage = np.array(sorted(expert_usage)) / 100.0
    n = len(expert_usage)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_usage))) / (n * np.sum(sorted_usage)) - (n + 1) / n
    min_usage = min(expert_usage)
    return gini, min_usage


def calculate_system_balance(gini, min_usage, beta=0.8):
    S_usage = min_usage * 5
    S_gini = 200 * (0.5 - gini)
    m = min(S_usage, S_gini)
    penalty = (1 / beta) * np.log(1 + np.exp(-beta * abs(S_usage - S_gini)))
    B_value = m - penalty
    return B_value


def calculate_training_trends(val_acc_history, k_fast=3, k_slow=5):
    T_fast_value = 0.0
    T_slow_value = 0.0
    # Paper Eq. (12): T = (1/W) * Σ_{i=1..W} (acc_t - acc_{t-i}), requires at least W+1 history points
    if len(val_acc_history) > k_fast:
        T_fast_value = np.mean([
            val_acc_history[-1] - val_acc_history[-1 - i]
            for i in range(1, k_fast + 1)
        ])
    if len(val_acc_history) > k_slow:
        T_slow_value = np.mean([
            val_acc_history[-1] - val_acc_history[-1 - i]
            for i in range(1, k_slow + 1)
        ])
    return T_fast_value, T_slow_value


def adaptive_tau_controller(val_acc_history, expert_usage_history, base_aux_weight):
    global _tau_history, _noise_scale_history
    MIN_TAU, MAX_TAU = 0.85, 1.02
    base_tau = _tau_history[-1] if _tau_history else 0.90

    if len(expert_usage_history) < 1:
        _noise_scale_history.append(0.1)
        return 0.90, base_aux_weight, 0.1

    expert_usage = expert_usage_history[-1]
    gini, min_usage = calculate_expert_health_metrics(expert_usage)
    B_value = calculate_system_balance(gini, min_usage, beta=0.8)
    T_fast_value, T_slow_value = calculate_training_trends(val_acc_history, k_fast=3, k_slow=5)

    print(f"  [Input] B={B_value:.1f}, T_fast={T_fast_value:.5f}, T_slow={T_slow_value:.5f}")
    print(f"  [Health] Gini={gini:.3f}, min_usage={min_usage:.1f}%")
    logging.info(f"  [Input] B={B_value:.1f}, T_fast={T_fast_value:.5f}, T_slow={T_slow_value:.5f}")
    logging.info(f"  [Health] Gini={gini:.3f}, min_usage={min_usage:.1f}%")

    fuzzy_sim = get_fuzzy_system()
    B_value = np.clip(B_value, 70, 100)
    T_fast_value = np.clip(T_fast_value, -0.02, 0.03)
    T_slow_value = np.clip(T_slow_value, -0.02, 0.05)

    fuzzy_sim.input['B'] = B_value
    fuzzy_sim.input['T_fast'] = T_fast_value
    fuzzy_sim.input['T_slow'] = T_slow_value
    fuzzy_sim.compute()

    delta_tau_out = fuzzy_sim.output['delta_tau']
    lambda_out = fuzzy_sim.output['lambda_out']
    noise_out = fuzzy_sim.output['noise']

    print(f"  [Fuzzy output] Δτ={delta_tau_out:.4f}, λ={lambda_out:.3f}, noise={noise_out:.4f}")
    logging.info(f"  [Fuzzy output] Δτ={delta_tau_out:.4f}, λ={lambda_out:.3f}, noise={noise_out:.4f}")

    final_tau = np.clip(base_tau + delta_tau_out, MIN_TAU, MAX_TAU)
    final_lambda = lambda_out
    final_noise = noise_out

    _tau_history.append(final_tau)
    _noise_scale_history.append(final_noise)

    _fuzzy_stats['B'].append(B_value)
    _fuzzy_stats['T_fast'].append(T_fast_value)
    _fuzzy_stats['T_slow'].append(T_slow_value)
    _fuzzy_stats['tau'].append(final_tau)
    _fuzzy_stats['aux_weight'].append(final_lambda)
    _fuzzy_stats['noise'].append(final_noise)

    print(f"  [Final] τ={final_tau:.4f}, aux_weight={final_lambda:.3f}, noise={final_noise:.4f}")
    logging.info(f"  [Final] τ={final_tau:.4f}, aux_weight={final_lambda:.3f}, noise={final_noise:.4f}")

    return final_tau, final_lambda, final_noise



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
    """Gating network: identifies road surface texture and dry/wet features"""

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

        self.register_buffer('noise_scale', torch.tensor(0.1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        logits = self.gap(x).squeeze(-1).squeeze(-1)

        logits = self.layernorm(logits)

        if self.training:
            noise = torch.randn_like(logits) * self.noise_scale
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

        self.register_buffer('tau', torch.tensor(1.0))

    def _initialize_experts_with_perturbation(self):
        if PRETRAINED:
            for expert in self.experts:
                if hasattr(expert.classifier, 'reset_parameters'):
                    expert.classifier.reset_parameters()
                else:
                    nn.init.normal_(expert.classifier.weight, mean=0, std=0.01)
                    if expert.classifier.bias is not None:
                        nn.init.constant_(expert.classifier.bias, 0)
        else:
            for i, expert in enumerate(self.experts):
                torch.manual_seed(42 + i)
                for m in expert.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, training=True):
        shared_features = self.shared_extractor(x)  # (B, C, H, W)

        router_logits = self.router(shared_features)  # (B, num_experts)

        if training:
            router_probs = F.softmax(router_logits / self.tau, dim=-1)
            topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
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
            router_probs = F.softmax(router_logits, dim=-1)
            topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
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

        p = router_probs.mean(dim=0)  # (num_experts,)

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
    global _tau_history, _noise_scale_history, _fuzzy_stats
    _tau_history = []
    _noise_scale_history = []
    _fuzzy_stats = {
        'B': [], 'T_fast': [], 'T_slow': [],
        'tau': [], 'aux_weight': [], 'noise': []
    }

    since = time.time()
    best_model_wts = copy.deepcopy(model_.state_dict())
    best_val_acc = float(initial_best_val_acc_) if initial_best_val_acc_ is not None else 0.0

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,
                                   delta=EARLY_STOPPING_DELTA,
                                   verbose=True)

    for epoch in range(num_epochs):
        adaptive_tau, dynamic_aux_weight, adaptive_noise = adaptive_tau_controller(
            val_accuracy, expert_usage_history, AUX_LOSS_WEIGHT
        )
        model_.tau.data = torch.tensor(adaptive_tau)
        model_.router.noise_scale.data = torch.tensor(adaptive_noise)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}, τ={adaptive_tau:.3f}, aux_weight={dynamic_aux_weight:.2f}, noise={adaptive_noise:.3f}")
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
                            warmup_factor = 0.1 if epoch < 20 else 1.0
                            current_aux_weight = dynamic_aux_weight * warmup_factor
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

    print("\n" + "=" * 60)
    print("Fuzzy Logic controller statistics")
    print("=" * 60)
    logging.info("Fuzzy Logic controller statistics")
    for var_name in ['B', 'T_fast', 'T_slow', 'tau', 'aux_weight', 'noise']:
        data = np.array(_fuzzy_stats[var_name])
        if len(data) > 0:
            print(f"{var_name:12s}: max={np.max(data):.4f}, min={np.min(data):.4f}, "
                  f"mean={np.mean(data):.4f}, std={np.std(data):.4f}, p90={np.percentile(data, 90):.4f}")
            logging.info(f"{var_name:12s}: max={np.max(data):.4f}, min={np.min(data):.4f}, "
                         f"mean={np.mean(data):.4f}, std={np.std(data):.4f}, p90={np.percentile(data, 90):.4f}")
    print("=" * 60 + "\n")

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
                                                  pin_memory=True if torch.cuda.is_available() else False,
                                                  generator=g if x == 'train' else None,
                                                  worker_init_fn=worker_init_fn)
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