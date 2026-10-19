import os
import torch
import torch.nn.functional as F
from torch import nn
import timm

# ======================
# Parameter configuration
# ======================
MODEL_NAME = "efficientnet_b0"
INPUT_SIZE = 224
NUM_CLASSES = 27
NUM_EXPERTS = 5
TOP_K = 2
PRETRAINED = False

PTH_PATH = f"../efficientnet_b0_300e_224x224_final.pth"
OUTPUT_PT_PATH = "moe_pi_optimized.pt"


# ======================
# Base network components (consistent with training)
# ======================
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

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        logits = self.gap(x).squeeze(-1).squeeze(-1)
        logits = self.layernorm(logits)
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


# ======================
# Core optimization: MoE designed specifically for BS=1
# ======================
class MoEModelPiOpt(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Shared feature extraction (x shape: 1, 3, 224, 224)
        shared_features = self.shared_extractor(x)

        # 2. Routing computation (no division by tau and no noise at inference)
        router_logits = self.router(shared_features)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        batch_size = x.size(0)
        out = torch.zeros(batch_size, self.num_classes, device=x.device, dtype=shared_features.dtype)

        idx0 = int(topk_indices[0, 0])
        idx1 = int(topk_indices[0, 1])
        w0 = topk_weights[0, 0]
        w1 = topk_weights[0, 1]

        for i, expert in enumerate(self.experts):
            if i == idx0:
                out += expert(shared_features) * w0
            elif i == idx1:
                out += expert(shared_features) * w1

        return out


# ======================
# Export pipeline
# ======================
def export(pth_path: str, output_pt_path: str):
    device = torch.device("cpu")
    print("Loading weights...")
    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)

    model = MoEModelPiOpt(
        num_experts=NUM_EXPERTS,
        num_classes=NUM_CLASSES,
        top_k=TOP_K,
        pretrained=PRETRAINED
    )

    # Backward-compatible loading of legacy weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("TorchScript compiling for edge devices...")
    scripted = torch.jit.script(model)
    scripted.save(output_pt_path)
    print(f"Export complete ✓  →  {output_pt_path}")

    # Verify inference
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = scripted(dummy)
    print(f"Verification output shape: {out.shape}")


if __name__ == "__main__":
    export(PTH_PATH, OUTPUT_PT_PATH)