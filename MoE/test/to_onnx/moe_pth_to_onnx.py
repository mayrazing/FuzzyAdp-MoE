import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ======================
# Parameter configuration (consistent with training/test scripts)
# ======================
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 27
NUM_EXPERTS = 5
TOP_K = 4
INPUT_SIZE = 224
PRETRAINED = False
PTH_PATH = "../efficientnet_b0_300e_224x224_final.pth"
ONNX_PATH = "moe_efficientnet_b0.onnx"
BATCH_SIZE = 4  # Consistent with test script


# ======================
# MoE component definitions (consistent with training/test scripts)
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
        # training=False during export, so the noise branch is not executed
        if self.training:
            noise = torch.randn_like(logits) * 0.1
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
    def __init__(self, num_experts, num_classes, top_k, pretrained=False):
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

    def forward(self, x):
        shared_features = self.shared_extractor(x)

        router_logits = self.router(shared_features)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Vectorized implementation: all experts computed in parallel, scatter weight mask is equivalent to TopK routing
        # Avoids dynamic indices being hard-coded during tracing, which would break the router
        all_expert_outputs = torch.stack(
            [expert(shared_features) for expert in self.experts], dim=1
        )  # (B, num_experts, num_classes)

        sparse_mask = torch.zeros_like(router_probs).scatter(1, topk_indices, topk_weights)
        # (B, num_experts), weights at non-TopK positions are 0

        final_output = torch.einsum('be,bec->bc', sparse_mask, all_expert_outputs)
        return final_output


# ======================
# Main flow
# ======================
if __name__ == "__main__":
    print("Building MoE model...")
    model = MoEModel(num_experts=NUM_EXPERTS,
                     num_classes=NUM_CLASSES,
                     top_k=TOP_K,
                     pretrained=PRETRAINED)

    print(f"Loading weights: {PTH_PATH}")
    checkpoint = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', None)
        print(f"Weights loaded successfully (best training validation accuracy: {best_val_acc:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("Weights loaded successfully (legacy format)")

    model.eval()

    dummy = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)

    print(f"Exporting ONNX: {ONNX_PATH} ...")
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=18,
    )
    print(f"Export complete: {ONNX_PATH}")