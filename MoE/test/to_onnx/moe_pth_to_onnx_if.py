import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import onnx

# ======================
# Parameter configuration
# ======================
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 27
NUM_EXPERTS = 5
TOP_K = 4
INPUT_SIZE = 224
PRETRAINED = False
PTH_PATH = "../efficientnet_b0_300e_224x224_final.pth"
ONNX_PATH = "moe_efficientnet_b0.onnx"
BATCH_SIZE = 1


# ======================
# MoE component definitions
# ======================
class SharedFeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv_stem = base_model.conv_stem
        self.bn1 = base_model.bn1
        self.act1 = base_model.act1 if hasattr(base_model, 'act1') else nn.SiLU(inplace=True)
        self.blocks = nn.Sequential(*base_model.blocks[:6])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ModuleList does not support TorchScript dynamic indexing, unrolled into fixed attributes
class MoEModel(nn.Module):
    def __init__(self, num_experts: int, num_classes: int, top_k: int, pretrained: bool = False):
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

        # Unrolled into fixed attributes so TorchScript can correctly handle conditional branch indexing
        experts = [
            ExpertNetwork(
                timm.create_model('efficientnet_b0', pretrained=pretrained),
                num_classes
            ) for _ in range(num_experts)
        ]
        self.expert0 = experts[0]
        self.expert1 = experts[1]
        self.expert2 = experts[2]
        self.expert3 = experts[3]
        self.expert4 = experts[4]

    def _run_expert(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        if idx == 0:
            return self.expert0(x)
        elif idx == 1:
            return self.expert1(x)
        elif idx == 2:
            return self.expert2(x)
        elif idx == 3:
            return self.expert3(x)
        else:
            return self.expert4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_extractor(x)

        router_logits = self.router(shared_features)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, self.num_classes,
                                   device=x.device, dtype=shared_features.dtype)
        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = topk_indices[i, j].item()
                weight = topk_weights[i, j]
                expert_out = self._run_expert(expert_idx, shared_features[i:i+1])
                final_output[i] = final_output[i] + weight * expert_out.squeeze(0)

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
        # Original weights use experts.0/1/2... key names, need to be remapped to expert0/1/2...
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("experts."):
                # "experts.0.blocks..." → "expert0.blocks..."
                parts = k.split(".", 2)
                new_key = f"expert{parts[1]}.{parts[2]}"
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        best_val_acc = checkpoint.get('best_val_acc', None)
        print(f"Weights loaded successfully (best training validation accuracy: {best_val_acc:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("Weights loaded successfully (legacy format)")

    model.eval()

    print("TorchScript compiling...")
    scripted = torch.jit.script(model)

    dummy = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE)

    print(f"Exporting ONNX: {ONNX_PATH} ...")
    torch.onnx.export(
        scripted,
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

    # Write metadata
    model_proto = onnx.load(ONNX_PATH)
    for k, v in [("num_experts", NUM_EXPERTS), ("top_k", TOP_K), ("num_classes", NUM_CLASSES)]:
        entry = model_proto.metadata_props.add()
        entry.key = k
        entry.value = str(v)
    onnx.save(model_proto, ONNX_PATH)

    print(f"Export complete: {ONNX_PATH}")
    print(f"Metadata written: num_experts={NUM_EXPERTS}, top_k={TOP_K}, num_classes={NUM_CLASSES}")