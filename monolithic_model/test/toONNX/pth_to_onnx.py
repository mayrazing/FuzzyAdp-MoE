import torch
import timm

MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 27
INPUT_SIZE = 224
PTH_PATH = "../efficientnet_b0_300e_224x224_final.pth"
ONNX_PATH = "efficientnet_b0.onnx"

model = timm.create_model(MODEL_NAME, pretrained=False)
model.reset_classifier(NUM_CLASSES)

checkpoint = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Key: export with batch=4 to match the test BATCH_SIZE, so the internal Reshape is not fixed to batch=1
dummy = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)

torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=18
)
print(f"Export complete: {ONNX_PATH}")