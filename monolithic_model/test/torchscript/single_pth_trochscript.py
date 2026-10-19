import torch
import timm

# ======================
# Configuration (consistent with the training script)
# ======================
MODEL_NAME   = "efficientnet_b0"
NUM_CLASSES  = 27
INPUT_SIZE   = 224
PTH_PATH     = "../efficientnet_b0_300e_224x224_final.pth"
OUTPUT_PT    = "efficientnet_b0.pt"


def export(pth_path: str, output_pt_path: str):
    device = torch.device("cpu")

    # 1. Rebuild model architecture
    print("Rebuilding model architecture...")
    model = timm.create_model(MODEL_NAME, pretrained=False)
    model.reset_classifier(NUM_CLASSES)

    # 2. Load weights
    print(f"Loading weights: {pth_path}")
    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names   = checkpoint.get('class_names', None)
        best_val_acc  = checkpoint.get('best_val_acc', None)
        print(f"Best validation accuracy: {best_val_acc:.4f}" if best_val_acc else "Weights loaded successfully")
        print(f"Classes: {class_names}")
    else:
        model.load_state_dict(checkpoint)
        class_names = None
        print("Weights loaded successfully (legacy format)")

    model.eval()

    # 3. Trace export
    print("Exporting via TorchScript trace...")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(output_pt_path)
    print(f"Export complete ✓  →  {output_pt_path}")

    # 4. Verify
    print("Verifying output...")
    loaded = torch.jit.load(output_pt_path, map_location=device)
    loaded.eval()
    with torch.no_grad():
        out = loaded(dummy)
    print(f"Output shape: {out.shape}")   # Expected (1, 27)
    assert out.shape == (1, NUM_CLASSES), "Unexpected output dimensions!"
    print("Verification passed ✓")

    # 5. Save class_names separately for test.py to load
    if class_names:
        import json
        meta_path = output_pt_path.replace(".pt", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"class_names": class_names, "best_val_acc": float(best_val_acc or 0)}, f, indent=4, ensure_ascii=False)
        print(f"Class info saved →  {meta_path}")


if __name__ == "__main__":
    export(PTH_PATH, OUTPUT_PT)