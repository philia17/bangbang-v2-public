"""
Export EnhancedCNN-V2 to ONNX format for Raspberry Pi deployment.

This script converts the PyTorch model to ONNX for faster inference
on ARM architecture using ONNX Runtime.
"""

import torch
import numpy as np
from src.models.enhanced_cnn import EnhancedCNN

# Paths
MODEL_PATH = "models/enhanced_cnn_fer2013.pth"
ONNX_PATH = "models/enhanced_cnn_fer2013.onnx"

# Input shape: (batch, channel, height, width)
INPUT_SHAPE = (1, 1, 48, 48)


def export_onnx():
    """Export EnhancedCNN to ONNX format."""
    print("Loading EnhancedCNN-V2 from PyTorch...")

    # Load model
    model = EnhancedCNN(num_classes=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # Create dummy input
    dummy_input = torch.randn(INPUT_SHAPE)

    # Export to ONNX using legacy API for better compatibility
    print(f"Exporting to ONNX: {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,  # Lower opset for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,
        verbose=False,
    )

    print(f"Export complete: {ONNX_PATH}")

    # Verify the exported model
    verify_onnx_model(ONNX_PATH, model, dummy_input)

    return ONNX_PATH


def verify_onnx_model(onnx_path: str, pytorch_model: torch.nn.Module, dummy_input: torch.Tensor):
    """Verify ONNX model produces same output as PyTorch model."""
    import onnx

    print("\nVerifying ONNX model...")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure OK")

    # Run inference with ONNX Runtime
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # ONNX input
    input_np = dummy_input.numpy().astype(np.float32)

    # ONNX inference
    onnx_output = session.run(None, {'input': input_np})[0]

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    # Compare
    diff = np.abs(onnx_output - pytorch_output).max()
    print(f"Max difference between ONNX and PyTorch output: {diff:.6f}")

    if diff < 1e-4:
        print("Verification PASSED - outputs match!")
    else:
        print("WARNING: Outputs differ more than expected")


if __name__ == '__main__':
    export_onnx()