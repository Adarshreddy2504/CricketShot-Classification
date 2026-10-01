"""
Transformer inference validation test.

Run from the repository root:

    python backend/test_transformer_inference.py

This test verifies:
1. The checkpoint exists.
2. The Transformer architecture imports correctly.
3. The checkpoint loads with strict=True.
4. A 30-frame input produces 10 class logits.
5. Softmax probabilities are valid.
6. The predicted class and confidence are printed.
"""

from pathlib import Path
import sys

import torch


# ============================================================
# PATHS
# ============================================================

# backend/
BACKEND_DIR = Path(__file__).resolve().parent

# repository root
PROJECT_DIR = BACKEND_DIR.parent

# Actual deployment checkpoint
CHECKPOINT_PATH = BACKEND_DIR / "models" / "cricket_model_transformer.ckpt"


# ============================================================
# MODEL IMPORT
# ============================================================

# Make sure backend/ is importable regardless of current directory.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.pipeline.efficientnet_transformer import ImprovedSOTAModel


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Cover",
    "Defense",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted",
    "Pull",
    "Square Cut",
    "Straight",
    "Sweep",
]

NUM_FRAMES = 30
IMAGE_SIZE = 224
NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# HELPERS
# ============================================================

def fail(message: str) -> None:
    print()
    print("=" * 70)
    print("TEST FAILED")
    print("=" * 70)
    print(message)
    print("=" * 70)
    raise SystemExit(1)


def get_state_dict(checkpoint):
    """
    Extract a state_dict from common checkpoint formats.

    Supports:
        - raw state_dict
        - {"state_dict": ...}
        - {"model_state_dict": ...}
    """
    if not isinstance(checkpoint, dict):
        fail(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}"
        )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        # Detect whether the checkpoint itself is already a state_dict.
        tensor_values = [
            value
            for value in checkpoint.values()
            if isinstance(value, torch.Tensor)
        ]

        if tensor_values:
            state_dict = checkpoint
        else:
            fail(
                "Could not find a model state_dict in the checkpoint."
            )

    # Remove a possible Lightning "model." prefix.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("CRICKET TRANSFORMER INFERENCE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Project directory : {PROJECT_DIR}")
    print(f"Backend directory : {BACKEND_DIR}")
    print(f"Checkpoint        : {CHECKPOINT_PATH}")
    print(f"Device            : {device}")

    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")

    print()

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    print("[1/6] Checking checkpoint...")

    if not CHECKPOINT_PATH.exists():
        fail(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint_size_mb = CHECKPOINT_PATH.stat().st_size / (
        1024 * 1024
    )

    print(
        f"Checkpoint found: "
        f"{checkpoint_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print("[2/6] Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = get_state_dict(checkpoint)

    print(
        f"Checkpoint tensors: {len(state_dict)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print("[3/6] Building Transformer model...")

    model = ImprovedSOTAModel(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Strict checkpoint validation
    # --------------------------------------------------------

    print()
    print("[4/6] Loading weights with strict=True...")

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        fail(
            "Checkpoint architecture does not exactly match "
            "ImprovedSOTAModel.\n\n"
            f"{exc}"
        )

    print("Strict checkpoint loading: PASS")

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    model = model.to(device)
    model.eval()

    print(f"Model device: {device}")

    # --------------------------------------------------------
    # Create realistic input shape
    # --------------------------------------------------------

    print()
    print("[5/6] Running 30-frame inference...")

    # Expected model input:
    #
    # [batch, frames, channels, height, width]
    #
    # [1, 30, 3, 224, 224]

    input_tensor = torch.randn(
        1,
        NUM_FRAMES,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    print(
        f"Input shape: {tuple(input_tensor.shape)}"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = model(input_tensor)

        # Some implementations may return a tuple.
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if not isinstance(logits, torch.Tensor):
            fail(
                "Model output is not a torch.Tensor."
            )

        print(
            f"Output shape: {tuple(logits.shape)}"
        )

        # Expected:
        #
        # [1, 10]

        expected_shape = (1, NUM_CLASSES)

        if tuple(logits.shape) != expected_shape:
            fail(
                "Unexpected model output shape.\n"
                f"Expected: {expected_shape}\n"
                f"Got     : {tuple(logits.shape)}"
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_index = int(
            torch.argmax(probabilities, dim=1).item()
        )

        confidence = float(
            probabilities[0, predicted_index].item()
        )

    # --------------------------------------------------------
    # Validate probabilities
    # --------------------------------------------------------

    probability_sum = float(
        probabilities[0].sum().item()
    )

    if not torch.isfinite(logits).all():
        fail("Model produced NaN or infinite logits.")

    if not torch.isfinite(probabilities).all():
        fail("Model produced NaN or infinite probabilities.")

    if abs(probability_sum - 1.0) > 1e-5:
        fail(
            "Softmax probabilities do not sum to 1.\n"
            f"Sum: {probability_sum}"
        )

    if not 0.0 <= confidence <= 1.0:
        fail(
            "Invalid confidence value:\n"
            f"{confidence}"
        )

    # --------------------------------------------------------
    # Print prediction
    # --------------------------------------------------------

    print()
    print("[6/6] Prediction")
    print("-" * 70)

    print(
        f"Predicted class : "
        f"{CLASS_NAMES[predicted_index]}"
    )

    print(
        f"Class index     : "
        f"{predicted_index}"
    )

    print(
        f"Confidence       : "
        f"{confidence * 100:.2f}%"
    )

    print()
    print("Class probabilities:")
    print("-" * 70)

    sorted_indices = torch.argsort(
        probabilities[0],
        descending=True,
    )

    for index in sorted_indices.tolist():

        probability = float(
            probabilities[0, index].item()
        )

        print(
            f"{CLASS_NAMES[index]:<15} "
            f"{probability * 100:>7.2f}%"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST PASSED")
    print("=" * 70)

    print("✓ Checkpoint exists")
    print("✓ Transformer architecture imported")
    print("✓ Checkpoint loaded with strict=True")
    print("✓ 30-frame input accepted")
    print("✓ Output shape is [1, 10]")
    print("✓ Probabilities are valid")
    print("✓ Inference completed successfully")

    print("=" * 70)


if __name__ == "__main__":
    main()