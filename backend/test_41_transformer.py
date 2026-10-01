"""
Cricket end-to-end delivery + shot-classification validation.

Pipeline:

    Full match video
        |
        v
    AutoClipper
        |
        v
    Delivery clips
        |
        v
    30 uniformly sampled frames
        |
        v
    Aspect-ratio-preserving resize/padding
        |
        v
    ImageNet normalization
        |
        v
    EfficientNet-B0 + Transformer
        |
        v
    Per-delivery shot prediction

Run from repository root:

    python backend/test_41_transformer.py

This validates inference compatibility and execution.
It does NOT calculate classification accuracy.
"""

from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ============================================================================
# PATHS
# ============================================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"

BALL_MODEL = MODELS_DIR / "Ball_Detection_Model.pt"
BAT_MODEL = MODELS_DIR / "Bat_Detection_Model.pt"
SHOT_MODEL = MODELS_DIR / "cricket_model_transformer.ckpt"

INPUT_VIDEO = Path(
    r"C:\Users\mamid\OneDrive\Desktop\matches\long_match.mp4"
)

OUTPUT_DIR = BACKEND_DIR / "test_41_output"


# ============================================================================
# PYTHON PATH
# ============================================================================

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from src.segmentation.auto_clipper import AutoClipper
from src.pipeline.efficientnet_transformer import ImprovedSOTAModel


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

SHOT_CLASSES = [
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

NUM_CLASSES = len(SHOT_CLASSES)

N_FRAMES = 30
IMAGE_SIZE = 224


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

class ResizeWithPadding:
    """
    Resize while preserving aspect ratio, then pad to a square.
    """

    def __init__(self, image_size=224, fill=0):
        self.image_size = image_size
        self.fill = fill

    def __call__(self, image):
        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image dimensions: {width}x{height}"
            )

        scale = min(
            self.image_size / width,
            self.image_size / height,
        )

        new_width = max(
            1,
            round(width * scale),
        )

        new_height = max(
            1,
            round(height * scale),
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        left = (self.image_size - new_width) // 2
        top = (self.image_size - new_height) // 2

        right = (
            self.image_size
            - new_width
            - left
        )

        bottom = (
            self.image_size
            - new_height
            - top
        )

        return transforms.functional.pad(
            image,
            [left, top, right, bottom],
            fill=self.fill,
        )


FRAME_TRANSFORM = transforms.Compose(
    [
        ResizeWithPadding(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================================
# ERROR HANDLING
# ============================================================================

def fail(message):
    print()
    print("=" * 72)
    print("TEST FAILED")
    print("=" * 72)
    print(message)
    print("=" * 72)
    raise SystemExit(1)


def require_file(path, description):
    if not path.exists():
        fail(
            f"{description} was not found:\n"
            f"{path}"
        )

    if not path.is_file():
        fail(
            f"{description} is not a file:\n"
            f"{path}"
        )


# ============================================================================
# INPUT VIDEO VALIDATION
# ============================================================================

def validate_input_video(video_path):
    print()
    print("=" * 72)
    print("VALIDATING INPUT VIDEO")
    print("=" * 72)

    require_file(
        video_path,
        "Input video",
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        fail(
            "OpenCV could not open the input video."
        )

    try:
        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    finally:
        capture.release()

    if frame_count <= 0:
        fail(
            "The input video contains no readable frames."
        )

    if fps <= 0:
        fail(
            "The input video reports an invalid FPS."
        )

    if width <= 0 or height <= 0:
        fail(
            "The input video reports invalid dimensions."
        )

    duration = frame_count / fps

    print(f"Video      : {video_path}")
    print(f"Frames     : {frame_count}")
    print(f"FPS        : {fps:.2f}")
    print(f"Resolution : {width} x {height}")
    print(f"Duration   : {duration:.2f} seconds")
    print("[✓] Input video is valid")


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def extract_state_dict(checkpoint):
    """
    Accept common checkpoint layouts:

        checkpoint["state_dict"]
        checkpoint["model_state_dict"]
        checkpoint itself as state_dict
    """

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Checkpoint must be a dictionary."
        )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Checkpoint state_dict is not a dictionary."
        )

    return state_dict


def clean_checkpoint_state_dict(state_dict):
    """
    Remove common training wrappers such as 'model.'.

    The class_weights buffer is deliberately ignored here.
    It is supplied from the freshly constructed model below.
    """

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("model."):
            key = key[len("model."):]

        if key == "class_weights":
            continue

        cleaned[key] = value

    return cleaned


def load_shot_model():
    print()
    print("=" * 72)
    print("LOADING TRANSFORMER CHECKPOINT")
    print("=" * 72)

    require_file(
        SHOT_MODEL,
        "Transformer checkpoint",
    )

    print(
        f"Checkpoint: {SHOT_MODEL}"
    )

    # ------------------------------------------------------------------------
    # Load checkpoint on CPU first.
    # ------------------------------------------------------------------------

    checkpoint = torch.load(
        SHOT_MODEL,
        map_location="cpu",
        weights_only=False,
    )

    raw_state_dict = extract_state_dict(
        checkpoint
    )

    cleaned_state_dict = (
        clean_checkpoint_state_dict(
            raw_state_dict
        )
    )

    print(
        f"Checkpoint tensors found: "
        f"{len(cleaned_state_dict)}"
    )

    # ------------------------------------------------------------------------
    # Construct the exact architecture used by the checkpoint.
    # ------------------------------------------------------------------------

    print(
        "[*] Building ImprovedSOTAModel..."
    )

    model = ImprovedSOTAModel(
        num_classes=NUM_CLASSES
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # ImprovedSOTAModel registers class_weights as a model buffer.
    # The saved checkpoint does not contain it.
    #
    # Therefore we use the model's own initialized buffer value.
    # This preserves the learned checkpoint parameters while satisfying
    # strict state-dict compatibility.
    # ------------------------------------------------------------------------

    model_state = model.state_dict()

    if "class_weights" not in model_state:
        fail(
            "ImprovedSOTAModel does not contain the expected "
            "'class_weights' buffer."
        )

    cleaned_state_dict["class_weights"] = (
        model_state["class_weights"]
        .detach()
        .clone()
    )

    # ------------------------------------------------------------------------
    # Verify key sets BEFORE loading.
    # ------------------------------------------------------------------------

    model_keys = set(
        model_state.keys()
    )

    checkpoint_keys = set(
        cleaned_state_dict.keys()
    )

    missing_keys = sorted(
        model_keys - checkpoint_keys
    )

    unexpected_keys = sorted(
        checkpoint_keys - model_keys
    )

    if missing_keys:
        fail(
            "Checkpoint is missing model keys before loading:\n"
            + "\n".join(
                f"  - {key}"
                for key in missing_keys
            )
        )

    if unexpected_keys:
        fail(
            "Checkpoint contains unexpected model keys before loading:\n"
            + "\n".join(
                f"  - {key}"
                for key in unexpected_keys
            )
        )

    print(
        "[✓] Checkpoint keys exactly match model keys"
    )

    # ------------------------------------------------------------------------
    # Strict load.
    # ------------------------------------------------------------------------

    print(
        "[*] Loading checkpoint with strict=True..."
    )

    try:
        incompatible = model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

    except RuntimeError as exc:
        fail(
            "The Transformer checkpoint does not exactly "
            "match ImprovedSOTAModel.\n\n"
            f"{exc}"
        )

    # strict=True should produce no incompatible keys.
    if incompatible.missing_keys:
        fail(
            "Unexpected missing keys after strict loading:\n"
            + "\n".join(
                f"  - {key}"
                for key in incompatible.missing_keys
            )
        )

    if incompatible.unexpected_keys:
        fail(
            "Unexpected keys after strict loading:\n"
            + "\n".join(
                f"  - {key}"
                for key in incompatible.unexpected_keys
            )
        )

    # ------------------------------------------------------------------------
    # Move model to inference device.
    # ------------------------------------------------------------------------

    model = model.to(DEVICE)
    model.eval()

    print(
        "[✓] Transformer loaded successfully"
    )

    print(
        f"[✓] Device: {DEVICE}"
    )

    return model


# ============================================================================
# DELIVERY FRAME LOADING
# ============================================================================

def load_delivery_frames(clip_path):
    """
    Read exactly 30 uniformly distributed frames.

    Returns:

        [1, 30, 3, 224, 224]
    """

    capture = cv2.VideoCapture(
        str(clip_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open delivery clip:\n"
            f"{clip_path}"
        )

    try:
        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:
            raise RuntimeError(
                f"Delivery clip contains no frames:\n"
                f"{clip_path}"
            )

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            N_FRAMES,
            dtype=np.int64,
        )

        tensors = []

        for frame_index in frame_indices:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            success, frame = capture.read()

            if not success or frame is None:
                raise RuntimeError(
                    f"Failed to read frame "
                    f"{frame_index} from "
                    f"{clip_path.name}"
                )

            # OpenCV gives BGR.
            # PIL preprocessing expects RGB.
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(
                rgb
            )

            tensor = FRAME_TRANSFORM(
                image
            )

            tensors.append(
                tensor
            )

    finally:
        capture.release()

    if len(tensors) != N_FRAMES:
        raise RuntimeError(
            f"Expected {N_FRAMES} frames, "
            f"got {len(tensors)}."
        )

    frames = torch.stack(
        tensors,
        dim=0,
    )

    frames = frames.unsqueeze(0)

    expected_shape = (
        1,
        N_FRAMES,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(frames.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected frame tensor shape: "
            f"{tuple(frames.shape)}. "
            f"Expected {expected_shape}."
        )

    return frames


# ============================================================================
# SINGLE DELIVERY INFERENCE
# ============================================================================

@torch.inference_mode()
def predict_delivery(model, clip_path):

    frames = load_delivery_frames(
        clip_path
    )

    frames = frames.to(
        DEVICE,
        non_blocking=(
            DEVICE.type == "cuda"
        ),
    )

    output = model(
        frames
    )

    # Some model implementations return
    # logits directly; support tuple/list as well.
    if isinstance(
        output,
        (tuple, list),
    ):
        if len(output) == 0:
            raise RuntimeError(
                "Model returned an empty tuple/list."
            )

        logits = output[0]

    else:
        logits = output

    if not isinstance(
        logits,
        torch.Tensor,
    ):
        raise RuntimeError(
            "Model output is not a torch.Tensor."
        )

    expected_shape = (
        1,
        NUM_CLASSES,
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected model output shape: "
            f"{tuple(logits.shape)}. "
            f"Expected {expected_shape}."
        )

    if not torch.isfinite(
        logits
    ).all():
        raise RuntimeError(
            "Model produced NaN or infinite logits."
        )

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    if not torch.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Model produced NaN or infinite probabilities."
        )

    probability_sum = float(
        probabilities[0].sum().item()
    )

    if abs(
        probability_sum - 1.0
    ) > 1e-5:
        raise RuntimeError(
            f"Probability sum is "
            f"{probability_sum:.8f}; "
            f"expected approximately 1.0."
        )

    confidence, class_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_index = int(
        class_index.item()
    )

    confidence_value = float(
        confidence.item()
    )

    if not 0 <= predicted_index < NUM_CLASSES:
        raise RuntimeError(
            f"Invalid predicted class index: "
            f"{predicted_index}"
        )

    return {
        "prediction": SHOT_CLASSES[
            predicted_index
        ],
        "class_index": predicted_index,
        "confidence": confidence_value,
        "class_probabilities": {
            class_name: float(
                probabilities[
                    0,
                    class_id,
                ].item()
            )
            for class_id, class_name in enumerate(
                SHOT_CLASSES
            )
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 72)
    print("41-DELIVERY END-TO-END TRANSFORMER TEST")
    print("=" * 72)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------------------
    # STEP 1
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 1: CHECKING REQUIRED FILES")
    print("=" * 72)

    require_file(
        BALL_MODEL,
        "Ball Detection Model",
    )

    require_file(
        BAT_MODEL,
        "Bat Detection Model",
    )

    require_file(
        SHOT_MODEL,
        "Transformer checkpoint",
    )

    require_file(
        INPUT_VIDEO,
        "Input video",
    )

    print(
        f"[✓] Ball model : {BALL_MODEL}"
    )

    print(
        f"[✓] Bat model  : {BAT_MODEL}"
    )

    print(
        f"[✓] Shot model : {SHOT_MODEL}"
    )

    print(
        f"[✓] Input video: {INPUT_VIDEO}"
    )

    # ------------------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------------------

    validate_input_video(
        INPUT_VIDEO
    )

    # ------------------------------------------------------------------------
    # STEP 3
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 2: PREPARING TEST OUTPUT")
    print("=" * 72)

    if OUTPUT_DIR.exists():
        print(
            f"[*] Removing previous output:\n"
            f"    {OUTPUT_DIR}"
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[✓] Output directory: {OUTPUT_DIR}"
    )

    # ------------------------------------------------------------------------
    # STEP 4
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 3: GENERATING DELIVERY CLIPS")
    print("=" * 72)

    print(
        "[*] Loading AutoClipper..."
    )

    clipper = AutoClipper(
        ball_model_path=str(
            BALL_MODEL
        ),
        bat_model_path=str(
            BAT_MODEL
        ),
        output_dir=str(
            OUTPUT_DIR
        ),
        confidence=0.30,
        use_ocr=True,
        ocr_gpu=False,
    )

    print(
        "[✓] AutoClipper loaded"
    )

    print()
    print(
        "[*] Processing match..."
    )

    extracted_clips = clipper.process_match(
        str(INPUT_VIDEO),
        clip_duration_sec=1.0,
    )

    if extracted_clips is None:
        extracted_clips = []

    if not isinstance(
        extracted_clips,
        list,
    ):
        fail(
            "AutoClipper.process_match() "
            "did not return a list."
        )

    print()
    print(
        f"[✓] AutoClipper returned "
        f"{len(extracted_clips)} clips"
    )

    # ------------------------------------------------------------------------
    # STEP 5
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 4: VALIDATING DELIVERY CLIPS")
    print("=" * 72)

    clip_paths = []

    for clip in extracted_clips:

        clip_path = Path(
            str(clip)
        )

        if not clip_path.is_absolute():
            clip_path = (
                Path.cwd()
                / clip_path
            )

        clip_path = clip_path.resolve()

        if not clip_path.exists():
            print(
                f"[WARNING] Missing clip: "
                f"{clip_path}"
            )
            continue

        if not clip_path.is_file():
            print(
                f"[WARNING] Not a file: "
                f"{clip_path}"
            )
            continue

        clip_paths.append(
            clip_path
        )

    if not clip_paths:
        fail(
            "AutoClipper generated no valid delivery clips."
        )

    print(
        f"[✓] Valid delivery clips: "
        f"{len(clip_paths)}"
    )

    for index, clip_path in enumerate(
        clip_paths,
        start=1,
    ):
        print(
            f"{index:03d}: {clip_path.name}"
        )

    # ------------------------------------------------------------------------
    # STEP 6
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 5: LOADING SHOT CLASSIFIER")
    print("=" * 72)

    model = load_shot_model()

    # ------------------------------------------------------------------------
    # STEP 7
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 6: CLASSIFYING EACH DELIVERY")
    print("=" * 72)

    successful = 0
    failed = 0

    results = []

    for delivery_number, clip_path in enumerate(
        clip_paths,
        start=1,
    ):

        print()
        print(
            f"Delivery "
            f"{delivery_number}/{len(clip_paths)}"
        )

        print(
            f"Clip: {clip_path.name}"
        )

        try:

            result = predict_delivery(
                model,
                clip_path,
            )

            successful += 1

            results.append(
                {
                    "delivery": delivery_number,
                    "clip": clip_path.name,
                    **result,
                }
            )

            print(
                f"Prediction : "
                f"{result['prediction']}"
            )

            print(
                f"Confidence : "
                f"{result['confidence'] * 100:.2f}%"
            )

            print(
                "Status     : PASS"
            )

        except Exception as exc:

            failed += 1

            print(
                "Status     : FAILED"
            )

            print(
                f"Error      : {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print(
        f"Delivery clips generated : {len(clip_paths)}"
    )

    print(
        f"Successful predictions   : {successful}"
    )

    print(
        f"Failed predictions       : {failed}"
    )

    if results:

        print()
        print(
            f"{'Delivery':<12}"
            f"{'Prediction':<18}"
            f"{'Confidence':<12}"
        )

        print(
            "-" * 42
        )

        for result in results:

            print(
                f"{result['delivery']:<12}"
                f"{result['prediction']:<18}"
                f"{result['confidence'] * 100:>8.2f}%"
            )

    # ------------------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------------------

    print()

    if failed > 0:

        print("=" * 72)
        print("TEST FAILED")
        print("=" * 72)

        raise SystemExit(1)

    print("=" * 72)
    print("END-TO-END TEST PASSED")
    print("=" * 72)

    print()
    print(
        f"✓ {successful} delivery clips "
        "successfully processed."
    )

    print("✓ AutoClipper")
    print("✓ Delivery clip validation")
    print("✓ 30-frame sampling")
    print("✓ Aspect-ratio-preserving preprocessing")
    print("✓ ImageNet normalization")
    print("✓ EfficientNet-B0 + Transformer")
    print("✓ Strict checkpoint loading")
    print("✓ Per-delivery prediction")
    print("✓ Probability validation")

    print()
    print(
        "NOTE: This test validates inference execution "
        "and checkpoint compatibility. It does not "
        "measure classification accuracy."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()