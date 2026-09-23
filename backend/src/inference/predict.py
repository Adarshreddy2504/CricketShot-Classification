import sys
import pathlib
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.training.sota_ours import ImprovedSOTAModel


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT = pathlib.Path(
    r"models\improved-sota-epoch=12-val_acc=0.7840.ckpt"
)

CLASS_NAMES = [
    "cover",
    "defense",
    "flick",
    "hook",
    "late_cut",
    "lofted",
    "pull",
    "square_cut",
    "straight",
    "sweep"
]

N_FRAMES = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames(video_path, n_frames=30):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise RuntimeError("Could not determine number of frames.")

    indices = np.linspace(
        0,
        total_frames - 1,
        n_frames
    ).astype(int)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    # If fewer than 30 frames were successfully read,
    # repeat the last frame.
    if len(frames) == 0:
        raise RuntimeError("Could not read any frames.")

    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return frames[:n_frames]


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("Loading model...")

    model = ImprovedSOTAModel(
        num_classes=10,
        backbone="efficientnet"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    # Lightning checkpoint
    state_dict = checkpoint["state_dict"]

    # Remove possible "model." prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    model.to(DEVICE)
    model.eval()

    print(f"Model loaded on: {DEVICE}")

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(video_path):

    video_path = pathlib.Path(video_path)

    if not video_path.exists():
        print(f"ERROR: Video not found:")
        print(video_path)
        return

    print()
    print("=" * 60)
    print("CRICKET SHOT PREDICTION")
    print("=" * 60)

    print(f"Video: {video_path.name}")

    frames = extract_frames(
        video_path,
        N_FRAMES
    )

    # Convert frames to tensors
    tensors = []

    for frame in frames:
        tensors.append(transform(frame))

    # Shape:
    # (30, 3, 224, 224)
    x = torch.stack(tensors)

    # Add batch dimension:
    # (1, 30, 3, 224, 224)
    x = x.unsqueeze(0)

    x = x.to(DEVICE)

    model = load_model()

    with torch.no_grad():
        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ].item() * 100

    print()
    print("PREDICTION")
    print("-" * 60)

    print(f"Shot       : {predicted_class.upper()}")
    print(f"Confidence : {confidence:.2f}%")

    print()
    print("ALL CLASS PROBABILITIES")
    print("-" * 60)

    results = []

    for i, class_name in enumerate(CLASS_NAMES):
        prob = probabilities[i].item() * 100
        results.append(
            (class_name, prob)
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for class_name, prob in results:
        print(
            f"{class_name:<15} {prob:6.2f}%"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print()
        print("Usage:")
        print(
            r'python predict.py "path\to\video.avi"'
        )
        print()
        sys.exit(1)

    predict(sys.argv[1])