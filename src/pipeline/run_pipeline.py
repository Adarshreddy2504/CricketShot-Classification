import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import your custom modules
from src.segmentation.auto_clipper import AutoClipper
from src.pipeline.extract_frames import extract_frames
from src.pipeline.efficientnet_gru import EfficientNetGRU # Update if your model file/class is named differently

# The 10 target classes
SHOT_CLASSES = [
    "Cover", "Defense", "Flick", "Hook", "Late Cut",
    "Lofted", "Pull", "Square Cut", "Straight", "Sweep"
]

def load_inference_model(checkpoint_path, device):
    """Loads the model checkpoint for inference."""
    model = EfficientNetGRU(num_classes=len(SHOT_CLASSES))
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Strip PyTorch Lightning 'model.' prefix if present
    if "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.to(device)
    model.eval()
    return model

def load_and_preprocess_frames(frames_dir):
    """Loads the 30 saved PNGs from disk and applies ImageNet normalization."""
    # Your extractor already resized them to 224x224, so we just need tensor conversion and normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_paths = sorted(Path(frames_dir).glob("*.png"))
    if len(frame_paths) != 30:
        return None
        
    tensor_list = []
    for fp in frame_paths:
        # Load image (PIL loads as RGB)
        img = Image.open(fp).convert("RGB")
        tensor_list.append(transform(img))
        
    # Stack into (30, 3, 224, 224) and add batch dimension -> (1, 30, 3, 224, 224)
    return torch.stack(tensor_list, dim=0).unsqueeze(0)

def main():
    project_root = Path(__file__).resolve().parents[2]
    
    # Paths setup
    raw_match_path = sys.argv[1] if len(sys.argv) > 1 else str(project_root / "data" / "raw_matches" / "long_match.mp4")
    clips_dir = project_root / "data" / "delivery_clips"
    frames_base_dir = project_root / "data" / "frames"
    output_csv = project_root / "data" / "shot_predictions.csv"
    
    checkpoint_path = project_root / "models" / "cricket_model.ckpt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Cricket model checkpoint not found: {checkpoint_path}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Executing pipeline on device: {device}")

    # ==========================================
    # STAGE 1: CLIP GENERATION
    # ==========================================
    print("\n--- STAGE 1: Slicing Deliveries (AutoClipper) ---")
    clipper = AutoClipper(
        ball_model_path=str(project_root / "models" / "Ball_Detection_Model.pt"),
        bat_model_path=str(project_root / "models" / "Bat_Detection_Model.pt"),
        output_dir=str(clips_dir)
    )
    extracted_clips = clipper.process_match(raw_match_path, clip_duration_sec=1.0)
    
    if not extracted_clips:
        print("[-] No clips detected. Exiting pipeline.")
        return

    # ==========================================
    # STAGE 2 & 3: EXTRACTION & PREDICTION
    # ==========================================
    print("\n--- STAGE 2: Loading Model ---")
    model = load_inference_model(str(checkpoint_path), device)
    
    print("\n--- STAGE 3: Extracting Frames & Classifying ---")
    prediction_records = []

    with torch.no_grad():
        for clip_path in extracted_clips:
            clip_path_obj = Path(clip_path)
            delivery_id = clip_path_obj.stem
            clip_frames_dir = frames_base_dir / delivery_id
            
            # Extract frames to disk using your custom script logic
            success = extract_frames(clip_path, clip_frames_dir)
            if not success:
                continue
                
            # Load frames from disk into tensor
            input_tensor = load_and_preprocess_frames(clip_frames_dir)
            if input_tensor is None:
                print(f"[WARNING] Skipping {delivery_id}: Frame count mismatch.")
                continue
                
            input_tensor = input_tensor.to(device)

            # Model Inference
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            top_class_idx = int(probabilities.argmax())
            top_class_name = SHOT_CLASSES[top_class_idx]
            confidence_score = float(probabilities[top_class_idx]) * 100.0

            # Store results
            row = {
                "delivery_id": delivery_id,
                "predicted_shot": top_class_name,
                "confidence": f"{confidence_score:.2f}%"
            }
            for idx, class_name in enumerate(SHOT_CLASSES):
                row[f"prob_{class_name}"] = f"{probabilities[idx] * 100.0:.2f}%"

            prediction_records.append(row)
            print(f"  -> {delivery_id} : {top_class_name} ({confidence_score:.2f}%)")

    # ==========================================
    # STAGE 4: EXPORT RESULTS
    # ==========================================
    if prediction_records:
        df = pd.DataFrame(prediction_records)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n[+] SUCCESS! Predictions exported to: {output_csv}")
    else:
        print("\n[-] Pipeline completed, but no predictions were generated.")

if __name__ == "__main__":
    main()