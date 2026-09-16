"""
Standalone ImprovedCricketDataset implementation for cricket shot classification.

This module provides a self-contained video dataset class that:
- Uses uniform frame sampling across the entire video
- Applies ImageNet-standard normalization
- Supports optional training augmentations
- Handles videos with variable lengths

Usage:
    from improved_cricket_dataset import ImprovedCricketDataset

    dataset = ImprovedCricketDataset(
        video_paths=['path/to/video1.mp4', 'path/to/video2.mp4'],
        class_names=['cover_drive', 'pull_shot', 'cut_shot'],
        classes=['cover_drive', 'pull_shot'],
        n_frames=30,
        training=True
    )

    video_tensor, label = dataset[0]
"""

import cv2
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import random
import torchvision.transforms.functional as TF


def extract_frames_uniform(video_path: str, n_frames: int = 30) -> List[np.ndarray]:
    """
    Extract frames using uniform sampling across the entire video.

    This approach ensures temporal coverage across the full video by sampling
    frames at evenly-spaced intervals.

    Args:
        video_path: Path to video file
        n_frames: Number of frames to extract

    Returns:
        List of RGB frames as numpy arrays (H, W, 3)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < n_frames:
        # If video has fewer frames than requested, use all frames and pad
        frame_indices = list(range(total_frames))
        # Pad with repeated frames if needed
        while len(frame_indices) < n_frames:
            frame_indices.extend(frame_indices[:n_frames - len(frame_indices)])
    else:
        # Sample uniformly across the video
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert from BGR (OpenCV format) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # Use last valid frame if read fails
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()
    return frames


class ResizeWithPadding:
    """Resize while preserving aspect ratio, then pad to a square."""
    def __init__(self, image_size=224, fill=0):
        self.image_size = image_size
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        scale = min(self.image_size / width, self.image_size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        left = (self.image_size - new_width) // 2
        top = (self.image_size - new_height) // 2
        right = self.image_size - new_width - left
        bottom = self.image_size - new_height - top
        return transforms.functional.pad(img, [left, top, right, bottom], fill=self.fill)


class ConsistentVideoTransform:
    """Applies identically parameterized augmentations to a sequence of 30 frames."""
    
    def __init__(self, training: bool = False, image_size: int = 224):
        self.training = training
        self.image_size = image_size
        self.resize = ResizeWithPadding(image_size)
        
        # Standard ImageNet normalization values
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, frames: List[np.ndarray]) -> torch.Tensor:
        # 1. Base conversion & resize (Deterministic)
        pil_frames = [TF.to_pil_image(frame) for frame in frames]
        pil_frames = [self.resize(frame) for frame in pil_frames]

        # 2. Sequence-Level Augmentation (Apply SAME params to ALL frames)
        if self.training:
            # -- Decide random parameters ONCE per video clip --
            apply_flip = random.random() < 0.5
            angle = transforms.RandomRotation.get_params([-10.0, 10.0])
            
            # ColorJitter params (base 1.0, range +/- 0.2, hue +/- 0.1)
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)

            # -- Apply identically to the entire sequence --
            transformed = []
            for img in pil_frames:
                if apply_flip:
                    img = TF.hflip(img)
                img = TF.rotate(img, angle)
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.adjust_hue(img, hue)
                transformed.append(img)
                
            pil_frames = transformed

        # 3. Final conversion & normalization
        tensor_frames = [TF.to_tensor(frame) for frame in pil_frames]
        tensor_frames = [TF.normalize(t, self.mean, self.std) for t in tensor_frames]

        # Stack into (30, 3, 224, 224)
        return torch.stack(tensor_frames)

class ImprovedCricketDataset(Dataset):
    """
    Improved cricket shot video dataset using uniform sampling and standard transforms.

    This dataset class:
    - Extracts frames uniformly across the entire video duration
    - Applies ImageNet-standard preprocessing
    - Supports optional training augmentations
    - Handles edge cases (short videos, read failures)

    Attributes:
        video_paths (List[str]): Paths to video files
        class_names (List[str]): List of all possible class names
        classes (List[str]): Class label for each video
        n_frames (int): Number of frames to extract per video
        training (bool): Whether to apply training augmentations
        image_size (int): Target image size
        class_to_idx (dict): Mapping from class names to indices
        transforms (transforms.Compose): Transform pipeline

    Example:
        >>> dataset = ImprovedCricketDataset(
        ...     video_paths=['video1.mp4', 'video2.mp4'],
        ...     class_names=['cover_drive', 'pull_shot'],
        ...     classes=['cover_drive', 'pull_shot'],
        ...     n_frames=30,
        ...     training=True
        ... )
        >>> video_tensor, label = dataset[0]
        >>> print(video_tensor.shape)  # (30, 3, 224, 224)
    """

    def __init__(self, video_paths: List[str], class_names: List[str],
                 classes: List[str], n_frames: int = 30, training: bool = False,
                 image_size: int = 224, use_augmentation: Optional[bool] = None):
        """
        Initialize the dataset.

        Args:
            video_paths: List of paths to video files
            class_names: List of all possible class names (for creating class_to_idx mapping)
            classes: List of class labels, one for each video in video_paths
            n_frames: Number of frames to extract from each video (default: 30)
            training: Whether to apply training augmentations (default: False)
            image_size: Target image size (default: 224)
            use_augmentation: Legacy parameter, same as 'training' (for backward compatibility)
        """
        self.video_paths = video_paths
        self.class_names = class_names
        self.classes = classes
        self.n_frames = n_frames
        self.image_size = image_size

        # Handle legacy parameter: use_augmentation is an alias for training
        if use_augmentation is not None:
            self.training = use_augmentation
        else:
            self.training = training

        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.sequence_transform = ConsistentVideoTransform(self.training, self.image_size)

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video using uniform sampling.

        Args:
            video_path: Path to video file

        Returns:
            List of RGB frames as numpy arrays
        """
        return extract_frames_uniform(video_path, self.n_frames)

    def transform_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Transform frames to tensor with temporally consistent normalization.
        """
        return self.sequence_transform(frames)

    def __len__(self) -> int:
        """Return the total number of videos in the dataset."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a video and its label.

        Args:
            idx: Index of the video

        Returns:
            Tuple of (video_tensor, label_tensor)
            - video_tensor: Shape (n_frames, 3, image_size, image_size)
            - label_tensor: Shape (1,) with class index
        """
        video_path = self.video_paths[idx]
        class_name = self.classes[idx]
        label = self.class_to_idx[class_name]

        # Extract and transform frames
        frames = self.extract_frames(video_path)
        video_tensor = self.transform_frames(frames)

        return video_tensor, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    """
    Example usage demonstrating how to use the ImprovedCricketDataset.
    """
    print("ImprovedCricketDataset - Standalone Implementation")
    print("=" * 60)

    # Example configuration
    example_video_paths = [
        'path/to/video1.mp4',
        'path/to/video2.mp4',
        'path/to/video3.mp4',
    ]

    example_class_names = [
        'cover_drive', 'pull_shot', 'cut_shot', 'straight_drive',
        'square_drive', 'flick', 'late_cut', 'sweep', 'lofted', 'defensive'
    ]

    example_classes = ['cover_drive', 'pull_shot', 'cut_shot']

    # Create dataset instances
    print("\n1. Creating training dataset...")
    train_dataset = ImprovedCricketDataset(
        video_paths=example_video_paths,
        class_names=example_class_names,
        classes=example_classes,
        n_frames=30,
        training=True  # Enables augmentations
    )

    print("\n2. Creating validation dataset...")
    val_dataset = ImprovedCricketDataset(
        video_paths=example_video_paths,
        class_names=example_class_names,
        classes=example_classes,
        n_frames=30,
        training=False  # No augmentations
    )

    # Test backward compatibility with legacy parameters
    print("\n3. Testing backward compatibility with legacy parameters...")
    legacy_dataset = ImprovedCricketDataset(
        video_paths=example_video_paths,
        class_names=example_class_names,
        classes=example_classes,
        n_frames=30,
        image_size=224,
        use_augmentation=True  # Legacy parameter
    )

    print(f"\nDataset size: {len(train_dataset)} videos")
    print(f"Number of classes: {len(example_class_names)}")
    print(f"Class to index mapping: {train_dataset.class_to_idx}")

    print("\n4. Dataset properties:")
    print(f"   - Frame extraction: Uniform sampling")
    print(f"   - Frames per video: {train_dataset.n_frames}")
    print(f"   - Frame size: {train_dataset.image_size}x{train_dataset.image_size}")
    print(f"   - Normalization: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
    print(f"   - Training augmentations: Random flip, rotation, color jitter")

    print("\n5. Usage with DataLoader:")
    print("""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    for batch_videos, batch_labels in train_loader:
        # batch_videos shape: (batch_size, n_frames, 3, 224, 224)
        # batch_labels shape: (batch_size,)
        pass
    """)

    print("\nNote: Update example_video_paths with actual video file paths to test.")