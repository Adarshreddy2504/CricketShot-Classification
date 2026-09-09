import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, resnet50
from src.data.improved_cricket_dataset import ImprovedCricketDataset

from common.evaluation_utils import (
    evaluate_model,
    save_results,
    print_evaluation_results
)

from common.seeding import set_seeds

# ============================================================
# Reproducibility
# ============================================================

set_seeds()


# ============================================================
# Spatial Attention
# ============================================================

class SpatialAttention(nn.Module):
    """
    Spatial/channel attention mechanism.

    Input:
        (B*T, C, H, W)

    Output:
        (B*T, C, H, W)
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(
                feature_dim,
                feature_dim // 8
            ),

            nn.ReLU(),

            nn.Linear(
                feature_dim // 8,
                feature_dim
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        attention_weights = self.attention(x)

        attention_weights = (
            attention_weights
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        return x * attention_weights


# ============================================================
# Temporal Attention
# ============================================================

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism.

    Input:
        (B, T, D)

    Output:
        attended features: (B, D)
        attention weights: (B, T, 1)
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                feature_dim,
                feature_dim // 4
            ),

            nn.ReLU(),

            nn.Linear(
                feature_dim // 4,
                1
            ),

            nn.Softmax(dim=1)
        )

    def forward(self, x):

        attention_weights = self.attention(x)

        attended_features = (
            x * attention_weights
        ).sum(dim=1)

        return (
            attended_features,
            attention_weights
        )


# ============================================================
# Improved SOTA Model
# ============================================================

class ImprovedSOTAModel(pl.LightningModule):
    """
    Improved cricket shot classification model.

    Architecture:

        Video
          ↓
        30 Frames
          ↓
        EfficientNet-B0
          ↓
        Spatial Attention
          ↓
        2-Layer Bidirectional GRU
          ↓
        Temporal Attention
          ↓
        Fully Connected Classifier
          ↓
        10 Cricket Shot Classes
    """

    def __init__(
        self,
        num_classes=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        backbone="efficientnet",
        temporal_dim=256,
        n_frames=30
    ):

        super().__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_frames = n_frames

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        if backbone == "efficientnet":

            self.backbone = efficientnet_b0(
                weights="IMAGENET1K_V1"
            )

            self.feature_dim = 1280

            # Remove original ImageNet classifier
            self.backbone.classifier = nn.Identity()

        elif backbone == "resnet50":

            self.backbone = resnet50(
                weights="IMAGENET1K_V1"
            )

            self.feature_dim = 2048

            # Remove original ImageNet classifier
            self.backbone.fc = nn.Identity()

        else:

            raise ValueError(
                f"Unsupported backbone: {backbone}"
            )

        # ----------------------------------------------------
        # Spatial Attention
        # ----------------------------------------------------

        self.spatial_attention = SpatialAttention(
            self.feature_dim
        )

        # ----------------------------------------------------
        # Bidirectional GRU
        # ----------------------------------------------------

        self.temporal_encoder = nn.GRU(

            input_size=self.feature_dim,

            hidden_size=temporal_dim,

            num_layers=2,

            batch_first=True,

            dropout=0.2,

            bidirectional=True
        )

        # ----------------------------------------------------
        # Temporal Attention
        # ----------------------------------------------------

        self.temporal_attention = TemporalAttention(
            temporal_dim * 2
        )

        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                temporal_dim * 2,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                512,
                num_classes
            )
        )

        # ----------------------------------------------------
        # Base criterion
        # ----------------------------------------------------

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1
        )

        # ----------------------------------------------------
        # Class weights
        # ----------------------------------------------------

        self.register_buffer(
            "class_weights",
            torch.ones(num_classes)
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, x):

        # Expected:
        #
        # B = batch size
        # T = number of frames
        # C = channels
        # H = height
        # W = width
        #
        # Shape:
        # (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # ----------------------------------------------------
        # Flatten batch and temporal dimensions
        # ----------------------------------------------------

        x = x.view(
            B * T,
            C,
            H,
            W
        )

        # ----------------------------------------------------
        # EfficientNet feature extraction
        # ----------------------------------------------------

        spatial_features = self.backbone(x)

        # Shape:
        # (B*T, 1280)

        # ----------------------------------------------------
        # Spatial attention
        # ----------------------------------------------------

        spatial_features = (
            spatial_features
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        spatial_features = (
            self.spatial_attention(
                spatial_features
            )
        )

        spatial_features = (
            spatial_features
            .squeeze(-1)
            .squeeze(-1)
        )

        # Shape:
        # (B*T, 1280)

        # ----------------------------------------------------
        # Restore temporal dimension
        # ----------------------------------------------------

        temporal_features = spatial_features.view(
            B,
            T,
            self.feature_dim
        )

        # Shape:
        # (B, T, 1280)

        # ----------------------------------------------------
        # Bidirectional GRU
        # ----------------------------------------------------

        temporal_output, _ = (
            self.temporal_encoder(
                temporal_features
            )
        )

        # Shape:
        # (B, T, 512)

        # ----------------------------------------------------
        # Temporal attention
        # ----------------------------------------------------

        attended_features, _ = (
            self.temporal_attention(
                temporal_output
            )
        )

        # Shape:
        # (B, 512)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            attended_features
        )

        # Shape:
        # (B, 10)

        return logits

    # ========================================================
    # Training Step
    # ========================================================

    def training_step(
        self,
        batch,
        batch_idx
    ):

        videos, labels = batch

        logits = self(videos)

        loss = F.cross_entropy(

            logits,

            labels,

            weight=self.class_weights,

            label_smoothing=0.1
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        acc = (
            torch.sum(
                preds == labels
            ).float()
            / len(labels)
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    # ========================================================
    # Validation Step
    # ========================================================

    def validation_step(
        self,
        batch,
        batch_idx
    ):

        videos, labels = batch

        logits = self(videos)

        loss = F.cross_entropy(

            logits,

            labels,

            weight=self.class_weights,

            label_smoothing=0.1
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        acc = (
            torch.sum(
                preds == labels
            ).float()
            / len(labels)
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return {
            "val_loss": loss,
            "val_acc": acc
        }

    # ========================================================
    # Test Step
    # ========================================================

    def test_step(
        self,
        batch,
        batch_idx
    ):

        videos, labels = batch

        logits = self(videos)

        loss = F.cross_entropy(

            logits,

            labels,

            weight=self.class_weights,

            label_smoothing=0.1
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        acc = (
            torch.sum(
                preds == labels
            ).float()
            / len(labels)
        )

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "test_acc",
            acc,
            on_step=False,
            on_epoch=True
        )

        return {
            "test_loss": loss,
            "test_acc": acc,
            "preds": preds,
            "labels": labels
        }

    # ========================================================
    # Optimizer
    # ========================================================

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(

            self.parameters(),

            lr=self.learning_rate,

            weight_decay=self.weight_decay,

            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.StepLR(

            optimizer,

            step_size=10,

            gamma=0.5
        )

        return {

            "optimizer": optimizer,

            "lr_scheduler": {

                "scheduler": scheduler,

                "interval": "epoch"
            }
        }

    # ========================================================
    # Calculate Class Weights
    # ========================================================

    def update_class_weights(
        self,
        labels,
        class_names
    ):
        """
        Calculate class weights directly from labels.

        IMPORTANT:
        This does NOT load videos.

        It only counts the already-known labels.
        """

        class_counts = torch.zeros(
            self.num_classes,
            dtype=torch.float32
        )

        class_to_idx = {
            name: idx
            for idx, name in enumerate(
                class_names
            )
        }

        # ----------------------------------------------------
        # Count samples
        # ----------------------------------------------------

        for label in labels:

            class_idx = class_to_idx[label]

            class_counts[class_idx] += 1

        # ----------------------------------------------------
        # Prevent division by zero
        # ----------------------------------------------------

        class_counts = torch.clamp(
            class_counts,
            min=1
        )

        # ----------------------------------------------------
        # Total number of samples
        # ----------------------------------------------------

        total_samples = (
            class_counts.sum()
        )

        # ----------------------------------------------------
        # Inverse-frequency weights
        # ----------------------------------------------------

        class_weights = (

            total_samples
            /
            (
                self.num_classes
                * class_counts
            )
        )

        # ----------------------------------------------------
        # Normalize weights
        # ----------------------------------------------------

        class_weights = (

            class_weights
            /
            class_weights.sum()
            *
            self.num_classes
        )

        # ----------------------------------------------------
        # Store weights
        # ----------------------------------------------------

        self.class_weights = (
            class_weights.to(
                self.device
            )
        )

        print()
        print(
            "Class counts:"
        )

        print(
            class_counts.tolist()
        )

        print()
        print(
            "Updated class weights:"
        )

        print(
            class_weights.tolist()
        )


# ============================================================
# READ EXISTING DATASET SPLIT
# ============================================================

def collect_fixed_split(
    dataset_path
):
    """
    Read the user's EXISTING:

        train/
        val/
        test/

    folders.

    IMPORTANT:
    No new split is created.
    No videos are moved.
    No videos are copied.
    """

    dataset_path = pathlib.Path(
        dataset_path
    )

    split_names = [
        "train",
        "val",
        "test"
    ]

    # --------------------------------------------------------
    # Get class names from train folder
    # --------------------------------------------------------

    train_dir = (
        dataset_path / "train"
    )

    if not train_dir.is_dir():

        raise FileNotFoundError(
            f"Training folder not found: "
            f"{train_dir}"
        )

    class_names = sorted([

        p.name

        for p in train_dir.iterdir()

        if p.is_dir()
    ])

    if len(class_names) == 0:

        raise RuntimeError(
            "No class folders found "
            "inside the train directory."
        )

    # --------------------------------------------------------
    # Collect every split
    # --------------------------------------------------------

    splits = {}

    for split in split_names:

        split_dir = (
            dataset_path / split
        )

        if not split_dir.is_dir():

            raise FileNotFoundError(
                f"Missing split folder: "
                f"{split_dir}"
            )

        paths = []
        labels = []

        # ----------------------------------------------------
        # Process every class
        # ----------------------------------------------------

        for class_name in class_names:

            class_dir = (
                split_dir / class_name
            )

            if not class_dir.is_dir():

                raise FileNotFoundError(
                    f"Missing class folder: "
                    f"{class_dir}"
                )

            # ------------------------------------------------
            # Find videos
            # ------------------------------------------------

            for path in sorted(
                class_dir.iterdir()
            ):

                if path.suffix.lower() in {

                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv"

                }:

                    paths.append(path)

                    labels.append(
                        class_name
                    )

        splits[split] = (
            paths,
            labels
        )

    return (
        splits,
        class_names
    )


# ============================================================
# MAIN
# ============================================================

def main():

    from pytorch_lightning import Trainer

    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor
    )

    from pytorch_lightning.loggers import (
        TensorBoardLogger
    )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    dataset_path = pathlib.Path(
        r"D:\cricket_project\CricketShot-Classification\cricketshot"
    )

    batch_size = 4

    n_frames = 30

    max_epochs = 15

    # Windows stability:
    # Start with 2workers.
    num_workers = 2

    # ========================================================
    # Check dataset
    # ========================================================

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{dataset_path}\n\n"
            f"Check that this path contains:\n"
            f"train\\\n"
            f"val\\\n"
            f"test\\"
        )

    # ========================================================
    # READ EXISTING SPLIT
    # ========================================================

    splits, class_names = (
        collect_fixed_split(
            dataset_path
        )
    )

    X_train, y_train = (
        splits["train"]
    )

    X_val, y_val = (
        splits["val"]
    )

    X_test, y_test = (
        splits["test"]
    )

    # ========================================================
    # Print dataset information
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "Dataset split "
        "(using existing folders)"
    )

    print(
        "NO new dataset split is being created."
    )

    print(
        "=" * 60
    )

    print(
        f"Train: {len(y_train)} samples"
    )

    print(
        f"Val:   {len(y_val)} samples"
    )

    print(
        f"Test:  {len(y_test)} samples"
    )

    print(
        f"Classes: {class_names}"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # DATASETS
    # ========================================================

    print()
    print(
        "Creating datasets..."
    )

    train_dataset = ImprovedCricketDataset(

        X_train,

        class_names,

        y_train,

        n_frames=n_frames,

        training=True,

        image_size=224,

        use_augmentation=True
    )

    val_dataset = ImprovedCricketDataset(

        X_val,

        class_names,

        y_val,

        n_frames=n_frames,

        training=False,

        image_size=224,

        use_augmentation=False
    )

    test_dataset = ImprovedCricketDataset(

        X_test,

        class_names,

        y_test,

        n_frames=n_frames,

        training=False,

        image_size=224,

        use_augmentation=False
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    print(
        "Creating data loaders..."
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=True,

        persistent_workers=(
            num_workers > 0
        )
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,

        persistent_workers=(
            num_workers > 0
        )
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,

        persistent_workers=(
            num_workers > 0
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Creating Improved SOTA model..."
    )

    model = ImprovedSOTAModel(

        num_classes=len(
            class_names
        ),

        learning_rate=1e-3,

        weight_decay=1e-4,

        backbone="efficientnet",

        temporal_dim=256,

        n_frames=n_frames
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print()
    print(
        "Calculating class weights..."
    )

    # IMPORTANT:
    # This uses y_train directly.
    #
    # It does NOT iterate through train_loader.
    # Therefore videos are NOT decoded here.

    model.update_class_weights(
        y_train,
        class_names
    )

    # ========================================================
    # CHECKPOINT DIRECTORY
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ========================================================
    # MODEL CHECKPOINT
    # ========================================================

    checkpoint_callback = ModelCheckpoint(

        monitor="val_acc",

        dirpath="models/",

        filename=(
            "improved-sota-"
            "{epoch:02d}-"
            "{val_acc:.4f}"
        ),

        save_top_k=3,

        save_last=True,

        mode="max",

        save_on_train_epoch_end=False
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stop_callback = EarlyStopping(

        monitor="val_loss",

        patience=4,

        mode="min",

        verbose=True
    )

    # ========================================================
    # LEARNING RATE MONITOR
    # ========================================================

    lr_monitor = LearningRateMonitor(
        logging_interval="epoch"
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(

        max_epochs=max_epochs,

        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor
        ],

        logger=TensorBoardLogger(

            "lightning_logs/",

            name="improved_sota_cricket"
        ),

        accelerator="auto",

        devices="auto",

        precision="16-mixed",

        gradient_clip_val=1.0,

        log_every_n_steps=10,

        accumulate_grad_batches=2
    )

    # ========================================================
    # GPU
    # ========================================================

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        print()
        print(
            "=" * 60
        )

        print(
            "GPU detected:"
        )

        print(
            torch.cuda.get_device_name(0)
        )

        print(
            "=" * 60
        )

    else:

        print()
        print(
            "WARNING: CUDA GPU not detected."
        )

        print(
            "Training will run on CPU."
        )

    # ========================================================
    # RESUME CHECKPOINT
    # ========================================================

    last_checkpoint = pathlib.Path(
        "models/last.ckpt"
    )

    if last_checkpoint.exists():

        resume_path = str(
            last_checkpoint
        )

        print()
        print(
            "=" * 60
        )

        print(
            "Previous checkpoint found."
        )

        print(
            f"Resuming from:"
        )

        print(
            resume_path
        )

        print(
            "=" * 60
        )

    else:

        resume_path = None

        print()
        print(
            "No previous checkpoint found."
        )

        print(
            "Starting from Epoch 1."
        )

    # ========================================================
    # START TRAINING
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "🚀 Starting Improved SOTA "
        "model training..."
    )

    print(
        "=" * 60
    )

    trainer.fit(

        model,

        train_loader,

        val_loader,

        ckpt_path=resume_path
    )

    # ========================================================
    # TEST BEST MODEL
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "🔍 Evaluating best model "
        "on test set..."
    )

    print(
        "=" * 60
    )

    trainer.test(

        model,

        test_loader,

        ckpt_path="best"
    )

    # ========================================================
    # BEST CHECKPOINT PATH
    # ========================================================

    best_path = (
        checkpoint_callback
        .best_model_path
    )

    print()
    print(
        f"Best checkpoint:"
    )

    print(
        best_path
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print(
        "Loading best checkpoint "
        "for detailed evaluation..."
    )

    best_model = (
        ImprovedSOTAModel
        .load_from_checkpoint(

            best_path,

            num_classes=len(
                class_names
            ),

            n_frames=n_frames
        )
    )

    best_model.eval()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    best_model = best_model.to(
        device
    )

    # ========================================================
    # DETAILED EVALUATION
    # ========================================================

    print()
    print(
        "Running detailed evaluation..."
    )

    results = evaluate_model(

        best_model,

        test_loader,

        class_names,

        device
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print_evaluation_results(

        results,

        "Improved SOTA",

        class_names
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_results(

        results,

        "improved_sota",

        output_dir="results"
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "✅ Improved SOTA model "
        "training and evaluation completed!"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()