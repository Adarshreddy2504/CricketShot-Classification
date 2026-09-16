import os
import pathlib
import json
import os

# Disable PyTorch internal FLOP-counter warning
os.environ["TORCH_LOGS"] = "-torch.utils.flop_counter"

import logging
logging.getLogger("torch.utils.flop_counter").disabled = True

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*treespec.*LeafSpec.*deprecated.*"
)
logging.getLogger("pytorch_lightning.utilities.seed").setLevel(logging.ERROR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from improved_dataset import ImprovedCricketDataset


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 27

pl.seed_everything(SEED, workers=True, verbose=False)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# SPATIAL / CHANNEL ATTENTION
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 8),
            nn.ReLU(),
            nn.Linear(feature_dim // 8, feature_dim),
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
# TEMPORAL ATTENTION
# ============================================================

class TemporalAttention(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, x):

        scores = self.attention(x).squeeze(-1)

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        attended_features = torch.sum(
            x * attention_weights.unsqueeze(-1),
            dim=1
        )

        return attended_features, attention_weights


# ============================================================
# MODEL
# ============================================================

class ImprovedSOTAModel(pl.LightningModule):

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

        # --------------------------------------------------------
        # BACKBONE
        # --------------------------------------------------------

        if backbone == "efficientnet":

            self.backbone = efficientnet_b0(
                weights="IMAGENET1K_V1"
            )

            self.feature_dim = 1280

            self.backbone.classifier = nn.Identity()

        elif backbone == "resnet50":

            self.backbone = resnet50(
                weights="IMAGENET1K_V1"
            )

            self.feature_dim = 2048

            self.backbone.fc = nn.Identity()

        else:

            raise ValueError(
                f"Unsupported backbone: {backbone}"
            )

        # --------------------------------------------------------
        # SPATIAL ATTENTION
        # --------------------------------------------------------

        self.spatial_attention = SpatialAttention(
            self.feature_dim
        )

        # --------------------------------------------------------
        # 2-LAYER BIDIRECTIONAL GRU
        # --------------------------------------------------------

        self.temporal_encoder = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=temporal_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # --------------------------------------------------------
        # TEMPORAL ATTENTION
        # --------------------------------------------------------

        self.temporal_attention = TemporalAttention(
            temporal_dim * 2
        )

        # --------------------------------------------------------
        # CLASSIFIER
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # CLASS WEIGHTS
        # --------------------------------------------------------

        self.register_buffer(
            "class_weights",
            torch.ones(num_classes)
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # x:
        # (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # ----------------------------------------------------
        # Combine batch and temporal dimensions
        # ----------------------------------------------------

        x = x.view(
            B * T,
            C,
            H,
            W
        )

        # ----------------------------------------------------
        # EfficientNet
        # ----------------------------------------------------

        spatial_features = self.backbone(x)

        # ----------------------------------------------------
        # Spatial attention
        # ----------------------------------------------------

        spatial_features = (
            spatial_features
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        spatial_features = self.spatial_attention(
            spatial_features
        )

        spatial_features = (
            spatial_features
            .squeeze(-1)
            .squeeze(-1)
        )

        # ----------------------------------------------------
        # Restore temporal dimension
        # ----------------------------------------------------

        temporal_features = spatial_features.view(
            B,
            T,
            self.feature_dim
        )

        # ----------------------------------------------------
        # BiGRU
        # ----------------------------------------------------

        temporal_output, _ = self.temporal_encoder(
            temporal_features
        )

        # ----------------------------------------------------
        # Temporal attention
        # ----------------------------------------------------

        attended_features, _ = self.temporal_attention(
            temporal_output
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        return self.classifier(
            attended_features
        )

    # ========================================================
    # LOSS
    # ========================================================

    def _loss(self, logits, labels):

        return F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=0.1
        )

    # ========================================================
    # TRAINING STEP
    # ========================================================

    def training_step(self, batch, batch_idx):

        videos, labels = batch

        logits = self(videos)

        loss = self._loss(
            logits,
            labels
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        acc = (
            preds == labels
        ).float().mean()

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
    # VALIDATION STEP
    # ========================================================

    def validation_step(self, batch, batch_idx):

        videos, labels = batch

        logits = self(videos)

        loss = self._loss(
            logits,
            labels
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        acc = (
            preds == labels
        ).float().mean()

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
    # TEST STEP
    # ========================================================

    def test_step(self, batch, batch_idx):

        videos, labels = batch

        logits = self(videos)

        loss = self._loss(
            logits,
            labels
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        acc = (
            preds == labels
        ).float().mean()

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
    # OPTIMIZER + E0 SCHEDULER
    # ========================================================

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # ====================================================
        # E2: KEEP THE SAME SCHEDULER AS THE EXISTING BASELINE CODE
        # ====================================================

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=30,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    def update_class_weights(
        self,
        labels,
        class_names
    ):

        class_counts = torch.zeros(
            self.num_classes,
            dtype=torch.float32
        )

        class_to_idx = {
            name: idx
            for idx, name in enumerate(class_names)
        }

        for label in labels:

            class_counts[
                class_to_idx[label]
            ] += 1

        class_counts = torch.clamp(
            class_counts,
            min=1
        )

        total_samples = class_counts.sum()

        class_weights = (
            total_samples
            /
            (
                self.num_classes
                *
                class_counts
            )
        )

        class_weights = (
            class_weights
            /
            class_weights.sum()
            *
            self.num_classes
        )

        self.class_weights = class_weights.to(
            self.device
        )

        print("\nClass counts:")
        print(class_counts.tolist())

        print("\nUpdated class weights:")
        print(class_weights.tolist())


# ============================================================
# EXISTING DATASET SPLIT
# ============================================================

def collect_fixed_split(dataset_path):

    dataset_path = pathlib.Path(
        dataset_path
    )

    split_names = [
        "train",
        "val",
        "test"
    ]

    train_dir = dataset_path / "train"

    if not train_dir.is_dir():

        raise FileNotFoundError(
            f"Training folder not found: {train_dir}"
        )

    # --------------------------------------------------------
    # Get class names from EXISTING train folder
    # --------------------------------------------------------

    class_names = sorted([
        p.name
        for p in train_dir.iterdir()
        if p.is_dir()
    ])

    if not class_names:

        raise RuntimeError(
            "No class folders found inside train."
        )

    splits = {}

    # --------------------------------------------------------
    # READ EXISTING TRAIN / VAL / TEST
    # --------------------------------------------------------

    for split in split_names:

        split_dir = (
            dataset_path / split
        )

        if not split_dir.is_dir():

            raise FileNotFoundError(
                f"Missing split folder: {split_dir}"
            )

        paths = []
        labels = []

        for class_name in class_names:

            class_dir = (
                split_dir / class_name
            )

            if not class_dir.is_dir():

                raise FileNotFoundError(
                    f"Missing class folder: {class_dir}"
                )

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
                    labels.append(class_name)

        splits[split] = (
            paths,
            labels
        )

    return splits, class_names


# ============================================================
# DETAILED EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataloader,
    class_names,
    device
):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for videos, labels in dataloader:

            videos = videos.to(device)

            logits = model(videos)

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    total = len(all_labels)

    correct = sum(
        p == y
        for p, y in zip(
            all_preds,
            all_labels
        )
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    num_classes = len(class_names)

    confusion_matrix = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    for true_label, pred_label in zip(
        all_labels,
        all_preds
    ):

        confusion_matrix[
            true_label
        ][
            pred_label
        ] += 1

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    per_class_accuracy = {}

    for i, class_name in enumerate(
        class_names
    ):

        class_total = sum(
            confusion_matrix[i]
        )

        class_correct = (
            confusion_matrix[i][i]
        )

        if class_total > 0:

            class_acc = (
                class_correct
                /
                class_total
                *
                100
            )

        else:

            class_acc = 0.0

        per_class_accuracy[
            class_name
        ] = class_acc

    # --------------------------------------------------------
    # Precision / Recall / F1
    # --------------------------------------------------------

    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):

        tp = confusion_matrix[i][i]

        fp = sum(
            confusion_matrix[j][i]
            for j in range(num_classes)
            if j != i
        )

        fn = sum(
            confusion_matrix[i][j]
            for j in range(num_classes)
            if j != i
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        class_support = sum(
            confusion_matrix[i]
        )

        precisions.append(
            precision * class_support
        )

        recalls.append(
            recall * class_support
        )

        f1_scores.append(
            f1 * class_support
        )

    weighted_precision = (
        sum(precisions) / total
        if total > 0
        else 0.0
    )

    weighted_recall = (
        sum(recalls) / total
        if total > 0
        else 0.0
    )

    weighted_f1 = (
        sum(f1_scores) / total
        if total > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix,
        "class_names": class_names
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_evaluation_results(
    results,
    title
):

    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Accuracy  : "
        f"{results['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{results['weighted_precision'] * 100:.4f}%"
    )

    print(
        f"Recall    : "
        f"{results['weighted_recall'] * 100:.4f}%"
    )

    print(
        f"F1 Score  : "
        f"{results['weighted_f1'] * 100:.4f}%"
    )

    print("\nPer-class accuracy:")

    for class_name, accuracy in (
        results["per_class_accuracy"].items()
    ):

        print(
            f"{class_name:<15}"
            f"{accuracy:.2f}%"
        )

    print("\nConfusion Matrix:")

    print(
        " " * 15 +
        " ".join(
            f"{name[:8]:>9}"
            for name in results["class_names"]
        )
    )

    for i, row in enumerate(
        results["confusion_matrix"]
    ):

        print(
            f"{results['class_names'][i]:<15}"
            +
            " ".join(
                f"{value:>9}"
                for value in row
            )
        )

    print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    output_dir,
    filename="model1_e0_results.json"
):

    output_dir = pathlib.Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / filename
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    from pytorch_lightning import Trainer

    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        Callback
    )

    class EpochMetricsPrinter(Callback):
        """Print the main training metrics after every validation epoch."""

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return

            metrics = trainer.callback_metrics

            def get_value(name):
                value = metrics.get(name)
                if value is None:
                    return None
                if hasattr(value, "detach"):
                    value = value.detach().cpu().item()
                return float(value)

            train_loss = get_value("train_loss")
            train_acc = get_value("train_acc")
            val_loss = get_value("val_loss")
            val_acc = get_value("val_acc")

            lr = None
            try:
                lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            except Exception:
                pass

            print("\\n" + "-" * 60)
            print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} Metrics")
            print("-" * 60)

            if train_loss is not None:
                print(f"Train Loss      : {train_loss:.4f}")
            if train_acc is not None:
                print(f"Train Accuracy  : {train_acc * 100:.2f}%")
            if val_loss is not None:
                print(f"Val Loss        : {val_loss:.4f}")
            if val_acc is not None:
                print(f"Val Accuracy    : {val_acc * 100:.2f}%")
            if lr is not None:
                print(f"Learning Rate   : {lr:.8f}")

            print("-" * 60)


    from pytorch_lightning.loggers import (
        TensorBoardLogger
    )

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    dataset_path = pathlib.Path(
        r"D:\Cricket\training\cricketshot"
    )

    batch_size = 4

    n_frames = 30

    max_epochs = 30

    num_workers = 2

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{dataset_path}"
        )

    # --------------------------------------------------------
    # USE EXISTING SPLIT
    # --------------------------------------------------------

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

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)

    print(
        f"Train: {len(y_train)}"
    )

    print(
        f"Val:   {len(y_val)}"
    )

    print(
        f"Test:  {len(y_test)}"
    )

    print(
        f"Classes: {class_names}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = ImprovedSOTAModel(
        num_classes=len(class_names),
        learning_rate=1e-3,
        weight_decay=1e-4,
        backbone="efficientnet",
        temporal_dim=256,
        n_frames=n_frames
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    model.update_class_weights(
        y_train,
        class_names
    )

    # --------------------------------------------------------
    # E0 OUTPUT DIRECTORIES
    # --------------------------------------------------------

    model_dir = pathlib.Path(
        "models_e2"
    )

    log_dir = pathlib.Path(
        "lightning_logs_e2"
    )

    result_dir = pathlib.Path(
        "results_e2"
    )

    model_dir.mkdir(
        exist_ok=True
    )

    result_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=str(model_dir),
        filename=(
            "model1-e2-{epoch:02d}-"
            "{val_acc:.4f}"
        ),
        save_top_k=3,
        save_last=True,
        mode="max",
        save_on_train_epoch_end=False
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=4,
    mode="min",
    verbose=True,
    check_on_train_epoch_end=False
)

    # --------------------------------------------------------
    # LEARNING RATE MONITOR
    # --------------------------------------------------------

    lr_monitor = LearningRateMonitor(
        logging_interval="epoch"
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(

        max_epochs=max_epochs,

        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            EpochMetricsPrinter()
        ],

        logger=TensorBoardLogger(
            str(log_dir),
            name="model1_e2_padding"
        ),

        accelerator="auto",

        devices="auto",

        precision="16-mixed",

        gradient_clip_val=1.0,

        log_every_n_steps=10,

        accumulate_grad_batches=2
    )

    # --------------------------------------------------------
    # E2 INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("MODEL 1 - E2 ASPECT-RATIO PADDING")
    print("CosineAnnealingLR (same as baseline code)")

    print("Automatic resume: ENABLED")

    print("Dataset split: EXISTING train/val/test")

    print("=" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    last_checkpoint = model_dir / "last.ckpt"

    if last_checkpoint.exists():
        print(f"\nResuming training from:\n{last_checkpoint}")
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=str(last_checkpoint)
        )
    else:
        print("\nNo previous checkpoint found. Starting fresh training.")
        trainer.fit(
            model,
            train_loader,
            val_loader
        )

    # --------------------------------------------------------
    # TEST BEST CHECKPOINT
    # --------------------------------------------------------

    print(
        "\nEvaluating best E2 checkpoint..."
    )

    trainer.test(
        model,
        test_loader,
        ckpt_path="best"
    )

    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    best_path = (
        checkpoint_callback.best_model_path
    )

    print(
        f"\nBest E2 checkpoint:\n"
        f"{best_path}"
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    best_model = (
        ImprovedSOTAModel.load_from_checkpoint(
            best_path,
            num_classes=len(class_names),
            n_frames=n_frames
        )
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    best_model = best_model.to(
        device
    )

    best_model.eval()

    # --------------------------------------------------------
    # DETAILED EVALUATION
    # --------------------------------------------------------

    results = evaluate_model(
        best_model,
        test_loader,
        class_names,
        device
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print_evaluation_results(
        results,
        "MODEL 1 - E2 ASPECT-RATIO PADDING"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_results(
        results,
        output_dir=str(result_dir),
        filename="model1_e2_results.json"
    )

    print("\n" + "=" * 60)
    print("E2 COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()