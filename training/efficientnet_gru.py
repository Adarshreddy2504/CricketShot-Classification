# Cricket shot classification model.
# Pipeline: EfficientNet-B0 frame features -> spatial attention -> 2-layer BiGRU
# -> temporal attention -> fully connected classifier for the 10 shot classes.

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class SpatialAttention(nn.Module):
    """
    Spatial/channel attention applied to EfficientNet feature vectors.
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 8, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        attention_weights = self.attention(x)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)

        return x * attention_weights


class TemporalAttention(nn.Module):
    """
    Attention over the temporal/frame dimension.
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, x):
        """
        x: (B, T, F)

        Returns:
            attended_features: (B, F)
            attention_weights: (B, T)
        """

        scores = self.attention(x).squeeze(-1)

        attention_weights = torch.softmax(scores, dim=1)

        attended_features = torch.sum(
            x * attention_weights.unsqueeze(-1),
            dim=1
        )

        return attended_features, attention_weights


class EfficientNetGRU(nn.Module):
    """
    Cricket Shot Classification Model

    Architecture:

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
        temporal_dim=256,
        n_frames=30
    ):
        super().__init__()

        self.num_classes = num_classes
        self.temporal_dim = temporal_dim
        self.n_frames = n_frames

        # --------------------------------------------------
        # EfficientNet-B0 backbone
        # --------------------------------------------------

        self.backbone = efficientnet_b0(
            weights="IMAGENET1K_V1"
        )

        feature_dim = 1280

        # Remove original ImageNet classifier
        self.backbone.classifier = nn.Identity()

        # --------------------------------------------------
        # Spatial Attention
        # --------------------------------------------------

        self.spatial_attention = SpatialAttention(
            feature_dim
        )

        # --------------------------------------------------
        # 2-Layer Bidirectional GRU
        # --------------------------------------------------

        self.temporal_encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=temporal_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # BiGRU output = temporal_dim × 2
        gru_output_dim = temporal_dim * 2

        # --------------------------------------------------
        # Temporal Attention
        # --------------------------------------------------

        self.temporal_attention = TemporalAttention(
            gru_output_dim
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),

            nn.Linear(
                gru_output_dim,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):
        """
        Input:
            x = (B, T, C, H, W)

        Example:
            (4, 30, 3, 224, 224)

        Output:
            logits = (B, num_classes)
        """

        batch_size, time_steps, channels, height, width = x.shape

        # --------------------------------------------------
        # Combine batch and temporal dimensions
        # --------------------------------------------------

        x = x.view(
            batch_size * time_steps,
            channels,
            height,
            width
        )

        # --------------------------------------------------
        # EfficientNet feature extraction
        # --------------------------------------------------

        features = self.backbone.features(x)

        # --------------------------------------------------
        # Spatial Attention
        # --------------------------------------------------

        features = self.spatial_attention(features)

        # Global average pooling
        features = self.backbone.avgpool(features)

        features = torch.flatten(
            features,
            start_dim=1
        )

        # --------------------------------------------------
        # Restore temporal dimension
        # --------------------------------------------------

        features = features.view(
            batch_size,
            time_steps,
            -1
        )

        # --------------------------------------------------
        # Bidirectional GRU
        # --------------------------------------------------

        gru_output, _ = self.temporal_encoder(features)

        # --------------------------------------------------
        # Temporal Attention
        # --------------------------------------------------

        attended_features, attention_weights = (
            self.temporal_attention(gru_output)
        )

        # --------------------------------------------------
        # Final classifier
        # --------------------------------------------------

        logits = self.classifier(
            attended_features
        )

        return logits