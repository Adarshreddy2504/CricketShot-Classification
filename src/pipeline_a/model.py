
import torch
import torch.nn as nn
import torchvision.models as models

class TemporalAttention(nn.Module):
    """
    Custom Temporal Attention mechanism.
    Takes GRU outputs across time-steps, calculates a dynamic weight for each frame,
    and returns a single aggregated context vector.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        # Maps the hidden_size (e.g., 512 for bidir GRU) to a single attention score per frame
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, gru_outputs):
        # gru_outputs shape: (batch_size, seq_length, hidden_size)
        
        # Calculate attention scores
        # scores shape: (batch_size, seq_length, 1)
        scores = self.attention(gru_outputs)
        
        # Apply softmax across the sequence dimension (dim=1) to get probabilities
        # weights shape: (batch_size, seq_length, 1)
        weights = torch.softmax(scores, dim=1)
        
        # Multiply weights with GRU outputs and sum across the sequence dimension
        # context shape: (batch_size, hidden_size)
        context = torch.sum(weights * gru_outputs, dim=1)
        
        return context

class CricketShotClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CricketShotClassifier, self).__init__()
        
        # 1. Backbone: EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Remove the final classification head
        # We keep the feature extractor, the adaptive average pool, and flatten it.
        # This will output a 1,280-dimensional feature vector per frame.
        self.backbone = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool,
            nn.Flatten()
        )
        
        # 2. Temporal Modeler: GRU
        # input_size=1280 (from EfficientNet), hidden_size=256, num_layers=2, bidir=True
        self.gru = nn.GRU(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Attention
        # Since GRU is bidirectional with hidden_size=256, its output size is 256 * 2 = 512
        self.attention = TemporalAttention(hidden_size=512)
        
        # 4. Classifier
        # Maps the context vector (512) to the 10 cricket shot classes
        self.classifier = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x: Tensor of shape (batch, num_frames, channels, height, width).
               e.g., (batch, 30, 3, 224, 224)
        """
        # Extract dimensions
        batch_size, num_frames, c, h, w = x.size()
        
        # Reshape to (batch_size * num_frames, c, h, w) to process all frames independently through the CNN
        x_flat = x.view(batch_size * num_frames, c, h, w)
        
        # Extract spatial features using the EfficientNet backbone
        # features shape: (batch_size * num_frames, 1280)
        features = self.backbone(x_flat)
        
        # Reshape back to sequence format for the temporal model
        # features shape: (batch_size, num_frames, 1280)
        features = features.view(batch_size, num_frames, -1)
        
        # Process the sequence through the GRU
        # gru_out shape: (batch_size, num_frames, 512)
        gru_out, hidden = self.gru(features)
        
        # Apply temporal attention to aggregate the sequence into a single context vector
        # context shape: (batch_size, 512)
        context = self.attention(gru_out)
        
        # Final classification
        # logits shape: (batch_size, num_classes)
        logits = self.classifier(context)
        
        return logits

if __name__ == "__main__":
    # Quick sanity check / dry run
    model = CricketShotClassifier(num_classes=10)
    print(model)
    
    # Dummy input: batch_size=2, frames=30, channels=3, 224x224 resolution
    dummy_input = torch.randn(2, 30, 3, 224, 224)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: 2, 10)")
