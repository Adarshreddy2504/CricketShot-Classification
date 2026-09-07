import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CricShotEfficientGRU(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=256):
        super(CricShotEfficientGRU, self).__init__()
        self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.cnn.classifier = nn.Identity()
        self.gru = nn.GRU(input_size=1280, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        b, seq, c, h, w = x.size()
        x = x.view(b * seq, c, h, w)
        features = self.cnn(x)
        features = features.view(b, seq, -1)
        
        # Pass features through the bidirectional GRU
        out, _ = self.gru(features)
        
        # Calculate attention weights
        attn_weights = self.attention(out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Multiply GRU outputs by attention weights and sum to create context vector
        context_vector = torch.sum(out * attn_weights, dim=1)
        
        # Pass through dropout and final classification layer
        x = self.dropout(context_vector)
        return self.fc(x)
