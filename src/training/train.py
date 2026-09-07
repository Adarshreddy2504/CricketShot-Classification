import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.models.efficientnet_gru import CricShotEfficientGRU

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

class CricketTensorDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        tensor = torch.tensor(data, dtype=torch.float32)
        
        # Extract label from filename (split by '_')
        filename = os.path.basename(file_path)
        parts = filename.replace('.npy', '').split('_')
        # We assume the label is the first part of the split, e.g., '3_video1.npy' -> 3
        label = int(parts[1])
        
        return tensor, label

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = 'data/tiny_training_tensors'
    dataset = CricketTensorDataset(data_dir=data_dir)
    
    if len(dataset) == 0:
        print(f"No .npy files found in {data_dir}. Please add data to train.")
        exit()
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = CricShotEfficientGRU(num_classes=10, hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            

            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
        
    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), 'data/cricshot_sota.pth')
    print("Training complete. Model saved to data/cricshot_sota.pth")
