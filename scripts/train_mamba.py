import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Simple Mamba-style Block in Pure PyTorch ---
class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x):
        # Simplified Selective Scan logic
        skip = x
        x = self.proj(x)
        x1, x2 = x.chunk(2, dim=-1)
        x1 = x1.transpose(1, 2)
        x1 = self.act(self.conv(x1))
        x1 = x1.transpose(1, 2)
        return self.out_proj(torch.cat([x1, x2], dim=-1)) + skip

class MambaClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.mamba_block = SimpleMambaBlock(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, genes) -> (batch, genes, 1)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.mamba_block(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.sigmoid(self.fc(x))

# --- Data Loading ---
class BRCAData(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def main():
    print("Loading data...")
    X = pd.read_csv("/data4t/projects/fs/data_processed/brca_ml_matrix.csv", index_col=0)
    y = pd.read_csv("/data4t/projects/fs/data_processed/labels.csv")['target'].values

    # Pre-processing: Log2 and Top 5000 genes
    print("Selecting Top 5000 genes...")
    X = np.log2(X + 1)
    top_genes = X.var().sort_values(ascending=False).head(5000).index
    X = X[top_genes].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = BRCAData(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaClassifier(input_dim=5000).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting Training on {device}...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    print("Training Complete!")

if __name__ == "__main__":
    main()
