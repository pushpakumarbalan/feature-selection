import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_model * 2, d_model)
    def forward(self, x):
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
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.mamba_block(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.sigmoid(self.fc(x))

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
    
    # Preprocessing
    X = np.log2(X + 1)
    top_genes = X.var().sort_values(ascending=False).head(5000).index
    X = X[top_genes].values
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # CALCULATE WEIGHTS: ratio of tumor to normal
    num_tumor = np.sum(y_train == 1)
    num_normal = np.sum(y_train == 0)
    weight_for_normal = num_tumor / num_normal
    print(f"Applying Class Weight: {weight_for_normal:.2f} to Normal samples.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaClassifier(input_dim=5000).to(device)
    
    # BCEWithLogitsLoss allows us to pass pos_weight (we invert it to help the minority class)
    # We will adjust our manual loop to use weighted loss
    criterion = nn.BCELoss(reduction='none') 
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) # Lower LR for stability
    train_loader = DataLoader(BRCAData(X_train, y_train), batch_size=16, shuffle=True)
    
    print("Balanced Training Phase...")
    for epoch in range(20): # More epochs to learn the rare class
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            
            # Apply weights manually: weight = weight_for_normal if label is 0
            loss = criterion(outputs, batch_y)
            weights = torch.where(batch_y == 0, weight_for_normal, 1.0)
            loss = (loss * weights).mean()
            
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(test_x).squeeze().cpu().numpy()
        preds_binary = (preds > 0.5).astype(int)
        print("\n--- NEW BALANCED EVALUATION ---")
        print(classification_report(y_test, preds_binary, target_names=['Normal', 'Tumor']))

    torch.save(model.state_dict(), "/data4t/projects/fs/models/brca_mamba_balanced.pt")

if __name__ == "__main__":
    main()
