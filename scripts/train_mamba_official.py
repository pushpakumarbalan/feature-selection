import os
from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from mamba_ssm.modules.mamba_simple import Mamba


class OfficialMambaClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_fast_path=True,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, 1) -> (batch, seq_len, d_model)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        # Mamba expects (batch, seq_len, d_model)
        x = self.mamba(x)
        # Pool across seq_len -> (batch, d_model, 1) -> (batch, d_model)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.sigmoid(self.fc(x))


class BRCAData(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def main():
    print("Loading data...")
    X = pd.read_csv("/data4t/projects/fs/data_processed/brca_ml_matrix.csv", index_col=0)
    y = pd.read_csv("/data4t/projects/fs/data_processed/labels.csv")["target"].values

    # Log-normalise
    X = np.log2(X + 1)

    # Hyperparameters
    top_gene_count = 5000
    epochs = 15
    batch_size = 8
    d_model = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Feature selection: top-variance genes
    top_genes = X.var().sort_values(ascending=False).head(top_gene_count).index
    X = StandardScaler().fit_transform(X[top_genes].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class-balance weight
    pos_weight = torch.tensor([np.sum(y_train == 1) / np.sum(y_train == 0)]).to(device)

    model = OfficialMambaClassifier(input_dim=top_gene_count, d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss(reduction="none")

    train_loader = DataLoader(
        BRCAData(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    print(f"Training {epochs} epochs, batch={batch_size}, genes={top_gene_count}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            weights = torch.where(batch_y == 0, pos_weight, torch.ones_like(batch_y))
            loss = (criterion(outputs, batch_y) * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = (model(test_x).squeeze().cpu().numpy() > 0.5).astype(int)

    print("\n--- OFFICIAL MAMBA RESULTS ---")
    print(classification_report(y_test, preds, target_names=["Normal", "Tumor"]))

    # Save checkpoint
    os.makedirs("/data4t/projects/fs/models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"/data4t/projects/fs/models/mamba_official_{timestamp}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "top_genes": list(top_genes),
            "device": device,
            "d_model": d_model,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        model_path,
    )
    print(f"Saved model checkpoint to: {model_path}")


if __name__ == "__main__":
    main()
