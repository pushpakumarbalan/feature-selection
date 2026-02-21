"""
Step 1: Extract gene importance scores from trained Mamba via gradient saliency.
Outputs top-N gene names to data_processed/top_genes_saliency.json
"""
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from mamba_ssm.modules.mamba_simple import Mamba

CHECKPOINT = "/data4t/projects/fs/models/mamba_official_20260219_234043.pt"
DATA_X     = "/data4t/projects/fs/data_processed/brca_ml_matrix.csv"
DATA_Y     = "/data4t/projects/fs/data_processed/labels.csv"
OUT_JSON   = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
TOP_N      = 50   # genes to pass to LLM


class OfficialMambaClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.sigmoid(self.fc(x))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    X_raw = pd.read_csv(DATA_X, index_col=0)
    y = pd.read_csv(DATA_Y)["target"].values
    X_raw = np.log2(X_raw + 1)

    # Load checkpoint — use same top_genes as training
    ckpt = torch.load(CHECKPOINT, map_location=device)
    top_genes = ckpt["top_genes"]
    print(f"Loaded checkpoint. top_genes count: {len(top_genes)}")

    X = StandardScaler().fit_transform(X_raw[top_genes].values)

    model = OfficialMambaClassifier(input_dim=len(top_genes), d_model=128).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Gradient saliency in mini-batches to avoid OOM
    # Accumulate |grad| per gene across all tumor samples
    saliency_acc = torch.zeros(len(top_genes), device=device)
    tumor_count  = 0
    BATCH = 16

    for i in range(0, len(X), BATCH):
        bx = torch.tensor(X[i:i+BATCH], dtype=torch.float32, device=device, requires_grad=True)
        by = y[i:i+BATCH]
        out = model(bx).squeeze()
        # Only backprop through tumor samples in this batch
        tumor_mask = torch.tensor(by == 1, dtype=torch.float32, device=device)
        if tumor_mask.sum() == 0:
            continue
        loss = (out * tumor_mask).sum()
        loss.backward()
        saliency_acc += bx.grad.abs().sum(dim=0).detach()
        tumor_count  += int(tumor_mask.sum().item())
        bx.grad = None

    saliency = (saliency_acc / max(tumor_count, 1)).cpu().numpy()

    # Map back to gene names and rank
    gene_saliency = pd.Series(saliency, index=top_genes).sort_values(ascending=False)
    top_genes_salience = gene_saliency.head(TOP_N)

    print(f"\nTop {TOP_N} genes by saliency:")
    for rank, (gene, score) in enumerate(top_genes_salience.items(), 1):
        print(f"  {rank:2d}. {gene:20s}  score={score:.6f}")

    result = {
        "top_genes": list(top_genes_salience.index),
        "scores": {g: float(s) for g, s in top_genes_salience.items()},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
