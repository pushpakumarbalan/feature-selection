"""
Step 3: Train Mamba on three gene subsets and compare held-out metrics.

    B1: Top-5000 variance genes (baseline)
    B2: Top-50 saliency genes (Mamba-selected, no LLM)
    B3: LLM-filtered saliency genes (Mamba + LLM)

This matches the paper table setup: same model architecture, same
80/20 stratified split (random_state=42), only the input gene set changes.

Outputs: results/comparison_results.json
"""
import json
import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from torch.utils.data import DataLoader, Dataset
from mamba_ssm.modules.mamba_simple import Mamba

DATA_X        = "/data4t/projects/fs/data_processed/brca_ml_matrix.csv"
DATA_Y        = "/data4t/projects/fs/data_processed/labels.csv"
SALIENCY_JSON = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
LLM_JSON      = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
OUT_JSON      = "/data4t/projects/fs/results/comparison_results.json"

MAMBA_TOP_LARGE = 50   # B2: full Mamba-selected list
EPOCHS          = 15
BATCH_SIZE      = 8
RANDOM_SEED     = 42


class MambaClassifier(nn.Module):
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


class BRCAData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_and_evaluate(X, y, label, device, seed=RANDOM_SEED):
    """Train and evaluate one condition on the fixed 80/20 split."""
    print(f"\n{'='*50}")
    print(f"Condition: {label}  |  genes={X.shape[1]}  |  seed={seed}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pos_weight = torch.tensor([np.sum(y_train==1)/np.sum(y_train==0)]).to(device)

    model = MambaClassifier(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss(reduction="none")
    loader = DataLoader(BRCAData(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze()
            w = torch.where(by==0, pos_weight, torch.ones_like(by))
            loss = (criterion(out, by) * w).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs = model(test_x).squeeze().cpu().numpy()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")
    auc = roc_auc_score(y_test, probs)
    print(f"\n  Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Normal", "Tumor"]))

    return {"label": label, "n_genes": X.shape[1],
            "accuracy": round(acc, 4), "f1": round(f1, 4), "auc": round(auc, 4),
            "report": classification_report(y_test, preds, target_names=["Normal","Tumor"])}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Using fixed split seed: {RANDOM_SEED}")

    X_raw = pd.read_csv(DATA_X, index_col=0)
    y = pd.read_csv(DATA_Y)["target"].values
    X_raw_log = np.log2(X_raw + 1)
    X_raw_df  = pd.DataFrame(X_raw_log, columns=X_raw.columns)

    # Load Mamba-selected gene list
    with open(SALIENCY_JSON) as f:
        saliency_data = json.load(f)
    all_saliency_genes = saliency_data["top_genes"]

    # Load LLM-selected gene list
    with open(LLM_JSON) as f:
        llm_data = json.load(f)
    llm_selected_genes = llm_data.get("mamba_selected_genes", [])
    if not llm_selected_genes:
        print("WARNING: LLM JSON has no mamba_selected_genes — run 02_llm_reasoning.py first")
        llm_selected_genes = all_saliency_genes[:17]

    # B1: Top-5000 variance genes
    top5k = X_raw_df.var().sort_values(ascending=False).head(5000).index.tolist()

    # B2: Top-50 saliency genes
    genes_b2 = [g for g in all_saliency_genes[:MAMBA_TOP_LARGE] if g in X_raw_df.columns]

    # B3: LLM-reasoned gene subset
    genes_b3 = [g for g in llm_selected_genes if g in X_raw_df.columns]
    if not genes_b3:
        print("WARNING: No LLM-selected genes found in feature matrix — check gene names")
        genes_b3 = [g for g in all_saliency_genes[:17] if g in X_raw_df.columns]

    conditions = [
        ("B1: Variance baseline", top5k),
        ("B2: Mamba saliency only (no LLM)", genes_b2),
        ("B3: Mamba + LLM structured CoT (ours)", genes_b3),
    ]

    all_results = []
    for label, gene_list in conditions:
        X_sub = StandardScaler().fit_transform(X_raw_df[gene_list].values)
        res = train_and_evaluate(X_sub, y, label, device, seed=RANDOM_SEED)
        all_results.append(res)

    os.makedirs("/data4t/projects/fs/results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved comparison results to {OUT_JSON}")

    print("\n" + "="*72)
    print(f"{'Condition':<40} {'Genes':>6} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-"*72)
    for r in all_results:
        print(f"{r['label']:<40} {r['n_genes']:>6} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f}")


if __name__ == "__main__":
    main()
