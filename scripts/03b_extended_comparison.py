"""
Extended comparison: adds classical ML baselines and multi-seed Mamba runs
to address reviewers' concerns.

New conditions (all on same train/test split logic):
  C1: LogReg-L1 (LASSO) on top-5000 variance genes
  C2: LogReg-L1 (LASSO) on LLM-selected 17 genes
  C3: ElasticNet LogReg on LLM-selected 17 genes
  Multi-seed Mamba (5 seeds) for B1, B2, B3 -> mean ± std

Outputs: results/extended_comparison.json
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from mamba_ssm.modules.mamba_simple import Mamba

DATA_X        = "/data4t/projects/fs/data_processed/brca_ml_matrix.csv"
DATA_Y        = "/data4t/projects/fs/data_processed/labels.csv"
SALIENCY_JSON = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
LLM_JSON      = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
OUT_JSON      = "/data4t/projects/fs/results/extended_comparison.json"

EPOCHS      = 15
BATCH_SIZE  = 8
SEEDS       = [42, 123, 7, 99, 2024]


# ─── Mamba model ─────────────────────────────────────────────────────────────

class MambaClassifier(nn.Module):
    def __init__(self, d_model=128):
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


def train_mamba_seed(X, y, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pos_weight = torch.tensor([np.sum(y_train==1)/np.sum(y_train==0)]).to(device)
    model = MambaClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss(reduction="none")
    loader = DataLoader(BRCAData(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(EPOCHS):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze()
            w = torch.where(by==0, pos_weight, torch.ones_like(by))
            loss = (criterion(out, by) * w).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average="weighted")),
        "auc": float(roc_auc_score(y_test, probs)),
    }


def eval_mamba_multiseed(X, y, label, device):
    runs = []
    for seed in SEEDS:
        print(f"  seed={seed} ...", end=" ", flush=True)
        m = train_mamba_seed(X, y, seed, device)
        runs.append(m)
        print(f"AUC={m['auc']:.4f}")
    aucs = [r["auc"] for r in runs]
    accs = [r["accuracy"] for r in runs]
    f1s  = [r["f1"] for r in runs]
    return {
        "label": label,
        "n_genes": X.shape[1],
        "n_seeds": len(SEEDS),
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std":  round(float(np.std(accs)),  4),
        "f1_mean":       round(float(np.mean(f1s)),  4),
        "f1_std":        round(float(np.std(f1s)),   4),
        "auc_mean":      round(float(np.mean(aucs)), 4),
        "auc_std":       round(float(np.std(aucs)),  4),
        "per_seed":      runs,
    }


def eval_classical(X, y, label, penalty="l1"):
    """Logistic regression with L1 or elasticnet, evaluated across same seeds."""
    runs = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
        kwargs = {"l1_ratio": 0.5} if penalty == "elasticnet" else {}
        clf = LogisticRegression(
            penalty=penalty, C=0.1, solver=solver,
            max_iter=2000, class_weight="balanced", random_state=seed,
            **kwargs
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        runs.append({
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds, average="weighted")),
            "auc": float(roc_auc_score(y_test, probs)),
        })
    aucs = [r["auc"] for r in runs]
    accs = [r["accuracy"] for r in runs]
    f1s  = [r["f1"] for r in runs]
    return {
        "label": label,
        "n_genes": X.shape[1],
        "n_seeds": len(SEEDS),
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std":  round(float(np.std(accs)),  4),
        "f1_mean":       round(float(np.mean(f1s)),  4),
        "f1_std":        round(float(np.std(f1s)),   4),
        "auc_mean":      round(float(np.mean(aucs)), 4),
        "auc_std":       round(float(np.std(aucs)),  4),
        "per_seed":      runs,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    X_raw = pd.read_csv(DATA_X, index_col=0)
    y = pd.read_csv(DATA_Y)["target"].values
    X_raw = np.log2(X_raw + 1)

    with open(SALIENCY_JSON) as f:
        saliency_genes = json.load(f)["top_genes"]
    with open(LLM_JSON) as f:
        llm_data = json.load(f)
    llm_genes = llm_data["mamba_selected_genes"]

    # Gene sets
    top5k   = X_raw.var().sort_values(ascending=False).head(5000).index.tolist()
    genes_b2 = [g for g in saliency_genes[:50] if g in X_raw.columns]
    genes_b3 = [g for g in llm_genes if g in X_raw.columns]

    sc5k  = StandardScaler().fit_transform(X_raw[top5k].values)
    scB2  = StandardScaler().fit_transform(X_raw[genes_b2].values)
    scB3  = StandardScaler().fit_transform(X_raw[genes_b3].values)

    results = []

    # ── Mamba multi-seed ────────────────────────────────────────────────────
    print("=== Mamba B1: Top-5000 Variance (multi-seed) ===")
    results.append(eval_mamba_multiseed(sc5k, y, "B1: Mamba Top-5000 Variance", device))

    print("\n=== Mamba B2: Top-50 Saliency (multi-seed) ===")
    results.append(eval_mamba_multiseed(scB2, y, "B2: Mamba Top-50 Saliency (no LLM)", device))

    print("\n=== Mamba B3: 17 LLM-Filtered (multi-seed) ===")
    results.append(eval_mamba_multiseed(scB3, y, "B3: Mamba + LLM CoT (ours)", device))

    # ── Classical: LASSO LogReg on same 3 gene sets ─────────────────────────
    print("\n=== LASSO LogReg C1: Top-5000 genes ===")
    results.append(eval_classical(sc5k, y, "C1: LASSO LogReg Top-5000 Variance", penalty="l1"))

    print("\n=== LASSO LogReg C2: Top-50 saliency genes ===")
    results.append(eval_classical(scB2, y, "C2: LASSO LogReg Top-50 Saliency", penalty="l1"))

    print("\n=== LASSO LogReg C3: 17 LLM-filtered genes ===")
    results.append(eval_classical(scB3, y, "C3: LASSO LogReg 17 LLM-Filtered", penalty="l1"))

    print("\n=== ElasticNet LogReg C4: 17 LLM-filtered genes ===")
    results.append(eval_classical(scB3, y, "C4: ElasticNet LogReg 17 LLM-Filtered", penalty="elasticnet"))

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print(f"{'Condition':<42} {'Genes':>6} {'AUC mean':>9} {'AUC std':>8}")
    print("-"*75)
    for r in results:
        print(f"{r['label']:<42} {r['n_genes']:>6} {r['auc_mean']:>9.4f} ±{r['auc_std']:.4f}")

    os.makedirs("/data4t/projects/fs/results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
