"""
09_shap_mi_baselines.py
Additional feature-selection baselines:
  S1  Mutual Information top-17 features (from top-5000 variance pool)
  S2  SHAP-importance top-17 (Random Forest SHAP values)
  S3  RF feature-importance top-17  (Gini impurity from RF trained on 5000)
Each evaluated with 5-fold Mamba CV (same architecture as other conditions).
Output: results/shap_mi_baselines.json
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from mamba_ssm.modules.mamba_simple import Mamba

DATA_X        = "/data4t/projects/fs/data_processed/brca_ml_matrix.csv"
DATA_Y        = "/data4t/projects/fs/data_processed/labels.csv"
LLM_JSON      = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
OUT_JSON      = "/data4t/projects/fs/results/shap_mi_baselines.json"

N_GENES   = 17
N_FOLDS   = 5
EPOCHS    = 15
BATCH     = 8
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


class MambaClf(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.emb   = nn.Linear(1, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=True)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(d_model, 1)
        self.sig   = nn.Sigmoid()
    def forward(self, x):
        x = self.emb(x.unsqueeze(-1))
        x = self.mamba(x).transpose(1,2)
        return self.sig(self.fc(self.pool(x).squeeze(-1)))

class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def mamba_fold(Xtr, ytr, Xte, yte, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    pw   = torch.tensor([ytr.sum()/(len(ytr)-ytr.sum())]).to(DEVICE)
    mdl  = MambaClf().to(DEVICE)
    opt  = torch.optim.AdamW(mdl.parameters(), lr=1e-4)
    crit = nn.BCELoss(reduction="none")
    ldr  = DataLoader(DS(Xtr, ytr), batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        mdl.train()
        for bx, by in ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            out = mdl(bx).view(-1)
            w   = torch.where(by==0, pw.expand_as(by), torch.ones_like(by))
            (crit(out, by)*w).mean().backward()
            opt.step()
    mdl.eval()
    with torch.no_grad():
        probs = mdl(torch.tensor(Xte, dtype=torch.float32).to(DEVICE)).view(-1).cpu().numpy()
    return roc_auc_score(yte, probs)

def cv_genes(X_mat, y, label):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    aucs = []
    for fold, (tri, tei) in enumerate(skf.split(X_mat, y)):
        sc = StandardScaler().fit(X_mat[tri])
        a  = mamba_fold(sc.transform(X_mat[tri]), y[tri],
                        sc.transform(X_mat[tei]), y[tei], seed=fold)
        aucs.append(a)
        print(f"  fold {fold+1}  AUC={a:.4f}")
    return {"label": label, "n_genes": X_mat.shape[1],
            "fold_aucs": aucs,
            "auc_mean": round(float(np.mean(aucs)),4),
            "auc_std":  round(float(np.std(aucs)), 4)}


def select_mi(X_pool, y, col_names, k):
    """Select top-k genes by mutual information (no leakage: use full dataset for ranking)."""
    mi = mutual_info_classif(X_pool, y, random_state=42, n_neighbors=5)
    idx = np.argsort(mi)[::-1][:k]
    return [col_names[i] for i in idx], mi[idx]

def select_rf_importance(X_pool, y, col_names, k):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_pool, y)
    idx = np.argsort(rf.feature_importances_)[::-1][:k]
    return [col_names[i] for i in idx], rf.feature_importances_[idx]

def select_shap(X_pool, y, col_names, k):
    try:
        import shap
    except ImportError:
        print("  shap not installed; skipping SHAP condition. pip install shap")
        return None, None
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_pool, y)
    idx_sub = np.random.default_rng(42).choice(len(X_pool), size=min(300, len(X_pool)), replace=False)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_pool[idx_sub])
    # Handle both old API (list of per-class arrays) and new API (3D array)
    if isinstance(shap_vals, list):
        # old API: list[n_classes] of (n_samples, n_features)
        sv = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # new API: (n_samples, n_features, n_classes)
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals
    mean_abs = np.abs(sv).mean(axis=0)   # shape: (n_features,)
    idx = np.argsort(mean_abs)[::-1][:k]
    selected = [str(col_names[i]) for i in idx]
    return selected, mean_abs[idx]


def main():
    X_raw = pd.read_csv(DATA_X, index_col=0)
    y     = pd.read_csv(DATA_Y)["target"].values
    Xl    = np.log2(X_raw + 1)

    with open(LLM_JSON) as f:
        llm_genes = json.load(f)["mamba_selected_genes"]

    top5k     = Xl.var().sort_values(ascending=False).head(5000).index.tolist()
    X5k       = StandardScaler().fit_transform(Xl[top5k].values)
    col_names = np.array(top5k)

    results = []

    # S1: Mutual Information top-17
    print("\n=== S1 Mutual Information top-17 ===")
    mi_genes, mi_scores = select_mi(X5k, y, col_names, N_GENES)
    print(f"  Selected: {mi_genes}")
    results.append(cv_genes(Xl[mi_genes].values, y, "S1: MI top-17"))
    results[-1]["selected_genes"] = mi_genes
    results[-1]["scores"] = mi_scores.tolist()

    # S2: RF feature importance top-17
    print("\n=== S2 RF Feature Importance top-17 ===")
    rf_genes, rf_imps = select_rf_importance(X5k, y, col_names, N_GENES)
    print(f"  Selected: {rf_genes}")
    results.append(cv_genes(Xl[rf_genes].values, y, "S2: RF-Importance top-17"))
    results[-1]["selected_genes"] = rf_genes
    results[-1]["scores"] = rf_imps.tolist()

    # S3: SHAP top-17
    print("\n=== S3 SHAP top-17 ===")
    shap_genes, shap_vals = select_shap(X5k, y, col_names, N_GENES)
    if shap_genes is not None:
        print(f"  Selected: {shap_genes}")
        results.append(cv_genes(Xl[shap_genes].values, y, "S3: SHAP top-17"))
        results[-1]["selected_genes"] = shap_genes
        results[-1]["scores"] = shap_vals.tolist()
    else:
        results.append({"label": "S3: SHAP top-17", "note": "shap not installed"})

    # S4: LLM-filtered (reference comparison)
    print("\n=== S4 LLM-filtered 17 genes (reference) ===")
    genes_b3 = [g for g in llm_genes if g in X_raw.columns]
    results.append(cv_genes(Xl[genes_b3].values, y, "B3: LLM CoT 17 genes (reference)"))
    results[-1]["selected_genes"] = genes_b3

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Condition':<40} {'AUC mean±std':>14}")
    print("-"*65)
    for r in results:
        if "auc_mean" in r:
            print(f"  {r['label']:<38}  {r['auc_mean']:.4f}±{r['auc_std']:.4f}")
        else:
            print(f"  {r['label']:<38}  [skipped]")

    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
