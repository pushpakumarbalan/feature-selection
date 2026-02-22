"""
07_reasoning_ablations.py
Ablation study on the reasoning contribution.
Conditions tested with identical 5-fold CV Mamba:
  A1  Top-17 saliency genes        (no LLM, just truncate)
  A2  Top-17 variance genes        (no LLM, top variance)
  A3  Random-17 from top-5000      (5 different random draws → mean±std)
  A4  Bottom-17 saliency           (worst saliency genes — sanity check)
  B3  17 LLM-filtered              (ours, loaded from JSON)
Outputs: results/reasoning_ablations.json
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from mamba_ssm.modules.mamba_simple import Mamba

DATA_X        = "/data4t/projects/fs/data_processed/brca_ml_matrix.csv"
DATA_Y        = "/data4t/projects/fs/data_processed/labels.csv"
SALIENCY_JSON = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
LLM_JSON      = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
OUT_JSON      = "/data4t/projects/fs/results/reasoning_ablations.json"

N_FOLDS = 5
EPOCHS  = 15
BATCH   = 8
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
N_RANDOM_DRAWS = 5


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

def cv_genes(X_matrix, y, label, verbose=True):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    aucs = []
    for fold, (tri, tei) in enumerate(skf.split(X_matrix, y)):
        sc = StandardScaler().fit(X_matrix[tri])
        a  = mamba_fold(sc.transform(X_matrix[tri]), y[tri],
                        sc.transform(X_matrix[tei]), y[tei], seed=fold)
        aucs.append(a)
        if verbose: print(f"  fold {fold+1}/{N_FOLDS}  AUC={a:.4f}")
    return {"label": label, "n_genes": X_matrix.shape[1],
            "fold_aucs": aucs,
            "auc_mean": round(float(np.mean(aucs)),4),
            "auc_std":  round(float(np.std(aucs)), 4)}


def main():
    X_raw = pd.read_csv(DATA_X, index_col=0)
    y     = pd.read_csv(DATA_Y)["target"].values
    X_raw = np.log2(X_raw + 1)

    with open(SALIENCY_JSON) as f:
        sal_genes = json.load(f)["top_genes"]
    with open(LLM_JSON) as f:
        llm_genes = json.load(f)["mamba_selected_genes"]

    top5k    = X_raw.var().sort_values(ascending=False).head(5000).index.tolist()
    top17_var= top5k[:17]
    genes_b2 = [g for g in sal_genes[:50] if g in X_raw.columns]
    top17_sal= [g for g in sal_genes[:17] if g in X_raw.columns]
    bot17_sal= [g for g in sal_genes[:50][-17:] if g in X_raw.columns]
    genes_b3 = [g for g in llm_genes if g in X_raw.columns]

    results = []

    # A1: Top-17 saliency (no LLM, just truncate rank)
    print(f"\n=== A1 Top-17 saliency (no LLM) ===")
    print(f"Genes: {top17_sal}")
    results.append(cv_genes(X_raw[top17_sal].values, y,
                             "A1: Top-17 Saliency (no LLM)"))

    # A2: Top-17 variance
    print(f"\n=== A2 Top-17 variance genes ===")
    results.append(cv_genes(X_raw[top17_var].values, y,
                             "A2: Top-17 Variance (no LLM)"))

    # A3: Random-17 from top-5000 (N_RANDOM_DRAWS draws, report mean across all folds×draws)
    print(f"\n=== A3 Random-17 from top-5000 (×{N_RANDOM_DRAWS} draws) ===")
    rand_fold_aucs = []
    rng = np.random.default_rng(0)
    for draw in range(N_RANDOM_DRAWS):
        rand_genes = list(rng.choice(top5k, size=17, replace=False))
        r = cv_genes(X_raw[rand_genes].values, y,
                     f"A3 draw {draw+1}", verbose=False)
        rand_fold_aucs.extend(r["fold_aucs"])
        print(f"  draw {draw+1}  AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}")
    results.append({
        "label": "A3: Random-17 from top-5000 (5 draws × 5 folds)",
        "n_genes": 17,
        "fold_aucs": rand_fold_aucs,
        "auc_mean": round(float(np.mean(rand_fold_aucs)),4),
        "auc_std":  round(float(np.std(rand_fold_aucs)), 4),
    })

    # A4: Bottom-17 saliency (sanity / lower bound)
    print(f"\n=== A4 Bottom-17 saliency (sanity check) ===")
    print(f"Genes: {bot17_sal}")
    results.append(cv_genes(X_raw[bot17_sal].values, y,
                             "A4: Bottom-17 Saliency (sanity check)"))

    # B3: LLM-filtered (reference)
    print(f"\n=== B3 LLM-filtered 17 genes (reference) ===")
    print(f"Genes: {genes_b3}")
    results.append(cv_genes(X_raw[genes_b3].values, y,
                             "B3: Mamba+LLM CoT 17 genes (ours)"))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Condition':<45} {'Genes':>5} {'AUC mean±std':>14}")
    print("-"*65)
    for r in results:
        print(f"{r['label']:<45} {r['n_genes']:>5}  {r['auc_mean']:.4f}±{r['auc_std']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
