"""
06_cv_significance.py
5-fold stratified CV across ALL conditions + paired bootstrap significance vs B1.
Conditions:
  B1  Mamba  top-5000 variance
  B2  Mamba  top-50 saliency (no LLM)
  B3  Mamba  17 LLM-filtered
  C1  LASSO  top-5000 variance
  C2  LASSO  top-50 saliency
  C3  LASSO  17 LLM-filtered
  C5  RF     17 LLM-filtered  (RF selected from top-5000)
Output: results/cv_significance.json
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from mamba_ssm.modules.mamba_simple import Mamba
from scipy import stats

DATA_X        = "/data4t/projects/fs/data_processed/brca_ml_matrix.csv"
DATA_Y        = "/data4t/projects/fs/data_processed/labels.csv"
SALIENCY_JSON = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
LLM_JSON      = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
OUT_JSON      = "/data4t/projects/fs/results/cv_significance.json"

N_FOLDS = 5
EPOCHS  = 15
BATCH   = 8
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT  = 2000
RNG     = np.random.default_rng(42)


# ─── Mamba ───────────────────────────────────────────────────────────────────
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


def lasso_fold(Xtr, ytr, Xte, yte, penalty="l1", seed=42):
    kw = {"l1_ratio": 0.5} if penalty=="elasticnet" else {}
    solver = "saga"
    clf = LogisticRegression(penalty=penalty, C=0.1, solver=solver,
                             max_iter=3000, class_weight="balanced",
                             random_state=seed, **kw)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:,1])


def rf_fold(Xtr, ytr, Xte, yte, seed=42):
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:,1])


def cv_condition(X, y, label, clf_fn):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    aucs = []
    for fold, (tri, tei) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tri])
        a = clf_fn(sc.transform(X[tri]), y[tri], sc.transform(X[tei]), y[tei], seed=fold)
        aucs.append(a)
        print(f"  fold {fold+1}/{N_FOLDS}  AUC={a:.4f}")
    return {"label": label, "n_genes": X.shape[1],
            "fold_aucs": aucs,
            "auc_mean": round(float(np.mean(aucs)),4),
            "auc_std":  round(float(np.std(aucs)), 4)}


def bootstrap_p(aucs_a, aucs_b, n=N_BOOT):
    """One-sided H0: mean(b) <= mean(a); returns p-value that b > a."""
    diffs = []
    aucs_a, aucs_b = np.array(aucs_a), np.array(aucs_b)
    for _ in range(n):
        ia = RNG.integers(0, len(aucs_a), len(aucs_a))
        ib = RNG.integers(0, len(aucs_b), len(aucs_b))
        diffs.append(aucs_b[ib].mean() - aucs_a[ia].mean())
    diffs = np.array(diffs)
    return float(np.mean(diffs <= 0))   # p-value: prob that diff<=0 under bootstrap


def main():
    X_raw = pd.read_csv(DATA_X, index_col=0)
    y     = pd.read_csv(DATA_Y)["target"].values
    X_raw = np.log2(X_raw + 1)

    with open(SALIENCY_JSON) as f:
        sal_genes = json.load(f)["top_genes"]
    with open(LLM_JSON) as f:
        llm_genes = json.load(f)["mamba_selected_genes"]

    top5k    = X_raw.var().sort_values(ascending=False).head(5000).index.tolist()
    genes_b2 = [g for g in sal_genes[:50] if g in X_raw.columns]
    genes_b3 = [g for g in llm_genes if g in X_raw.columns]

    X5k  = X_raw[top5k].values
    Xb2  = X_raw[genes_b2].values
    Xb3  = X_raw[genes_b3].values

    results = []

    print(f"\n=== B1 Mamba 5k ({N_FOLDS}-fold CV) ===")
    r_b1 = cv_condition(X5k, y, "B1: Mamba Top-5000 Variance", mamba_fold)
    results.append(r_b1)

    print(f"\n=== B2 Mamba 50-saliency ({N_FOLDS}-fold CV) ===")
    r_b2 = cv_condition(Xb2, y, "B2: Mamba Top-50 Saliency (no LLM)", mamba_fold)
    results.append(r_b2)

    print(f"\n=== B3 Mamba 17-LLM ({N_FOLDS}-fold CV) ===")
    r_b3 = cv_condition(Xb3, y, "B3: Mamba+LLM CoT (ours)", mamba_fold)
    results.append(r_b3)

    print(f"\n=== C1 LASSO 5k ({N_FOLDS}-fold CV) ===")
    r_c1 = cv_condition(X5k, y, "C1: LASSO Top-5000 Variance",
                        lambda *a, **kw: lasso_fold(*a, penalty="l1", **kw))
    results.append(r_c1)

    print(f"\n=== C2 LASSO 50-saliency ({N_FOLDS}-fold CV) ===")
    r_c2 = cv_condition(Xb2, y, "C2: LASSO Top-50 Saliency",
                        lambda *a, **kw: lasso_fold(*a, penalty="l1", **kw))
    results.append(r_c2)

    print(f"\n=== C3 LASSO 17-LLM ({N_FOLDS}-fold CV) ===")
    r_c3 = cv_condition(Xb3, y, "C3: LASSO 17 LLM-Filtered",
                        lambda *a, **kw: lasso_fold(*a, penalty="l1", **kw))
    results.append(r_c3)

    print(f"\n=== C4 RF 5k ({N_FOLDS}-fold CV) ===")
    r_c4 = cv_condition(X5k, y, "C4: RandomForest Top-5000 Variance", rf_fold)
    results.append(r_c4)

    print(f"\n=== C5 RF 17-LLM ({N_FOLDS}-fold CV) ===")
    r_c5 = cv_condition(Xb3, y, "C5: RandomForest 17 LLM-Filtered", rf_fold)
    results.append(r_c5)

    # ── Bootstrap significance: each condition vs B1 (Mamba baseline) ────────
    print("\n=== Bootstrap significance tests vs B1 ===")
    ref = r_b1["fold_aucs"]
    for r in results[1:]:
        p = bootstrap_p(ref, r["fold_aucs"])
        r["bootstrap_p_vs_B1"] = round(p, 4)
        direction = "better" if np.mean(r["fold_aucs"]) > np.mean(ref) else "worse"
        print(f"  {r['label']:<42}  p={p:.4f}  ({direction} than B1)")

    # Also paired t-test for Mamba conditions (same folds → paired)
    for r in [r_b2, r_b3]:
        t, p_t = stats.ttest_rel(r_b1["fold_aucs"], r["fold_aucs"])
        r["paired_ttest_vs_B1"] = {"t": round(float(t),4), "p": round(float(p_t),4)}
        print(f"  Paired t-test B1 vs {r['label'][:20]}: t={t:.3f} p={p_t:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'Condition':<42} {'Genes':>6} {'AUC':>10}  p(vs B1)")
    print("-"*72)
    for r in results:
        p_str = f"{r.get('bootstrap_p_vs_B1','—'):>8}" if isinstance(r.get('bootstrap_p_vs_B1'), float) else "        —"
        print(f"{r['label']:<42} {r['n_genes']:>6} {r['auc_mean']:.4f}±{r['auc_std']:.4f}{p_str}")

    os.makedirs("/data4t/projects/fs/results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
