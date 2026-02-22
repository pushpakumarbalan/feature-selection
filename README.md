# LLM CoT Reasoning for Genomic Feature Selection: When Performance Gains Do Not Imply Faithful Reasoning

**ICLR 2026 Workshop on Logical Reasoning of Large Language Models — submission**

> **Paper:** `paper/main.tex`

## Summary

A Mamba SSM trained on TCGA-BRCA RNA-seq extracts top-50 gradient-saliency genes.
DeepSeek-R1 (7B, local Ollama) applies structured chain-of-thought to filter the list
to 17 biologically motivated genes.

Rigorous evaluation with 5-fold stratified CV, bootstrap significance tests, full
classical baselines, reasoning ablations, a decision-level faithfulness audit, and a
consistency experiment reveals:

- The Mamba AUC gain from LLM filtering (B3 0.949 vs B1 0.926, p=0.045) is due to
  **dimensionality reduction, not gene-specific reasoning** — any 17-gene subset
  achieves comparable AUC (0.946–0.962).
- **Classical methods dominate**: LASSO / RF / SHAP / MI reach AUC ≥ 0.999 with
  their own 17-gene selections.
- **LLM faithfulness is modest**: Precision=0.60, Recall=0.375, TNR=0.60 on
  validated BRCA genes.
- **LLM outputs are unstable**: 3/5 re-runs produce no structured gene list;
  mean pairwise Jaccard=0.30; zero stable genes.

## Pipeline

```
TCGA-BRCA (20k genes, 1208 samples)
    → top-5000 by variance
    → Mamba SSM (d_model=128, 5-fold CV)        ← B1 / B2 / B3
    → gradient saliency → top-50 candidates
    → DeepSeek-R1 CoT (structured prompt)
    → 17-gene set → Mamba / LASSO / RF eval
```

## Key Results (5-fold stratified CV)

### Downstream classification

| ID | Condition | Genes | AUC mean±std | p vs B1 |
|----|-----------|------:|--------------|---------|
| B1 | Mamba — variance baseline | 5000 | 0.926±0.026 | — |
| B2 | Mamba — top-50 saliency (no LLM) | 50 | 0.810±0.034 | 1.000 |
| **B3** | **Mamba + LLM CoT (ours)** | **17** | **0.949±0.016** | **0.045** |
| C1 | LASSO — variance (5k) | 5000 | 0.9996±0.001 | 0.000 |
| C3 | LASSO — 17 LLM-filtered | 17 | 0.9757±0.010 | 0.000 |
| C4 | RF — variance (5k) | 5000 | 0.9995±0.001 | 0.000 |
| C5 | RF — 17 LLM-filtered | 17 | 0.9928±0.005 | 0.000 |

### Reasoning ablations (Mamba, 5-fold CV)

| ID | Condition | AUC mean±std |
|----|-----------|--------------|
| A1 | Top-17 saliency (no LLM) | 0.948±0.027 |
| A2 | Top-17 variance (no LLM) | 0.946±0.015 |
| A3 | Random-17 from top-5000 (5 draws) | 0.953±0.035 |
| A4 | Bottom-17 saliency (sanity) | 0.962±0.013 |
| B3 | LLM CoT (ours) | 0.949±0.016 |
| S1 | MI top-17 (data-driven) | 0.9997±0.001 |
| S2 | RF feature-importance top-17 | 0.9996±0.001 |
| S3 | SHAP top-17 | 0.9993±0.001 |

### Decision-level faithfulness (B3 gene set)

| Metric | Value |
|--------|-------|
| TP — validated BRCA genes selected | 6 |
| FP — known non-BRCA genes selected | 4 |
| TN — known non-BRCA correctly rejected | 6 |
| FN — validated BRCA genes missed | 10 |
| Precision | 0.60 |
| Recall | 0.375 |
| TNR | 0.60 |

### LLM consistency (5 re-runs at T=0.3)

| Prompt variant | Valid/5 | Jaccard mean±std | Stable genes |
|----------------|--------:|-----------------|--------------|
| Standard | 2/5 | 0.30±0.46 | none |
| Shuffled gene order | 0/5 | — | none |
| Names only (no scores) | 1/5 | 0.10±0.30 | none |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_mamba_official.py` | Train Mamba classifier |
| `scripts/01_extract_saliency.py` | Gradient saliency → top-50 genes |
| `scripts/02_llm_reasoning.py` | DeepSeek-R1 structured CoT prompt |
| `scripts/02b_llm_gene_explain.py` | Per-gene biological reasoning (CoT) |
| `scripts/03_train_comparison.py` | Original 3-way benchmark |
| `scripts/03b_extended_comparison.py` | 5-seed multi-run + LASSO/ElasticNet |
| `scripts/04_generate_figures.py` | Paper figures |
| `scripts/05_fix_reasoning_parser.py` | Decision-level TP/FP/TN/FN audit |
| `scripts/06_cv_significance.py` | 5-fold CV + bootstrap p-values |
| `scripts/07_reasoning_ablations.py` | A1–A4 ablation conditions |
| `scripts/08_llm_consistency.py` | LLM consistency (original parser) |
| `scripts/08b_llm_consistency_debug.py` | Consistency with robust multi-strategy parser |
| `scripts/09_shap_mi_baselines.py` | MI / RF / SHAP 17-gene data-driven baselines |

## Setup

```bash
conda activate fs311
# mamba-ssm and causal-conv1d must be built from source for sm_120 (RTX 50-series)
# DeepSeek-R1 7B served locally: ollama run deepseek-r1:7b
```

## Data

TCGA-BRCA RNA-seq downloaded via GDC Data Transfer Tool. Not included in repo (5 GB+).

## Outputs

| File | Contents |
|------|----------|
| `data_processed/top_genes_saliency.json` | Mamba saliency scores — top-50 genes |
| `data_processed/llm_gene_reasoning.json` | LLM output + per-gene reasoning + decision-level metrics |
| `results/cv_significance.json` | 5-fold CV + bootstrap p-values |
| `results/reasoning_ablations.json` | A1–A4 + B3 ablation AUCs |
| `results/shap_mi_baselines.json` | S1/S2/S3 data-driven baselines |
| `results/llm_consistency.json` | Jaccard / valid-run counts across re-runs |
| `results/extended_comparison.json` | 5-seed holdout results |
| `paper/main.tex` | Full revised paper (ICLR 2026 workshop) |
