# Mamba SSM + LLM Reasoning for Gene Selection in BRCA

## Summery:

Uses a Mamba (SSM) model trained on TCGA-BRCA RNA-seq data to identify disease-associated genes via gradient saliency, then applies an LLM (DeepSeek-R1) to reason over the candidates and provide biological interpretability.

```
TCGA-BRCA (20k genes)
    → Mamba SSM (trained on tumor/normal as pretext task)
    → Gradient Saliency (top-50 candidate genes)
    → DeepSeek-R1 (reasons over candidates, selects top-20 with CoT rationale)
    → Benchmark comparison
```

## Results

| Method | Genes | Accuracy | F1 | AUC |
|---|---|---|---|---|
| Variance baseline | 5000 | 0.846 | 0.871 | 0.930 |
| Mamba saliency | 50 | 0.640 | 0.715 | 0.734 |
| **Mamba + LLM reasoning (ours)** | **20** | **0.915** | **0.926** | **0.976** |

250× fewer genes, +6.9% accuracy over the 5000-gene baseline.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train_mamba_official.py` | Train Mamba classifier on TCGA-BRCA |
| `scripts/01_extract_saliency.py` | Extract top-50 genes via gradient saliency |
| `scripts/02_llm_reasoning.py` | LLM causal gene selection prompt |
| `scripts/02b_llm_gene_explain.py` | LLM per-gene biological reasoning (CoT) |
| `scripts/03_train_comparison.py` | 3-way benchmark (B1/B2/B3) |
| `scripts/04_generate_figures.py` | Generate all paper figures |

## Setup

```bash
conda activate fs311
# causal-conv1d and mamba-ssm must be built from source for sm_120 (RTX 50-series)
# see setup notes in environment.yml
```

## Data

TCGA-BRCA RNA-seq downloaded via GDC Data Transfer Tool. Not included in repo (5GB+).

## Outputs

- `data_processed/top_genes_saliency.json` — Mamba-selected genes + scores
- `data_processed/llm_gene_reasoning.json` — LLM reasoning + selected gene set
- `results/comparison_results.json` — Benchmark numbers
- `results/figures/` — Paper figures (PNG)
