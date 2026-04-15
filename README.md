# Mamba SSM + LLM Reasoning for Biomarker Discovery (TCGA-BRCA)

## Summary

This repository implements the pipeline described in:

[Mamba_SSM_with_LLM_Reasoning_for_Biomarker_Discovery.pdf](paper/Mamba_SSM_with_LLM_Reasoning_for_Biomarker_Discovery.pdf)

The workflow:

```
TCGA-BRCA RNA-seq (~20k genes)
    -> Mamba SSM classifier (tumor vs normal)
    -> Gradient saliency ranking (top-50 genes)
    -> DeepSeek-R1 structured CoT filtering
    -> Final 17-gene BRCA-specific subset
    -> 3-way benchmark (B1/B2/B3)
```

### System Pipeline Figure

![System Pipeline](results/figures/f1g_pipeline1.png)

## Paper Results (Held-Out Test Split)

| Method | Genes | Accuracy | F1 | AUC |
|---|---|---|---|---|
| B1: Variance baseline | 5000 | 0.8785 | 0.8941 | 0.903 |
| B2: Mamba saliency only (no LLM) | 50 | 0.7247 | 0.7813 | 0.832 |
| **B3: Mamba + LLM structured CoT (ours)** | **17** | **0.8907** | **0.9033** | **0.927** |

B3 uses 294x fewer genes than B1 while improving AUC by +0.024.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train_mamba_official.py` | Train Mamba classifier on TCGA-BRCA |
| `scripts/01_extract_saliency.py` | Extract top-50 genes via gradient saliency |
| `scripts/02_llm_reasoning.py` | DeepSeek-R1 structured CoT gene filtering |
| `scripts/02b_llm_gene_explain.py` | Optional LLM gene-interpretation pass |
| `scripts/03_train_comparison.py` | 3-way benchmark (B1/B2/B3) |
| `scripts/04_generate_figures.py` | Generate all paper figures |
| `scripts/07_reasoning_faithfulness.py` | Faithfulness audit against curated BRCA gene sets |

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
- `results/reasoning_faithfulness.json` — Faithfulness audit summary
- `results/figures/` — Paper figures (PNG)
