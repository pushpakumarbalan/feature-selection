"""
Step 2b: LLM reasons about Mamba-selected genes.
Mamba selected these genes from data. LLM explains WHY they are biologically
meaningful using its encoded knowledge (pathways, literature, clinical context).
This is the interpretability contribution — not filtering.
"""
import json
import requests

SALIENCY_JSON = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
LLM_JSON      = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "deepseek-r1:7b"
TOP_N         = 20   # same as B3 in 03_train_comparison.py

with open(SALIENCY_JSON) as f:
    saliency_data = json.load(f)

# These are the Mamba-selected genes — LLM explains them, does NOT filter
mamba_genes = saliency_data["top_genes"][:TOP_N]
scores      = saliency_data["scores"]

prompt = f"""You are an expert breast cancer biologist.

A Mamba selective state space model trained on TCGA-BRCA RNA-seq data identified the following {TOP_N} genes as the most important for distinguishing breast tumor from normal tissue, using gradient saliency (ranked by importance score):

{chr(10).join(f'  {i+1}. {g} (saliency={scores[g]:.4f})' for i, g in enumerate(mamba_genes))}

For EACH gene above, write exactly one line explaining its specific known role in breast cancer biology. Be precise — mention the relevant pathway, molecular function, or clinical significance. If a gene has no known breast cancer role, say so honestly.

Format (one line per gene):
GENE_NAME: [biological role in breast cancer]

After all genes, write 2-3 sentences summarising the key biological pathways represented in this Mamba-selected gene set."""

print(f"DeepSeek-R1 reasoning about {TOP_N} Mamba-selected genes...")
print("-" * 60)

resp = requests.post(OLLAMA_URL, json={
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 4096},
}, timeout=300)
resp.raise_for_status()
reasoning = resp.json()["response"]
print(reasoning)
print("-" * 60)

result = {
    "model": MODEL,
    "mamba_selected_genes": mamba_genes,
    "saliency_scores": {g: scores[g] for g in mamba_genes},
    "chain_of_thought": reasoning,
    "note": "Genes selected by Mamba gradient saliency. LLM provides biological reasoning only.",
}
with open(LLM_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {LLM_JSON}")
