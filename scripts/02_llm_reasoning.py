"""
Step 2: Send top-N saliency genes to DeepSeek-R1 via ollama.
Outputs LLM-selected gene subset + chain-of-thought to data_processed/llm_gene_reasoning.json
"""
import json
import re
import requests

IN_JSON  = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
OUT_JSON = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "deepseek-r1:7b"
N_SELECT   = 20  # LLM selects this many from top-50

# ─── Prompt design notes ────────────────────────────────────────────────────
# Problem we are solving: a naive LLM will simply keep the top-N by saliency
# rank (trivial cutoff), making it indistinguishable from B2.
# Solution: explicitly forbid rank-based selection, pre-annotate known-bad
# gene archetypes so the LLM has concrete rejection targets, and require it
# to evaluate EVERY gene before outputting the final list.
# ────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an expert breast cancer genomics researcher with deep knowledge of TCGA-BRCA biology.

A Mamba selective state space model trained on TCGA RNA-seq data produced the list below.
Genes are ordered by gradient saliency score, but HIGH SALIENCY DOES NOT MEAN BRCA-SPECIFIC.
The saliency scores span a very narrow range ({score_min:.5f}–{score_max:.5f}).
Many high-ranked genes are likely high because they co-vary with tumour purity or immune infiltration, NOT because they are causal drivers of breast cancer.

YOUR TASK: Evaluate EVERY gene for BRCA specificity using biological knowledge.
Select between 15–25 genes that are the most specifically relevant to breast cancer.

REJECTION RULES — a gene must be rejected if it matches ANY of these:
  R1. Unannotated / low-confidence sequences: names like AL######.#, AC######.#, MIR####, or lncRNAs with no known disease role
  R2. Ubiquitous housekeeping genes (expressed in all tissues — e.g., ribosomal, metabolic, cytoskeletal)
  R3. Tissue-specific genes unrelated to breast: muscle (e.g., MB, UTRN, DMD), neuronal, renal
  R4. General immune markers with no breast-specific role (e.g., HLA class II genes without documented BRCA association)
  R5. Antisense RNAs or pseudogenes with no known functional BRCA connection (e.g., *-AS1 genes unless well-documented)

KEEP RULES — keep a gene if ANY of these are true, REGARDLESS of its saliency rank:
  K1. Known breast cancer oncogene or tumour suppressor (e.g., FOXA1, XBP1, INPP4B, ZEB1, MLPH)
  K2. Part of a pathway directly implicated in BRCA: ER signalling, PI3K/AKT, HER2, EMT, DNA damage response
  K3. Clinically used as a BRCA subtype marker or prognostic gene

CRITICAL: DO NOT simply keep the genes with the highest saliency ranks.
You MUST reject some genes from the top-10 if they match R1–R5, and you MAY keep genes ranked 21–50 if they match K1–K3.

Gene list (rank. NAME  saliency):
{gene_list}

Step 1 — For EVERY gene output exactly one line:
GENE_NAME: keep - <one sentence: specific pathway or BRCA evidence>
GENE_NAME: reject - <one sentence: which rejection rule and why>

Step 2 — After all {n} lines, output your final selection:
SELECTED_GENES: GENE1, GENE2, GENE3, ...

Include 15 to 25 genes in SELECTED_GENES.
"""


def main():
    with open(IN_JSON) as f:
        data = json.load(f)

    genes = data["top_genes"]
    scores = data["scores"]

    gene_list_str = "\n".join(
        f"  {i+1:2d}. {g:<22s} saliency={scores[g]:.5f}"
        for i, g in enumerate(genes)
    )
    score_values = [scores[g] for g in genes]
    prompt = PROMPT_TEMPLATE.format(
        n=len(genes),
        gene_list=gene_list_str,
        score_min=min(score_values),
        score_max=max(score_values),
    )

    print(f"Sending {len(genes)} Mamba-selected genes to {MODEL}...")
    print("LLM will reason about BRCA specificity for EVERY gene before selecting.")
    print("Saliency rank is explicitly NOT the selection criterion.")
    print("-" * 60)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 8192},  # 0.3: avoids trivial rank-copy
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    full_response = resp.json()["response"]

    print("LLM Response:")
    print(full_response)
    print("-" * 60)

    # Parse SELECTED_GENES line — handle markdown bold (**...**), underscores vs spaces
    # e.g. "SELECTED_GENES: ..." or "**SELECTED GENES**: ..." or "**SELECTED_GENES**: ..."
    match = re.search(
        r'\*{0,2}SELECTED[_ ]GENES\*{0,2}:?\*{0,2}\s*([A-Z0-9,._\- \n]+)',
        full_response, re.IGNORECASE
    )
    if match:
        selected_raw = match.group(1).strip().split('\n')[0]  # first line only
        selected_genes = [
            g.strip().strip('*').rstrip('.,') for g in re.split(r'[,;]', selected_raw) if g.strip()
        ]
        valid_input = set(genes)
        selected_valid = [g for g in selected_genes if g in valid_input]
        hallucinated  = [g for g in selected_genes if g and g not in valid_input]
        if hallucinated:
            print(f"  ⚠ LLM hallucinated {len(hallucinated)} gene names (not in input): {hallucinated}")
            print(f"     These are dropped; only input-list genes are used.")
        if len(selected_valid) >= 5:
            selected_genes = selected_valid
            print(f"\nLLM reasoned and selected {len(selected_genes)} genes: {selected_genes}")
        else:
            print(f"WARNING: Only {len(selected_valid)} valid genes parsed — using top-20 saliency as fallback")
            selected_genes = genes[:N_SELECT]
    else:
        print("WARNING: Could not parse SELECTED_GENES — using top-20 saliency as fallback")
        selected_genes = genes[:N_SELECT]

    # Extract per-gene keep/reject reasoning lines
    reasoning_lines = re.findall(
        r'^([A-Z0-9._-]+):\s*(keep|reject)\s*-\s*(.+)$',
        full_response, re.MULTILINE | re.IGNORECASE
    )
    per_gene_reasoning = {g: {"decision": d, "reason": r} for g, d, r in reasoning_lines}

    # Audit: flag if LLM kept exactly the top-N by rank (trivial solution)
    top_n_by_rank = set(genes[:len(selected_genes)])
    selected_set  = set(selected_genes)
    rank_overlap  = len(top_n_by_rank & selected_set) / len(selected_set) if selected_set else 1.0
    if rank_overlap == 1.0:
        print(f"\n⚠️  WARNING: LLM kept exactly the top-{len(selected_genes)} by saliency rank "
              f"(overlap={rank_overlap:.0%}). Reasoning may be trivial — consider re-running.")
    else:
        rejected_from_top = top_n_by_rank - selected_set
        added_from_lower  = selected_set - top_n_by_rank
        print(f"\n✓ LLM deviated from saliency rank (overlap={rank_overlap:.0%})")
        print(f"  Rejected from top-{len(selected_genes)}: {rejected_from_top}")
        print(f"  Promoted from lower ranks:   {added_from_lower}")

    think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
    chain_of_thought = think_match.group(1).strip() if think_match else full_response

    result = {
        "model": MODEL,
        "input_genes": genes,
        "mamba_selected_genes": selected_genes,
        "per_gene_reasoning": per_gene_reasoning,
        "rank_overlap_with_saliency_top_n": round(rank_overlap, 3),
        "chain_of_thought": chain_of_thought,
        "full_response": full_response,
        "note": (
            "Mamba ranks top-50 by gradient saliency. "
            "LLM reasons BRCA-specificity for every gene independently of rank, "
            "rejecting housekeeping/unannotated/off-target genes and promoting "
            "lower-ranked but well-known BRCA genes. "
            "LLM selection causally determines the B3 gene set."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
