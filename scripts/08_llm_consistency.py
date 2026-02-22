"""
08_llm_consistency.py
Test LLM reasoning consistency:
  - Run the same structured prompt 5 times at T=0.3 (default)
  - Run 5 times at T=0.7 (higher entropy)
Measure:
  - Jaccard overlap between every pair of runs
  - Mean set size, std set size
  - Stability index (mean pairwise Jaccard)
  - Which genes are "stable" (selected in ≥4/5 runs) vs "unstable"
Also runs the counterfactual conditions (no-rank, shuffled-order) — see below.
Output: results/llm_consistency.json
"""
import json, os, itertools, random
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("WARNING: ollama python package not installed; run: pip install ollama")

DATA_JSON   = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
OUT_JSON    = "/data4t/projects/fs/results/llm_consistency.json"
MODEL       = "deepseek-r1:7b"

# ─── Build prompt variants ────────────────────────────────────────────────────

def make_prompt(gene_list: list, scores: dict, variant: str = "standard") -> str:
    """
    variant:
      standard   — gene name, rank, score (original)
      no_rank    — gene name + score, no rank number
      no_score   — gene name + rank, no score
      shuffled   — standard but gene order shuffled randomly
      names_only — only gene names, no rank or score
    """
    random.seed(42)
    if variant == "shuffled":
        gl = gene_list[:]
        random.shuffle(gl)
    else:
        gl = gene_list

    if variant == "standard":
        gene_lines = "\n".join(
            f"  {i+1:2d}. {g}  (saliency={scores.get(g,'N/A'):.6f})"
            for i, g in enumerate(gl)
        )
    elif variant == "no_rank":
        gene_lines = "\n".join(
            f"  {g}  (saliency={scores.get(g,'N/A'):.6f})"
            for g in gl
        )
    elif variant == "no_score":
        gene_lines = "\n".join(f"  {i+1:2d}. {g}" for i, g in enumerate(gl))
    elif variant == "shuffled":
        gene_lines = "\n".join(
            f"  {i+1:2d}. {g}  (saliency={scores.get(g,'N/A'):.6f})"
            for i, g in enumerate(gl)
        )
    elif variant == "names_only":
        gene_lines = "\n".join(f"  {g}" for g in gl)
    else:
        raise ValueError(variant)

    return f"""You are a cancer genomics expert.
Below are {len(gl)} candidate genes ranked by gradient saliency from a Mamba SSM trained on TCGA-BRCA RNA-seq data.
HIGH saliency does NOT imply BRCA specificity — many high-scoring genes are tissue/immune confounders.

{gene_lines}

REJECTION CRITERIA (reject if any apply):
R1: Unannotated/lncRNA/pseudogene with no documented cancer role
R2: Housekeeping gene with no cancer-specific function
R3: Off-tissue gene (muscle, neuronal, renal) with no BRCA evidence
R4: Non-specific immune marker without documented BRCA specificity
R5: Antisense RNA or non-coding RNA with no established BRCA function

KEEP CRITERIA (keep only if at least one applies):
K1: Known breast cancer oncogene or tumour suppressor (COSMIC CGC or OncoKB)
K2: Established role in BRCA-relevant pathway (ER, HER2, PI3K/AKT, EMT)
K3: PAM50 intrinsic subtype gene or validated BRCA biomarker

INSTRUCTIONS:
- Evaluate EVERY gene individually. Write one line per gene: "GENE: KEEP/REJECT — reason"
- Do NOT simply select the top-N by saliency rank.
- After evaluating all genes output exactly: SELECTED_GENES: gene1, gene2, ...
"""


def parse_selected(response_text: str) -> list:
    for line in response_text.split("\n"):
        if "SELECTED_GENES:" in line:
            part = line.split("SELECTED_GENES:")[1]
            genes = [g.strip().upper() for g in part.split(",") if g.strip()]
            return genes
    return []


def jaccard(s1, s2):
    a, b = set(s1), set(s2)
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)


def pairwise_jaccard(runs):
    scores = [jaccard(runs[i], runs[j])
              for i, j in itertools.combinations(range(len(runs)), 2)]
    return float(np.mean(scores)), float(np.std(scores))


def run_llm_batch(prompt: str, n: int, temperature: float, label: str) -> list:
    if not OLLAMA_AVAILABLE:
        print(f"  [SKIP — ollama not installed] {label}")
        return []
    runs = []
    for i in range(n):
        print(f"  run {i+1}/{n} T={temperature} ...", end=" ", flush=True)
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": 4096},
        )
        text = resp["message"]["content"]
        selected = parse_selected(text)
        runs.append(selected)
        print(f"→ {len(selected)} genes: {selected}")
    return runs


def analyze_runs(runs: list, label: str) -> dict:
    if not runs:
        return {"label": label, "note": "skipped (ollama unavailable)"}
    all_genes  = [g for r in runs for g in r]
    gene_freq  = {g: sum(1 for r in runs if g in r) for g in set(all_genes)}
    stable     = sorted([g for g,f in gene_freq.items() if f >= len(runs)*0.8],
                        key=lambda g: -gene_freq[g])
    unstable   = sorted([g for g,f in gene_freq.items() if f < len(runs)*0.5],
                        key=lambda g: -gene_freq[g])
    mj, sj     = pairwise_jaccard(runs)
    return {
        "label": label,
        "n_runs": len(runs),
        "set_sizes": [len(r) for r in runs],
        "mean_set_size": float(np.mean([len(r) for r in runs])),
        "std_set_size":  float(np.std([len(r) for r in runs])),
        "mean_pairwise_jaccard": round(mj, 4),
        "std_pairwise_jaccard":  round(sj, 4),
        "stable_genes":  stable,
        "unstable_genes": unstable,
        "gene_frequency": gene_freq,
        "all_runs": runs,
    }


def main():
    with open(DATA_JSON) as f:
        data = json.load(f)

    input_genes = data["input_genes"]
    # Build dummy scores from position (saliency values not stored per gene;
    # use rank-based proxy as in original pipeline)
    scores = {g: 1.0/(i+1) * 0.001 for i, g in enumerate(input_genes)}
    # If actual scores were saved in JSON use them:
    if "saliency_scores" in data:
        scores = data["saliency_scores"]

    N_RUNS = 5
    results = {}

    # ── Condition 1: standard prompt × 5, T=0.3 ─────────────────────────────
    print("\n=== [1] Standard prompt, T=0.3, n=5 ===")
    prompt_std = make_prompt(input_genes, scores, "standard")
    runs_std_03 = run_llm_batch(prompt_std, N_RUNS, 0.3, "standard T=0.3")
    results["standard_T03"] = analyze_runs(runs_std_03, "Standard prompt, T=0.3")

    # ── Condition 2: standard prompt × 5, T=0.7 ─────────────────────────────
    print("\n=== [2] Standard prompt, T=0.7, n=5 ===")
    runs_std_07 = run_llm_batch(prompt_std, N_RUNS, 0.7, "standard T=0.7")
    results["standard_T07"] = analyze_runs(runs_std_07, "Standard prompt, T=0.7")

    # ── Condition 3: shuffled gene order, T=0.3 ─────────────────────────────
    print("\n=== [3] Shuffled gene order, T=0.3, n=5 ===")
    prompt_shuf = make_prompt(input_genes, scores, "shuffled")
    runs_shuf = run_llm_batch(prompt_shuf, N_RUNS, 0.3, "shuffled T=0.3")
    results["shuffled_T03"] = analyze_runs(runs_shuf, "Shuffled order, T=0.3")

    # ── Condition 4: names only (no rank, no score), T=0.3 ───────────────────
    print("\n=== [4] Names-only prompt (no rank/score), T=0.3, n=5 ===")
    prompt_names = make_prompt(input_genes, scores, "names_only")
    runs_names = run_llm_batch(prompt_names, N_RUNS, 0.3, "names_only T=0.3")
    results["names_only_T03"] = analyze_runs(runs_names, "Names-only (no rank/score), T=0.3")

    # ── Condition 5: no score (rank only), T=0.3 ─────────────────────────────
    print("\n=== [5] Rank-only prompt (no saliency scores), T=0.3, n=5 ===")
    prompt_noscr = make_prompt(input_genes, scores, "no_score")
    runs_noscr = run_llm_batch(prompt_noscr, N_RUNS, 0.3, "no_score T=0.3")
    results["no_score_T03"] = analyze_runs(runs_noscr, "No saliency scores (rank only), T=0.3")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Condition':<42} {'Jaccard mean':>13} {'Set size':>9}")
    print("-"*70)
    for k, v in results.items():
        if "note" in v:
            print(f"  {v['label']:<40}  [skipped]")
        else:
            print(f"  {v['label']:<40}  {v['mean_pairwise_jaccard']:.4f}±{v['std_pairwise_jaccard']:.4f}  "
                  f"{v['mean_set_size']:.1f}±{v['std_set_size']:.1f}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
