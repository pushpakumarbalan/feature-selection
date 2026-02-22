"""
08b_llm_consistency_debug.py - inspect raw LLM responses and try robust parsing
"""
import json
import re
import ollama

DATA_JSON = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
OUT_JSON  = "/data4t/projects/fs/results/llm_consistency.json"
MODEL     = "deepseek-r1:7b"

with open(DATA_JSON) as f:
    data = json.load(f)
input_genes = data["input_genes"]
scores = {g: 1.0/(i+1)*0.001 for i, g in enumerate(input_genes)}

PROMPT = f"""You are a cancer genomics expert.
Below are 50 candidate genes from a Mamba SSM trained on TCGA-BRCA RNA-seq.
HIGH saliency does NOT imply BRCA specificity.

{chr(10).join(f"  {i+1:2d}. {g}  (saliency={scores[g]:.6f})" for i,g in enumerate(input_genes))}

REJECTION CRITERIA (reject if any):
R1: Unannotated/lncRNA with no cancer role
R2: Housekeeping gene
R3: Off-tissue gene (muscle, neuronal, renal)
R4: Non-specific immune marker
R5: Antisense RNA

KEEP CRITERIA (keep if at least one):
K1: Known BRCA oncogene/TSG (COSMIC/OncoKB)
K2: BRCA pathway (ER, HER2, PI3K/AKT, EMT)
K3: PAM50 gene or validated BRCA biomarker

Evaluate every gene with one line: "GENE: KEEP/REJECT — reason"
Then output: SELECTED_GENES: gene1, gene2, ...
"""

def robust_parse(text: str, input_genes: list) -> list:
    """Try multiple extraction strategies."""
    text_upper = text.upper()
    input_set = set(g.upper() for g in input_genes)

    # Strategy 1: SELECTED_GENES: line
    for line in text.split("\n"):
        if "SELECTED_GENES:" in line.upper():
            part = line.split(":", 1)[1]
            genes = [g.strip().upper() for g in re.split(r"[,\s]+", part) if g.strip()]
            return [g for g in genes if g in input_set]

    # Strategy 2: "Final selection" or "Selected genes" section
    m = re.search(r"(?:final.{0,20}select|selected.{0,10}genes?)\s*:?\s*\n?(.*?)(?:\n\n|\Z)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        block = m.group(1)
        genes = [g.strip().upper() for g in re.split(r"[,\n*\d\.\-]+", block)
                 if g.strip() and g.strip().upper() in input_set]
        if genes:
            return genes

    # Strategy 3: extract all gene names mentioned in KEEP lines
    keep_genes = []
    for line in text.split("\n"):
        if re.search(r"\bKEEP\b", line, re.IGNORECASE):
            for g in input_genes:
                if g.upper() in line.upper():
                    keep_genes.append(g.upper())
    if keep_genes:
        return list(dict.fromkeys(keep_genes))  # deduplicated, order-preserved

    return []


import itertools
import numpy as np

def jaccard(s1, s2):
    a, b = set(s1), set(s2)
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


print("=== Standard T=0.3, n=5 ===")
runs_std = []
for i in range(5):
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        options={"temperature": 0.3, "num_predict": 6000},
    )
    text = resp["message"]["content"]
    genes = robust_parse(text, input_genes)
    runs_std.append(genes)
    print(f"  run {i+1}: {len(genes)} genes → {genes[:5]}...")
    if i == 0:
        # Save first raw response for inspection
        with open("/data4t/projects/fs/results/llm_raw_response_sample.txt", "w") as f:
            f.write(text)
        print("  [saved raw response to results/llm_raw_response_sample.txt]")

print("\n=== Shuffled order T=0.3, n=5 ===")
import random; random.seed(99)
shuffled = input_genes[:]
random.shuffle(shuffled)
prompt_shuf = PROMPT.replace(
    chr(10).join(f"  {i+1:2d}. {g}  (saliency={scores[g]:.6f})" for i,g in enumerate(input_genes)),
    chr(10).join(f"  {i+1:2d}. {g}  (saliency={scores[g]:.6f})" for i,g in enumerate(shuffled))
)
runs_shuf = []
for i in range(5):
    resp = ollama.chat(model=MODEL, messages=[{"role":"user","content":prompt_shuf}],
                       options={"temperature":0.3,"num_predict":6000})
    genes = robust_parse(resp["message"]["content"], input_genes)
    runs_shuf.append(genes)
    print(f"  run {i+1}: {len(genes)} genes → {genes[:5]}...")

print("\n=== Names-only T=0.3, n=5 ===")
prompt_names = re.sub(r"\(saliency=[^\)]+\)", "", PROMPT)
prompt_names = re.sub(r"\s+\d+\.\s+", "\n  ", prompt_names)
runs_names = []
for i in range(5):
    resp = ollama.chat(model=MODEL, messages=[{"role":"user","content":prompt_names}],
                       options={"temperature":0.3,"num_predict":6000})
    genes = robust_parse(resp["message"]["content"], input_genes)
    runs_names.append(genes)
    print(f"  run {i+1}: {len(genes)} genes → {genes[:5]}...")

def summarize(runs, label):
    sizes = [len(r) for r in runs]
    pairs = [jaccard(runs[i],runs[j]) for i,j in itertools.combinations(range(len(runs)),2)]
    all_genes = set(g for r in runs for g in r)
    freq = {g: sum(1 for r in runs if g in r) for g in all_genes}
    stable = [g for g,f in freq.items() if f >= 4]
    return {
        "label": label,
        "mean_set_size": round(float(np.mean(sizes)),2),
        "std_set_size": round(float(np.std(sizes)),2),
        "mean_jaccard": round(float(np.mean(pairs)),4) if pairs else 0.0,
        "std_jaccard": round(float(np.std(pairs)),4) if pairs else 0.0,
        "stable_genes_4of5": stable,
        "all_runs": runs,
    }

results = {
    "standard_T03":    summarize(runs_std,   "Standard prompt, T=0.3"),
    "shuffled_T03":    summarize(runs_shuf,  "Shuffled order, T=0.3"),
    "names_only_T03":  summarize(runs_names, "Names-only, T=0.3"),
}

print("\n=== Summary ===")
for k,v in results.items():
    print(f"  {v['label']:<40} J={v['mean_jaccard']:.3f}±{v['std_jaccard']:.3f}  "
          f"size={v['mean_set_size']:.1f}±{v['std_set_size']:.1f}  "
          f"stable={v['stable_genes_4of5']}")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {OUT_JSON}")
