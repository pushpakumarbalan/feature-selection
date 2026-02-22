"""
Fix per_gene_reasoning parser: extract decision-level metrics from
the chain_of_thought field in llm_gene_reasoning.json.
Also computes decision-level TP, FP, TN, FN against ground-truth gene sets.
Outputs updated llm_gene_reasoning.json with per_gene_reasoning populated.
"""
import json
import re

LLM_JSON = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"

# ─── Ground-truth gene sets ───────────────────────────────────────────────────
# Compiled from COSMIC CGC Tier1, OncoKB BRCA, PAM50, ER/PI3K/EMT pathways
VALIDATED_BRCA = {
    # COSMIC CGC Tier 1
    "TP53", "PIK3CA", "CDH1", "GATA3", "PTEN", "BRCA1", "BRCA2",
    "ERBB2", "CCND1", "AKT1", "RB1", "ATM", "CHEK2", "NF1",
    # ER signalling
    "ESR1", "PGR", "FOXA1", "XBP1", "NRIP1", "MLPH", "INPP4B",
    # PI3K/AKT
    "INPP4B", "PTEN", "AKT1",
    # EMT/TNBC
    "ZEB1", "ZEB2", "VIM", "TWIST1", "SNAI1", "SNAI2",
    # PAM50 relevant
    "RHOB", "ESR1", "PGR", "FOXA1", "MLPH",
    # Well-documented BRCA-associated
    "PTK6", "ETS1", "NRIP1", "COL8A1", "LTBP2", "CTHRC1",
    "GPX7", "PTPRK", "FGL2", "THY1",
}

KNOWN_NON_BRCA = {
    # Off-tissue / muscle
    "MB", "UTRN",
    # Non-specific immune
    "HLA-DRB1", "ITGAL",
    # Neural / renal transcription factor
    "LMX1B",
    # Antisense RNA no BRCA function
    "PRKAG2-AS1",
    # Globin regulator (no cancer relevance)
    "RHEX",
    # Unannotated/lncRNA
    "AL035661.1", "AP000553.7", "MIR5581",
}


def parse_cot(cot_text: str, input_genes: list) -> dict:
    """
    Extract per-gene reasoning text from numbered list in the CoT block.
    Returns dict: gene -> {"rationale": str, "decision": "selected" | "not_in_output"}
    """
    per_gene = {}
    # Match lines like "1. **GENE**: rationale text"
    pattern = re.compile(
        r"\d+\.\s+\*\*([A-Z0-9_\.\-]+)\*\*[:\s]+(.+?)(?=\n\d+\.\s+\*\*|\nSELECTED|$)",
        re.DOTALL
    )
    for m in pattern.finditer(cot_text):
        gene = m.group(1).strip()
        rationale = m.group(2).strip().replace("\n", " ")
        per_gene[gene] = {"rationale": rationale}
    return per_gene


def compute_decision_metrics(input_genes, selected_genes, per_gene_reasoning):
    """Compute TP/FP/TN/FN at decision level for genes with known GT."""
    tp = fp = tn = fn = unverifiable = 0
    details = {}
    for gene in input_genes:
        selected = gene in selected_genes
        if gene in VALIDATED_BRCA:
            gt = "validated_brca"
            if selected:
                tp += 1
                outcome = "TP"
            else:
                fn += 1
                outcome = "FN"
        elif gene in KNOWN_NON_BRCA:
            gt = "known_non_brca"
            if selected:
                fp += 1
                outcome = "FP"
            else:
                tn += 1
                outcome = "TN"
        else:
            gt = "unknown"
            unverifiable += 1
            outcome = "UNK"

        details[gene] = {
            "ground_truth": gt,
            "selected": selected,
            "outcome": outcome,
            "rationale": per_gene_reasoning.get(gene, {}).get("rationale", ""),
        }

    n_verifiable = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr       = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "unverifiable": unverifiable,
        "n_verifiable_input_genes": n_verifiable,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity_TNR": round(tnr, 4),
        "per_gene": details,
    }


def main():
    with open(LLM_JSON) as f:
        data = json.load(f)

    input_genes   = data["input_genes"]
    selected_genes = set(data["mamba_selected_genes"])
    cot_text      = data.get("chain_of_thought", "")

    # Parse per-gene reasoning from CoT
    per_gene = parse_cot(cot_text, input_genes)
    print(f"Parsed reasoning for {len(per_gene)} / {len(input_genes)} input genes")

    # Annotate with selection decision
    for gene in input_genes:
        if gene not in per_gene:
            per_gene[gene] = {"rationale": "[not parsed from CoT]"}
        per_gene[gene]["selected"] = gene in selected_genes

    # Decision-level metrics
    metrics = compute_decision_metrics(input_genes, selected_genes, per_gene)

    print("\n=== Decision-Level Faithfulness Metrics ===")
    print(f"TP (validated BRCA, selected):         {metrics['TP']}")
    print(f"FP (known non-BRCA, selected):         {metrics['FP']}")
    print(f"TN (known non-BRCA, not selected):     {metrics['TN']}")
    print(f"FN (validated BRCA, not selected):     {metrics['FN']}")
    print(f"Unverifiable (unknown GT):             {metrics['unverifiable']}")
    print(f"Precision (on verifiable):             {metrics['precision']:.4f}")
    print(f"Recall (on validated BRCA in input):   {metrics['recall']:.4f}")
    print(f"Specificity/TNR (non-BRCA rejection):  {metrics['specificity_TNR']:.4f}")
    print("\n=== Per-Gene Outcomes ===")
    for gene, d in metrics["per_gene"].items():
        sel = "KEEP" if d["selected"] else "REJECT"
        print(f"  {gene:<20} GT={d['ground_truth']:<18} {sel:<8} -> {d['outcome']}  | {d['rationale'][:80]}")

    # Update JSON
    data["per_gene_reasoning"] = per_gene
    data["decision_level_metrics"] = metrics
    with open(LLM_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nUpdated {LLM_JSON}")


if __name__ == "__main__":
    main()
