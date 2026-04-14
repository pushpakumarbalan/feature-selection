"""
Step 7: LLM Reasoning Faithfulness Analysis
=============================================
Research question: Is the LLM's biological reasoning accurate?
  - When the LLM says "keep — part of ER signalling", is that correct?
  - When it says "reject — housekeeping gene", is that correct?
  - How often does it MISS well-known BRCA genes from the input list?
  - How often does it INVENT pathway connections not supported by literature?

Method:
  We cross-reference the LLM's keep/reject decisions against a curated set of
  known BRCA-relevant genes (compiled from COSMIC CGC, OncoKB, TCGA marker
  papers). This gives ground-truth precision/recall for the LLM's reasoning.

Outputs: results/reasoning_faithfulness.json
"""

import json
import os

LLM_JSON   = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
SAL_JSON   = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
OUT_JSON   = "/data4t/projects/fs/results/reasoning_faithfulness.json"

# ──────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH: Known BRCA-relevant genes, grouped by evidence class
# Sources: COSMIC CGC, OncoKB, TCGA BRCA 2012, PAM50, key pathway papers
# ──────────────────────────────────────────────────────────────────────────────
KNOWN_BRCA_GENES = {

    "tier1_drivers": {
        "TP53", "PIK3CA", "CDH1", "GATA3", "MAP3K1", "CBFB", "PTEN",
        "BRCA1", "BRCA2", "ERBB2", "CCND1", "MYC", "FGFR1", "AKT1",
        "RB1", "ATM", "CHEK2", "PALB2", "STK11", "NF1",
    },

    "er_signalling": {
        "ESR1", "PGR", "FOXA1", "GATA3", "XBP1", "NRIP1", "TFF1",
        "CCND1", "TFF3", "AGR2", "MLPH", "INPP4B",
    },

    "pi3k_akt": {
        "PIK3CA", "PTEN", "AKT1", "AKT2", "INPP4B", "TSC1", "TSC2",
        "MTOR", "PIK3R1", "PREX1",
    },

    "her2_pathway": {
        "ERBB2", "ERBB3", "GRB7", "STARD3", "EGFR", "ERBB4",
    },

    "emt_tnbc": {
        "ZEB1", "ZEB2", "VIM", "TWIST1", "SNAI1", "SNAI2", "FN1",
        "CDH2", "CDH1", "FOXC1", "LAMC2",
    },

    "ddr": {
        "BRCA1", "BRCA2", "ATM", "CHEK2", "PALB2", "RAD51", "RAD51C",
        "BRIP1", "BARD1", "FANCL",
    },

    "pam50": {
        "UBE2T", "BIRC5", "NUF2", "CDC6", "CCNB1", "TYMS", "MYBL2",
        "CEP55", "MELK", "NDC80", "RRM2", "UBE2C", "CENPF", "PTTG1",
        "EXO1", "ORC6", "ANLN", "RHOB",
        "FOXA1", "BLVRA", "MMP11", "CDH3", "MAPT", "ESR1", "SFRP1",
        "KRT14", "KRT5", "MLPH", "CCNB1", "CDC20", "CXXC5",
        "NAT1", "TMEM45B", "BAG1", "PGR", "ACTR3B", "MIA", "NCE2",
        "GPR160", "FGFR4", "KRT17", "SFRP1", "CCND1",
    },

    "brca_associated": {
        "RHOB", "PTK6", "ETS1", "NRIP1", "ZEB1", "MLPH", "INPP4B",
        "FOXA1", "XBP1", "THY1", "COL8A1", "LTBP2", "CTHRC1",
        "GPX7", "PTPRK", "FGL2",
    },
}

ALL_VALIDATED = set().union(*KNOWN_BRCA_GENES.values())

KNOWN_NON_BRCA = {
    "MB", "UTRN", "DMD",
    "HLA-DRB1",
    "AL035661.1", "AP000553.7", "MIR5581",
    "PRKAG2-AS1",
    "LMX1B",
    "ITGAL",
    "DNAH5",
    "RTN4RL1",
}


def classify_gene(gene):
    if gene in KNOWN_NON_BRCA:
        return False, "known_non_brca"
    if gene in ALL_VALIDATED:
        cats = [k for k, v in KNOWN_BRCA_GENES.items() if gene in v]
        return True, "+".join(cats)
    return None, "unknown"


def evaluate_llm_decisions(per_gene_reasoning, all_input_genes):
    TP = FP = TN = FN = UNKNOWN_KEEP = UNKNOWN_REJECT = 0
    details = {}

    for gene in all_input_genes:
        decision = per_gene_reasoning.get(gene, {}).get("decision", "NOT_EVALUATED")
        reason   = per_gene_reasoning.get(gene, {}).get("reason", "")
        is_valid, cat = classify_gene(gene)

        record = {
            "llm_decision": decision,
            "llm_reason": reason,
            "ground_truth_validated": is_valid,
            "ground_truth_category": cat,
        }

        if is_valid is True:
            if decision == "keep":
                TP += 1; record["faithfulness_verdict"] = "CORRECT_KEEP"
            elif decision == "reject":
                FN += 1; record["faithfulness_verdict"] = "WRONG_REJECT"
            else:
                record["faithfulness_verdict"] = "NOT_EVALUATED"
        elif is_valid is False:
            if decision == "keep":
                FP += 1; record["faithfulness_verdict"] = "WRONG_KEEP"
            elif decision == "reject":
                TN += 1; record["faithfulness_verdict"] = "CORRECT_REJECT"
            else:
                record["faithfulness_verdict"] = "NOT_EVALUATED"
        else:
            if decision == "keep":
                UNKNOWN_KEEP += 1; record["faithfulness_verdict"] = "UNVERIFIABLE_KEEP"
            else:
                UNKNOWN_REJECT += 1; record["faithfulness_verdict"] = "UNVERIFIABLE_REJECT"

        details[gene] = record

    precision   = TP / (TP + FP)         if (TP + FP) > 0       else None
    recall      = TP / (TP + FN)         if (TP + FN) > 0       else None
    f1          = (2*precision*recall / (precision+recall)
                   if precision and recall else None)
    specificity = TN / (TN + FP)         if (TN + FP) > 0       else None

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "UNKNOWN_KEEP": UNKNOWN_KEEP, "UNKNOWN_REJECT": UNKNOWN_REJECT,
        "precision":   round(precision,   4) if precision   is not None else None,
        "recall":      round(recall,      4) if recall      is not None else None,
        "f1":          round(f1,          4) if f1          is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
    }, details


def check_selected_quality(selected, validated_set, non_brca_set):
    val  = [g for g in selected if g in validated_set]
    bad  = [g for g in selected if g in non_brca_set]
    unk  = [g for g in selected if g not in validated_set and g not in non_brca_set]
    return {
        "total_selected": len(selected),
        "validated_brca": len(val),
        "known_non_brca": len(bad),
        "unverifiable":   len(unk),
        "pct_validated":  round(len(val)/len(selected), 4) if selected else None,
        "validated_genes":    val,
        "non_brca_genes":     bad,
        "unverifiable_genes": unk,
    }


def check_missed(input_genes, selected_genes, validated_set):
    input_val   = [g for g in input_genes if g in validated_set]
    kept        = [g for g in input_val   if g in set(selected_genes)]
    missed      = [g for g in input_val   if g not in set(selected_genes)]
    return {
        "validated_in_input":        input_val,
        "n_validated_in_input":      len(input_val),
        "correctly_kept_by_llm":     kept,
        "missed_by_llm":             missed,
        "recall_of_validated_input": round(len(kept)/len(input_val), 4) if input_val else None,
    }


def main():
    with open(LLM_JSON) as f:
        llm_data = json.load(f)
    with open(SAL_JSON) as f:
        sal_data = json.load(f)

    all_input_genes    = sal_data["top_genes"]
    selected_genes     = llm_data.get("mamba_selected_genes",
                         llm_data.get("selected_genes", []))
    per_gene_reasoning = llm_data.get("per_gene_reasoning", {})

    print("LLM Reasoning Faithfulness Analysis")
    print(f"Input genes      : {len(all_input_genes)}")
    print(f"LLM selected     : {len(selected_genes)}")
    print(f"Reasoning entries: {len(per_gene_reasoning)}")
    print(f"BRCA ground-truth: {len(ALL_VALIDATED)} genes")
    print("="*60)

    if per_gene_reasoning:
        metrics, details = evaluate_llm_decisions(per_gene_reasoning, all_input_genes)
        print("\n1. Keep/Reject Decision Faithfulness:")
        print(f"   TP (correctly kept validated BRCA genes) : {metrics['TP']}")
        print(f"   FP (incorrectly kept non-BRCA genes)     : {metrics['FP']}")
        print(f"   TN (correctly rejected non-BRCA genes)   : {metrics['TN']}")
        print(f"   FN (wrongly rejected known BRCA genes)   : {metrics['FN']}")
        print(f"   Unverifiable keeps   : {metrics['UNKNOWN_KEEP']}")
        print(f"   Unverifiable rejects : {metrics['UNKNOWN_REJECT']}")
        print(f"   Precision  : {metrics['precision']}")
        print(f"   Recall     : {metrics['recall']}")
        print(f"   F1         : {metrics['f1']}")
        print(f"   Specificity: {metrics['specificity']}")
    else:
        metrics, details = None, {}
        print("\n1. Keep/Reject Decision Faithfulness:")
        print("   No structured per-gene keep/reject lines were parsed from LLM output.")
        print("   Decision-level precision/recall metrics are unavailable for this run.")

    sq = check_selected_quality(selected_genes, ALL_VALIDATED, KNOWN_NON_BRCA)
    print(f"\n2. Final Selection Quality:")
    print(f"   {sq['validated_brca']}/{sq['total_selected']} ({sq['pct_validated']:.1%}) are validated BRCA genes")
    print(f"   {sq['known_non_brca']} known non-BRCA genes kept")
    print(f"   {sq['unverifiable']} genes with no ground-truth label")
    print(f"   Validated : {sq['validated_genes']}")
    if sq['non_brca_genes']:
        print(f"   ⚠ Non-BRCA kept: {sq['non_brca_genes']}")

    ma = check_missed(all_input_genes, selected_genes, ALL_VALIDATED)
    print(f"\n3. Missed BRCA Genes:")
    print(f"   Known BRCA genes in top-50 input: {ma['n_validated_in_input']}")
    print(f"   Correctly kept : {ma['correctly_kept_by_llm']}")
    print(f"   Missed         : {ma['missed_by_llm']}")
    print(f"   Recall of validated input genes: {ma['recall_of_validated_input']}")

    # Spot-check pathway accuracy for key genes
    pathway_checks = {}
    known_keywords = {
        "MLPH":   ["luminal", "melanophilin", "ER", "subtype"],
        "ZEB1":   ["EMT", "epithelial", "mesenchymal", "TNBC", "ZEB"],
        "INPP4B": ["PI3K", "AKT", "phosphatase", "lipid"],
        "XBP1":   ["ER", "unfolded", "endoplasmic", "luminal", "UPR"],
        "FOXA1":  ["ER", "pioneer", "luminal", "transcription"],
        "NRIP1":  ["ER", "nuclear receptor", "RIP140", "luminal"],
        "RHOB":   ["Rho", "RhoB", "GTPase", "tumour suppressor", "tumor suppressor"],
    }
    print("\n4. Pathway Accuracy Spot-check:")
    for gene, keywords in known_keywords.items():
        if gene in per_gene_reasoning:
            reason = per_gene_reasoning[gene].get("reason", "").lower()
            matched = [kw for kw in keywords if kw.lower() in reason]
            accurate = len(matched) > 0
            pathway_checks[gene] = {
                "llm_reason": per_gene_reasoning[gene].get("reason", ""),
                "expected_keywords": keywords,
                "keywords_found": matched,
                "pathway_accurate": accurate,
            }
            status = "✓" if accurate else "✗"
            print(f"   {status} {gene}: matched {matched}")
            if not accurate:
                print(f"       LLM said: '{per_gene_reasoning[gene].get('reason','')[:100]}'")
        else:
            print(f"   — {gene}: not evaluated by LLM (per_gene_reasoning missing)")

    if metrics is not None:
        summary = (
            f"The LLM correctly identified {metrics['TP']} of "
            f"{metrics['TP']+metrics['FN']} validated BRCA genes "
            f"(recall={metrics['recall']}). "
            f"{sq['pct_validated']:.1%} of finally selected genes are validated BRCA genes. "
            f"{len(ma['missed_by_llm'])} known BRCA genes in the input were missed."
        )
    else:
        summary = (
            f"Selection-level audit: {sq['validated_brca']}/{sq['total_selected']} "
            f"({sq['pct_validated']:.1%}) selected genes are validated BRCA genes, "
            f"{sq['known_non_brca']} known non-BRCA genes were kept, and "
            f"{len(ma['missed_by_llm'])} known BRCA genes in the top-50 input were missed "
            f"(recall on validated input = {ma['recall_of_validated_input']})."
        )

    output = {
        "model": llm_data.get("model", "unknown"),
        "n_input_genes": len(all_input_genes),
        "n_selected_genes": len(selected_genes),
        "n_per_gene_reasoning_captured": len(per_gene_reasoning),
        "ground_truth_set_size": len(ALL_VALIDATED),
        "decision_faithfulness_metrics": metrics,
        "decision_details_per_gene": details,
        "selected_gene_quality": sq,
        "missed_brca_gene_analysis": ma,
        "pathway_accuracy_spot_check": pathway_checks,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
