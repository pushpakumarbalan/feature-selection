"""
Step 4: Generate all paper figures.
  Fig 1: Pipeline flowchart (system diagram)
  Fig 2: Bar chart - accuracy/F1/AUC across B1/B2/B3
  Fig 3: Gene saliency heatmap (top-20)
  Fig 4: Chain-of-thought excerpt from DeepSeek-R1
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

SALIENCY_JSON  = "/data4t/projects/fs/data_processed/top_genes_saliency.json"
LLM_JSON       = "/data4t/projects/fs/data_processed/llm_gene_reasoning.json"
RESULTS_JSON   = "/data4t/projects/fs/results/comparison_results.json"
FIG_DIR        = "/data4t/projects/fs/results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {"B1": "#6c757d", "B2": "#2196F3", "B3": "#4CAF50"}


# ─── Figure 1: Pipeline Flowchart ────────────────────────────────────────────
def fig_pipeline():
    """Two-branch pipeline: shows LLM as a CAUSAL re-ranking step (B2 vs B3)."""
    BW, BH = 2.3, 1.2   # box width, height
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor("#f0f4f8")
    fig.patch.set_facecolor("#f0f4f8")

    def box(cx, cy, label, color, fs=8.5, bold=True, ec="#333", lw=1.5):
        """Draw a rounded box centred at (cx, cy)."""
        p = mpatches.FancyBboxPatch(
            (cx - BW/2, cy - BH/2), BW, BH,
            boxstyle="round,pad=0.12", linewidth=lw,
            edgecolor=ec, facecolor=color, zorder=3
        )
        ax.add_patch(p)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                zorder=4, multialignment="center")

    def arrow(x0, y0, x1, y1, color="#333", lw=1.5, label="", label_side="bottom"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color))
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dy = -0.28 if label_side == "bottom" else 0.28
            dx = -0.35 if label_side == "left" else 0
            ax.text(mx + dx, my + dy, label, ha="center", va="center",
                    fontsize=7, color="#555", style="italic")

    # ── Row 1: main pipeline ──────────────────────────────────────
    # centres at y=7.0
    Y_MAIN = 7.0
    main_boxes = [
        (1.4,  "TCGA-BRCA\nRNA-seq\n(~1100 samples,\n20 k genes)",    "#90CAF9"),
        (4.1,  "Variance\nFilter\n(top 5 000\ngenes)",                  "#B0BEC5"),
        (6.8,  "Mamba SSM\nClassifier\n(initial\ntraining)",            "#7986CB"),
        (9.5,  "Gradient\nSaliency\nExtraction\n(top-50 genes)",        "#4DB6AC"),
    ]
    for cx, lbl, col in main_boxes:
        box(cx, Y_MAIN, lbl, col)

    # arrows along main row
    gaps = [
        (main_boxes[0][0] + BW/2, main_boxes[1][0] - BW/2, ""),
        (main_boxes[1][0] + BW/2, main_boxes[2][0] - BW/2, ""),
        (main_boxes[2][0] + BW/2, main_boxes[3][0] - BW/2, "saliency"),
    ]
    for x0, x1, lbl in gaps:
        arrow(x0, Y_MAIN, x1, Y_MAIN, label=lbl)

    # ── Fork from saliency box ────────────────────────────────────
    X_FORK = main_boxes[3][0] + BW/2    # right edge of saliency box ≈ 10.65
    Y_B2   = 4.8
    Y_B3   = 2.2

    # vertical splitter line
    ax.plot([X_FORK + 0.5, X_FORK + 0.5], [Y_B3, Y_MAIN],
            color="#777", lw=1.4, ls="--", zorder=2)
    # dots at split
    ax.plot(X_FORK + 0.5, Y_B2, "o", color="#777", ms=5, zorder=3)
    ax.plot(X_FORK + 0.5, Y_B3, "o", color="#777", ms=5, zorder=3)
    # horizontal arrows from splitter to B2 and B3 first boxes
    arrow(X_FORK, Y_MAIN, X_FORK + 0.5, Y_MAIN)   # connect saliency box to splitter
    arrow(X_FORK + 0.5, Y_B2, X_FORK + 1.15, Y_B2,
          color="#2196F3", label="50 genes", label_side="top")
    arrow(X_FORK + 0.5, Y_B3, X_FORK + 1.15, Y_B3,
          color="#E53935", label="50 genes", label_side="bottom")

    # ── B2 path (upper branch) ────────────────────────────────────
    X_B2_1 = X_FORK + 1.15 + BW/2   # ≈ 12.95
    box(X_B2_1, Y_B2,
        "B2: Mamba\nRetrain\n(50 Mamba-saliency\ngenes)",
        "#BBDEFB", ec="#2196F3", lw=2)

    # ── B3 path (lower branch) ────────────────────────────────────
    X_LLM = X_FORK + 1.15 + BW/2    # LLM box same x as B2 box
    box(X_LLM, Y_B3,
        "DeepSeek-R1\nReasoning\nkeep / reject each gene\n→ selects 20 BRCA-specific",
        "#FFCDD2", ec="#E53935", lw=2)

    # causal annotation badge
    ax.text(X_LLM + BW/2 + 0.15, Y_B3 + 0.25,
            "⚡ CAUSAL\nSTEP", ha="left", va="center",
            fontsize=7.5, fontweight="bold", color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFEBEE", ec="#E53935", lw=1))

    # arrow: LLM → B3 retrain
    X_B3_TRAIN = X_LLM + BW/2 + 1.1 + BW/2    # ≈ 15.55
    arrow(X_LLM + BW/2, Y_B3, X_B3_TRAIN - BW/2, Y_B3,
          color="#E53935", label="20 LLM-selected", label_side="top")
    box(X_B3_TRAIN, Y_B3,
        "B3: Mamba\nRetrain\n(20 LLM-reasoned\nBRCA-specific genes)",
        "#FFCCBC", ec="#E53935", lw=2)

    # ── Comparison output box ─────────────────────────────────────
    X_CMP = X_B3_TRAIN
    Y_CMP = 0.75
    box(X_CMP, Y_CMP,
        "Performance\nComparison\nB1 / B2 / B3",
        "#C8E6C9", ec="#388E3C", lw=2, fs=8)
    # arrows from B2 and B3 to comparison
    arrow(X_B2_1, Y_B2 - BH/2, X_CMP, Y_CMP + BH/2 + 0.1,
          color="#2196F3", lw=1.3)
    arrow(X_B3_TRAIN, Y_B3 - BH/2, X_CMP, Y_CMP + BH/2,
          color="#E53935", lw=1.3)

    # B1 baseline note
    ax.text(X_CMP, Y_CMP - BH/2 - 0.18,
            "+ B1 Baseline (top-5000 variance genes)",
            ha="center", va="top", fontsize=7.5, color="#555", style="italic")

    # ── Section labels ────────────────────────────────────────────
    ax.text(0.15, Y_MAIN + BH/2 + 0.22, "① Data & Feature Extraction",
            fontsize=8, fontweight="bold", color="#1A237E", va="bottom")
    ax.text(X_FORK + 1.0, Y_B2 + BH/2 + 0.25, "② B2 — Mamba Only",
            fontsize=8, fontweight="bold", color="#1565C0", va="bottom")
    ax.text(X_FORK + 1.0, Y_B3 + BH/2 + 0.25,
            "③ B3 — Mamba + LLM Reasoning (causal)",
            fontsize=8, fontweight="bold", color="#B71C1C", va="bottom")

    ax.set_title(
        "Mamba-SSM + LLM Reasoning Pipeline for BRCA Gene Biomarker Discovery\n"
        "LLM reasoning causally determines B3 gene set — not post-hoc explanation",
        fontsize=12, fontweight="bold", pad=10
    )
    plt.tight_layout(pad=1.0)
    path = f"{FIG_DIR}/fig1_pipeline.png"
    plt.savefig(path, bbox_inches="tight", dpi=180)
    print(f"Saved Fig 1 → {path}")
    plt.close()


# ─── Figure 2: Comparison Bar Chart ──────────────────────────────────────────
def fig_comparison():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    labels   = [r["label"].split(":")[0] for r in results]
    full_lbl = [r["label"].split(":", 1)[1].strip() for r in results]
    acc  = [r["accuracy"] for r in results]
    f1   = [r["f1"]       for r in results]
    auc  = [r["auc"]      for r in results]

    x = np.arange(len(labels))
    w = 0.25
    colors = [COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w, acc, w, label="Accuracy", color=[c+"cc" for c in colors], edgecolor="white")
    b2 = ax.bar(x,     f1,  w, label="F1 (weighted)", color=colors, edgecolor="white")
    b3 = ax.bar(x + w, auc, w, label="AUC-ROC", color=[c+"88" for c in colors], edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Performance Comparison Across Gene Selection Strategies", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({fl})" for l, fl in zip(labels, full_lbl)],
                       fontsize=8.5)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    plt.tight_layout()
    path = f"{FIG_DIR}/fig2_comparison.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved Fig 2 → {path}")
    plt.close()


# ─── Figure 3: Gene Saliency Heatmap (top-50, LLM selection highlighted) ─────
def fig_saliency():
    with open(SALIENCY_JSON) as f:
        data = json.load(f)
    with open(LLM_JSON) as f:
        llm_data = json.load(f)

    # Show all top-50 Mamba genes; green = kept by LLM, blue = rejected by LLM
    all_genes = data["top_genes"][:50]
    scores    = [data["scores"][g] for g in all_genes]
    llm_kept  = set(llm_data.get("mamba_selected_genes", []))

    fig_height = max(8, len(all_genes) * 0.28)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    bar_colors = ["#4CAF50" if g in llm_kept else "#90CAF9" for g in all_genes]
    ax.barh(range(len(all_genes)), scores, color=bar_colors, edgecolor="white", height=0.75)

    ax.set_yticks(range(len(all_genes)))
    ax.set_yticklabels(all_genes, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Gradient (Saliency Score)", fontsize=10)

    legend_patches = [
        mpatches.Patch(color="#4CAF50",
                       label="LLM-reasoned & selected (B3 gene set — causal)"),
        mpatches.Patch(color="#2196F3",
                       label="Mamba-saliency only — rejected by LLM reasoning"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    ax.set_title("Top-50 Mamba Gradient Saliency Genes\n"
                 "(green = LLM-selected for B3; blue = Mamba-only, LLM-rejected)",
                 fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    plt.tight_layout()
    path = f"{FIG_DIR}/fig3_saliency.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved Fig 3 → {path}")
    plt.close()


# ─── Figure 4: LLM Reasoning Decision Table (B&W, print-ready) ───────────────
def fig_cot():
    """Black-and-white per-gene decision table. Print-ready for Overleaf."""
    import re as _re
    with open(LLM_JSON) as f:
        data = json.load(f)

    full_resp   = data.get("full_response", "")
    kept_set    = set(data.get("mamba_selected_genes", []))
    input_genes = data.get("input_genes", [])

    # Parse numbered list "N. **GENE**: reason" from full response
    rows = []
    pattern = _re.compile(
        r'\d+\.\s+\*{0,2}([A-Z0-9._-]+)\*{0,2}:?\s*(.+?)(?=\n\d+\.|\nSELECTED|\Z)',
        _re.DOTALL
    )
    for m in pattern.finditer(full_resp):
        gene   = m.group(1).strip()
        reason = m.group(2).strip().replace('\n', ' ')
        reason = _re.sub(r'\s+', ' ', reason)
        if gene in set(input_genes):
            decision = "Keep" if gene in kept_set else "Reject"
            rows.append((gene, decision, reason[:95] + ('...' if len(reason) > 95 else '')))

    # Fallback: build rows from input_genes if parser found nothing
    if not rows:
        for g in input_genes:
            decision = "Keep" if g in kept_set else "Reject"
            rows.append((g, decision, "(see full_response in llm_gene_reasoning.json)"))

    n_rows  = len(rows)
    row_h   = 0.38
    hdr_h   = 1.5
    fig_h   = hdr_h + n_rows * row_h + 0.5

    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, fig_h)

    # ── Title ──
    ax.text(6.5, fig_h - 0.35,
            f"DeepSeek-R1 Per-Gene Biological Reasoning  (model: {data['model']})",
            ha="center", va="top", fontsize=12, fontweight="bold", color="black")
    ax.text(6.5, fig_h - 0.78,
            "Mamba selected top-50 genes by gradient saliency. "
            "LLM evaluated each gene for BRCA specificity — "
            "decisions causally determine the B3 training set.",
            ha="center", va="top", fontsize=9.5, color="#222", style="italic")

    # ── Column headers ──
    COL  = [0.25, 2.55, 4.35, 13.0]   # left-edge x positions
    CTRX = [1.38, 3.45, 8.65]          # centre x per column
    HDR  = ["Gene", "Decision", "Biological Justification / Pathway"]
    hdr_y = fig_h - 1.22
    for hdr, cx in zip(HDR, CTRX):
        ax.text(cx, hdr_y, hdr, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="black")
    ax.axhline(hdr_y - 0.20, xmin=0.015, xmax=0.985, color="black", lw=1.1)

    # ── Rows ──
    for r_idx, (gene, decision, reason) in enumerate(rows):
        y = hdr_y - 0.24 - r_idx * row_h
        if r_idx % 2 == 0:
            bg = mpatches.FancyBboxPatch(
                (0.15, y - row_h * 0.50), 12.7, row_h * 0.96,
                boxstyle="square,pad=0", facecolor="#f4f4f4",
                edgecolor="none", zorder=1
            )
            ax.add_patch(bg)

        # Gene name
        ax.text(COL[0] + 0.1, y, gene, ha="left", va="center",
                fontsize=9.5, fontweight="bold", color="black", zorder=2)

        # Decision symbol — bold, slightly larger
        sym  = "✓ Keep" if decision == "Keep" else "✗ Reject"
        ax.text(CTRX[1], y, sym, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="black", zorder=2)

        # Reason
        ax.text(COL[2] + 0.1, y, reason, ha="left", va="center",
                fontsize=8.5, color="#111", zorder=2)

    # ── Bottom border ──
    bot = hdr_y - 0.24 - n_rows * row_h + row_h * 0.46
    ax.axhline(bot, xmin=0.015, xmax=0.985, color="black", lw=0.8)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.15, bot), 12.7, (hdr_y - bot + 0.24),
        boxstyle="square,pad=0", facecolor="none",
        edgecolor="black", lw=0.9, zorder=5
    ))

    plt.tight_layout(pad=0.4)
    path = f"{FIG_DIR}/fig4_cot.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    print(f"Saved Fig 4 → {path}")
    plt.close()


# ─── Figure 5: Full Architectural Diagram ─────────────────────────────────────
def fig_architecture():
    """Full system architecture: data ingestion, Mamba internals,
    LLM reasoning, and three-way comparison. Print-ready (white bg)."""
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("white")

    def rbox(cx, cy, w, h, label, fc, ec="#333", lw=1.4,
             fs=9, bold=True, fc_text="black", sublabel=None, sub_fs=7.8):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx-w/2, cy-h/2), w, h,
            boxstyle="round,pad=0.1", linewidth=lw,
            edgecolor=ec, facecolor=fc, zorder=3
        ))
        ty = cy + (0.15 if sublabel else 0)
        ax.text(cx, ty, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=fc_text, zorder=4, multialignment="center")
        if sublabel:
            ax.text(cx, cy-0.22, sublabel, ha="center", va="center",
                    fontsize=sub_fs, color="#444", zorder=4,
                    multialignment="center", style="italic")

    def arr(x0, y0, x1, y1, color="#333", lw=1.5,
            label="", lbl_dx=0, lbl_dy=-0.25):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color))
        if label:
            ax.text((x0+x1)/2+lbl_dx, (y0+y1)/2+lbl_dy, label,
                    ha="center", va="center",
                    fontsize=7.5, color="#444", style="italic")

    def sec(x, y, text, color):
        ax.text(x, y, text, ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.2))

    # ══ A: Data & Feature Extraction  y=12.0 ══════════════════════════════
    Y_A = 12.0
    sec(0.2, 13.3, "①  Data & Feature Extraction", "#1A237E")
    A = [(1.7, "TCGA-BRCA\nRNA-seq", "1,097 samples\n20,531 genes", "#BBDEFB", "#1565C0"),
         (5.0, "Log₂(x+1)\nNorm.", "variance\nstabilisation",    "#E3F2FD", "#1565C0"),
         (8.3, "Variance\nFilter",  "top 5,000\ngenes",           "#CFD8DC", "#37474F"),
         (11.6,"Train/Test\nSplit", "80%/20%\nstratified",        "#E0E0E0", "#424242")]
    for cx, lbl, sub, fc, ec in A:
        rbox(cx, Y_A, 2.6, 1.1, lbl, fc, ec=ec, sublabel=sub, fs=9.5)
    for i in range(len(A)-1):
        arr(A[i][0]+1.3, Y_A, A[i+1][0]-1.3, Y_A)

    # ══ B: Mamba SSM Architecture  y=9.2 ═════════════════════════════════
    Y_B = 9.2
    sec(0.2, 10.6, "②  Mamba-SSM Classifier Architecture", "#4A148C")
    arr(11.6, Y_A-0.55, 1.5, Y_B+0.55, color="#555",
        label="5k gene\nexpression\nvector", lbl_dx=-0.7)
    B = [(1.5,  "Input\nVector",        None,                "#E8EAF6","#3949AB"),
         (3.9,  "Linear\nEmbedding",    "d_model=128",       "#D1C4E9","#512DA8"),
         (6.4,  "Mamba\nSSM Block",     "d_state=16\nd_conv=4\nexpand=2","#B39DDB","#4527A0"),
         (9.0,  "Adaptive\nAvg Pool",   "seq → 1 vec",       "#CE93D8","#6A1B9A"),
         (11.3, "FC + Sigmoid",         "binary output",     "#F3E5F5","#6A1B9A"),
         (13.7, "BCELoss\n+class wt",   "AdamW lr=1e-4",     "#EDE7F6","#4527A0")]
    for cx, lbl, sub, fc, ec in B:
        rbox(cx, Y_B, 2.0, 0.9, lbl, fc, ec=ec, sublabel=sub, fs=8.5)
    for i in range(len(B)-1):
        arr(B[i][0]+1.0, Y_B, B[i+1][0]-1.0, Y_B)
    ax.annotate("", xy=(3.9, Y_B-0.45), xytext=(13.7, Y_B-0.45),
                arrowprops=dict(arrowstyle="<-", lw=1.1, color="#7E57C2"))
    ax.text(8.8, Y_B-0.72, "back-propagation (15 epochs)",
            ha="center", fontsize=7.5, color="#7E57C2", style="italic")

    # ══ C: Gradient Saliency  y=6.8 ═══════════════════════════════════════
    Y_C = 6.8
    sec(0.2, 7.9, "③  Gradient Saliency Extraction", "#006064")
    arr(1.5, Y_B-0.45, 2.5, Y_C+0.45, color="#00897B",
        label="trained\nmodel", lbl_dx=-0.7)
    C = [(2.5, "|∂L/∂x_i|",     "per-gene\ngradient",  "#B2DFDB","#00695C"),
         (5.3, "Mean Abs.\nGrad","across\nsamples",     "#80CBC4","#00796B"),
         (8.1, "Rank &\nTop-50", "B2 gene set",         "#4DB6AC","#00838F")]
    for cx, lbl, sub, fc, ec in C:
        rbox(cx, Y_C, 2.3, 0.9, lbl, fc, ec=ec, sublabel=sub, fs=9)
    for i in range(len(C)-1):
        arr(C[i][0]+1.15, Y_C, C[i+1][0]-1.15, Y_C)

    # ══ D: LLM Reasoning (causal)  y=4.5 ═════════════════════════════════
    Y_D = 4.5
    sec(0.2, 5.7, "④  DeepSeek-R1 Biological Reasoning  [⚡ CAUSAL]", "#B71C1C")
    arr(8.1+1.15, Y_C-0.45, 11.2, Y_D+0.55, color="#c62828",
        label="top-50 gene\nnames+scores", lbl_dx=1.0)
    D = [(11.2, "DeepSeek-R1\n7B (Ollama)", "temp=0.3",              "#FFCDD2","#C62828"),
         (14.6, "Keep / Reject\nper gene",  "rejects: housekeeping,\nunannotated, off-target","#FFCCBC","#BF360C"),
         (18.2, "B3 Gene Set\n(17 genes)",  "BRCA-specific\nsubset", "#FFAB91","#BF360C")]
    for cx, lbl, sub, fc, ec in D:
        rbox(cx, Y_D, 2.8, 1.0, lbl, fc, ec=ec, sublabel=sub, fs=9)
    for i in range(len(D)-1):
        arr(D[i][0]+1.4, Y_D, D[i+1][0]-1.4, Y_D, color="#c62828")
    ax.text(14.6, Y_D-0.82,
            "Rejected (6): MB, AL035661.1, HLA-DRB1, NFE4, CMBL, DNAH5",
            ha="center", fontsize=7.5, color="#8D1A0A",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFF3E0", ec="#BF360C", lw=0.8))

    # ══ E: Three-way Evaluation  y=2.2 ════════════════════════════════════
    Y_E = 2.2
    sec(0.2, 3.3, "⑤  Three-way Evaluation", "#1B5E20")
    E = [(3.5,  "B1 Baseline",        "Top-5000 variance\n(no Mamba, no LLM)", "#ECEFF1","#546E7A"),
         (9.0,  "B2: Mamba-Selected", "Top-50 saliency\n(Mamba only)",         "#BBDEFB","#1565C0"),
         (14.5, "B3: LLM-Reasoned",   "17 BRCA-specific\n(Mamba + LLM)",       "#FFE0B2","#E65100")]
    for cx, lbl, sub, fc, ec in E:
        rbox(cx, Y_E, 3.8, 1.1, lbl, fc, ec=ec, sublabel=sub, fs=10, sub_fs=8.5)
    arr(8.3, Y_A-0.55, 3.5, Y_E+0.55, color="#546E7A")  # B1 from variance
    arr(8.1+1.15, Y_C-0.45, 9.0, Y_E+0.55, color="#1565C0")  # B2 from saliency
    arr(18.2, Y_D-0.5, 14.5, Y_E+0.55, color="#E65100")       # B3 from LLM

    rbox(9.0, 0.72, 6.5, 0.82,
         "Accuracy  |  F1 (weighted)  |  AUC-ROC",
         "#C8E6C9", ec="#2E7D32", lw=1.8, fs=10.5)
    for cx in [3.5, 9.0, 14.5]:
        arr(cx, Y_E-0.55, 9.0, 0.72+0.41, color="#2E7D32", lw=1.2)

    ax.text(10, 13.7,
            "System Architecture: Mamba-SSM + LLM Reasoning for BRCA Biomarker Discovery",
            ha="center", va="center", fontsize=14, fontweight="bold", color="black")

    path = f"{FIG_DIR}/fig5_architecture.png"
    plt.savefig(path, bbox_inches="tight", dpi=180)
    print(f"Saved Fig 5 → {path}")
    plt.close()


if __name__ == "__main__":
    fig_pipeline()
    fig_comparison()
    fig_saliency()
    fig_cot()
    fig_architecture()
    print("\nAll figures saved to", FIG_DIR)
