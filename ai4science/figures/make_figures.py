"""Generate three ICML-style figures for the AI4Science workshop paper.

Outputs (PDF + PNG):
  F1_diagnostic_schematic.pdf    conceptual: the three counterfactual interventions
  F2_decoupling_scatter.pdf      headline:   IMG vs test accuracy, trained + zero-shot
  F3_support_score_distributions.pdf  mechanism: score distributions per config

Every number comes from the paper's Table 1 / Table 2 or from
/tmp/cg_state/support_scores_*.json.  No invented data.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = Path(__file__).parent
DATA = Path("/tmp/cg_state")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
})

# Restrained palette — muted tones, high contrast only where semantic
BLIND_RED = "#B85450"        # warm muted red
NONBLIND_BLUE = "#2E5C8A"    # deep muted blue
BASELINE_GREY = "#7A7A7A"    # medium grey
THRESH = "#BBBBBB"
IMG_ORANGE = "#C17A3A"       # muted terracotta
ESG_GREEN = "#3F7D5A"        # muted forest green
IPG_PURPLE = "#6B4E7D"       # muted plum
INK = "#2D3748"              # body text color
PAPER = "#F7F5F0"            # warm off-white for soft backgrounds

CONFIGS = [
    ("v5.0-base",     0.890, 0.909, 2.03,  19.71, True),
    ("v5.1-ground",   0.911, 0.921, 2.17,  21.13, True),
    ("v5.2-real",     0.892, 0.923, 62.90, 18.74, False),
    ("v5.3-contrast", 0.894, 0.925, 69.24, 18.56, False),
    ("v6.0-retrain",  0.954, 0.911, 70.21, 20.01, False),
]

BASELINES = [
    ("BiomedCLIP",         0.538,  8.0,  0.0),
    ("MedGemma-4B-IT",     0.758, 14.0,  2.0),
    ("MAIRA-2 (verifier)", 0.498, 32.2,  6.6),
]

SCORE_FILES = {
    "v5.0-base":     DATA / "support_scores_v5_0_base.json",
    "v5.1-ground":   DATA / "support_scores_v5_1_ground.json",
    "v5.2-real":     DATA / "support_scores_v5_2_real.json",
    "v5.3-contrast": DATA / "support_scores_v5_3_contrast.json",
    "v6.0-retrain":  DATA / "support_scores_v6_0_retrain.json",
}


# ---------------------------------------------------------------------------
# FIGURE 1 — Conceptual schematic of the three counterfactual interventions
# ---------------------------------------------------------------------------

def figure_schematic():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # --- Top row: the verifier
    f_box = FancyBboxPatch((3.5, 3.1), 3.0, 0.85,
                           boxstyle="round,pad=0.04,rounding_size=0.08",
                           linewidth=1.1, edgecolor=INK, facecolor="#EEF1F5")
    ax.add_patch(f_box)
    ax.text(5.0, 3.52, r"verifier $f\!\left(x_{\mathrm{img}},\,x_{\mathrm{cl}},\,x_{\mathrm{ev}}\right)$",
            ha="center", va="center", fontsize=10.5, color=INK)
    ax.text(5.0, 4.07, "aggregate test accuracy $= 90.9\\%$ (v5.0-base baseline)",
            ha="center", va="center", fontsize=8.2, color="#5A6578", style="italic")

    # Arrow down to three interventions
    ax.annotate("", xy=(5.0, 2.72), xytext=(5.0, 3.08),
                arrowprops=dict(arrowstyle="-|>", color="#6B7280", lw=0.9))

    # --- Three intervention blocks, consistent geometry
    interventions = [
        ("IMG", r"image $\to$ zero tensor",            2.03,  IMG_ORANGE),
        ("ESG", r"evidence $\to$ shuffled within batch", 19.71, ESG_GREEN),
        ("IPG", r"image $\to$ horizontal flip",        0.88,  IPG_PURPLE),
    ]
    xs = [1.7, 5.0, 8.3]
    for x, (name, desc, gap, color) in zip(xs, interventions):
        ib = FancyBboxPatch((x - 1.35, 1.55), 2.7, 0.80,
                            boxstyle="round,pad=0.04,rounding_size=0.08",
                            linewidth=1.1, edgecolor=color, facecolor="white")
        ax.add_patch(ib)
        ax.text(x, 2.12, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(x, 1.78, desc, ha="center", va="center", fontsize=8.3, color=INK)

        # down arrow to gap value
        ax.annotate("", xy=(x, 1.15), xytext=(x, 1.52),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9))
        gap_str = f"{gap:.2f}\\,pp"
        ax.text(x, 0.85, f"gap $= {gap_str}$", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

    # Single bottom-line annotation (no per-column "(example: v5.0)" repetition)
    ax.text(5.0, 0.22,
            "High aggregate accuracy does not imply the verifier uses the image.",
            ha="center", va="center", fontsize=8.5, style="italic", color="#3B4655")

    plt.savefig(OUT / "F1_diagnostic_schematic.pdf")
    plt.savefig(OUT / "F1_diagnostic_schematic.png")
    plt.close(fig)
    print(f"wrote {OUT/'F1_diagnostic_schematic.pdf'}")


# ---------------------------------------------------------------------------
# FIGURE 2 — Decoupling scatter: IMG vs Test Accuracy
# ---------------------------------------------------------------------------

def figure_decoupling():
    """Clean scatter with label-lane routing, zero text-on-text overlap."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    # Subtle horizontal guides
    ax.grid(axis="y", alpha=0.18, linewidth=0.5, zorder=0)

    # Evidence-blind shaded region
    ax.axhspan(-6, 5, facecolor="#F3E5E4", alpha=0.75, zorder=0)
    ax.axhline(5, color=BLIND_RED, linestyle="--", linewidth=0.7,
               alpha=0.5, zorder=1)

    # Zone label anchored to y-axis, never collides with data
    ax.text(0.005, 0.08, "evidence-blind zone (IMG $<5$pp)",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=8, color=BLIND_RED, style="italic")

    # --- Data ---
    for label, val_acc, test_acc, img, esg, blind in CONFIGS:
        color = BLIND_RED if blind else NONBLIND_BLUE
        ax.scatter(test_acc, img, s=140, color=color, marker="o",
                   edgecolor="white", linewidths=1.5, zorder=5)
    for label, acc, img, esg in BASELINES:
        ax.scatter(acc, img, s=115, color=BASELINE_GREY, marker="s",
                   edgecolor="white", linewidths=1.3, zorder=4)

    # --- Label placements: strict label lanes, no overlaps ---
    # Trained configs: non-blind cluster stacks LEFT of the points; blind cluster
    # labels go ABOVE (v5.0) and BELOW (v5.1) to separate the near-coincident points.
    trained_labels = {
        "v6.0-retrain":  (0.83,  72.0, "right"),  # stacked-left of upper cluster
        "v5.3-contrast": (0.83,  68.0, "right"),
        "v5.2-real":     (0.83,  63.0, "right"),
        "v5.0-base":     (0.86,  18.0, "right"),  # above the point, out of zone
        "v5.1-ground":   (0.93,  -4.0, "left"),   # below the shaded zone
    }
    anchors = {lbl: (t_acc, img) for (lbl, _, t_acc, img, _, _) in CONFIGS}
    for lbl, (lx, ly, ha) in trained_labels.items():
        t_acc, img = anchors[lbl]
        blind = [c for c in CONFIGS if c[0] == lbl][0][5]
        color = BLIND_RED if blind else NONBLIND_BLUE
        ax.annotate(lbl, xy=(t_acc, img), xytext=(lx, ly),
                    fontsize=9, color=color, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                                    alpha=0.6, shrinkA=0, shrinkB=6))

    # Baselines: labels placed on clean side, different horizontal lanes so they
    # never collide with v5.0-base.
    baseline_labels = {
        "BiomedCLIP":         (0.57, 8.0,   "left"),
        "MedGemma-4B-IT":     (0.64, 14.0,  "right"),
        "MAIRA-2 (verifier)": (0.515, 35.0, "left"),
    }
    for label, acc, img, esg in BASELINES:
        lx, ly, ha = baseline_labels[label]
        ax.annotate(label, xy=(acc, img), xytext=(lx, ly),
                    fontsize=9, color="#3A3A3A", ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.5,
                                    alpha=0.6, shrinkA=0, shrinkB=5))

    # --- Axes ---
    ax.set_xlabel("Test accuracy", fontsize=10)
    ax.set_ylabel("Image-masking gap (IMG), pp", fontsize=10)
    ax.set_xlim(0.45, 1.00)
    ax.set_ylim(-8, 84)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # --- Legend, top-left corner, clean ---
    legend_handles = [
        mpatches.Patch(color=NONBLIND_BLUE, label="trained, not evidence-blind"),
        mpatches.Patch(color=BLIND_RED,     label="trained, evidence-blind"),
        mpatches.Patch(color=BASELINE_GREY, label="zero-shot baseline"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
              bbox_to_anchor=(0.01, 0.99), fontsize=8.5, handlelength=1.2)

    # --- Subtitle (set as xaxis label extension, not a floating text) ---
    ax.text(0.5, -0.15,
            "Aggregate accuracy and image-grounding are independent axes.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#3B4655", style="italic")

    plt.savefig(OUT / "F2_decoupling_scatter.pdf")
    plt.savefig(OUT / "F2_decoupling_scatter.png")
    plt.close(fig)
    print(f"wrote {OUT/'F2_decoupling_scatter.pdf'}")


# ---------------------------------------------------------------------------
# FIGURE 3 — Support-score distributions per config, split by true label
# ---------------------------------------------------------------------------

def figure_distributions():
    # Load per-config support scores
    configs_order = ["v5.0-base", "v5.1-ground", "v5.2-real", "v5.3-contrast", "v6.0-retrain"]
    data = {}
    for name in configs_order:
        with open(SCORE_FILES[name]) as fh:
            d = json.load(fh)
        # Reconstruct p_SUP from scores_true_class: if label==0 (SUP), that IS p_SUP;
        # if label==1 (CON), p_SUP = 1 - scores_true_class.
        labels = np.asarray(d["labels"])
        s_true = np.asarray(d["scores_true_class"])
        p_sup = np.where(labels == 0, s_true, 1.0 - s_true)
        data[name] = (p_sup, labels)

    fig, axes = plt.subplots(1, 5, figsize=(10.2, 2.9), sharey=True)
    bins = np.linspace(0, 1, 31)

    sup_patch = None
    con_patch = None
    for ax, name in zip(axes, configs_order):
        p_sup, labels = data[name]
        mask_sup = labels == 0
        mask_con = labels == 1
        ax.hist(p_sup[mask_sup], bins=bins, color=NONBLIND_BLUE, alpha=0.78,
                edgecolor="none", density=True)
        ax.hist(p_sup[mask_con], bins=bins, color=BLIND_RED, alpha=0.72,
                edgecolor="none", density=True)
        if sup_patch is None:
            sup_patch = mpatches.Patch(color=NONBLIND_BLUE, alpha=0.78, label="true SUP")
            con_patch = mpatches.Patch(color=BLIND_RED, alpha=0.72, label="true CON")
        blind = CONFIGS[configs_order.index(name)][5]
        title_color = BLIND_RED if blind else NONBLIND_BLUE
        ax.set_title(name, fontsize=10, color=title_color, pad=4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_ylim(0, 28)
        if ax is axes[0]:
            ax.set_ylabel("density")
        ax.set_xlabel(r"$p(\mathrm{SUP}\mid x)$")
        ax.grid(axis="y", alpha=0.15, linewidth=0.4)

    # Legend above the whole figure, not inside the last panel.
    fig.legend(handles=[sup_patch, con_patch],
               loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=2, frameon=False, fontsize=9)

    plt.savefig(OUT / "F3_support_score_distributions.pdf")
    plt.savefig(OUT / "F3_support_score_distributions.png")
    plt.close(fig)
    print(f"wrote {OUT/'F3_support_score_distributions.pdf'}")


if __name__ == "__main__":
    # figure_schematic()  # F1 is a user-provided image (F1_user_provided.png);
                          # do not regenerate here or the user's version is overwritten.
    figure_decoupling()
    figure_distributions()
