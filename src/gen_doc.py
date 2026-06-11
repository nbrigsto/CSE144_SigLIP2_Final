"""Generate architecture + model-comparison figures and a compiled PDF document."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

OUT = "docs"
os.makedirs(OUT, exist_ok=True)

# ---- palette ----
FAM = {
    "SigLIP":   "#2563eb",
    "FineTune": "#7c3aed",
    "DINOv2":   "#059669",
    "EVA":      "#d97706",
    "CLIP":     "#dc2626",
    "Baseline": "#6b7280",
}

BACKBONES = [
    ("SoViT-400m SigLIP2 @378 (p14)  ★ champion", 0.9462, "SigLIP"),
    ("SoViT-400m SigLIP2 @512 (p16)", 0.9416, "SigLIP"),
    ("SigLIP2-L @512", 0.9379, "SigLIP"),
    ("SigLIP-L @384", 0.9323, "SigLIP"),
    ("SigLIP-L @384  (end-to-end fine-tuned)", 0.9306, "FineTune"),
    ("DINOv2-L @336", 0.8906, "DINOv2"),
    ("EVA-02-L @448", 0.8452, "EVA"),
    ("CLIP-L @336", 0.8406, "CLIP"),
    ("ConvNeXt-Tiny @224  (fine-tuned baseline)", 0.76, "Baseline"),
]

JOURNEY = [
    ("ConvNeXt-Tiny\nfine-tune", 0.76),
    ("DINOv2-L probe\n@224", 0.862),
    ("SigLIP-L probe\n@384", 0.922),
    ("SoViT SigLIP2\n@378", 0.931),
    ("+ multi-view TTA\n(champion)", 0.946),
]


def box(ax, x, y, w, h, text, fc, ec="#1f2937", fs=9, tc="white", lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#374151"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=color, zorder=1))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(6, 8.6, "Inference Pipeline — Frozen Foundation-Feature Ensemble",
            ha="center", fontsize=15, fontweight="bold", color="#111827")
    ax.text(6, 8.15, "100-class fine-grained classification (food / flowers / cars), ~10 images per class",
            ha="center", fontsize=10, color="#6b7280")

    # Row 1: per-image flow
    box(ax, 0.3, 6.6, 1.9, 1.0, "Input image\n(variable size)", "#374151", fs=9)
    arrow(ax, 2.2, 7.1, 2.7, 7.1)
    box(ax, 2.7, 6.3, 2.4, 1.6,
        "Test-Time Augmentation\n• center-crop  • full-frame squash\n× horizontal flip\n= up to 4 views",
        "#0ea5e9", fs=8.5)
    arrow(ax, 5.1, 7.1, 5.6, 7.1)
    box(ax, 5.6, 6.3, 2.7, 1.6,
        "FROZEN backbone\n(e.g. SoViT-400m SigLIP2 @378)\nno gradients → fits 8 GB\npooled embedding / view",
        "#2563eb", fs=8.5)
    arrow(ax, 8.3, 7.1, 8.8, 7.1)
    box(ax, 8.8, 6.3, 2.9, 1.6,
        "L2-normalize each view\n→ average over views\n→ L2-normalize\n= one embedding / image",
        "#1d4ed8", fs=8.5)

    # down arrow
    arrow(ax, 10.25, 6.3, 10.25, 5.55)

    # Row 2: probe + ensemble (right to left)
    box(ax, 8.8, 4.5, 2.9, 1.0,
        "Logistic-Regression probe\n(class-balanced, C tuned by CV)", "#1d4ed8", fs=8.5)
    arrow(ax, 8.8, 5.0, 8.3, 5.0)
    box(ax, 5.3, 4.3, 3.0, 1.4,
        "Repeated 5×5-fold CV\ngreedy ensemble selection\n(Caruana) over backbones",
        "#7c3aed", fs=8.5)
    arrow(ax, 5.3, 5.0, 4.8, 5.0)
    box(ax, 2.1, 4.5, 2.7, 1.0, "Average softmax\nprobabilities", "#6d28d9", fs=9)
    arrow(ax, 2.1, 5.0, 1.6, 5.0)
    box(ax, 0.3, 4.5, 1.3, 1.0, "argmax", "#5b21b6", fs=9)
    arrow(ax, 0.95, 4.5, 0.95, 3.95)
    box(ax, 0.3, 3.35, 2.2, 0.6, "submission.csv", "#111827", fs=9)

    # Model pool panel
    ax.add_patch(FancyBboxPatch((3.1, 0.4), 8.6, 2.6, boxstyle="round,pad=0.05,rounding_size=0.05",
                                linewidth=1.2, edgecolor="#9ca3af", facecolor="#f9fafb", zorder=0))
    ax.text(7.4, 2.75, "Backbone pool evaluated (frozen feature extractors)", ha="center",
            fontsize=10.5, fontweight="bold", color="#111827")
    pool = [("SigLIP / SoViT-400m  (language-aligned)", "SigLIP"),
            ("DINOv2-L  (self-supervised)", "DINOv2"),
            ("EVA-02-L  (masked-image)", "EVA"),
            ("CLIP-L  (language-aligned)", "CLIP"),
            ("ConvNeXt / fine-tuned ViT  (supervised)", "Baseline")]
    for i, (txt, fam) in enumerate(pool):
        yy = 2.35 - i * 0.42
        ax.add_patch(plt.Rectangle((3.4, yy), 0.28, 0.28, color=FAM[fam], zorder=1))
        ax.text(3.85, yy + 0.14, txt, va="center", fontsize=9, color="#1f2937")
    ax.text(7.4, 0.2, "Greedy CV selection keeps only backbones that improve held-out accuracy "
            "(the strong SigLIP family dominates this task).",
            ha="center", fontsize=8, color="#6b7280", style="italic")

    # key callout
    ax.text(0.3, 2.7, "Why frozen?\n10 imgs/class →\nlinear probe on\nhuge-pretrained\nfeatures beats\nfine-tuning, and\nfits 8 GB.",
            fontsize=8.3, color="#374151", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fef3c7", ec="#f59e0b"))
    fig.tight_layout()
    fig.savefig(f"{OUT}/architecture_pipeline.png", dpi=160, bbox_inches="tight")
    return fig


def fig_models():
    data = sorted(BACKBONES, key=lambda t: t[1])
    names = [d[0] for d in data]; accs = [d[1] for d in data]; cols = [FAM[d[2]] for d in data]
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    bars = ax.barh(range(len(names)), accs, color=cols, edgecolor="#1f2937", linewidth=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0.55, 1.0)
    ax.axvline(0.60, color="#dc2626", ls="--", lw=1.5)
    ax.text(0.602, 0.2, "60% baseline", color="#dc2626", fontsize=9, rotation=90, va="bottom")
    for b, a in zip(bars, accs):
        ax.text(a + 0.003, b.get_y() + b.get_height() / 2, f"{a*100:.1f}%",
                va="center", fontsize=8.5, color="#111827")
    ax.set_xlabel("Repeated 5×5-fold cross-validation accuracy", fontsize=10)
    ax.set_title("Models evaluated — CV accuracy (frozen probe unless noted)",
                 fontsize=13, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM.values()]
    ax.legend(handles, FAM.keys(), title="Pretraining family", loc="lower right", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison.png", dpi=160, bbox_inches="tight")
    return fig


def fig_journey():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    xs = range(len(JOURNEY)); ys = [j[1] for j in JOURNEY]
    ax.plot(xs, ys, "-o", color="#2563eb", lw=2.5, ms=9, mfc="#1d4ed8", mec="white")
    for x, (lbl, y) in zip(xs, JOURNEY):
        ax.annotate(f"{y*100:.1f}%", (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, fontweight="bold", color="#111827")
    ax.set_xticks(list(xs)); ax.set_xticklabels([j[0] for j in JOURNEY], fontsize=9)
    ax.axhline(0.60, color="#dc2626", ls="--", lw=1.3); ax.text(0, 0.605, "60% baseline", color="#dc2626", fontsize=9)
    ax.set_ylim(0.55, 1.0); ax.set_ylabel("CV accuracy", fontsize=10)
    ax.set_title("Accuracy journey: from baseline to champion", fontsize=13, fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/accuracy_journey.png", dpi=160, bbox_inches="tight")
    return fig


def main():
    f1, f2, f3 = fig_architecture(), fig_models(), fig_journey()
    with PdfPages(f"{OUT}/CSE144_model_architecture.pdf") as pdf:
        pdf.savefig(f1); pdf.savefig(f2); pdf.savefig(f3)
    print("Wrote:")
    for f in ["architecture_pipeline.png", "model_comparison.png", "accuracy_journey.png",
              "CSE144_model_architecture.pdf"]:
        print(" ", os.path.join(OUT, f))


if __name__ == "__main__":
    main()
