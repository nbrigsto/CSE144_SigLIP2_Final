// Builds docs/CSE144_Final_Report.docx  (then convert to PDF with LibreOffice).
// Run:  NODE_PATH="C:\\nvm4w\\nodejs\\node_modules" node docs/report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, ExternalHyperlink, PageNumber, Header, Footer,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const img = (p) => fs.readFileSync(path.join(ROOT, p));

// ---- palette ----
const NAVY = "21356B", INK = "17223B", MUTED = "6B7793", HEAD = "D5E2F0";

// ---- helpers ----
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const P = (runs, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 276 },
  children: Array.isArray(runs) ? runs : [new TextRun(runs)],
  ...opts,
});
const T = (t, o = {}) => new TextRun({ text: t, ...o });
const bullet = (runs) => new Paragraph({
  numbering: { reference: "b", level: 0 }, spacing: { after: 60, line: 268 },
  children: Array.isArray(runs) ? runs : [new TextRun(runs)],
});
const numItem = (ref, runs) => new Paragraph({
  numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 268 },
  children: Array.isArray(runs) ? runs : [new TextRun(runs)],
});

function figure(file, wPx, caption) {
  const { width, height } = (() => {
    const sizes = {
      "docs/accuracy_journey.png": 0.483,
      "docs/model_comparison.png": 0.567,
      "docs/figs/probe_training_curve.png": 0.610,
      "docs/figs/probe_performance.png": 0.588,
    };
    return { width: wPx, height: Math.round(wPx * sizes[file]) };
  })();
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({
        type: "png", data: img(file), transformation: { width, height },
        altText: { title: caption, description: caption, name: caption },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: MUTED })],
    }),
  ];
}

// ---- tables ----
const border = { style: BorderStyle.SINGLE, size: 1, color: "C9D3E0" };
const borders = { top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border };
const cellM = { top: 60, bottom: 60, left: 110, right: 110 };

function table(headers, rows, colW) {
  const total = colW.reduce((a, b) => a + b, 0);
  const mkCell = (txt, w, opts = {}) => new TableCell({
    width: { size: w, type: WidthType.DXA }, margins: cellM,
    shading: opts.head ? { fill: HEAD, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: opts.right ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: [new TextRun({ text: txt, bold: !!opts.head || !!opts.bold,
        size: 18, color: INK })],
    })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: colW, borders,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) =>
        mkCell(h, colW[i], { head: true, right: i > 0 && i === headers.length - 1 })) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) =>
        mkCell(String(c), colW[i], { right: i === r.length - 1 && r.length > 1,
          bold: i === 0 && r._b })) })),
    ],
  });
}

// =====================================================================
const children = [];

// ---- Title block ----
children.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 40 },
    children: [new TextRun({ text: "CSE 144 Final Project Report", bold: true, size: 40, color: INK })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
    children: [new TextRun({ text: "Transfer Learning Challenge", size: 28, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: [new TextRun({ text: "Group Members (max 3):", bold: true, size: 22, color: INK })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 6 },
    children: [new TextRun({ text: "[Your Name]  (CruzID: [cruzid])", size: 22, color: INK })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
    children: [new TextRun({ text: "Spring 2026", size: 22, color: MUTED })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [
      new TextRun({ text: "Kaggle public leaderboard: ", size: 22, color: INK }),
      new TextRun({ text: "0.96363  (2nd place)", bold: true, size: 22, color: NAVY }),
    ] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 6 } },
    spacing: { after: 200 }, children: [] }),
);

// ---- 1 Introduction ----
children.push(H1("1  Introduction"));
children.push(P([
  T("Goal and setting. "),
  T("This project is the CSE 144 transfer-learning challenge: a 100-class fine-grained image " +
    "classification task with an extreme few-shot training budget of roughly 10–11 labeled images " +
    "per class (1,079 training images total) and 1,036 unlabeled test images. The official baseline " +
    "to beat is 60% test accuracy. Submissions are scored on classification accuracy on the Kaggle " +
    "public leaderboard."),
]));
children.push(P([
  T("Why transfer learning is appropriate. "),
  T("With only ~10 images per class, training a deep network from scratch — or even fully " +
    "fine-tuning a medium network — overfits badly before it learns useful features. The data, " +
    "however, comes from common visual domains (food, flowers, cars, aircraft) that are richly " +
    "represented in large-scale image–text pretraining. Transfer learning lets us reuse the " +
    "representations of a frozen foundation model and spend our tiny label budget only on a small, " +
    "well-regularized classifier on top, which generalizes far better in this regime."),
]));
children.push(P([
  T("Approach and main result. "),
  T("Our final system is a frozen SigLIP-2 SoViT-400m backbone (@378 px) that produces image " +
    "embeddings, a multinomial logistic-regression probe trained on those embeddings, multi-view " +
    "test-time augmentation, fusion with a zero-shot text signal from the same model’s text tower, " +
    "and a transductive Sinkhorn step that exploits the assumption of a class-balanced test set. " +
    "This reaches ≈96% cross-validation accuracy and a public-leaderboard score of "),
  T("0.96363 (2nd place)", { bold: true }),
  T(", versus 76% for an end-to-end fine-tuned ConvNeXt baseline and the 60% target."),
]));

// ---- 2 Dataset ----
children.push(H1("2  Dataset"));
children.push(P([
  T("Sizes and classes. "),
  T("100 classes; 1,079 training images (≈10–11 per class); 1,036 test images. Because the training " +
    "set is so small we do not hold out a fixed validation split for model selection — instead all " +
    "reported numbers use repeated 5×5-fold stratified cross-validation over the training set (see " +
    "§4), which uses the data far more efficiently than a single split."),
]));
children.push(P([
  T("Composition (an analysis finding). "),
  T("Inspecting class-mean embeddings against canonical label vocabularies revealed that the 100 " +
    "classes are a sample of four well-known fine-grained datasets, not three as the prompt text " +
    "suggested. The undocumented fourth quarter is aircraft:"),
]));
children.push(table(
  ["Label range", "Source dataset", "Domain"],
  [
    ["0 – 24", "Food-101", "Food"],
    ["25 – 49", "Oxford-102 Flowers", "Flowers"],
    ["50 – 74", "Stanford Cars", "Cars"],
    ["75 – 99", "FGVC-Aircraft", "Aircraft"],
  ], [2400, 4360, 2600]));
children.push(P(""));
children.push(P([
  T("Label mapping. "),
  T("Training images live in folders named by integer class. We map label = int(folder_name) so " +
    "folder “0” → 0 … “99” → 99 (src/dataset.py). This avoids the classic ImageFolder pitfall where " +
    "string-sorting folder names (“10” before “2”) scrambles the label space and collapses accuracy " +
    "to chance. Test predictions are written in the exact ID order of sample_submission.csv."),
]));
children.push(P([
  T("Preprocessing and augmentation. "),
  T("Images are resized to the backbone’s native 378-px input and normalized with the model’s " +
    "pretraining mean/std (handled by the timm data config). Because the backbone is frozen, we use " +
    "deterministic test-time augmentation rather than stochastic train-time augmentation: each image " +
    "is encoded as up to four views — a center-crop and a full-frame squash-resize, each also " +
    "horizontally flipped. Views are L2-normalized, averaged, then L2-normalized again into one " +
    "robust embedding. The squash-resize view alone added +1.6% (subjects that fill the frame are " +
    "not cut off by cropping)."),
]));

// ---- 3 Implementation ----
children.push(H1("3  Implementation"));
children.push(H2("3.1  Model"));
children.push(P([
  T("Pretrained backbone and why. "),
  T("The champion backbone is SigLIP-2 SoViT-400m, patch-14, 378-px (timm id " +
    "vit_so400m_patch14_siglip_378.v2_webli). We swept four pretraining paradigms — language-aligned " +
    "(SigLIP/SigLIP-2/CLIP), self-supervised (DINOv2), masked-image modeling (EVA-02), and supervised " +
    "ImageNet (ConvNeXt). The SigLIP family dominated this semantic, fine-grained task because its " +
    "image–text contrastive pretraining yields features that linearly separate fine categories. " +
    "SoViT-400m gives the best accuracy/memory trade-off: at 8-bit forward passes it runs on an " +
    "8 GB GPU, which a fully fine-tuned model of similar capacity would not."),
]));
children.push(P([
  T("Architecture / head. "),
  T("We keep the backbone entirely frozen and attach a single linear classifier: a multinomial " +
    "logistic-regression probe mapping the 1,152-dim embedding to 100 class logits. No extra pooling " +
    "or dropout is needed — the probe is the only trained component, and its L2 regularization " +
    "is the main capacity control."),
]));
children.push(P([
  T("Fine-tuning strategy. "),
  T("All backbone layers are frozen (no unfreezing schedule). This is a deliberate choice for the " +
    "few-shot regime: we tested end-to-end fine-tuning of SigLIP-L and it peaked at 93.1% and then " +
    "overfit, below the 94.6% of the frozen probe. Two extra inference-time stages add new signal " +
    "without touching the backbone: (i) zero-shot text fusion using the SigLIP-2 text tower, and " +
    "(ii) a transductive Sinkhorn balanced-prior step. Both are detailed in §4 and §5."),
]));

children.push(H2("3.2  Training"));
children.push(P([
  T("Loss and optimizer. "),
  T("The probe is a convex multinomial logistic regression (cross-entropy loss) solved with L-BFGS " +
    "(scikit-learn, max_iter=3000). Because the problem is convex there is a single global optimum " +
    "and no stochastic-training instability."),
]));
children.push(P([
  T("Hyperparameters. "),
  T("Inverse-regularization C = 8 (chosen by cross-validation), class_weight = “balanced”, " +
    "multinomial/softmax objective. There is no learning-rate schedule, batch size, or epoch count " +
    "in the usual sense — L-BFGS optimizes the full objective directly. The feature step uses the " +
    "frozen backbone with batch size 8 (8 GB GPU). Fusion weight w = 0.35 (zero-shot share) and " +
    "Sinkhorn temperature τ = 0.4 are the only inference hyperparameters."),
]));
children.push(P([
  T("Environment. "),
  T("Single NVIDIA RTX 2080 (8 GB); PyTorch 2.5.1 + CUDA 12.1; timm for backbones; open_clip / " +
    "transformers for the SigLIP-2 text tower; scikit-learn for the probe and Sinkhorn. Feature " +
    "extraction is the only GPU step; everything downstream is CPU and seconds-fast on cached features."),
]));

// ---- 4 Experiments ----
children.push(H1("4  Experiments"));
children.push(P([
  T("Baseline. "),
  T("Our initial baseline was an end-to-end fine-tuned ConvNeXt-Tiny (@224), which reached ~76% CV " +
    "accuracy — comfortably above the 60% target but clearly limited by overfitting on 10 images/class."),
]));
children.push(P([
  T("Validation method. "),
  T("Every design decision was scored by repeated 5×5-fold stratified cross-validation (5 repeats × " +
    "5 folds, seed 42) on the training set, reporting mean accuracy. For the transductive step we use " +
    "a balanced held-out query protocol that mimics the assumed class-balanced test set, so the " +
    "Sinkhorn gain is measured honestly rather than on a skewed split."),
]));
children.push(P("Hyperparameter / design sweeps. We tuned and ablated:"));
children.push(bullet("Backbone choice and resolution (four pretraining paradigms; 224 → 512 px)."));
children.push(bullet("TTA views (center-crop, squash-resize, hflip, zoomed crop)."));
children.push(bullet("Probe regularization C and class weighting."));
children.push(bullet("Zero-shot fusion weight w (0 → 0.5) and Sinkhorn temperature τ."));
children.push(P("Key ablations (all measured by the same repeated CV):"));
children.push(table(
  ["Change vs. champion", "Effect on CV accuracy"],
  [
    ["Squash-resize TTA view (added)", "+1.6%"],
    ["Zero-shot text fusion (added)", "≈ +2.6%"],
    ["Transductive Sinkhorn balanced prior (added)", "+1.5%"],
    ["Bigger / higher-res backbones (SigLIP2-L@512, patch16@512)", "≤ 0 (no gain)"],
    ["Richer TTA (zoomed crop) / higher res 448–512", "−0.2 to −0.6%"],
    ["End-to-end fine-tuning of SigLIP-L", "−1.5% (overfits)"],
    ["Multi-backbone late fusion (add DINOv2/EVA-02/CLIP)", "−3.8% (dilutes champion)"],
    ["Per-group fusion weights / 2nd text model (DFN-5B)", "≈ 0 (noise / overfit)"],
  ], [6600, 2760]));

// ---- 5 Results ----
children.push(H1("5  Results"));
children.push(P([
  T("Headline. "),
  T("The final pipeline reaches ≈96% repeated-CV accuracy and "),
  T("0.96363 on the Kaggle public leaderboard (2nd place)", { bold: true }),
  T(" — a +20-point absolute improvement over the 76% fine-tuned baseline and +36 over the 60% target."),
]));
children.push(P("Accuracy journey (each stage measured by repeated CV):"));
children.push(table(
  ["Stage", "Method", "CV acc"],
  [
    ["1", "ConvNeXt-Tiny, end-to-end fine-tune (baseline)", "76.0%"],
    ["2", "Pivot to frozen features: DINOv2-L probe", "86.2%"],
    ["3", "Stronger backbone: SigLIP-L probe @384", "92.2%"],
    ["4", "Champion backbone: SoViT-400m SigLIP-2 @378 + multi-TTA", "94.6%"],
    ["5", "+ zero-shot text fusion + Sinkhorn balanced prior", "≈96%"],
  ], [900, 6300, 2160]));
children.push(P(""));
children.push(...figure("docs/accuracy_journey.png", 540,
  "Figure 1. Accuracy journey from the fine-tuned baseline to the final fused, transductive pipeline."));
children.push(...figure("docs/model_comparison.png", 470,
  "Figure 2. Backbone comparison (frozen features → logistic probe). The SigLIP family dominates this fine-grained task."));
children.push(P([
  T("Probe training curve. "),
  T("Because the probe is convex, its training curve traces the L-BFGS optimizer converging. Both " +
    "training and held-out cross-entropy fall sharply and flatten by ~20–30 iterations; the " +
    "validation curve sits a small, stable gap above training and never turns upward, confirming the " +
    "C = 8 regularization prevents overfitting."),
]));
children.push(...figure("docs/figs/probe_training_curve.png", 450,
  "Figure 3. Probe training vs. validation cross-entropy (frozen SigLIP-2 features, C = 8)."));
children.push(...figure("docs/figs/probe_performance.png", 450,
  "Figure 4. Probe-alone CV accuracy by domain: strong on food/flowers, harder on fine-grained cars/aircraft."));
children.push(P([
  T("Error analysis. "),
  T("Per-domain CV shows the remaining errors are concentrated in the hardest fine-grained pairs — " +
    "visually near-identical car trims and aircraft variants (cars/aircraft probe accuracy ≈89–91% " +
    "vs. ≈95–100% for food/flowers). Cross-domain confusion is essentially zero, so the four source " +
    "datasets are perfectly separable; the open challenge is within-domain fine-grained discrimination."),
]));

// ---- 6 Discussion ----
children.push(H1("6  Discussion"));
children.push(P([
  T("What worked best and why. "),
  T("Three moves drove almost all the gain: (1) switching from fine-tuning a small net to a "),
  T("frozen foundation model + linear probe", { bold: true }),
  T(" (the right inductive bias for 10 images/class); (2) choosing the "),
  T("SigLIP family", { bold: true }),
  T(", whose image–text pretraining linearly separates fine categories; and (3) adding "),
  T("zero-shot text fusion and a transductive balanced-prior step", { bold: true }),
  T(", which inject genuinely new signal (language alignment; the balanced-test assumption) rather " +
    "than re-weighting existing features."),
]));
children.push(P([
  T("Failure cases / fitting behavior. "),
  T("End-to-end fine-tuning overfit (93.1% then declining) — the classic under-data, over-capacity " +
    "failure. Adding more backbones to an ensemble hurt (−3.8%): the non-SigLIP models are both " +
    "weaker and correlated, so they dilute the champion. The frozen probe itself shows only a small, " +
    "stable train/val gap (Figure 3), i.e. it neither under- nor over-fits."),
]));
children.push(P([
  T("Limitations and next steps. "),
  T("(i) The Sinkhorn gain depends on the test set actually being class-balanced; if it is not, the " +
    "--soft flag hedges by blending back toward the raw probabilities. (ii) Zero-shot accuracy is " +
    "bounded by class-name quality; a few car/aircraft names are auto-matched and human verification " +
    "is the main remaining lever. (iii) The clearest future improvement is a pairwise specialist " +
    "re-ranker for the few near-duplicate fine-grained pairs the probe still confuses."),
]));

// ---- 7 Reproducibility ----
children.push(H1("7  Reproducibility"));
children.push(P([
  T("Seeds / determinism. "),
  T("A fixed seed (42) is set across NumPy, PyTorch, and scikit-learn (set_seed in src/utils.py). " +
    "Feature extraction uses eval-only transforms and is deterministic; the convex probe has a unique " +
    "optimum. Extracted features are cached to outputs/feats/, so the entire probe + fusion + Sinkhorn " +
    "stage reproduces exactly and in seconds."),
]));
children.push(P([
  T("Environment. "),
  T("Python 3 with PyTorch 2.5.1 + CUDA 12.1, timm, scikit-learn, open_clip / transformers " +
    "(≥ 4.49 for the SigLIP-2 Gemma tokenizer), pandas, pillow. Full pin list in requirements.txt; " +
    "install with pip install -r requirements.txt."),
]));
children.push(P([T("Exact commands.", { bold: true })]));
children.push(numItem("c", [T("Extract frozen champion features (cached):", {}),]));
children.push(P([T("python src/extract_features.py --backbone vit_so400m_patch14_siglip_378.v2_webli " +
  "--batch_size 8 --tta multi --out_suffix _tta", { font: "Consolas", size: 17, color: NAVY })],
  { spacing: { after: 80 }, indent: { left: 360 } }));
children.push(numItem("c", [T("Cache the zero-shot text probabilities:")]));
children.push(P([T("python src/zeroshot_siglip.py --names_csv data/class_names.csv",
  { font: "Consolas", size: 17, color: NAVY })], { spacing: { after: 80 }, indent: { left: 360 } }));
children.push(numItem("c", [T("Generate the final submission (probe + fusion + Sinkhorn):")]));
children.push(P([T("python src/predict_fused.py --w 0.35 --tau 0.4 --out outputs/submission_fused.csv",
  { font: "Consolas", size: 17, color: NAVY })], { spacing: { after: 80 }, indent: { left: 360 } }));
children.push(numItem("c", [T("(Optional) save the fitted probe as a distributable weight bundle:")]));
children.push(P([T("python src/save_model.py", { font: "Consolas", size: 17, color: NAVY })],
  { spacing: { after: 120 }, indent: { left: 360 } }));
children.push(P([
  T("The fine-tuned ConvNeXt baseline is reproducible via python src/train.py --config " +
    "configs/default.yaml and src/predict.py. Trained weights (fitted probe + cached champion " +
    "features) are provided via the Google Drive link in the repository README."),
]));

// ---- 8 Team Contributions ----
children.push(H1("8  Team Contributions"));
children.push(P([
  T("[Your Name] (solo). ", { bold: true }),
  T("Sole contributor: dataset analysis (identifying the four source datasets and fixing the " +
    "label mapping), backbone selection and feature extraction, the logistic probe and multi-view " +
    "TTA, zero-shot text fusion, the transductive Sinkhorn step, all cross-validation experiments " +
    "and ablations, the Kaggle submissions, and this report."),
]));

// ---- 9 References ----
children.push(H1("9  References"));
const ref = (runs) => new Paragraph({ spacing: { after: 100, line: 264 },
  indent: { left: 360, hanging: 360 }, children: runs });
children.push(ref([
  T("Zhai, X., et al. “Sigmoid Loss for Language Image Pre-Training (SigLIP).” "),
  T("ICCV", { italics: true }), T(", 2023. "),
  new ExternalHyperlink({ link: "https://arxiv.org/abs/2303.15343",
    children: [new TextRun({ text: "arxiv.org/abs/2303.15343", style: "Hyperlink", size: 20 })] }),
]));
children.push(ref([
  T("Tschannen, M., et al. “SigLIP 2: Multilingual Vision-Language Encoders.” 2025. "),
  new ExternalHyperlink({ link: "https://arxiv.org/abs/2502.14786",
    children: [new TextRun({ text: "arxiv.org/abs/2502.14786", style: "Hyperlink", size: 20 })] }),
]));
children.push(ref([
  T("Cuturi, M. “Sinkhorn Distances: Lightspeed Computation of Optimal Transport.” "),
  T("NeurIPS", { italics: true }), T(", 2013. "),
  new ExternalHyperlink({ link: "https://arxiv.org/abs/1306.0895",
    children: [new TextRun({ text: "arxiv.org/abs/1306.0895", style: "Hyperlink", size: 20 })] }),
]));
children.push(ref([
  T("Wightman, R. “PyTorch Image Models (timm).” GitHub. "),
  new ExternalHyperlink({ link: "https://github.com/huggingface/pytorch-image-models",
    children: [new TextRun({ text: "github.com/huggingface/pytorch-image-models", style: "Hyperlink", size: 20 })] }),
]));
children.push(ref([
  T("Pedregosa, F., et al. “Scikit-learn: Machine Learning in Python.” "),
  T("JMLR", { italics: true }), T(", 2011. "),
  new ExternalHyperlink({ link: "https://jmlr.org/papers/v12/pedregosa11a.html",
    children: [new TextRun({ text: "jmlr.org/papers/v12/pedregosa11a.html", style: "Hyperlink", size: 20 })] }),
]));

// =====================================================================
const doc = new Document({
  creator: "CSE 144 Final Project",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: NAVY },
        paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Calibri", color: INK },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
      { reference: "c", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 420, hanging: 260 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "", size: 18, color: MUTED }),
        new TextRun({ children: [PageNumber.CURRENT], size: 18, color: MUTED })],
    })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "docs", "CSE144_Final_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote " + out + " (" + buf.length + " bytes)");
});
