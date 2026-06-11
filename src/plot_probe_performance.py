"""Probe-only accuracy (no fusion, no Sinkhorn) by domain, via repeated 5-fold CV
on the champion features.
"""
import os, warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")

TAG = "vit_so400m_patch14_siglip_378_378_v2_tta"
GROUP = np.array([i // 25 for i in range(100)])
GNAME = ["Food", "Flowers", "Cars", "Aircraft"]
COL = {"Overall": "#17223B", "Food": "#2E5BFF", "Flowers": "#0FA39A",
       "Cars": "#F5A623", "Aircraft": "#21356B"}

d = np.load(f"outputs/feats/{TAG}_train.npz")
X = normalize(d["feats"].astype(np.float32)); y = d["labels"].astype(np.int64)

# repeated 5-fold out-of-fold predictions (probe alone)
oof_pred = np.zeros((5, len(y)), dtype=int)
for r in range(5):
    skf = StratifiedKFold(5, shuffle=True, random_state=42 + r)
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
        oof_pred[r, va] = clf.predict(X[va])

correct = (oof_pred == y[None, :])                      # (5 repeats, N)
overall = correct.mean() * 100
labels = ["Overall"] + GNAME
vals = [overall]
errs = [correct.mean(1).std() * 100]
for g in range(4):
    sel = GROUP[y] == g
    per_rep = correct[:, sel].mean(1)
    vals.append(per_rep.mean() * 100); errs.append(per_rep.std() * 100)

fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
bars = ax.bar(labels, vals, yerr=errs, capsize=4,
              color=[COL[l] for l in labels], width=0.62, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1.1, f"{v:.1f}%",
            ha="center", fontsize=12, fontweight="bold", color="#17223B")
ax.set_ylim(60, 102)
ax.set_ylabel("cross-validated accuracy (%)", fontsize=11)
ax.set_title("Logistic probe alone — accuracy by domain\n"
             "(frozen SigLIP 2 features; no text fusion, no Sinkhorn)",
             fontsize=13, fontweight="bold", color="#17223B")
ax.axhline(overall, color="#17223B", lw=1, ls="--", alpha=0.4, zorder=1)
ax.grid(axis="y", color="#E2E8F0", lw=0.6, zorder=0)
for s in (ax.spines["top"], ax.spines["right"]): s.set_visible(False)
fig.text(0.5, -0.02, "5× repeated 5-fold CV  ·  error bars = std across repeats  ·  "
         "the probe is strong on food/flowers, weaker on fine-grained cars/aircraft",
         ha="center", fontsize=9, color="#6B7793")
os.makedirs("docs/figs", exist_ok=True)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig("docs/figs/probe_performance.png", bbox_inches="tight")
print("wrote docs/figs/probe_performance.png")
print("overall %.1f%% | " % overall + " ".join(f"{n} {v:.1f}%" for n, v in zip(GNAME, vals[1:])))
