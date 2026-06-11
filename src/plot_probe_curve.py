"""Probe training curve: cross-entropy vs L-BFGS iteration, train + held-out.
Steps the real probe (C=8) one iteration at a time on the champion features.
"""
import os, warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")

TAG = "vit_so400m_patch14_siglip_378_378_v2_tta"
NAVY, AMBER, MUTED = "#21356B", "#F5A623", "#6B7793"
LBL = list(range(100))

d = np.load(f"outputs/feats/{TAG}_train.npz")
X = normalize(d["feats"].astype(np.float32)); y = d["labels"].astype(np.int64)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(C=8.0, solver="lbfgs", max_iter=1, warm_start=True,
                         class_weight="balanced", tol=1e-12)
iters, tr_loss, va_loss = [], [], []
for it in range(1, 61):
    clf.max_iter = 1
    clf.fit(Xtr, ytr)
    Ptr = np.zeros((len(Xtr), 100)); Ptr[:, clf.classes_] = clf.predict_proba(Xtr)
    Pva = np.zeros((len(Xva), 100)); Pva[:, clf.classes_] = clf.predict_proba(Xva)
    iters.append(it)
    tr_loss.append(log_loss(ytr, Ptr, labels=LBL))
    va_loss.append(log_loss(yva, Pva, labels=LBL))

fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
ax.plot(iters, tr_loss, color=NAVY, lw=2.6, label="training loss")
ax.plot(iters, va_loss, color=AMBER, lw=2.6, label="validation loss (held-out 20%)")
ax.set_title("Logistic-probe training curve\n(cross-entropy on frozen SigLIP 2 SoViT-400m features)",
             fontsize=13, fontweight="bold", color="#17223B")
ax.set_xlabel("L-BFGS iteration", fontsize=11)
ax.set_ylabel("cross-entropy loss", fontsize=11)
ax.legend(frameon=False, fontsize=11, loc="upper right")
ax.grid(True, color="#E2E8F0", lw=0.6)
for s in (ax.spines["top"], ax.spines["right"]): s.set_visible(False)
fig.text(0.5, -0.01,
         f"final train {tr_loss[-1]:.3f}  ·  validation {va_loss[-1]:.3f}  ·  "
         "C=8, multinomial, class-weight balanced",
         ha="center", fontsize=9.5, color=MUTED)
os.makedirs("docs/figs", exist_ok=True)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig("docs/figs/probe_training_curve.png", bbox_inches="tight")
print("wrote docs/figs/probe_training_curve.png")
print(f"final train_loss={tr_loss[-1]:.3f} val_loss={va_loss[-1]:.3f} min_val={min(va_loss):.3f}@{iters[int(np.argmin(va_loss))]}")
