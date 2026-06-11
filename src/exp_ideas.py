"""Stress-test creative improvements on the balanced holdout (vs current best:
single-champion probe + global zero-shot fusion w=0.35 + Sinkhorn).

Ideas: (1) cross-group error analysis, (2) group-conditional masking,
(3) per-group fusion weights, (4) multi-backbone late fusion, (5) the combo.
"""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.dirname(__file__))
from exp_transductive import sinkhorn, balanced_split

FEAT_DIR = "outputs/feats"
SEED = 42
NC = 100
GROUP = np.array([i // 25 for i in range(NC)])          # 0 food,1 flower,2 car,3 aircraft
GNAME = ["food", "flower", "car", "aircraft"]
CHAMP = "vit_so400m_patch14_siglip_378_378_v2_tta"
EXTRA = ["vit_so400m_patch16_siglip_512_512_v2_tta",
         "vit_large_patch16_siglip_512_512_v2_tta",
         "vit_large_patch14_reg4_dinov2_336_lvd142_tta"]


def load(tag):
    d = np.load(os.path.join(FEAT_DIR, f"{tag}_train.npz"))
    return normalize(d["feats"].astype(np.float32)), d["labels"].astype(np.int64)


def probe_probs(X, sup, qry, ys, C=8.0):
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X[sup], ys)
    P = np.zeros((len(qry), NC)); P[:, clf.classes_] = clf.predict_proba(X[qry])
    return P


def sink(F, tau=0.4):
    return sinkhorn(np.log(F + 1e-9), tau=tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--w", type=float, default=0.35)
    args = ap.parse_args()

    feats = {CHAMP: load(CHAMP)}
    y = feats[CHAMP][1]
    for t in EXTRA:
        feats[t] = load(t)
    Z = np.load(os.path.join(FEAT_DIR, "zeroshot378_train.npz"))["proba"].astype(np.float64)

    res = {}
    def add(k, v): res.setdefault(k, []).append(v)
    cross_group_err = []
    # accumulate per-group accuracy at each w to choose per-group weights
    wgrid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    grp_w_acc = {g: {w: [] for w in wgrid} for g in range(4)}

    for r in range(args.reps):
        rng = np.random.default_rng(SEED + r)
        sup, qry = balanced_split(y, args.q, rng)
        ys, yq = y[sup], y[qry]; Zq = Z[qry]; gq = GROUP[yq]
        Pc = probe_probs(feats[CHAMP][0], sup, qry, ys)

        # ---- current best: champion + global w + sinkhorn ----
        base = sink((1 - args.w) * Pc + args.w * Zq)
        add("BASE champ+w.35", (base.argmax(1) == yq).mean())
        # cross-group error fraction
        bp = base.argmax(1); err = bp != yq
        cross_group_err.append((GROUP[bp][err] != gq[err]).mean() if err.any() else 0.0)

        # ---- group-conditional masking (zero out other groups) ----
        Fg = (1 - args.w) * Pc + args.w * Zq
        mask = (GROUP[None, :] == gq[:, None]).astype(float)
        Fm = Fg * mask
        add("group-mask", (sink(Fm).argmax(1) == yq).mean())

        # ---- per-group fusion weight: record acc per group per w ----
        for g in range(4):
            sel = gq == g
            if not sel.any(): continue
            for w in wgrid:
                Fw = sink((1 - w) * Pc + w * Zq)
                grp_w_acc[g][w].append((Fw.argmax(1)[sel] == yq[sel]).mean())

        # ---- multi-backbone late fusion (avg probe probs) ----
        Pens = np.mean([probe_probs(feats[t][0], sup, qry, ys)
                        for t in [CHAMP] + EXTRA], axis=0)
        add("multibackbone probe (no zs)", (sink(Pens).argmax(1) == yq).mean())
        add("multibackbone+zs+sink", (sink((1 - args.w) * Pens + args.w * Zq).argmax(1) == yq).mean())

    # choose per-group best w (on aggregate), then report a combined run estimate
    best_wg = {g: max(wgrid, key=lambda w: np.mean(grp_w_acc[g][w])) for g in range(4)}
    print("per-group best fusion w:", {GNAME[g]: best_wg[g] for g in range(4)})
    # reconstruct per-group-w accuracy by mixing each group's best-w accuracy
    pgw = np.mean([np.mean(grp_w_acc[g][best_wg[g]]) for g in range(4)])  # macro over groups

    print(f"\ncross-group error fraction of base: {np.mean(cross_group_err):.3f}")
    print(f"\n{'method':28s} acc      delta")
    base = np.mean(res["BASE champ+w.35"])
    for k in res:
        m = np.mean(res[k]); print(f"{k:28s} {m:.4f}  {m-base:+.4f}")
    print(f"{'per-group-w (macro est)':28s} {pgw:.4f}  (group-macro, not directly comparable)")


if __name__ == "__main__":
    main()
