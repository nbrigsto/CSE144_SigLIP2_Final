"""Does a SECOND text model (DFN-5B) add signal over SigLIP zero-shot?

Balanced holdout (same honest protocol). Fusion = (1-w)P + w*Ztext, then Sinkhorn.
Ztext is one of: SigLIP, DFN, or their average. If the two text models are
decorrelated, the average should beat either alone.
"""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.dirname(__file__))
from exp_transductive import sinkhorn, balanced_split

FEAT_DIR = "outputs/feats"
NC = 100
CHAMP = "vit_so400m_patch14_siglip_378_378_v2_tta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.4)
    args = ap.parse_args()

    d = np.load(os.path.join(FEAT_DIR, f"{CHAMP}_train.npz"))
    X = normalize(d["feats"].astype(np.float32)); y = d["labels"].astype(np.int64)
    Zsig = np.load(os.path.join(FEAT_DIR, "zeroshot378_train.npz"))["proba"].astype(np.float64)
    Zd378 = np.load(os.path.join(FEAT_DIR, "zeroshot_dfn5b378_train.npz"))["proba"].astype(np.float64)

    agree = (Zsig.argmax(1) == Zd378.argmax(1)).mean()
    print(f"DFN378 train acc={(Zd378.argmax(1)==y).mean():.4f}  "
          f"SigLIP acc={(Zsig.argmax(1)==y).mean():.4f}  top1 agreement={agree:.3f}")

    sources = {
        "sig": lambda q: Zsig[q],
        "d378": lambda q: Zd378[q],
        "avg": lambda q: 0.5 * (Zsig[q] + Zd378[q]),
        "avg6040": lambda q: 0.6 * Zsig[q] + 0.4 * Zd378[q],
    }
    ws = [0.2, 0.3, 0.35, 0.4, 0.5]
    res = {f"{s}_w{w}": [] for s in sources for w in ws}
    res["probe_only"] = []

    for r in range(args.reps):
        rng = np.random.default_rng(r)
        sup, qry = balanced_split(y, args.q, rng)
        clf = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced").fit(X[sup], y[sup])
        P = np.zeros((len(qry), NC)); P[:, clf.classes_] = clf.predict_proba(X[qry])
        yq = y[qry]
        res["probe_only"].append((sinkhorn(np.log(P + 1e-9), tau=args.tau).argmax(1) == yq).mean())
        for s, fn in sources.items():
            Z = fn(qry)
            for w in ws:
                F = (1 - w) * P + w * Z
                a = (sinkhorn(np.log(F + 1e-9), tau=args.tau).argmax(1) == yq).mean()
                res[f"{s}_w{w}"].append(a)

    base = np.mean(res["sig_w0.35"])
    print(f"\n{'method':12s} acc      delta vs sig_w0.35")
    print(f"{'probe_only':12s} {np.mean(res['probe_only']):.4f}  {np.mean(res['probe_only'])-base:+.4f}")
    for s in sources:
        for w in ws:
            k = f"{s}_w{w}"; m = np.mean(res[k])
            print(f"{k:12s} {m:.4f}  {m-base:+.4f}")


if __name__ == "__main__":
    main()
