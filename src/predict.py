"""Generate submission.csv from trained checkpoints (TTA + ensembling).

    python src/predict.py --config configs/default.yaml --ckpt outputs/model_fold*.pt
"""
import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_eval_transform, TestDataset
from model import build_model


@torch.no_grad()
def predict_logits(model, loader, device, tta=True):
    model.eval()
    all_ids, all_probs = [], []
    for x, ids in loader:
        x = x.to(device)
        views = [x]
        if tta:
            views.append(torch.flip(x, dims=[3]))  # horizontal flip
        prob = 0.0
        for v in views:
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(v)
            prob = prob + torch.softmax(logits.float(), dim=1)
        prob /= len(views)
        all_probs.append(prob.cpu().numpy())
        all_ids.extend(list(ids))  # ids are filename strings
    return all_ids, np.concatenate(all_probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="Checkpoint path(s) or glob(s). Probabilities are averaged.")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # expand globs
    ckpts = []
    for pat in args.ckpt:
        ckpts.extend(sorted(glob.glob(pat)) or [pat])
    assert ckpts, "No checkpoints found"
    print(f"Ensembling {len(ckpts)} checkpoint(s): {ckpts}")

    # IDs and order come from sample_submission.csv (test/ may hold extra images)
    sample = os.path.join(cfg["data_dir"], "sample_submission.csv")
    sub = pd.read_csv(sample)
    ids_order = sub["ID"].tolist()

    test_dir = os.path.join(cfg["data_dir"], "test")
    eval_tf = build_eval_transform(cfg["img_size"])
    loader = DataLoader(TestDataset(test_dir, ids_order, eval_tf),
                        batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)

    ids_ref, mean_probs = None, None
    for ck in ckpts:
        state = torch.load(ck, map_location=device)
        ck_cfg = state.get("cfg", cfg)
        model = build_model(ck_cfg["backbone"], ck_cfg["num_classes"],
                            pretrained=False).to(device)
        model.load_state_dict(state["model"])
        ids, probs = predict_logits(model, loader, device, tta=not args.no_tta)
        if ids_ref is None:
            ids_ref = ids
            mean_probs = np.zeros_like(probs)
        assert ids == ids_ref, "ID order mismatch across checkpoints"
        mean_probs += probs
    mean_probs /= len(ckpts)

    preds = mean_probs.argmax(axis=1)
    pred_map = dict(zip(ids_ref, preds))
    out = args.out or os.path.join(cfg["output_dir"], "submission.csv")
    df = pd.DataFrame({"ID": ids_order,
                       "Label": [int(pred_map[i]) for i in ids_order]})
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} rows, labels in "
          f"[{df['Label'].min()},{df['Label'].max()}]")

    # sanity vs sample_submission
    assert list(df.columns) == list(sub.columns), \
        f"columns {list(df.columns)} != {list(sub.columns)}"
    assert len(df) == len(sub), f"row count {len(df)} != {len(sub)}"
    assert df["ID"].tolist() == sub["ID"].tolist(), "ID order mismatch with sample_submission"
    print("Validated against sample_submission.csv ✓")


if __name__ == "__main__":
    main()
