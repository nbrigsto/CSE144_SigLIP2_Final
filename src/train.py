"""End-to-end fine-tune baseline (k-fold). Fixed seed, deterministic loaders;
saves a checkpoint per fold. Reference only; the frozen-probe pipeline wins.

    python src/train.py --config configs/default.yaml [--train_all]
"""
import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from dataset import (list_train_samples, build_train_transform, build_eval_transform,
                     TrainDataset)
from model import build_model, set_backbone_trainable
from utils import (set_seed, seed_worker, ModelEMA, mixup_cutmix,
                   SoftTargetCrossEntropy, accuracy)


def make_loader(samples, transform, batch_size, workers, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        TrainDataset(samples, transform), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, drop_last=False,
        worker_init_fn=seed_worker, generator=g,
    )


def build_param_groups(model, head_lr, backbone_lr, wd):
    head = set(model.get_classifier().parameters())
    head_p, body_p = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (head_p if p in head else body_p).append(p)
    groups = [{"params": head_p, "lr": head_lr}]
    if body_p:
        groups.append({"params": body_p, "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=wd)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train_one_fold(cfg, train_samples, val_samples, fold_tag, device):
    img_size = cfg["img_size"]
    train_tf = build_train_transform(img_size, cfg["aug_strength"])
    eval_tf = build_eval_transform(img_size)

    train_loader = make_loader(train_samples, train_tf, cfg["batch_size"],
                               cfg["num_workers"], True, cfg["seed"])
    val_loader = (make_loader(val_samples, eval_tf, cfg["batch_size"],
                              cfg["num_workers"], False, cfg["seed"])
                  if val_samples else None)

    model = build_model(cfg["backbone"], cfg["num_classes"], cfg["drop_rate"]).to(device)
    ema = ModelEMA(model, cfg["ema_decay"])
    soft_ce = SoftTargetCrossEntropy()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and torch.cuda.is_available())

    total_epochs = cfg["head_epochs"] + cfg["full_epochs"]
    steps_per_epoch = max(1, len(train_loader) // cfg["grad_accum"])
    best_acc, best_path = -1.0, os.path.join(cfg["output_dir"], f"model_{fold_tag}.pt")

    optimizer = scheduler = None
    for epoch in range(total_epochs):
        phase_full = epoch >= cfg["head_epochs"]
        # (Re)build optimizer at phase boundaries.
        if epoch == 0 or epoch == cfg["head_epochs"]:
            set_backbone_trainable(model, phase_full)
            optimizer = build_param_groups(model, cfg["head_lr"], cfg["backbone_lr"],
                                           cfg["weight_decay"])
            remaining = total_epochs - epoch
            warm = cfg["warmup_epochs"] if phase_full else 0
            def lr_lambda(e, warm=warm, total=remaining):
                if e < warm:
                    return (e + 1) / max(1, warm)
                prog = (e - warm) / max(1, total - warm)
                return 0.5 * (1 + np.cos(np.pi * prog))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            phase_epoch = 0

        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x, targets = mixup_cutmix(x, y, cfg["num_classes"], cfg["mixup_alpha"],
                                      cfg["cutmix_alpha"], cfg["mix_prob"],
                                      cfg["label_smoothing"])
            with torch.autocast("cuda", enabled=cfg["amp"] and torch.cuda.is_available()):
                logits = model(x)
                loss = soft_ce(logits, targets) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (i + 1) % cfg["grad_accum"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            running += loss.item() * cfg["grad_accum"]
        scheduler.step()
        phase_epoch += 1

        msg = f"[{fold_tag}] epoch {epoch+1}/{total_epochs} " \
              f"({'full' if phase_full else 'head'}) loss={running/len(train_loader):.4f}"
        if val_loader is not None:
            raw_acc = evaluate(model, val_loader, device)
            ema_acc = evaluate(ema.ema, val_loader, device)
            msg += f" val_acc raw={raw_acc:.4f} ema={ema_acc:.4f}"
            # Save the better of raw vs EMA weights.
            cand = ema.ema if ema_acc >= raw_acc else model
            acc = max(raw_acc, ema_acc)
            if acc >= best_acc:
                best_acc = acc
                torch.save({"model": cand.state_dict(), "cfg": cfg}, best_path)
        print(msg, flush=True)

    if val_loader is None:  # train_all: save final EMA weights
        torch.save({"model": ema.ema.state_dict(), "cfg": cfg}, best_path)
        best_acc = float("nan")
    print(f"[{fold_tag}] best_val_acc={best_acc:.4f} saved={best_path}", flush=True)
    return best_acc, best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train_all", action="store_true",
                    help="Train one model on ALL training data (final submission model).")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.train_all:
        cfg["train_all"] = True

    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} backbone={cfg['backbone']} img_size={cfg['img_size']}")

    train_dir = os.path.join(cfg["data_dir"], "train")
    samples = list_train_samples(train_dir)
    labels = np.array([lab for _, lab in samples])
    assert labels.min() == 0 and labels.max() == cfg["num_classes"] - 1, \
        "Label range mismatch — check folder->int mapping"
    print(f"train images={len(samples)} classes={len(set(labels))}")

    if cfg.get("train_all"):
        train_one_fold(cfg, samples, None, "all", device)
        return

    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    accs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if fold not in cfg["folds_to_run"]:
            continue
        tr = [samples[i] for i in tr_idx]
        va = [samples[i] for i in va_idx]
        acc, _ = train_one_fold(cfg, tr, va, f"fold{fold}", device)
        accs.append(acc)
    if accs:
        print(f"CV val_acc mean={np.mean(accs):.4f} std={np.std(accs):.4f}")


if __name__ == "__main__":
    main()
