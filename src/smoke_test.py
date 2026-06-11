"""Quick sanity checks: label mapping, model build, one GPU train step."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from dataset import (list_train_samples, build_train_transform, TrainDataset)
from model import build_model, set_backbone_trainable
from utils import set_seed, mixup_cutmix, SoftTargetCrossEntropy

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

# 1) Label mapping
samples = list_train_samples("data/train")
labels = sorted(set(l for _, l in samples))
print("n_samples:", len(samples), "n_classes:", len(labels),
      "min:", min(labels), "max:", max(labels))
assert len(labels) == 100 and min(labels) == 0 and max(labels) == 99
# folder "0" -> 0, "99" -> 99
by_label = {}
for p, l in samples:
    by_label.setdefault(l, p)
assert os.path.basename(os.path.dirname(by_label[0])) == "0"
assert os.path.basename(os.path.dirname(by_label[99])) == "99"
print("Label mapping OK (folder 0->0, 99->99)")

# 2) Model build + 3) one train step on a small batch
model = build_model("convnext_tiny.fb_in22k_ft_in1k", 100, 0.1).to(device)
set_backbone_trainable(model, True)
tf = build_train_transform(224, "strong")
ds = TrainDataset(samples[:16], tf)
xs = torch.stack([ds[i][0] for i in range(8)]).to(device)
ys = torch.tensor([ds[i][1] for i in range(8)]).to(device)
soft_ce = SoftTargetCrossEntropy()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
x, t = mixup_cutmix(xs, ys, 100)
with torch.autocast("cuda", enabled=device == "cuda"):
    out = model(x)
    loss = soft_ce(out, t)
scaler.scale(loss).backward()
scaler.step(opt); scaler.update()
print("Forward/backward OK, loss:", round(loss.item(), 4), "out shape:", tuple(out.shape))
if device == "cuda":
    print("peak GPU mem (MB):", round(torch.cuda.max_memory_allocated() / 1e6, 1))
print("SMOKE TEST PASSED")
