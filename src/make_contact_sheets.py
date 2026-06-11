"""Render per-class contact sheets + compact batch sheets to docs/class_sheets/ for
human review and class naming.
"""
import os, sys, math
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from dataset import list_train_samples

OUT = "docs/class_sheets"
THUMB = 200          # per-image thumbnail size for human sheets
BTHUMB = 200         # thumbnail size for batch sheets
PER_BATCH = 10       # classes per batch sheet
REPS = 3             # representative thumbs per class on batch sheet


def font(sz):
    try:
        return ImageFont.truetype("arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def thumb(path, size):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (30, 30, 30))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = list_train_samples("data/train")
    by_class = {}
    for p, lab in samples:
        by_class.setdefault(lab, []).append(p)

    # --- per-class sheets (human review) ---
    for lab in sorted(by_class):
        paths = sorted(by_class[lab])
        cols = min(5, len(paths)); rows = math.ceil(len(paths) / cols)
        head = 30
        sheet = Image.new("RGB", (cols * THUMB, rows * THUMB + head), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        d.text((6, 6), f"class {lab}  ({len(paths)} imgs)", fill=(255, 255, 0), font=font(20))
        for i, p in enumerate(paths):
            r, c = divmod(i, cols)
            sheet.paste(thumb(p, THUMB), (c * THUMB, head + r * THUMB))
        sheet.save(os.path.join(OUT, f"class_{lab:02d}.jpg"), quality=85)

    # --- compact batch sheets (for model naming) ---
    classes = sorted(by_class)
    n_batches = math.ceil(len(classes) / PER_BATCH)
    label_w = 60
    for b in range(n_batches):
        chunk = classes[b * PER_BATCH:(b + 1) * PER_BATCH]
        sheet = Image.new("RGB", (label_w + REPS * BTHUMB, len(chunk) * BTHUMB), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        for r, lab in enumerate(chunk):
            paths = sorted(by_class[lab])
            # spread picks across the class
            picks = [paths[int(k * (len(paths) - 1) / max(1, REPS - 1))] for k in range(REPS)]
            d.text((6, r * BTHUMB + BTHUMB // 2 - 10), str(lab), fill=(255, 255, 0), font=font(28))
            for c, p in enumerate(picks):
                sheet.paste(thumb(p, BTHUMB), (label_w + c * BTHUMB, r * BTHUMB))
        sheet.save(os.path.join(OUT, f"_batch_{b}.jpg"), quality=80)
    print(f"wrote {len(classes)} class sheets + {n_batches} batch sheets to {OUT}/")


if __name__ == "__main__":
    main()
