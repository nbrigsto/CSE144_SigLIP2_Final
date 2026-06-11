"""Label-correct datasets and transforms.

ImageFolder sorts folder names as strings, so we set label = int(folder_name)
explicitly to keep "0"->0 ... "99"->99.
"""
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_train_samples(train_dir):
    """Return (path, int_label) pairs. Label is parsed from the folder name."""
    samples = []
    class_names = sorted(os.listdir(train_dir), key=lambda s: int(s))
    for name in class_names:
        cdir = os.path.join(train_dir, name)
        if not os.path.isdir(cdir):
            continue
        label = int(name)  # the gotcha fix
        for p in glob.glob(os.path.join(cdir, "*")):
            samples.append((p, label))
    return samples


def build_train_transform(img_size, aug_strength="strong"):
    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0),
                                      interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if aug_strength == "strong":
        ops.append(transforms.TrivialAugmentWide(
            interpolation=transforms.InterpolationMode.BICUBIC))
        ops.append(transforms.ColorJitter(0.3, 0.3, 0.3, 0.1))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if aug_strength == "strong":
        ops.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(ops)


def build_eval_transform(img_size):
    resize = int(round(img_size * 1.14))
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TrainDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class TestDataset(Dataset):
    """Reads test images for a list of filename IDs, in the given order."""

    def __init__(self, test_dir, ids, transform):
        self.test_dir = test_dir
        self.ids = list(ids)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(os.path.join(self.test_dir, img_id)).convert("RGB")
        return self.transform(img), img_id
