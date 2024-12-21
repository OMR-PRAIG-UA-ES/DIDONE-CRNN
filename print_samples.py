import os
import cv2
import argparse
import torch
import numpy as np
import random

from dataset import DidoneDataset, PrimusDataset

random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # For reproducibility, but slower

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Parse arguments
parser = argparse.ArgumentParser(description="Train a CRNN model")
parser.add_argument(
    "--dataset",
    choices=["didone", "primus"],
    default="didone",
    help="Type of dataset to use",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="./datasets/didone",
    help="Path to the dataset",
)
parser.add_argument(
    "--split",
    action="store_true",
    help="Whether to use the split encoding or not",
)
parser.add_argument(
    "--camera",
    action="store_true",
    help="Whether to use the camera augmentation or not (only for Primus)",
)
parser.add_argument("--img_height", "-ih", type=int, default=128, help="Image height")
args = parser.parse_args()

if args.dataset == "didone":
    dataset = DidoneDataset(
        root_path=args.data_path,
        encoding="split" if args.split else "std",
        img_height=args.img_height,
    )
else:
    dataset = PrimusDataset(
        camera=args.camera,
        root_path=args.data_path,
        encoding="split" if args.split else "std",
        img_height=args.img_height,
    )

train_split, val_split = torch.utils.data.random_split(dataset, [0.8, 0.2])
val_split.training = False

os.makedirs("samples/train", exist_ok=True)
lbls = []
for i, (img, label) in enumerate(train_split):
    if i == 10:
        break
    img *= 255
    img = img.permute(1, 2, 0).numpy()
    lbls.append(" ".join([dataset.vocab.i2c[i.item()] for i in label]))
    cv2.imwrite(f"samples/train/{i}.png", img)
with open("samples/train/labels.txt", "w") as f:
    f.write("\n".join(lbls))

os.makedirs("samples/val", exist_ok=True)
lbls = []
for i, (img, label) in enumerate(val_split):
    if i == 10:
        break
    img *= 255
    img = img.permute(1, 2, 0).numpy()
    lbls.append(" ".join([dataset.vocab.i2c[i.item()] for i in label]))
    cv2.imwrite(f"samples/val/{i}.png", img)
with open("samples/val/labels.txt", "w") as f:
    f.write("\n".join(lbls))
