"""
Test the model on the validation set.
author: Adrián Roselló Pedraza (RosiYo)
"""

import os
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from torchinfo import summary

from dataset import DidoneDataset, PrimusDataset
from end2end import CRNN
from metrics import compute_metrics, ctc_greedy_decoder, split_sequence

random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # For reproducibility, but slower

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

_, val_split = torch.utils.data.random_split(dataset, [0.8, 0.2])


val_loader = torch.utils.data.DataLoader(
    val_split,
    batch_size=1,
    shuffle=False,
)

model = CRNN(num_channels=1, img_height=128, output_size=len(dataset.vocab.c2i)).to(
    device
)
model.load_state_dict(torch.load("models/std/best.pt", weights_only=True))
model.eval()
summary(model)

preds = []
gts = []
for i, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Testing"):
    x, y = x.to(device), y.to(device)
    y_hat = model(x)
    y_hat = y_hat.log_softmax(dim=-1)
    y_hat = ctc_greedy_decoder(y_hat, dataset.vocab)
    preds.extend(y_hat)
    y = [[dataset.vocab.i2c[i.item()] for i in sample] for sample in y]
    y = split_sequence(y)
    gts.extend(y)

metrics = compute_metrics(gts, preds)
print(f"Test Metrics:")
for k, v in metrics.items():
    print(f"    - {k}: {v}")

# Print random samples
index = random.randint(0, len(gts) - 1)
print(f"Ground truth - {gts[index]}")
print(f"Prediction - {preds[index]}")
