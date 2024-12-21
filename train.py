"""
Simple training script for PyTorch models.
author: Adrián Roselló Pedraza (RosiYo)
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import random
from torchinfo import summary

from dataset import DidoneDataset, PrimusDataset
from batch_prep import ctc_batch_preparation
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

# limit samples to test the script
# train_split = torch.utils.data.Subset(train_split, range(10))
# val_split = torch.utils.data.Subset(val_split, range(5))


def train_collate_fn(batch):
    x, y = zip(*batch)
    input_lengths = torch.tensor([i.shape[2] for i in x], dtype=torch.int32)
    target_lengths = torch.tensor([len(y) for i in y], dtype=torch.int32)
    batch_prep = list(zip(x, input_lengths, y, target_lengths))
    return ctc_batch_preparation(batch_prep, dataset.vocab.c2i["<PAD>"])


train_loader = torch.utils.data.DataLoader(
    train_split,
    batch_size=24,
    collate_fn=train_collate_fn,
    shuffle=True,
)


val_loader = torch.utils.data.DataLoader(
    val_split,
    batch_size=1,
    shuffle=False,
)

model = CRNN(num_channels=1, img_height=128, output_size=len(dataset.vocab.c2i)).to(
    device
)
summary(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ctc_loss = torch.nn.CTCLoss(blank=dataset.vocab.c2i["<BLANK>"]).to(device)

patience = 25
best_loss = float("inf")
counter = 0
skip_val = 50

epoch = 0
while True:
    if counter == patience:
        break

    model.train()
    for i, (x, in_len, y, tgt_len) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="Training"
    ):
        optimizer.zero_grad()
        x, y, in_len, tgt_len = (
            x.to(device),
            y.to(device),
            in_len.to(device),
            tgt_len.to(device),
        )
        in_len = in_len // model.cnn.width_reduction
        y_hat = model(x)
        y_hat = y_hat.log_softmax(dim=-1)
        y_hat = y_hat.permute(1, 0, 2).contiguous()
        loss = ctc_loss(y_hat, y, in_len, tgt_len)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} - Batch {i} - Loss {loss.item()}")

    if epoch >= skip_val:
        preds = []
        gts = []
        for i, (x, y) in tqdm(
            enumerate(val_loader), total=len(val_loader), desc="Validation"
        ):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            y_hat = y_hat.log_softmax(dim=-1)
            y_hat = ctc_greedy_decoder(y_hat, dataset.vocab)
            preds.extend(y_hat)
            y = [[dataset.vocab.i2c[i.item()] for i in sample] for sample in y]
            y = split_sequence(y)
            gts.extend(y)

        metrics = compute_metrics(gts, preds)
        print(f"Epoch {epoch} - Validation - Metrics:")
        for k, v in metrics.items():
            print(f"    - {k}: {v}")

        loss = metrics["ser"]

        # Print random samples
        if True:
            index = random.randint(0, len(gts) - 1)
            print(f"Ground truth - {gts[index]}")
            print(f"Prediction - {preds[index]}")

        if loss < best_loss:
            os.makedirs("models/split", exist_ok=True)
            torch.save(model.state_dict(), f"models/split/best.pt")
            print(f"Model saved. SER: {loss}")
            best_loss = loss
            counter = 0
        else:
            counter += 1
            print(f"Not improving. Counter: {counter}")

        torch.save(model.state_dict(), f"models/split/last.pt")
    epoch += 1


print("Finished training")
print(f"Best SER: {best_loss}")
