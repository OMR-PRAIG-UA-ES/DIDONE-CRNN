import os
import gc
import random

from tqdm import tqdm
import cv2
import fire
from end2end import CRNN
from dataset import DidoneDataset
from batch_prep import ctc_batch_preparation
import torch
import numpy as np

mode = "std"
GRAD_CAM = f"analysis/{mode}"
os.makedirs(GRAD_CAM, exist_ok=True)

random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU setups
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # For reproducibility, but slower

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    Implementation adapted from:  https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter09_part03_interpreting-what-convnets-learn.ipynb
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        super(GradCAM, self).__init__()
        self.model = model
        self.device = device

    def make_gradcam(self, img: any) -> torch.Tensor:
        # Preprocess input image
        x = img.unsqueeze(0).to(self.device)

        # 1. Retrieving the gradients of the top predicted class:
        # Encoder (CNN)
        ypred_encoder = self.model.cnn(x)
        ypred_encoder = ypred_encoder.requires_grad_(
            True
        )  # Detach the tensor before calculating gradients
        # Prepare for RNN
        _, _, _, w = ypred_encoder.size()  # 1, 128, 64 / 16 = 4, w
        ypred = ypred_encoder.permute(0, 3, 1, 2).contiguous()
        ypred = ypred.reshape(1, w, self.model.decoder_input_size)
        # Decoder (RNN) -> ypred.shape = (batch_size, seq_len, num_classes)
        ypred = self.model.rnn(ypred)
        ypred = ypred.log_softmax(dim=-1)
        # Extract the top class channels
        top_class_channels = torch.topk(
            ypred, k=1, dim=-1, sorted=False
        ).values.squeeze()
        # Compute gradients
        torch.autograd.backward(
            top_class_channels, torch.clone(top_class_channels.detach())
        )
        # Extract gradients with respect to encoder output
        grads = ypred_encoder.grad

        # 2. Gradient pooling and channel-importance weighting:
        pooled_grads = torch.mean(
            grads, dim=(0, 2, 3)
        ).detach()  # pooled_grads.shape = (128,)
        ypred_encoder = ypred_encoder.detach()
        # Multiply pooled gradients with encoder output
        for i in range(ypred_encoder.shape[1]):
            ypred_encoder[:, i, :, :] *= pooled_grads[i]
        heatmap = torch.mean(ypred_encoder, dim=1).squeeze()

        # 3. Heatmap postprocessing:
        heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
        heatmap /= torch.max(heatmap)
        return {"heatmap": heatmap, "img": x[0].permute(1, 2, 0)}

    def get_and_save_gradcam_heatmap(self, img: any, grad_img_output_path: str):
        # Apply Grad-CAM
        gc_output = self.make_gradcam(img=img)
        heatmap = gc_output["heatmap"].cpu().numpy()
        img = gc_output["img"].repeat(1, 1, 3).cpu().numpy()

        # Combine heatmap with original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img_cam = heatmap + img
        img_cam = img_cam / np.max(img_cam)
        # img_cam = cv2.cvtColor(img_cam,cv2.COLOR_RGB2BGR)

        # Save Grad-CAM image
        cv2.imwrite(grad_img_output_path, np.uint8(255 * img_cam))

def run_grad_cam(checkpoint_path: str, num_batches: int = 1):
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    # Load dataset
    dataset = DidoneDataset("./datasets/didone", mode)

    def train_collate_fn(batch):
        x, y = zip(*batch)
        input_lengths = torch.tensor([i.shape[2] for i in x], dtype=torch.int32)
        target_lengths = torch.tensor([len(y) for i in y], dtype=torch.int32)
        batch_prep = list(zip(x, input_lengths, y, target_lengths))
        return ctc_batch_preparation(batch_prep, dataset.vocab.c2i["<PAD>"])

    _, val_split = torch.utils.data.random_split(dataset, [0.8, 0.2])

    # Model
    model = CRNN(num_channels=1, img_height=128, output_size=len(dataset.vocab.c2i)).to(
        device
    )
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    for param in model.parameters():
        param.requires_grad = False

    val_loader = torch.utils.data.DataLoader(
        val_split,
        batch_size=5,
        shuffle=False,
        collate_fn=train_collate_fn,
    )

    # Grad-CAM
    for i, batch in tqdm(enumerate(val_loader), total=num_batches, desc="Grad-CAM"):
        if i == num_batches:
            break
        samples, _, _, _ = batch
        for j, x in enumerate(samples):
            grad_cam = GradCAM(model=model.to(device), device=device)
            output_path = os.path.join(GRAD_CAM, f"grad_cam_{i}_{j}.png")
            grad_cam.get_and_save_gradcam_heatmap(
                img=x, grad_img_output_path=output_path
            )

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(run_grad_cam)
