# DeepLabv3-Resnet50-Peak Performance
"""
Training script for DeepLabV3-ResNet50 on Cityscapes.

Design goals:
- Keep the overall structure similar to the provided U-Net training script.
- Preserve the main W&B logging keys: train_loss, valid_loss, learning_rate, epoch,
  predictions, labels.
- Use a stronger and more standard Cityscapes training recipe than the baseline:
  SGD + momentum + polynomial learning-rate decay + random scale/crop/flip.
"""

import math
import os
import random
from argparse import ArgumentParser
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image, ImageOps
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid

from model import Model


# -----------------------------
# Label conversion utilities
# -----------------------------
ID_TO_TRAINID = {cls.id: cls.train_id for cls in Cityscapes.classes}
TRAIN_ID_TO_COLOR = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
TRAIN_ID_TO_COLOR[255] = (0, 0, 0)


def build_id_to_trainid_lut() -> torch.Tensor:
    lut = torch.full((256,), 255, dtype=torch.long)
    for k, v in ID_TO_TRAINID.items():
        if 0 <= k < 256:
            lut[k] = v
    return lut


ID_TO_TRAINID_LUT = build_id_to_trainid_lut()


# -----------------------------
# Dataset
# -----------------------------
class CityscapesSegmentation(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        crop_size: Tuple[int, int] = (512, 1024),
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        hflip_prob: float = 0.5,
        train: bool = True,
    ):
        self.base = Cityscapes(
            root=root,
            split=split,
            mode="fine",
            target_type="semantic",
        )
        self.crop_h, self.crop_w = crop_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.hflip_prob = hflip_prob
        self.train = train
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.base)

    def _resize_pair(self, image: Image.Image, target: Image.Image, new_h: int, new_w: int):
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        target = target.resize((new_w, new_h), resample=Image.NEAREST)
        return image, target

    def _pad_if_needed(self, image: Image.Image, target: Image.Image):
        pad_h = max(0, self.crop_h - image.height)
        pad_w = max(0, self.crop_w - image.width)
        if pad_h > 0 or pad_w > 0:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            target = ImageOps.expand(target, border=(0, 0, pad_w, pad_h), fill=255)
        return image, target

    def _random_crop(self, image: Image.Image, target: Image.Image):
        max_top = image.height - self.crop_h
        max_left = image.width - self.crop_w
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        box = (left, top, left + self.crop_w, top + self.crop_h)
        return image.crop(box), target.crop(box)

    def _center_crop_or_resize(self, image: Image.Image, target: Image.Image):
        # Validation path: resize so the shorter side is at least crop size, then center-crop.
        scale = max(self.crop_h / image.height, self.crop_w / image.width)
        new_h = int(round(image.height * scale))
        new_w = int(round(image.width * scale))
        image, target = self._resize_pair(image, target, new_h, new_w)
        image, target = self._pad_if_needed(image, target)

        top = max(0, (image.height - self.crop_h) // 2)
        left = max(0, (image.width - self.crop_w) // 2)
        box = (left, top, left + self.crop_w, top + self.crop_h)
        return image.crop(box), target.crop(box)

    def __getitem__(self, index: int):
        image, target = self.base[index]

        if self.train:
            scale = random.uniform(self.scale_min, self.scale_max)
            new_h = max(1, int(round(image.height * scale)))
            new_w = max(1, int(round(image.width * scale)))
            image, target = self._resize_pair(image, target, new_h, new_w)
            image, target = self._pad_if_needed(image, target)
            image, target = self._random_crop(image, target)

            if random.random() < self.hflip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            image, target = self._center_crop_or_resize(image, target)

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - self.mean) / self.std

        target_np = np.array(target, dtype=np.uint8)
        target_tensor = torch.from_numpy(target_np.astype(np.int64))
        target_tensor = ID_TO_TRAINID_LUT[target_tensor.clamp(0, 255)]

        return image_tensor, target_tensor


# -----------------------------
# Metrics and visualization
# -----------------------------
def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in TRAIN_ID_TO_COLOR.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def compute_mean_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 19, ignore_index: int = 255) -> float:
    dices = []
    valid_mask = targets != ignore_index

    for cls in range(num_classes):
        pred_c = (preds == cls) & valid_mask
        target_c = (targets == cls) & valid_mask

        pred_sum = pred_c.sum().item()
        target_sum = target_c.sum().item()

        if pred_sum == 0 and target_sum == 0:
            continue

        intersection = (pred_c & target_c).sum().item()
        denom = pred_sum + target_sum
        dice = (2.0 * intersection) / max(denom, 1)
        dices.append(dice)

    if not dices:
        return 0.0
    return float(sum(dices) / len(dices))


class PolyLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, power: float = 0.9, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.power = power
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, current_step: int):
        factor = (1.0 - min(current_step, self.total_steps) / self.total_steps) ** self.power
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = max(base_lr * factor, self.min_lr)


# -----------------------------
# CLI
# -----------------------------
def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch DeepLabV3-ResNet50 model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resize-h", type=int, default=512, help="Training/validation crop height")
    parser.add_argument("--resize-w", type=int, default=1024, help="Training/validation crop width")
    parser.add_argument("--scale-min", type=float, default=0.5, help="Minimum random scaling factor")
    parser.add_argument("--scale-max", type=float, default=2.0, help="Maximum random scaling factor")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--aux-loss-weight", type=float, default=0.4, help="Weight for DeepLab auxiliary loss")
    parser.add_argument("--experiment-id", type=str, default="deeplabv3-resnet50", help="Experiment ID for Weights & Biases")
    return parser


# -----------------------------
# Main training loop
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float):
    # Standard segmentation setup: lower LR for backbone, higher LR for classifier/aux head.
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "aux_classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = SGD(
        [
            {"params": backbone_params, "lr": lr},
            {"params": classifier_params, "lr": lr * 10.0},
        ],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
    )
    return optimizer


def main(args):
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    crop_size = (args.resize_h, args.resize_w)
    train_dataset = CityscapesSegmentation(
        root=args.data_dir,
        split="train",
        crop_size=crop_size,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        hflip_prob=args.hflip_prob,
        train=True,
    )
    valid_dataset = CityscapesSegmentation(
        root=args.data_dir,
        split="val",
        crop_size=crop_size,
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Model(in_channels=3, n_classes=19, pretrained_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = create_optimizer(model, args.lr, args.momentum, args.weight_decay)
    total_steps = args.epochs * len(train_dataloader)
    scheduler = PolyLRScheduler(optimizer, total_steps=total_steps, power=0.9)

    best_valid_loss = float("inf")
    current_best_model_path = None
    global_step = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs: Dict[str, torch.Tensor] = model(images)
                main_loss = criterion(outputs["out"], labels)
                aux_loss = criterion(outputs["aux"], labels) if "aux" in outputs and outputs["aux"] is not None else 0.0
                loss = main_loss + args.aux_loss_weight * aux_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            scheduler.step(global_step)

            wandb.log(
                {
                    "train_loss": float(loss.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        model.eval()
        with torch.no_grad():
            losses = []
            dice_scores = []

            for i, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs: Dict[str, torch.Tensor] = model(images)
                    main_loss = criterion(outputs["out"], labels)
                    aux_loss = criterion(outputs["aux"], labels) if "aux" in outputs and outputs["aux"] is not None else 0.0
                    loss = main_loss + args.aux_loss_weight * aux_loss

                losses.append(float(loss.item()))
                preds = outputs["out"].argmax(dim=1)
                dice_scores.append(compute_mean_dice(preds, labels, num_classes=19, ignore_index=255))

                if i == 0:
                    predictions = preds.unsqueeze(1)
                    label_vis = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions.cpu())
                    label_vis = convert_train_id_to_color(label_vis.cpu())

                    predictions_img = make_grid(predictions, nrow=min(4, predictions.shape[0]))
                    labels_img = make_grid(label_vis, nrow=min(4, label_vis.shape[0]))

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log(
                        {
                            "predictions": [wandb.Image(predictions_img)],
                            "labels": [wandb.Image(labels_img)],
                        },
                        step=global_step,
                    )

            valid_loss = sum(losses) / max(len(losses), 1)
            valid_mean_dice = sum(dice_scores) / max(len(dice_scores), 1)
            wandb.log(
                {
                    "valid_loss": valid_loss,
                    "valid_mean_dice": valid_mean_dice,
                },
                step=global_step,
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path and os.path.exists(current_best_model_path):
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt",
                )
                torch.save(model.state_dict(), current_best_model_path)

    print("Training complete!")
    final_model_path = os.path.join(output_dir, f"final_model-epoch={args.epochs - 1:04}-val_loss={best_valid_loss:.4f}.pt")
    torch.save(model.state_dict(), final_model_path)
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
