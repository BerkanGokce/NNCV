# DeepLabV3-ResNet50-v2
import math
import os
import random
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image, ImageEnhance, ImageOps
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid

from model import Model

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
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class CityscapesSegmentationTrain(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        crop_size: Tuple[int, int] = (512, 1024),
        scale_min: float = 0.75,
        scale_max: float = 1.75,
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.5,
    ):
        self.base = Cityscapes(root=root, split=split, mode="fine", target_type="semantic")
        self.crop_h, self.crop_w = crop_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _resize_pair(image: Image.Image, target: Image.Image, new_h: int, new_w: int):
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

    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.color_jitter_prob:
            return image
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        return image

    def __getitem__(self, index: int):
        image, target = self.base[index]

        scale = random.uniform(self.scale_min, self.scale_max)
        new_h = max(1, int(round(image.height * scale)))
        new_w = max(1, int(round(image.width * scale)))
        image, target = self._resize_pair(image, target, new_h, new_w)
        image, target = self._pad_if_needed(image, target)
        image, target = self._random_crop(image, target)

        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        image = self._apply_color_jitter(image)

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - _MEAN) / _STD

        target_np = np.array(target, dtype=np.uint8)
        target_tensor = torch.from_numpy(target_np.astype(np.int64))
        target_tensor = ID_TO_TRAINID_LUT[target_tensor.clamp(0, 255)]
        return image_tensor, target_tensor


class CityscapesSegmentationVal(Dataset):
    def __init__(self, root: str, split: str):
        self.base = Cityscapes(root=root, split=split, mode="fine", target_type="semantic")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - _MEAN) / _STD

        target_np = np.array(target, dtype=np.uint8)
        target_tensor = torch.from_numpy(target_np.astype(np.int64))
        target_tensor = ID_TO_TRAINID_LUT[target_tensor.clamp(0, 255)]
        return image_tensor, target_tensor


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 19, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid_mask = (target != self.ignore_index).float()

        target_clamped = target.clone()
        target_clamped[target_clamped == self.ignore_index] = 0
        one_hot = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid_mask.unsqueeze(1)
        probs = probs * valid_mask.unsqueeze(1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        cardinality = torch.sum(probs + one_hot, dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        present = torch.sum(one_hot, dim=dims) > 0
        if present.any():
            return 1.0 - dice[present].mean()
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes: int = 19, ignore_index: int = 255, dice_weight: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.dice = SoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        total = ce_loss + self.dice_weight * dice_loss
        return total, ce_loss, dice_loss


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mean_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 19, ignore_index: int = 255) -> float:
    dices: List[float] = []
    valid_mask = targets != ignore_index
    for cls in range(num_classes):
        pred_c = (preds == cls) & valid_mask
        target_c = (targets == cls) & valid_mask
        pred_sum = pred_c.sum().item()
        target_sum = target_c.sum().item()
        if pred_sum == 0 and target_sum == 0:
            continue
        intersection = (pred_c & target_c).sum().item()
        dice = (2.0 * intersection) / max(pred_sum + target_sum, 1)
        dices.append(dice)
    return float(sum(dices) / len(dices)) if dices else 0.0


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in TRAIN_ID_TO_COLOR.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image


def build_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float) -> torch.optim.Optimizer:
    named_params = list(model.named_parameters())

    groups = {
        "stem_layer1": [],
        "layer2": [],
        "layer3": [],
        "layer4": [],
        "heads": [],
    }

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if "classifier" in name or "aux_classifier" in name:
            groups["heads"].append(param)
        elif ".backbone.conv1." in name or ".backbone.bn1." in name or ".backbone.layer1." in name:
            groups["stem_layer1"].append(param)
        elif ".backbone.layer2." in name:
            groups["layer2"].append(param)
        elif ".backbone.layer3." in name:
            groups["layer3"].append(param)
        elif ".backbone.layer4." in name:
            groups["layer4"].append(param)
        else:
            groups["stem_layer1"].append(param)

    param_groups = [
        {"params": groups["stem_layer1"], "lr": lr * 0.1},
        {"params": groups["layer2"], "lr": lr * 0.25},
        {"params": groups["layer3"], "lr": lr * 0.5},
        {"params": groups["layer4"], "lr": lr},
        {"params": groups["heads"], "lr": lr * 5.0},
    ]
    param_groups = [group for group in param_groups if len(group["params"]) > 0]

    return SGD(
        param_groups,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )


def forward_with_flip_tta(model: nn.Module, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    outputs = model(images)
    logits = outputs["out"]

    flipped_images = torch.flip(images, dims=[3])
    flipped_outputs = model(flipped_images)
    flipped_logits = torch.flip(flipped_outputs["out"], dims=[3])

    logits = 0.5 * (logits + flipped_logits)
    return {"out": logits}


def get_args_parser() -> ArgumentParser:
    parser = ArgumentParser("Training script for DeepLabV3-ResNet50 on Cityscapes")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-h", type=int, default=512)
    parser.add_argument("--resize-w", type=int, default=1024)
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.75)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--color-jitter-prob", type=float, default=0.5)
    parser.add_argument("--aux-loss-weight", type=float, default=0.4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--tta-val", action="store_true")
    parser.add_argument("--experiment-id", type=str, default="deeplabv3-resnet50-peak-v2")
    return parser


def log_visuals(images: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor, step: int):
    vis_predictions = convert_train_id_to_color(preds.unsqueeze(1).cpu())
    vis_labels = convert_train_id_to_color(labels.unsqueeze(1).cpu())
    predictions_img = make_grid(vis_predictions, nrow=min(4, len(vis_predictions)))
    labels_img = make_grid(vis_labels, nrow=min(4, len(vis_labels)))
    wandb.log(
        {
            "predictions": [wandb.Image(predictions_img.permute(1, 2, 0).numpy())],
            "labels": [wandb.Image(labels_img.permute(1, 2, 0).numpy())],
        },
        step=step,
    )


def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation", name=args.experiment_id, config=vars(args))

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_dataset = CityscapesSegmentationTrain(
        root=args.data_dir,
        split="train",
        crop_size=(args.resize_h, args.resize_w),
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        hflip_prob=args.hflip_prob,
        color_jitter_prob=args.color_jitter_prob,
    )
    valid_dataset = CityscapesSegmentationVal(root=args.data_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Valid batches per epoch: {len(valid_loader)}")

    model = Model(in_channels=3, n_classes=19, pretrained_backbone=True).to(device)
    criterion = SegmentationLoss(
        num_classes=19,
        ignore_index=255,
        dice_weight=args.dice_weight,
        label_smoothing=args.label_smoothing,
    )
    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay)
    scheduler = PolyLRScheduler(optimizer, total_steps=args.epochs * len(train_loader), power=0.9)

    best_valid_dice = -1.0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    final_model_path = os.path.join(output_dir, "final_model.pt")
    global_step = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04d}/{args.epochs:04d}")
        model.train()

        train_loss_meter = 0.0
        train_ce_meter = 0.0
        train_dice_meter = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                main_total, main_ce, main_dice = criterion(outputs["out"], labels)
                if "aux" in outputs and outputs["aux"] is not None:
                    aux_total, aux_ce, aux_dice = criterion(outputs["aux"], labels)
                    loss = main_total + args.aux_loss_weight * aux_total
                    ce_loss = main_ce + args.aux_loss_weight * aux_ce
                    dice_loss = main_dice + args.aux_loss_weight * aux_dice
                else:
                    loss = main_total
                    ce_loss = main_ce
                    dice_loss = main_dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            scheduler.step(global_step)

            train_loss_meter += float(loss.item())
            train_ce_meter += float(ce_loss.item())
            train_dice_meter += float(dice_loss.item())

            wandb.log(
                {
                    "train_loss": float(loss.item()),
                    "train_ce_loss": float(ce_loss.item()),
                    "train_dice_loss": float(dice_loss.item()),
                    "learning_rate": optimizer.param_groups[-1]["lr"],
                    "lr_stem_layer1": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        model.eval()
        valid_losses: List[float] = []
        valid_ces: List[float] = []
        valid_dices_losses: List[float] = []
        valid_scores: List[float] = []
        vis_preds: List[torch.Tensor] = []
        vis_labels: List[torch.Tensor] = []
        vis_images: List[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = forward_with_flip_tta(model, images) if args.tta_val else model(images)
                    total_loss, ce_loss, dice_loss = criterion(outputs["out"], labels)

                preds = outputs["out"].argmax(dim=1)
                score = compute_mean_dice(preds, labels, num_classes=19, ignore_index=255)

                valid_losses.append(float(total_loss.item()))
                valid_ces.append(float(ce_loss.item()))
                valid_dices_losses.append(float(dice_loss.item()))
                valid_scores.append(score)

                if len(vis_preds) < 8:
                    vis_preds.append(preds[0].cpu())
                    vis_labels.append(labels[0].cpu())
                    vis_images.append(images[0].cpu())

        valid_loss = float(np.mean(valid_losses)) if valid_losses else math.inf
        valid_ce = float(np.mean(valid_ces)) if valid_ces else math.inf
        valid_dice_loss = float(np.mean(valid_dices_losses)) if valid_dices_losses else math.inf
        valid_mean_dice = float(np.mean(valid_scores)) if valid_scores else 0.0

        wandb.log(
            {
                "epoch_train_loss": train_loss_meter / max(len(train_loader), 1),
                "epoch_train_ce_loss": train_ce_meter / max(len(train_loader), 1),
                "epoch_train_dice_loss": train_dice_meter / max(len(train_loader), 1),
                "valid_loss": valid_loss,
                "valid_ce_loss": valid_ce,
                "valid_dice_loss": valid_dice_loss,
                "valid_mean_dice": valid_mean_dice,
            },
            step=global_step,
        )

        print(
            f"valid_loss={valid_loss:.4f} | valid_ce={valid_ce:.4f} | "
            f"valid_dice_loss={valid_dice_loss:.4f} | valid_mean_dice={valid_mean_dice:.4f}"
        )

        if vis_preds:
            log_visuals(torch.stack(vis_images), torch.stack(vis_preds), torch.stack(vis_labels), global_step)

        if valid_mean_dice > best_valid_dice:
            best_valid_dice = valid_mean_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} with valid_mean_dice={best_valid_dice:.4f}")

    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Best valid_mean_dice={best_valid_dice:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
