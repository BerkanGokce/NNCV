# DeepLabv3 - Fast

import math
import os
import random
from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import Normalize, ToDtype, ToImage
from torchvision.utils import make_grid

from model import Model


id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)


class CityscapesTrainTransform:
    def __init__(self, output_size: Tuple[int, int], scale_range: Tuple[float, float], hflip_prob: float) -> None:
        self.output_h, self.output_w = output_size
        self.scale_min, self.scale_max = scale_range
        self.hflip_prob = hflip_prob
        self.to_image = ToImage()
        self.to_float = ToDtype(torch.float32, scale=True)
        self.normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image: Image.Image, label: Image.Image):
        scale = random.uniform(self.scale_min, self.scale_max)
        scaled_h = max(self.output_h, int(round(image.height * scale)))
        scaled_w = max(self.output_w, int(round(image.width * scale)))

        image = image.resize((scaled_w, scaled_h), resample=Image.BILINEAR)
        label = label.resize((scaled_w, scaled_h), resample=Image.NEAREST)

        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        max_top = scaled_h - self.output_h
        max_left = scaled_w - self.output_w
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0

        image = image.crop((left, top, left + self.output_w, top + self.output_h))
        label = label.crop((left, top, left + self.output_w, top + self.output_h))

        image = self.to_image(image)
        image = self.to_float(image)
        image = self.normalize(image)

        label = torch.from_numpy(np.array(label, dtype=np.int64)).unsqueeze(0)
        return image, label


class CityscapesEvalTransform:
    def __init__(self, output_size: Tuple[int, int]) -> None:
        self.output_h, self.output_w = output_size
        self.to_image = ToImage()
        self.to_float = ToDtype(torch.float32, scale=True)
        self.normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image: Image.Image, label: Image.Image):
        image = image.resize((self.output_w, self.output_h), resample=Image.BILINEAR)
        label = label.resize((self.output_w, self.output_h), resample=Image.NEAREST)

        image = self.to_image(image)
        image = self.to_float(image)
        image = self.normalize(image)

        label = torch.from_numpy(np.array(label, dtype=np.int64)).unsqueeze(0)
        return image, label


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, transform) -> None:
        self.dataset = Cityscapes(root, split=split, mode="fine", target_type="semantic")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return self.transform(image, label)


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    label_img = label_img.clone()
    for city_id, train_id in id_to_trainid.items():
        label_img[label_img == city_id] = train_id
    return label_img


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for channel in range(3):
            color_image[:, channel][mask] = color[channel]

    return color_image


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    indices = target * num_classes + pred
    hist = torch.bincount(indices, minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes)


def compute_miou(confmat: torch.Tensor) -> float:
    intersection = torch.diag(confmat)
    union = confmat.sum(dim=1) + confmat.sum(dim=0) - intersection
    valid = union > 0
    iou = intersection[valid].float() / union[valid].float()
    return iou.mean().item() if valid.any() else 0.0


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch DeepLabV3 model")

    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="deeplabv3-resnet50-fast")
    parser.add_argument("--resize-h", type=int, default=512)
    parser.add_argument("--resize-w", type=int, default=1024)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.25)
    parser.add_argument("--hflip-prob", type=float, default=0.5)

    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = CityscapesTrainTransform(
        output_size=(args.resize_h, args.resize_w),
        scale_range=(args.scale_min, args.scale_max),
        hflip_prob=args.hflip_prob,
    )
    eval_transform = CityscapesEvalTransform(output_size=(args.resize_h, args.resize_w))

    train_dataset = SegmentationDataset(args.data_dir, split="train", transform=train_transform)
    valid_dataset = SegmentationDataset(args.data_dir, split="val", transform=eval_transform)

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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Model(
        in_channels=3,
        n_classes=19,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    total_iters = args.epochs * len(train_loader)
    current_iter = 0

    best_valid_miou = -1.0
    best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")
        model.train()

        for step, (images, labels) in enumerate(train_loader):
            labels = convert_to_train_id(labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().squeeze(1)

            poly_lr = args.lr * (1 - current_iter / total_iters) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = poly_lr

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": poly_lr,
                    "epoch": epoch + 1,
                },
                step=current_iter,
            )
            current_iter += 1

        model.eval()
        confmat = torch.zeros((19, 19), dtype=torch.int64)
        valid_losses = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_loader):
                labels = convert_to_train_id(labels)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long().squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

                predictions = outputs.argmax(dim=1)
                confmat += compute_confusion_matrix(predictions.cpu(), labels.cpu(), num_classes=19)

                if batch_idx == 0:
                    pred_vis = convert_train_id_to_color(predictions.unsqueeze(1).cpu())
                    label_vis = convert_train_id_to_color(labels.unsqueeze(1).cpu())

                    pred_grid = make_grid(pred_vis, nrow=4).permute(1, 2, 0).numpy()
                    label_grid = make_grid(label_vis, nrow=4).permute(1, 2, 0).numpy()

                    wandb.log(
                        {
                            "predictions": [wandb.Image(pred_grid)],
                            "labels": [wandb.Image(label_grid)],
                        },
                        step=current_iter,
                    )

        valid_loss = sum(valid_losses) / len(valid_losses)
        valid_miou = compute_miou(confmat)

        wandb.log(
            {
                "valid_loss": valid_loss,
                "valid_mIoU": valid_miou,
                "epoch": epoch + 1,
            },
            step=current_iter,
        )

        if valid_miou > best_valid_miou:
            best_valid_miou = valid_miou
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = os.path.join(
                output_dir,
                f"best_model-epoch={epoch + 1:04}-miou={valid_miou:.4f}.pt",
            )
            torch.save(model.state_dict(), best_model_path)

        print(f"  valid_loss={valid_loss:.4f} | valid_mIoU={valid_miou:.4f}")

    final_model_path = os.path.join(output_dir, f"final_model-epoch={args.epochs:04}-miou={best_valid_miou:.4f}.pt")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
