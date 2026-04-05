import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode
)

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}


def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch DeepLabV3 model")

    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="deeplabv3-resnet50", help="Experiment ID for Weights & Biases")
    parser.add_argument("--resize-h", type=int, default=512, help="Training image height")
    parser.add_argument("--resize-w", type=int, default=1024, help="Training image width")
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use ImageNet-pretrained ResNet50 backbone")

    return parser


def main(args):
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ImageNet normalization because backbone is ImageNet-pretrained
    img_transform = Compose([
        ToImage(),
        Resize((args.resize_h, args.resize_w), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    target_transform = Compose([
        ToImage(),
        Resize((args.resize_h, args.resize_w), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
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

    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    # simple polynomial-like decay per epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9
    )

    best_valid_loss = float("inf")
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1:04}/{args.epochs:04}")

        # --------------------
        # Training
        # --------------------
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # --------------------
        # Validation
        # --------------------
        model.eval()
        with torch.no_grad():
            losses = []

            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)
                    predictions = predictions.unsqueeze(1)
                    labels_vis = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels_vis = convert_train_id_to_color(labels_vis)

                    predictions_img = make_grid(predictions.cpu(), nrow=4)
                    labels_img = make_grid(labels_vis.cpu(), nrow=4)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

            valid_loss = sum(losses) / len(losses)

            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                if current_best_model_path and os.path.exists(current_best_model_path):
                    os.remove(current_best_model_path)

                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)

        scheduler.step()

    print("Training complete!")

    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt"
        )
    )

    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)