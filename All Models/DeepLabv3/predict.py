from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)

from model import Model


IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(512, 1024), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    img = transform(img)
    img = img.unsqueeze(0)
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)

    prediction = Resize(
        size=original_shape,
        interpolation=InterpolationMode.NEAREST
    )(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        in_channels=3,
        n_classes=19,
        pretrained_backbone=False,  # at inference we load trained weights, not backbone init
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            pred = model(img_tensor)
            seg_pred = postprocess(pred, original_shape)

            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()