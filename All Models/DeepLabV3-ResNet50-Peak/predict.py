# DeepLabv3-Resnet50-Peak Performance
"""
Prediction pipeline for DeepLabV3-ResNet50.

This file is intended for challenge submission. It keeps the same fixed input/output
paths as the provided baseline and loads weights from /app/model.pt.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode, Resize

from model import Model

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def preprocess(img: Image.Image) -> torch.Tensor:
    img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    img_tensor = (img_tensor - _MEAN) / _STD
    return img_tensor.unsqueeze(0)


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    if isinstance(pred, dict):
        pred = pred["out"]

    pred_max = torch.argmax(pred, dim=1, keepdim=True).float()
    pred_resized = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)
    prediction_numpy = pred_resized.cpu().numpy().squeeze().astype(np.uint8)
    return prediction_numpy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
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
            Image.fromarray(seg_pred).save(out_path)


if __name__ == "__main__":
    main()
