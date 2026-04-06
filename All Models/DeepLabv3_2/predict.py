# DeepLabv3 Fast
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
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
    return transform(img).unsqueeze(0)


def postprocess(pred: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:
    pred_labels = pred.argmax(dim=1, keepdim=True).to(torch.uint8)
    pred_labels = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_labels)
    return pred_labels.squeeze(0).squeeze(0).cpu().numpy()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        in_channels=3,
        n_classes=19,
        pretrained_backbone=False,
    )

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = sorted(Path(IMAGE_DIR).glob("*.png"))
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
