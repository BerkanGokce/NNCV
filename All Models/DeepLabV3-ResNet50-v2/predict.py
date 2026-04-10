# DeepLabV3-ResNet50-v2
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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


def predict_with_flip_tta(model: Model, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)["out"]
    flipped_x = torch.flip(x, dims=[3])
    flipped_logits = model(flipped_x)["out"]
    flipped_logits = torch.flip(flipped_logits, dims=[3])
    return 0.5 * (logits + flipped_logits)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    image_files = sorted(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).to(device)
            logits = predict_with_flip_tta(model, x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred).save(out_path)


if __name__ == "__main__":
    main()
