import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from rockart_model import build_rockart

Image.MAX_IMAGE_PIXELS = None

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match validation transforms in train_rockart_model.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def main():
    ckpt_path = BASE_DIR / "petroglyphnet.pth"
    raw = _load_checkpoint(ckpt_path)

    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        class_names = raw["classes"]
    else:
        state_dict = raw
        with open(BASE_DIR / "class_names.json") as f:
            class_names = json.load(f)

    model = build_rockart(num_classes=len(class_names))
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    test_dir = r"D:\Sunayana\PHD\Project\CNN\Datasets\test"

    for img_file in sorted(os.listdir(test_dir)):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(test_dir, img_file)
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)[0]
                pred = int(out.argmax(1).item())

            top_p, top_i = probs.topk(min(3, len(class_names)))
            top_str = ", ".join(
                f"{class_names[i]} {p:.3f}" for p, i in zip(top_p.tolist(), top_i.tolist())
            )
            print(img_file, "→", class_names[pred], f"  (top: {top_str})")

if __name__ == "__main__":
    main()