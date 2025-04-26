import argparse
import json
import logging
from pathlib import Path
from typing import Callable, cast

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from utils.helpers import load_class_id_to_label_map

logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config.json")) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def predict(image_path: Path, config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform: Callable[[Image.Image], Tensor] = Compose(
        [
            Resize((32, 32)),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor_image = cast(Tensor, transform(image))
    input_tensor = tensor_image.unsqueeze(0).to(device)
    label_map = load_class_id_to_label_map()

    weights_enum = getattr(models.ResNet18_Weights, config["model"]["weights"])
    model = models.resnet18(weights=weights_enum)
    model.fc = nn.Linear(model.fc.in_features, config["model"]["num_classes"])
    model.load_state_dict(
        torch.load(config["training"]["save_path"], map_location=device)
    )
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        topk = torch.topk(probabilities, k=3)

    logger.info("Top 3 Predictions:")
    for i in range(3):
        pred_class = int(topk.indices[0][i].item())
        confidence = topk.values[0][i].item()
        label = label_map.get(pred_class, f"Class {pred_class}")
        logger.info(
            f"  {i+1}. {label} (id: {pred_class}) | Confidence: {confidence:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    args = parser.parse_args()

    config = load_config()
    predict(args.image, config)
