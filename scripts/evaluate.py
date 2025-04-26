import json
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from utils.data_loader import GTSRB, test_transforms

logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)

log_file = logs_dir / f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config.json")) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> None:
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    report = cast(
        str, classification_report(y_true, y_pred, digits=4, output_dict=False)
    )
    matrix = confusion_matrix(y_true, y_pred)

    logger.info("\n=== Classification Report ===\n" + report)
    logger.info("\n=== Confusion Matrix ===\n" + str(matrix))


def main() -> None:
    config = load_config()

    if config["training"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["training"]["device"])

    test_dataset = GTSRB(
        csv_path=Path(config["dataset"]["test_csv"]), transform=test_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    weights_enum = getattr(models.ResNet18_Weights, config["model"]["weights"])
    model = models.resnet18(weights=weights_enum)
    model.fc = nn.Linear(model.fc.in_features, config["model"]["num_classes"])
    model.load_state_dict(
        torch.load(config["training"]["save_path"], map_location=device)
    )
    model.to(device)

    logger.info("[*] Evaluating model...")
    evaluate(model, test_loader, device)
    logger.info("[+] Evaluation complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Evaluation failed with exception:")
        logger.info(e)
