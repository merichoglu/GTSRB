import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from utils.data_loader import get_dataloaders
from utils.helpers import get_dataset_size

logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)

log_file = logs_dir / f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config.json")) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    dataset_size = get_dataset_size(dataloader)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / dataset_size


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    dataset_size = get_dataset_size(dataloader)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = running_loss / dataset_size
    accuracy = correct / dataset_size
    return avg_loss, accuracy


def main() -> None:
    config = load_config()

    if config["training"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["training"]["device"])

    train_loader, test_loader = get_dataloaders(
        train_csv=Path(config["dataset"]["train_csv"]),
        test_csv=Path(config["dataset"]["test_csv"]),
        batch_size=config["training"]["batch_size"],
    )

    if config["model"]["backbone"] == "resnet18":
        weights_enum = getattr(models.ResNet18_Weights, config["model"]["weights"])
        model = models.resnet18(weights=weights_enum)
        model.fc = nn.Linear(model.fc.in_features, config["model"]["num_classes"])
    else:
        raise NotImplementedError("Only resnet18 is supported for now.")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        logger.info(
            f"[Epoch {epoch}/{config['training']['epochs']}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )

    save_path = Path(config["training"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"[+] Model saved to {save_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with exception:")
        logger.info(e)
