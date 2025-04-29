#!/usr/bin/env python3
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from ultralytics import YOLO

from utils.data_loader import get_dataloaders
from utils.helpers import get_dataset_size

# ── Logging setup ─────────────────────────────────────────────────────────────
logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"train_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────────────────────────


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


def train_resnet(cfg: dict) -> None:
    # device
    device = torch.device(
        "cuda"
        if (cfg["common"]["device"] == "auto" and torch.cuda.is_available())
        else cfg["common"]["device"]
    )
    logger.info(f"Using device: {device}")

    # dataloaders
    train_loader, test_loader = get_dataloaders(
        train_csv=Path(cfg["dataset"]["train_csv"]),
        test_csv=Path(cfg["dataset"]["test_csv"]),
        batch_size=cfg["common"]["batch_size"],
    )

    # build model
    weights_enum = getattr(models.ResNet18_Weights, cfg["resnet"]["pretrained_weights"])
    model = models.resnet18(weights=weights_enum)
    model.fc = nn.Linear(model.fc.in_features, cfg["resnet"]["num_classes"])
    model.to(device)

    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["common"]["learning_rate"])

    # training loop
    for epoch in range(1, cfg["common"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(
            f"[Epoch {epoch}/{cfg['common']['epochs']}] Train Loss: {train_loss:.4f}"
        )

    # save
    save_path = Path(cfg["resnet"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"[+] ResNet model saved to {save_path}")


def train_yolo(cfg: dict) -> None:
    # instantiate from pretrained YOLO stub
    yolo = YOLO(cfg["yolo"]["pretrained_weights"])
    run_name = f"yolov8_train_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    yolo.train(
        data=cfg["dataset"]["yolo_data"],
        epochs=cfg["common"]["epochs"],
        imgsz=cfg["common"]["imgsz"],
        batch=cfg["common"]["batch_size"],
        project="logs",
        workers=0,
        name=f"yolov8_train_{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )
    logger.info("[+] YOLOv8 classification training complete")

    # copy best.pt into models/
    src = Path("logs") / run_name / "weights" / "best.pt"
    dst = Path(cfg["yolo"]["finetuned_weights"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    logger.info(f"[+] Copied YOLOv8 best.pt to {dst}")


def main() -> None:
    cfg = load_config()

    # train ResNet if enabled
    if cfg.get("resnet", {}).get("enabled", False):
        logger.info("Starting ResNet18 training…")
        train_resnet(cfg)

    # train YOLO if enabled
    if cfg.get("yolo", {}).get("enabled", False):
        logger.info("Starting YOLOv8 classification training…")
        train_yolo(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with exception:")
        logger.info(e)
