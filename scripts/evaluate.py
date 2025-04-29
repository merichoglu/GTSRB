#!/usr/bin/env python3
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from ultralytics import YOLO

from utils.data_loader import GTSRB, test_transforms

# ── Logging setup ─────────────────────────────────────────────────────────────
logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"eval_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
# ───────────────────────────────────────────────────────────────────────────────


def load_config(config_path: Path = Path("config.json")) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def evaluate_resnet(cfg: dict, device: torch.device) -> None:
    test_dataset = GTSRB(
        csv_path=Path(cfg["dataset"]["test_csv"]), transform=test_transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["common"]["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    weights_enum = getattr(models.ResNet18_Weights, cfg["resnet"]["pretrained_weights"])
    model = models.resnet18(weights=weights_enum)
    model.fc = nn.Linear(model.fc.in_features, cfg["resnet"]["num_classes"])
    model.load_state_dict(torch.load(cfg["resnet"]["save_path"], map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    report = classification_report(y_true, y_pred, digits=4)
    matrix = confusion_matrix(y_true, y_pred)
    logger.info("\n=== ResNet Classification Report ===\n" + str(report))
    logger.info("\n=== ResNet Confusion Matrix ===\n" + str(matrix))


def evaluate_yolo(cfg: dict, device: torch.device) -> None:
    yolo = YOLO(cfg["yolo"]["finetuned_weights"])
    idx2label = {idx: int(name) for idx, name in yolo.names.items()}

    df = pd.read_csv(cfg["dataset"]["test_csv"])
    paths = df["Filename"].tolist()
    y_true = df["ClassId"].tolist()

    y_pred = []
    chunk_size = 512
    for i in range(0, len(paths), chunk_size):
        batch_paths = paths[i : i + chunk_size]
        results = yolo.predict(
            source=batch_paths,
            imgsz=cfg["common"]["imgsz"],
            batch=cfg["common"]["batch_size"],
            device=device,
            workers=0,
            verbose=False,
        )
        for r in results:
            assert r.probs is not None, "No classification probabilities returned"
            idx = int(r.probs.top1)
            class_id = idx2label[idx]
            y_pred.append(class_id)

    report = classification_report(y_true, y_pred, digits=4)
    matrix = confusion_matrix(y_true, y_pred)
    logger.info("\n=== YOLOv8 Classification Report ===\n" + str(report))
    logger.info("\n=== YOLOv8 Confusion Matrix ===\n" + str(matrix))


def main() -> None:
    cfg = load_config()
    device = torch.device(
        "cuda"
        if (cfg["common"]["device"] == "auto" and torch.cuda.is_available())
        else cfg["common"]["device"]
    )

    # eval ResNet if enabled
    if cfg.get("resnet", {}).get("enabled", False):
        logger.info("[*] Evaluating ResNet18 model...")
        evaluate_resnet(cfg, device)

    # eval YOLO if enabled
    if cfg.get("yolo", {}).get("enabled", False):
        logger.info("[*] Evaluating YOLOv8 classifier...")
        evaluate_yolo(cfg, device)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Evaluation failed with exception:")
        logger.info(e)
