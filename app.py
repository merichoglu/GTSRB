import json
from pathlib import Path
from typing import Callable, cast

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from ultralytics import YOLO

from utils.helpers import load_class_id_to_label_map


# ── Config & Label Map ──────────────────────────────────────────────────────
@st.cache_data
def load_config(config_path: Path = Path("config.json")) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


@st.cache_data
def get_label_map() -> dict[int, str]:
    return load_class_id_to_label_map()


# ── Model Loaders ──────────────────────────────────────────────────────────
@st.cache_resource
def load_resnet_model(config: dict) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config["resnet"]["num_classes"])
    model.load_state_dict(torch.load(config["resnet"]["save_path"], map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_yolo_model(config: dict) -> YOLO:
    return YOLO(config["yolo"]["finetuned_weights"])


# ── Prediction Functions ───────────────────────────────────────────────────
def predict_resnet(
    image: Image.Image, model: nn.Module, label_map: dict[int, str]
) -> list[tuple[str, float, int]]:
    transform: Callable[[Image.Image], Tensor] = Compose(
        [
            Resize((32, 32)),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tensor_image = cast(Tensor, transform(image))
    input_tensor = tensor_image.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
    top3_idx = np.argsort(probs)[-3:][::-1]
    results = []
    for idx in top3_idx:
        class_id = int(idx)
        label = label_map.get(class_id, f"Class {class_id}")
        confidence = float(probs[idx])
        results.append((label, confidence, class_id))
    return results


def predict_yolo(
    image: Image.Image,
    model: YOLO,
    label_map: dict[int, str],
    config: dict,
    device: str = "cpu",
) -> list[tuple[str, float, int]]:
    arr = np.array(image)
    results = model.predict(
        source=[arr], imgsz=config["common"]["imgsz"], device=device, verbose=False
    )
    r = results[0]
    if r.probs is None:
        raise RuntimeError(
            "No classification probabilities returned; loaded wrong model?"
        )
    idx2label = {i: int(n) for i, n in model.names.items()}
    tensor = torch.as_tensor(r.probs.data)
    probs = tensor.cpu().numpy().flatten()
    top3_idx = np.argsort(probs)[-3:][::-1]
    out = []
    for idx in top3_idx:
        class_id = idx2label[int(idx)]
        label = label_map.get(class_id, f"Class {class_id}")
        confidence = float(probs[idx])
        out.append((label, confidence, class_id))
    return out


# ── Streamlit App ─────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")
    st.markdown(
        "<h1 style='text-align: center;'>Traffic Sign Classifier</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>Upload an image and get predictions from both ResNet18 and YOLOv8</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if not uploaded_file:
        return

    image = Image.open(uploaded_file).convert("RGB")
    config = load_config()
    label_map = get_label_map()
    resnet_model = load_resnet_model(config)
    yolo_model = load_yolo_model(config)

    col_h1, col_h2 = st.columns(2)
    col_h1.subheader("ResNet18 Predictions")
    col_h2.subheader("YOLOv8 Classifier Predictions")

    # predictions side by side
    col_res, col_yolo = st.columns(2)

    with col_res:
        st.image(image, use_container_width=True)
        res_results = predict_resnet(image, resnet_model, label_map)
        for i, (label, confidence, class_id) in enumerate(res_results):
            pct = confidence * 100
            st.write(f"{i+1}. **{label}** (ID: {class_id}) — {pct:.2f}%")
            st.progress(confidence)

    with col_yolo:
        st.image(image, use_container_width=True)
        yolo_results = predict_yolo(image, yolo_model, label_map, config)
        for i, (label, confidence, class_id) in enumerate(yolo_results):
            pct = confidence * 100
            st.write(f"{i+1}. **{label}** (ID: {class_id}) — {pct:.2f}%")
            st.progress(confidence)


if __name__ == "__main__":
    main()
