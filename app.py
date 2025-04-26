import sys

if "torch.__path__" in sys.modules:
    del sys.modules["torch.__path__"]

import json
from pathlib import Path
from typing import Callable, cast

import streamlit as st
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from utils.helpers import load_class_id_to_label_map


@st.cache_resource
def load_model(config: dict) -> nn.Module:
    weights_enum = getattr(models.ResNet18_Weights, config["model"]["weights"])
    model = models.resnet18(weights=weights_enum)
    model.fc = nn.Linear(model.fc.in_features, config["model"]["num_classes"])
    model.load_state_dict(
        torch.load(config["training"]["save_path"], map_location="cpu")
    )
    model.eval()
    return model


def load_config(config_path: Path = Path("config.json")) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


@st.cache_data
def get_label_map() -> dict[int, str]:
    return load_class_id_to_label_map()


def predict(
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
        probabilities = torch.softmax(output, dim=1)
        topk = torch.topk(probabilities, k=3)

    results = []
    for i in range(3):
        class_id = int(topk.indices[0][i].item())
        confidence = float(topk.values[0][i].item())
        label = label_map.get(class_id, f"Class {class_id}")
        results.append((label, confidence, class_id))

    return results


def main():
    st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")
    st.markdown(
        "<h1 style='text-align: center;'>Traffic Sign Classifier</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>Upload a traffic sign image and see the top-3 predictions</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        config = load_config()
        model = load_model(config)
        label_map = get_label_map()

        # 2-column layout, image narrower
        left_col, right_col = st.columns([0.4, 0.6])

        with left_col:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with right_col:
            st.markdown("### Top 3 Predictions")
            results = predict(image, model, label_map)

            for i, (label, confidence, class_id) in enumerate(results):
                confidence_percent = confidence * 100
                if i == 0:
                    st.markdown(
                        f"""
                        <div style='
                            padding: 12px;
                            background-color: rgba(0, 123, 255, 0.15);
                            border-left: 4px solid #007bff;
                            border-radius: 6px;
                            color: #ffffff;
                            font-weight: 500;
                        '>
                            <div><strong>{i+1}. {label}</strong></div>
                            <div style='margin-top: 4px;'>Class ID: <code style="color:#00ffaa;">{class_id}</code></div>
                            <div>Confidence: <span style="color:#00ffaa;">{confidence_percent:.2f}%</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"**{i+1}. {label}** (Class ID: {class_id})  \n"
                        f"Confidence: `{confidence_percent:.2f}%`"
                    )
                st.progress(confidence)


if __name__ == "__main__":
    main()
