# Traffic Sign Classifier

A deep learning model for classifying German traffic signs using the GTSRB dataset. Implements both a fine‑tuned ResNet18 and a YOLOv8‑based classifier.

---

## Training Configuration

### ResNet18

- **Pretrained weights:** ImageNet (`IMAGENET1K_V1`)
- **Epochs:** 10
- **Batch size:** 32
- **Learning rate:** 0.001
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Input size:** 32×32 RGB (via `utils.data_loader` transforms)

### YOLOv8 Classification (YOLOv8n‑cls)

- **Pretrained weights:** `yolov8n-cls.pt`
- **Epochs:** 10
- **Batch size:** 32
- **Initial learning rate:** 0.01
- **Optimizer & loss:** Built-in YOLOv8 classification head
- **Input size:** 224×224
- **Augmentations:** mosaic, mixup, cutmix, random flips (configurable)

---

## Evaluation Results

### ResNet18

- **Test Accuracy:** 94.99%
- **Macro F1-score:** 0.9253
- **Weighted F1-score:** 0.9497
- _Perfect precision & recall on class:_ **32**
- _Lower performance on:_ **19**, **20**, **24**, **36**, **37**

### YOLOv8 Classification

- **Test Accuracy:** 96.74%
- **Macro F1-score:** 0.9223
- **Weighted F1-score:** 0.9672
- **Perfect on classes:** 0, 9, 10, 14, 16, 17, 24, 27, 28, 29, 32, 41
- **Lower recall on:** 20 (54.44%), 37 (13.33%), 36 (50.83%)

---

## Setup Instructions

Clone the repository and set up the environment:

```bash
git clone https://github.com/merichoglu/GTSRB.git
cd GTSRB

# create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Streamlit Inference UI

A Streamlit web app is provided to test trained models on custom images through a browser interface.

### Launch the app:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Features

- Upload any traffic sign image (`.png`, `.jpg`, `.jpeg`)
- View the uploaded image and top-3 predictions side by side
- Each prediction includes:
  - Human-readable label
  - Class ID
  - Confidence (percentage + progress bar)

---

## Notes

- ResNet18 weights saved to `models/resnet18.pth` after training
- YOLOv8 best checkpoint copied to `models/yolov8_best.pt`
- All training/evaluation logs are in the `logs/` directory
- Dataset must be placed in `data/raw/` and preprocessed via `utils/data_loader` before training
