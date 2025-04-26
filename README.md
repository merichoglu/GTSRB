# Traffic Sign Classifier

A deep learning model for classifying German traffic signs using the GTSRB dataset. Trained on 43 classes with a fine-tuned ResNet18 architecture using PyTorch.

---

## Training Configuration

- Architecture: ResNet18 (pretrained on ImageNet)
- Epochs: 10
- Batch size: 64
- Learning rate: 0.0001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Input size: 32x32 RGB
- Augmentations: Resize, random rotation, color jitter

---

## Evaluation Results

- Final Test Accuracy: **92.91%**
- Macro F1-score: **0.8948**
- Weighted F1-score: **0.9283**
- Perfect class-wise precision and recall on labels: 14, 16, 32
- Lower recall observed in rare classes: 0, 19, 21, 27

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

A Streamlit web app is provided to test the trained model on custom images through a browser interface.

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
  - Confidence as percentage and progress bar

---

## Notes

- Model weights are saved to `models/resnet18.pth` after training
- All training/evaluation logs are written to the `logs/` directory
- The dataset must be placed in `data/raw/` and preprocessed before training
