# Traffic Sign Classifier

A deep learning model for classifying German traffic signs using the GTSRB dataset. Trained on 43 classes with a fine-tuned ResNet18 architecture using PyTorch.

## Training Configuration

- Architecture: ResNet18 (pretrained on ImageNet)
- Epochs: 10
- Batch size: 64
- Learning rate: 0.0001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Input size: 32x32 RGB
- Augmentations: Resize, random rotation, color jitter

## Evaluation Results

- Final Test Accuracy: **92.91%**
- Macro F1-score: **0.8948**
- Weighted F1-score: **0.9283**
- Perfect class-wise precision and recall on labels: 14, 16, 32
- Lower recall observed in rare classes: 0, 19, 21, 27
