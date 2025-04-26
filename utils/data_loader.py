from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class GTSRB(Dataset):
    def __init__(
        self,
        csv_path: Path,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ) -> None:
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img_path = Path(self.data.iloc[idx]["Filename"])
        label = int(self.data.iloc[idx]["ClassId"])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


train_transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def get_dataloaders(
    train_csv: Path,
    test_csv: Path,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = GTSRB(train_csv, transform=train_transforms)
    test_dataset = GTSRB(test_csv, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
