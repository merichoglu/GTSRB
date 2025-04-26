from typing import Sized, cast

from torch.utils.data import DataLoader


def get_dataset_size(dataloader: DataLoader) -> int:
    return len(cast(Sized, dataloader.dataset))
