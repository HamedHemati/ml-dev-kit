from torch.utils.data import Dataset
import torch
from typing import List, Tuple
import numpy as np


# Implementation of a dummy dataset to be used for testing purposes only
class DummyDataset(Dataset):
    def __init__(
        self, n_samples: int, inp_shape: Tuple[int, int], n_classes: int
    ) -> None:
        super().__init__()
        self.inp_shape = inp_shape
        self.n_classes = n_classes
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.randn(self.inp_shape, dtype=torch.float32)
        label = np.random.randint(0, self.n_classes)

        return {"image": image, "label": label}
