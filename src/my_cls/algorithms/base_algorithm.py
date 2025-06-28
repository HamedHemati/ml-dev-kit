from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from torch.utils.data import Dataset

from ..models import get_model


class BaseAlgorithm(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config["device"])

        # Initialize model
        self.model = get_model(config)
        self.model.to(self.device)

        # Initialize logger
        self.log = config["log"]
        self.global_iter = 0

    @abstractmethod
    def train_epoch(self, epoch: int, train_set: Dataset) -> None:
        pass

    @abstractmethod
    def validate(self, val_set) -> None:
        pass

    @abstractmethod
    def test(self, test_set: Dataset) -> None:
        pass
