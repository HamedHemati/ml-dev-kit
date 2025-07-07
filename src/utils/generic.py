from typing import Dict, Any
import random
import numpy as np
import torch

from ..algorithms import BaseAlgorithm
from .logging import init_logging, finalize_logging


def set_manual_seed(seed: int) -> None:
    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Numpy
    np.random.seed(seed)

    # Python
    random.seed(seed)

    print(f"Seed set to {seed}")


def initialize_run(config: Dict[str, Any]) -> None:
    set_manual_seed(config["random_seed"])
    if config["log"]:
        init_logging(config)


def finalize_run(algorithm: BaseAlgorithm) -> None:
    if algorithm.config["log"]:
        finalize_logging(algorithm)
