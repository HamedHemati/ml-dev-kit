from typing import Dict, Any
import random
import numpy as np
import torch

from ..algorithms import BaseAlgorithm
from .logging import init_logging, finalize_logging


def get_device(config: Dict[str, Any]) -> torch.device:
    if config["device"] == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    elif config["device"] == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    print(f"Compute device: {device}")

    return device


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


def initialize_experiment(config: Dict[str, Any]) -> None:
    config["device"] = get_device(config)
    set_manual_seed(config["seed"])
    if config["log"]:
        init_logging(config)


def finalize_experiment(algorithm: BaseAlgorithm) -> None:
    if algorithm.config["log"]:
        finalize_logging(algorithm)