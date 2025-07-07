from typing import Dict, Any

from .base_algorithm import BaseAlgorithm
from .supervised_segmentation import SupervisedSegmentation


def get_algorithm(config: Dict[str, Any]) -> BaseAlgorithm:
    if config["algorithm"] == "supervised_classification":
        return SupervisedSegmentation(config)
    else:
        raise ValueError(f"Algorithm {config.algorithm} is not implemented.")