from typing import Dict, Any
import torch.nn as nn
import numpy as np

from .mlp import MLP


def get_model(config: Dict[str, Any]) -> nn.Module:
    if config["model"] == "mlp":
        return MLP(
            inp_dim=np.prod(config["dataset_params"]["inp_shape"]),
            n_classes=config["dataset_params"]["n_classes"],
        )
    else:
        raise ValueError(f"Model {config['model']} is not implemented.")


__all__ = ["get_model"]
