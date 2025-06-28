import os
from typing import Dict, Any
import wandb
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn

from ..algorithms import BaseAlgorithm


def init_logging(config: Dict[str, Any]) -> None:
    wandb.init(project=config["wandb_project"], config=config, mode="online")
    run_name = wandb.run.name  # type: ignore

    # Add checkpoint dir to config
    checkpoint_dir = "./outputs/" + run_name + "/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    config["checkpoint_dir"] = checkpoint_dir  # type: ignore

    # Save config as a yaml file
    config_path = "./outputs/" + run_name + "/config.yaml"
    OmegaConf.save(config, config_path)


def finalize_logging(
    algorithm: BaseAlgorithm,
) -> None:
    wandb.finish()

    # Save the model checkpoint
    checkpoint_path = os.path.join(algorithm.config["checkpoint_dir"], "model.pt")
    torch.save(algorithm.model.state_dict(), checkpoint_path)


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return config  # type: ignore
