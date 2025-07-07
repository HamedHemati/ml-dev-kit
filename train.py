import sys
import os
import hydra
from omegaconf import OmegaConf, DictConfig

from src.datasets import get_dataset
from src.utils.generic import initialize_run, finalize_run
from src.algorithms import get_algorithm


@hydra.main(config_path="configs", config_name="generic", version_base="1.3.2")
def main(config: DictConfig) -> None:
    # Convert config to dict
    config = OmegaConf.to_container(config, resolve=True)

    # Initialize experiment
    initialize_run(config)

    # Intialize algorithm and dataset
    algorithm = get_algorithm(config)
    train_set, val_set, test_set = get_dataset(config)

    # Train for multiple epochs
    for epoch in range(config["num_epochs"]):
        algorithm.train_epoch(epoch, train_set)
        if epoch % config["val_interval"] == 0:
            algorithm.validate(epoch, val_set)

    # Test the model after training the model
    algorithm.test(test_set)

    # Finalize experiment
    finalize_run(algorithm)


if __name__ == "__main__":
    main()
