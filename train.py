import sys
import os
import hydra
from omegaconf import OmegaConf, DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from my_cls.datasets import get_dataset
from my_cls.utils.generic import initialize_experiment, finalize_experiment
from my_cls.algorithms import get_algorithm


@hydra.main(config_path="configs", config_name="generic", version_base="1.3.2")
def main(config: DictConfig) -> None:
    # Convert config to dict
    config = OmegaConf.to_container(config, resolve=True)

    # Initialize experiment
    initialize_experiment(config)

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
    finalize_experiment(algorithm)


if __name__ == "__main__":
    main()
