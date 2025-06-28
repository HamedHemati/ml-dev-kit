from .dummy_ds import DummyDataset


def get_dataset(config):
    if config["dataset"] == "dummy":
        train_set = DummyDataset(
            n_samples=config["n_train_samples"], **config["dataset_params"]
        )
        val_set = DummyDataset(
            n_samples=config["n_val_samples"], **config["dataset_params"]
        )
        test_set = DummyDataset(
            n_samples=config["n_test_samples"], **config["dataset_params"]
        )

        print(f"Loaded Dummy dataset ...")

        return (train_set, val_set, test_set)

    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")


__all__ = ["get_dataset"]
