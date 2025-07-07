import wandb
from tqdm import tqdm
from typing import Dict, Any
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from .base_algorithm import BaseAlgorithm


class SupervisedSegmentation(BaseAlgorithm):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__(config)

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, epoch: int, train_set: Dataset) -> None:
        self.model.train()
        print(f"Training epoch {epoch}")

        # Initialize data loader
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )

        # Initialize progress bar
        train_loader_progress = tqdm(train_loader)
        total_loss = 0.0
        for batch in train_loader_progress:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Calculate loss and update model parameters
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            train_loader_progress.set_description(f"Step loss: {loss.item():.3f}")
            self.global_iter += 1
            if self.log:
                wandb.log({"Step loss": loss.item()}, step=self.global_iter)

        # Log epoch loss
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{self.config['num_epochs']}, Loss: {epoch_loss:0.3f}")
        if self.log:
            wandb.log({"Epoch loss": epoch_loss}, step=self.global_iter)
            wandb.log({"Epoch": epoch}, step=self.global_iter)

    def validate(self, epoch: int, val_set: Dataset) -> None:
        """Validates model on a given validation set."""
        print(f"Running validation")
        self.model.eval()
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=128,
            shuffle=False,
            num_workers=self.config["num_workers"],
        )
        val_loader_progress = tqdm(val_loader)
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader_progress:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Compute loss
                preds = self.model(images)
                loss = self.criterion(preds, labels)

                # Logging
                total_loss += loss.item()

        # Log validation loss
        avg_val_loss = total_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:0.3f}")
        if self.log:
            wandb.log({"Validation loss": avg_val_loss}, step=self.global_iter)

    def test(self, test_set: Dataset) -> None:
        pass