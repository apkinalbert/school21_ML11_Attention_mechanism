from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    device: torch.device
    epochs: int = 100
    patience: int | None = 5
    verbose: int = 2


@dataclass
class SearchConfig:
    n_trials: int = 20
    n_jobs: int = 1
    plot_importances: bool = True


@dataclass
class DataLoaders:
    train: DataLoader
    valid: DataLoader
    test: DataLoader