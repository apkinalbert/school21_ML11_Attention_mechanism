import copy
from collections.abc import Callable

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import TrainConfig


StepFn = Callable[
    [nn.Module, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]

EpochCallback = Callable[[nn.Module, int], None]

def generation_step(
    model: nn.Module,
    latin: torch.Tensor,
    cyrillic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    names = latin[:, :-1]
    targets = latin[:, 1:]
    logits = model(names)

    return logits, targets

def translation_step(
    model: nn.Module,
    latin: torch.Tensor,
    cyrillic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_input = cyrillic[:, :-1]
    targets = cyrillic[:, 1:]
    logits = model(latin, target_input)

    return logits, targets


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    step_fn: StepFn,
    device: torch.device,
    pad_idx: int,
) -> float:
    model.train()

    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)

    for latin, cyrillic in train_loader:
        latin = latin.to(device)
        cyrillic = cyrillic.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, targets = step_fn(
            model, latin, cyrillic
        )

        num_tokens = (targets != pad_idx).sum()

        loss_sum = loss_fn(logits.transpose(1, 2), targets)
        (loss_sum / num_tokens).backward()

        optimizer.step()

        total_loss += loss_sum.detach()
        total_tokens += num_tokens

    return (total_loss / total_tokens).item()


@torch.no_grad()
def validate(
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    step_fn: StepFn,
    device: torch.device,
    pad_idx: int,
) -> float:
    model.eval()

    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)

    for latin, cyrillic in valid_loader:
        latin = latin.to(device)
        cyrillic = cyrillic.to(device)


        logits, targets = step_fn(
            model, latin, cyrillic
        )

        num_tokens = (targets != pad_idx).sum()

        loss_sum = loss_fn(logits.transpose(1, 2), targets)

        total_loss += loss_sum
        total_tokens += num_tokens

    return (total_loss / total_tokens).item()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    step_fn: StepFn,
    pad_idx: int,
    train_config: TrainConfig,
    on_epoch_end: EpochCallback | None = None,
) -> tuple[pd.DataFrame, int, float]:

    history = {
        'train_loss' : [],
        'valid_loss' : [],
    }

    best_epoch = 0
    best_loss = float('inf')
    epoch_without_improvement = 0
    best_state = None

    for epoch in range(train_config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, step_fn, train_config.device, pad_idx,
        )

        valid_loss = validate(
            model, valid_loader, loss_fn, step_fn, train_config.device, pad_idx,
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            epoch_without_improvement = 0

            best_state = copy.deepcopy(model.state_dict())

        else:
            epoch_without_improvement += 1

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if train_config.verbose == 2:
            print(
                f"Epoch {epoch + 1}/{train_config.epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"valid loss: {valid_loss:.4f} | "
            )

        if on_epoch_end is not None:
            on_epoch_end(model, epoch + 1)

        if train_config.patience is not None and epoch_without_improvement >= train_config.patience:
            if train_config.verbose == 2:
                print(f"Early stopping at epoch {epoch + 1}")
            break


    if best_state is None:
        raise RuntimeError("No valid checkpoint was saved")
    model.load_state_dict(best_state)

    return pd.DataFrame(history), best_epoch, best_loss