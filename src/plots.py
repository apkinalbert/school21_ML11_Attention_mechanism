import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
from optuna.visualization.matplotlib import plot_param_importances


def plot_history(
    history: pd.DataFrame,
    best_epoch: int,
    name: str,
) -> None:

    fig, ax = plt.subplots(
        1, 1, figsize=(8,6),
    )

    epochs = range(1, len(history) + 1)

    ax.plot(epochs, history['train_loss'], label='Train')
    ax.plot(epochs, history['valid_loss'], label='Valid')
    ax.axvline(x=best_epoch, linestyle='--', label='Best epoch')
    ax.set_title(f"Loss {name}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_tuning_results(
    study: optuna.study.Study,
) -> None:
    plot_param_importances(study)
    plt.tight_layout()
    plt.show()


def plot_positional_encoding(
    plots: list[tuple[torch.Tensor, int, str]],
) -> None:
    if not plots:
        raise ValueError("plots must contain at least one item")

    values_list = [
        weights[:n_positions].detach().cpu().numpy()
        for weights, n_positions, title in plots
    ]

    limit = max(abs(values).max() for values in values_list)
    if limit == 0:
        limit = 1.0

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(plots),
        figsize=(6 * len(plots), 5),
        squeeze=False,
        layout='constrained',
    )

    for ax, values, (_, _, title) in zip(
        axes[0], values_list, plots
    ):
        heatmap = ax.imshow(
            values,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-limit,
            vmax=limit,
        )

        ax.set_xlabel('Embedding dimension')
        ax.set_ylabel('Token position')
        ax.set_title(title)

    fig.colorbar(
        heatmap,
        ax=axes[0].tolist(),
        label='Weight value',
    )

    plt.show()


def plot_attention(
    weights: torch.Tensor,
    source_tokens: list[str],
    query_tokens: list[str],
    title: str = 'Attention',
    subplot_titles: list[str] | None = None,
) -> None:
    values = weights.detach().cpu()

    if values.ndim == 2:
        values = values.unsqueeze(0)

    if values.ndim != 3:
        raise ValueError("Expected [T_query, T_source] or [heads, T_query, T_source]")

    n_heads, n_queries, n_sources = values.shape

    if n_queries != len(query_tokens) or n_sources != len(source_tokens):
        raise ValueError("Token labels must match attention matrix dimensions")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_heads,
        figsize=(6 * n_heads, 6),
        squeeze=False,
        layout='constrained',
    )

    for head_idx, ax in enumerate(axes[0]):
        heatmap = ax.imshow(
            values[head_idx].numpy(),
            aspect='auto',
            cmap='Blues',
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(
            range(n_sources),
            labels=source_tokens,
            rotation=90,
        )
        ax.set_yticks(range(n_queries), labels=query_tokens)

        ax.set_xlabel('English tokens')
        ax.set_ylabel('Russian decoder input')
        if subplot_titles is None:
            ax.set_title(f'Head {head_idx + 1}')
        else:
            ax.set_title(subplot_titles[head_idx])

    fig.suptitle(title)
    fig.colorbar(heatmap, ax=axes[0].tolist(), label='Attention weight')

    plt.show()