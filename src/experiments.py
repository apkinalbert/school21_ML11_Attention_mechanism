import copy
import math
from collections.abc import Callable
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from torch.utils.data import DataLoader

from configs import DataLoaders, SearchConfig, TrainConfig
from models import (
    AttentionDecoder,
    AttentionSeq2Seq,
    Decoder,
    Encoder,
    Seq2Seq,
)
from plots import plot_attention, plot_history, plot_tuning_results
from train import EpochCallback, StepFn, fit, validate


SuggestParamsFn = Callable[
    [optuna.trial.Trial],
    tuple[dict[str, int | float], float],
]


def run_experiment(
    name: str,
    build_model: Callable[..., nn.Module],
    model_params: dict[str, int | float],
    loaders: DataLoaders,
    step_fn: StepFn,
    pad_idx: int,
    train_config: TrainConfig,
    lr: float = 0.001,
    plot: bool = False,
    on_epoch_end: EpochCallback | None = None,
) -> dict[str, Any]:

    if train_config.verbose == 2 or train_config.verbose == 1:
        print(f"\n{'=' * 60}")
        print(f"Experiment: {name}")
        print(f"{'=' * 60}")

    model = build_model(**model_params)
    model.to(train_config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

    history, best_epoch, best_valid_loss = fit(
        model=model, 
        train_loader=loaders.train,
        valid_loader=loaders.valid,
        optimizer=optimizer,
        loss_fn=loss_fn,
        step_fn=step_fn,
        pad_idx=pad_idx,
        train_config=train_config,
        on_epoch_end=on_epoch_end,
    )

    if plot:
        plot_history(
            history=history,
            best_epoch=best_epoch,
            name=name
        )

    result = {
        'name' : name,
        'model' : model,
        'history' : history,
        'best_epoch' : best_epoch,
        'best_valid_loss' : best_valid_loss
    }

    return result


def suggest_encoder_params(
    trial: optuna.trial.Trial,
) -> tuple[dict[str, int | float], float]:
    
    model_params = {
        'embedding_dim': trial.suggest_int('embedding_dim', 8, 128, log=True),
        'hidden_size': trial.suggest_int('hidden_size', 16, 512, log=True),
        'dropout' : trial.suggest_float('dropout', 0.0, 0.4),
    }

    lr = trial.suggest_float('lr', 1e-5, 3e-2, log=True)

    return model_params, lr


def suggest_attention_params(
    trial: optuna.trial.Trial,
) -> tuple[dict[str, int | float], float]:
    
    model_params, lr = suggest_encoder_params(trial)
    model_params['attention_dim'] = trial.suggest_int('attention_dim', 16, 256, log=True)

    return model_params, lr


def suggest_transformer_params(
    trial: optuna.trial.Trial,
) -> tuple[dict[str, int | float], float]:
    model_params = {
        'model_dim': trial.suggest_categorical('model_dim', [64, 128, 256]),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
        'feedforward_dim': trial.suggest_categorical(
            'feedforward_dim', [128, 256, 512]
        ),
        'n_encoder_layers': trial.suggest_int('n_encoder_layers', 1, 3),
        'n_decoder_layers': trial.suggest_int('n_decoder_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
    }

    lr = trial.suggest_float('lr', 1e-5, 3e-2, log=True)

    return model_params, lr
    

def tune(
    study_name: str,
    suggest_params: SuggestParamsFn,
    build_model: Callable[..., nn.Module],
    fixed_model_params: dict[str, int | float],
    loaders: DataLoaders,
    step_fn: StepFn,
    pad_idx: int,
    train_config: TrainConfig,
    search_config: SearchConfig,
) -> optuna.study.Study:

    if train_config.verbose == 2:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective_for_trial(trial: optuna.trial.Trial) -> float:
        suggested_params, lr = suggest_params(trial)
        
        model_params = fixed_model_params.copy()
        model_params.update(suggested_params)
    
        trial_config = replace(train_config, verbose=0)
    
        result = run_experiment(
            name=f'trial_{trial.number}',
            build_model=build_model,
            model_params=model_params,
            loaders=loaders,
            step_fn=step_fn,
            pad_idx=pad_idx,
            lr=lr,
            train_config=trial_config
        )
    
        return result['best_valid_loss']

    Path('optuna').mkdir(exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///./optuna/{study_name}.db',
        load_if_exists=True,
        direction='minimize'
    )

    study.optimize(objective_for_trial, n_trials=search_config.n_trials, n_jobs=search_config.n_jobs)

    if train_config.verbose == 2:
        print(f'Best validation: {study.best_value:.6f}')
        print('Best parameters:')
        for name, value in study.best_params.items():
            print(f'  {name}: {value}')

    if search_config.plot_importances:
        plot_tuning_results(study)

    return study


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    step_fn: StepFn,
    device: torch.device,
    pad_idx: int,
) -> dict[str, float]:
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

    test_loss = validate(
        model=model,
        valid_loader=loader,
        loss_fn=loss_fn,
        step_fn=step_fn,
        device=device,
        pad_idx=pad_idx,
    )

    result = {
        'test_loss' : test_loss,
        'test_perplexity' : math.exp(test_loss)
    }

    return result


def run(
    name: str,
    suggest_params: SuggestParamsFn,
    build_model: Callable[..., nn.Module],
    fixed_model_params: dict[str, int | float],
    loaders: DataLoaders,
    step_fn: StepFn,
    pad_idx: int,
    train_config: TrainConfig,
    search_config: SearchConfig, 
    plot: bool = False,
    on_epoch_end: EpochCallback | None = None,
) -> dict[str, Any]:

    study = tune(
        study_name=name,
        suggest_params=suggest_params,
        build_model=build_model,
        fixed_model_params=fixed_model_params,
        loaders=loaders,
        step_fn=step_fn,
        pad_idx=pad_idx,
        train_config=train_config,
        search_config=search_config,
    )

    best_params = study.best_params.copy()
    lr = best_params.pop('lr')

    model_params = fixed_model_params.copy()
    model_params.update(best_params)

    result = run_experiment(
        name=name,
        build_model=build_model,
        model_params=model_params,
        loaders=loaders,
        step_fn=step_fn,
        pad_idx=pad_idx,
        lr=lr,
        train_config=train_config,
        plot=plot,
        on_epoch_end=on_epoch_end,
    )

    result['study'] = study
    result['model_params'] = model_params
    result['lr'] = lr

    test_results = evaluate(
        model=result['model'],
        loader=loaders.test,
        step_fn=step_fn,
        device=train_config.device,
        pad_idx=pad_idx,
    )

    result.update(test_results)

    if train_config.verbose == 2 or train_config.verbose == 1:
        print(f"Experiment: {name}")
        print(f"Best epoch: {result['best_epoch']}/{result['best_epoch'] + train_config.patience} completed")
        print(f"Best validation loss: {result['best_valid_loss']:.4f}")
        print(f"Test loss: {result['test_loss']:.4f}")
        print(f"Test perplexity: {result['test_perplexity']:.4f}")
    
    return result


def generate_names(
    prefixes: list[str],
    temperatures: list[float],
    model: nn.Module,
    token_to_id: dict[str, int],
    id_to_token: dict[int, str],
    train_df: pd.DataFrame
) -> pd.DataFrame:
    names = []

    train_names = set(train_df['latin'].str.lower())

    for prefix, temp in product(prefixes, temperatures):
        name = model.generate(
            token_to_id,
            id_to_token,
            prefix,
            temp,
        )
        names.append({
            'prefix': prefix,
            'temperature': temp,
            'name': name,
            'in_train': name in train_names,
        })

    return pd.DataFrame(names)


def build_translation_model(
    pretrained_encoder: Encoder,
    vocab_size: int,
    embedding_dim: int,
    hidden_size: int,
    pad_idx: int,
    dropout: float = 0.0,
    positional_encoding: str = 'none',
    max_len: int = 64,
) -> Seq2Seq:
    
    encoder = copy.deepcopy(pretrained_encoder)
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        pad_idx=pad_idx,
        dropout=dropout,
        positional_encoding=positional_encoding,
        max_len=max_len,
    )
    seq2seq = Seq2Seq(
        encoder=encoder,
        decoder=decoder
    )

    return seq2seq


def build_attention_translation_model(
    pretrained_encoder: Encoder,
    vocab_size: int,
    embedding_dim: int,
    hidden_size: int,
    pad_idx: int,
    attention_dim: int,
    dropout: float = 0.0,
    positional_encoding: str = 'none',
    max_len: int = 64,
    n_heads: int = 1,
) -> AttentionSeq2Seq:
    
    encoder = copy.deepcopy(pretrained_encoder)
    decoder = AttentionDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        pad_idx=pad_idx,
        attention_dim=attention_dim,
        encoder_hidden_size=encoder.cell.hidden_size,
        dropout=dropout,
        positional_encoding=positional_encoding,
        max_len=max_len,
        n_heads=n_heads,
    )
    attention_seq2seq = AttentionSeq2Seq(
        encoder=encoder,
        decoder=decoder
    )

    return attention_seq2seq


def make_translation_callback(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    latin_to_id: dict[str, int],
    cyrillic_to_id: dict[str, int],
    id_to_cyrillic: dict[int, str],
    random_state: int | None,
    every_n_epochs: int = 5,
    n_samples: int = 5,
) -> EpochCallback:
    if every_n_epochs < 1:
        raise ValueError("every_n_epochs must be >= 1")

    train_examples = train_df.sample(n=n_samples, random_state=random_state).copy()
    train_examples['split'] = 'train'

    valid_examples = valid_df.sample(n=n_samples, random_state=random_state).copy()
    valid_examples['split'] = 'valid'

    examples = pd.concat([train_examples, valid_examples], ignore_index=True)

    table = examples[['split', 'latin', 'cyrillic']]
    table = table.rename(columns={'latin': 'name', 'cyrillic': 'expected'})

    display_handle = None
    epoch_columns = []

    def highlight_matches(column: pd.Series) -> list[str]:
        return [
            'background-color: #245c36; color: white;'
            if prediction == expected else ''
            for prediction, expected in zip(column, table['expected'])
        ]

    def show_translations(model: nn.Module, epoch: int) -> None:
        nonlocal display_handle

        if epoch % every_n_epochs != 0:
            return

        predictions = []
        for name in table['name']:
            prediction = model.translate(
                name=name,
                latin_to_id=latin_to_id,
                cyrillic_to_id=cyrillic_to_id,
                id_to_cyrillic=id_to_cyrillic,
            )
            predictions.append(prediction)

        column_name = f'Epoch {epoch}'
        table[column_name] = predictions

        if column_name not in epoch_columns:
            epoch_columns.append(column_name)

        styled_table = (
            table.style
            .hide(axis='index')
            .set_caption('Translations during training')
            .set_properties(**{
                'text-align': 'left',
                'padding': '8px 12px',
            })
            .apply(
                highlight_matches,
                axis=0,
                subset=epoch_columns,
            )
        )

        if display_handle is None:
            display_handle = display(styled_table, display_id=True)
        else:
            display_handle.update(styled_table)

    return show_translations


def make_attention_callback(
    name: str,
    translation: str,
    latin_to_id: dict[str, int],
    cyrillic_to_id: dict[str, int],
    every_n_epochs: int = 5,
) -> EpochCallback:
    if every_n_epochs < 1:
        raise ValueError("every_n_epochs must be >= 1")

    def show_attention(model: nn.Module, epoch: int) -> None:
        if epoch % every_n_epochs != 0:
            return

        weights, source_tokens, query_tokens = model.get_attention(
            name=name,
            translation=translation,
            latin_to_id=latin_to_id,
            cyrillic_to_id=cyrillic_to_id,
        )

        plot_attention(
            weights=weights,
            source_tokens=source_tokens,
            query_tokens=query_tokens,
            title=f"Epoch {epoch}: {name} → {translation}",
        )

    return show_attention


def combine_callbacks(
    callbacks: list[EpochCallback],
) -> EpochCallback:
    
    def show_callbacks(model: nn.Module, epoch: int) -> None:
        for callback in callbacks:
            callback(model, epoch)

    return show_callbacks
    

def summarize_results(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for name, result in results.items():
        train_loss = result['history'].iloc[result['best_epoch'] - 1]['train_loss']
        valid_loss = result['best_valid_loss']

        rows.append({
            'experiment': name,
            'best_epoch': result['best_epoch'],
            'epochs_completed': len(result['history']),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'gap' : valid_loss - train_loss,
            'test_loss': result['test_loss'],
            'test_perplexity': result['test_perplexity'],
        })

    return pd.DataFrame(rows).sort_values(by='test_loss', ascending=True)