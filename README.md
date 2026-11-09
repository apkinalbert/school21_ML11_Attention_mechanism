# Attention and Transformers

This project explores attention mechanisms and sequence-to-sequence models through English-to-Russian name transliteration. It combines GRU models with attention implemented from scratch and a custom Transformer built with PyTorch.

## Installation

Install all required dependencies:

```bash
pip3 install -r requirements.txt
```

Run the notebook or Python scripts after installing the dependencies.

## What was done

* Prepared the names dataset and split it into  **training, validation, and test sets** .
* Trained **GRU models** to generate English names and transliterate them into Russian.
* Implemented **attention and multi-head attention from scratch** and visualized their weights.
* Compared  **sinusoidal and learned positional encodings** .
* Built a custom  **Transformer encoder-decoder using PyTorch** .
* Used **early stopping** and tuned hyperparameters with  **Optuna** .
* Compared models using **loss and perplexity** and examined overfitting.
* Saved trained models and exported test predictions to CSV.

## Main techniques

`Sequence-to-Sequence` · `GRU` · `Attention` · `Multi-Head Attention` · `Transformers` · `Positional Encoding` · `Teacher Forcing` · `Greedy Decoding` · `Masking` · `Cross-Entropy` · `Perplexity` · `Adam` · `Optuna` · `PyTorch`

## Dataset

* Baby names — paired English names and their Russian transliterations, stored in `baby_names.csv`.
