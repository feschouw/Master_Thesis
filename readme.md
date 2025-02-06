# Temporal Fusion Transformer for Causal Discovery

This repository contains the implementation of using Temporal Fusion Transformers (TFT) for discovering causal relationships in multivariate time series data.

## Overview

This project explores using TFT's inherent architectural features - variable selection networks and multi-head attention mechanisms - to identify temporal dependencies and causal relationships in time series data. The approach combines:

- TFT's built-in feature selection capabilities
- Both deterministic and probabilistic forecasting
- Permutation Feature Importance (PFI) analysis for validation

## Key Features

- Implements TFT model for time series prediction
- Supports both point estimates and interval scores
- Includes comprehensive evaluation metrics (F1 score, Precision, Recall, SHD)
- Provides visualization tools for causal graphs and matrices

## Project Structure

```
├── data/               # Dataset directory
├── example.ipynb       # Example notebook 
├── src/                # Source code
│   ├── analysis/       # Analysis tools and PFI implementation
│   ├── config/         # Configuration files
│   ├── data/           # Data loading utilities
│   ├── models/         # TFT model implementation
│   └── visualization/  # Plotting utilities
├── main.py             # Main script to run experiments
├── README.md
```

## Dataset

- Synthetic datasets: [Diamond shape causal structures with additive noise](https://dataverse.harvard.edu/dataverse/basic_causal_structures_additive_noise)

## Dependencies
Install required packages:
- pandas==2.2.2
- darts==0.32.0
- seaborn==0.13.2

## Usage

1. Prepare your data in CSV format with a timestamp column and variables of interest.

2. Use Command Line Interface

Run experiments using the main script with various configuration options:

```bash
python main.py --dataset harvard_diamond_data_0 \
               --ground_truth harvard_diamond_data_ground_truth \
               --input_chunk_length 30 \
               --output_chunk_length 14 \
               --hidden_size 64 \
               --lstm_layers 2 \
               --attention_heads 2 \
               --batch_size 64 \
               --dropout 0.25 \
               --epochs 200 \
               --device mps
```
  > [!TIP]
  > If you encounter training issues on Apple Silicon (M-series processors) with `train_loss=nan.0`, switch to "--device cpu"

## Results

The script generates several outputs in the specified output directory:
- Causal graphs visualization
- Adjacency matrices
- Performance metrics plots
- Evaluation metrics (F1, Precision, Recall, SHD)

## Contact

Floris Schouw - florisschouw@gmail.com
