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
├── data/                # Dataset directory
├── scripts/            # Helper scripts
├── src/                # Source code
│   ├── analysis/      # Analysis tools
│   ├── config/        # Configuration files
│   ├── data/          # Data loading utilities
│   ├── models/        # TFT model implementation
│   └── visualization/ # Plotting utilities
└── example_notebook.ipynb  # Example usage notebook
```

## Dataset

- Synthetic datasets: [Diamond shape causal structures with additive noise](https://dataverse.harvard.edu/dataverse/basic_causal_structures_additive_noise)

## Usage

1. Prepare your data in CSV format with a timestamp column and variables of interest.

2. Configure the experiment parameters in `src/config/default_config.py`.

3. Run the analysis:

see example_notebook.ipynb

## Contact

Floris Schouw - florisschouw@gmail.com