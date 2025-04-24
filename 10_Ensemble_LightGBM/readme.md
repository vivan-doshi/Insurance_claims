# LightGBM Ensemble with Optuna Hyperparameter Tuning

This project implements an ensemble learning approach using LightGBM models for predicting insurance metrics, specifically:
- Loss Cost per Exposure Unit (LC)
- Historically Adjusted Loss Cost (HALC)
- Claim Status (CS)

The implementation uses a two-level stacking ensemble:
1. **Base Models (Model 0)**: LightGBM models with optimized hyperparameters
2. **Meta Models (Model 1)**: LightGBM models trained on the predictions from base models plus original features

## File Structure

```
.
├── ensemble-pipeline.py            # Complete pipeline script combining both models
├── hyperparameter-optimization.py  # Dedicated script for hyperparameter optimization
├── feature_selected_train.csv      # Training features
├── feature_selected_y_train.csv    # Training targets
├── feature_selected_test.csv       # Test features
├── models/                         # Directory for storing base models
├── meta_models/                    # Directory for storing meta models
└── plots/                          # Directory for storing feature importance plots
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- lightgbm
- optuna
- matplotlib
- seaborn

Install the required packages using:

```bash
pip install pandas numpy scikit-learn lightgbm optuna matplotlib seaborn
```

## Usage

### Step 1: Hyperparameter Optimization

To optimize hyperparameters for all models:

```bash
python hyperparameter-optimization.py --n_trials 30 --model_type all
```

Options:
- `--n_trials`: Number of optimization trials (default: 30)
- `--n_folds`: Number of folds for cross-validation (default: 5)
- `--model_type`: Type of model to optimize (options: lc, halc, cs, meta_lc, meta_halc, meta_cs, all)
- `--optimize_meta`: Flag to optimize meta-models

### Step 2: Training Models

To train all models using the complete ensemble pipeline:

```bash
python ensemble-pipeline.py --train
```

### Step 3: Making Predictions

To make predictions on the test set:

```bash
python ensemble-pipeline.py --predict
```

You can also perform both training and prediction in one go:

```bash
python ensemble-pipeline.py --train --predict
```

## Implementation Details

### Base Models (Model 0)

The base models are LightGBM models optimized using Optuna. Three separate models are trained:
1. Regression model for Loss Cost (LC)
2. Regression model for Historically Adjusted Loss Cost (HALC)
3. Classification model for Claim Status (CS)

Each model uses 5-fold cross-validation during hyperparameter optimization to ensure robust performance.

### Meta Models (Model 1)

The meta models use the predictions from the base models as additional features. The meta-models are also LightGBM models with their own optimized hyperparameters.

## Model Performance

Out-of-fold (OOF) predictions are used to evaluate model performance:
- For regression models (LC, HALC): RMSE (Root Mean Squared Error)
- For classification model (CS): AUC-ROC (Area Under the Receiver Operating Characteristic Curve)

## Feature Importance

Feature importance plots are generated and saved in the `plots/` directory, allowing for interpretability of the models.