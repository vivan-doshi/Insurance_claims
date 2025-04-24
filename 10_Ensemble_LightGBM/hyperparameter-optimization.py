# =====================================================================
# UPDATED VERSION - Tweedie Objective is now ALWAYS used for LC and HALC
# tweedie_variance_power is always optimized when using tweedie objective
# This script supports strategic optimization via command-line arguments
# =====================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import os
import argparse
import warnings
from functools import partial
import time
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LightGBM Hyperparameter Optimization')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials. Use higher for base models, lower for meta models.')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--model_type', type=str, default='all',
                        help='Type of model to optimize (lc, halc, cs, meta_lc, meta_halc, meta_cs, all). Use this to run optimization strategically for specific models.')
    parser.add_argument('--optimize_meta', action='store_true', help='Flag to indicate optimization for meta-models. Loads meta-features.')
    parser.add_argument('--n_bins_regression_stratification', type=int, default=10,
                        help='Number of bins for stratifying regression targets')

    return parser.parse_args()

def clean_feature_names(df):
    """Clean feature names to avoid LightGBM errors"""
    # Replace characters that cause LightGBM errors
    clean_columns = {}
    for col in df.columns:
        # Replace special characters with underscores
        new_col = re.sub(r'[^A-Za-z0-9_]', '_', col)
        # Ensure name starts with a letter or underscore
        if not new_col[0].isalpha() and new_col[0] != '_':
            new_col = 'f_' + new_col
        clean_columns[col] = new_col

    # Rename columns
    df = df.rename(columns=clean_columns)
    return df

def bin_regression_target(y, n_bins=10):
    """Bins a continuous target variable for stratification."""
    # Use qcut to create quantile-based bins
    # Handle cases with very few unique values or NaNs
    try:
        # Check for NaNs and remove them for binning, then re-index
        y_not_na = y.dropna()
        if len(np.unique(y_not_na)) < n_bins:
             # If not enough unique values, use fewer bins or just unique values
             bins = np.unique(y_not_na)
        else:
             # Use qcut for quantile-based binning
             bins = pd.qcut(y_not_na, q=n_bins, labels=False, duplicates='drop')

        # Create a new series with the same index as original y, filling NaNs if any
        y_binned = pd.Series(np.nan, index=y.index)
        y_binned[y_not_na.index] = bins
        return y_binned.astype(float) # Return as float to handle potential NaNs

    except Exception as e:
        print(f"Warning: Could not bin regression target with {n_bins} bins. Using default binning or handling: {e}")
        # Fallback: return a simple binning or handle as appropriate for your data
        if len(np.unique(y.dropna())) > 0:
             return pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
        else:
             return pd.Series(0, index=y.index) # Default to a single bin if no valid data


def load_data():
    """Load the base features and target variables"""
    print("Loading base data...")
    X_train = pd.read_csv('feature_selected_train.csv', index_col=0)
    y_df = pd.read_csv('feature_selected_y_train.csv', index_col=0)

    # Clean feature names to avoid LightGBM errors
    X_train = clean_feature_names(X_train)

    # Extract target variables
    y_lc = y_df['Loss_Cost']
    y_halc = y_df['Historically_Adjusted_Loss_Cost']
    y_cs = y_df['Claim_Status']

    print(f"Base Features shape: {X_train.shape}")
    print(f"Base Target shapes - Loss_Cost: {y_lc.shape}, HALC: {y_halc.shape}, Claim_Status: {y_cs.shape}")

    return X_train, y_lc, y_halc, y_cs

def load_meta_features(n_bins_regression_stratification=10):
    """Load meta features (original features + OOF predictions) and bin regression targets"""
    print("Loading meta features...")
    # Load base data first to get original features and true targets
    X_train, y_lc, y_halc, y_cs = load_data()

    if os.path.exists('oof_predictions.csv'):
        oof_df = pd.read_csv('oof_predictions.csv')

        # Ensure the index of oof_df matches X_train before combining
        # This is important if row order might have changed or indices differ
        oof_df = oof_df.set_index(X_train.index)

        # Create meta-features by combining original features with OOF predictions
        meta_X = X_train.copy()
        # Add OOF predictions as new features
        meta_X['oof_lc'] = oof_df['oof_lc']
        meta_X['oof_halc'] = oof_df['oof_halc']
        meta_X['oof_cs'] = oof_df['oof_cs']

        # Get true target variables for meta-model training (should be same as base targets)
        # Using targets from oof_df ensures alignment if oof_df was re-indexed
        meta_y_lc = oof_df['true_lc']
        meta_y_halc = oof_df['true_halc']
        meta_y_cs = oof_df['true_cs']


        # Bin regression targets for stratified cross-validation of meta-models
        meta_y_lc_binned = bin_regression_target(meta_y_lc, n_bins_regression_stratification)
        meta_y_halc_binned = bin_regression_target(meta_y_halc, n_bins_regression_stratification)

        print(f"Meta features shape: {meta_X.shape}")
        # Return meta features, true targets, and binned targets for stratification
        return meta_X, meta_y_lc, meta_y_halc, meta_y_cs, meta_y_lc_binned, meta_y_halc_binned
    else:
        print("Error: 'oof_predictions.csv' file not found. Run base models training first to generate OOF predictions.")
        return None, None, None, None, None, None

# Define objective function for Loss Cost (LC) regression
def objective_lc(trial, X, y, y_binned, n_folds=5):
    """Objective function for LC model optimization targeting RMSE with stratified CV"""
    # Objective is fixed to 'tweedie' as requested for LC
    objective_type = 'tweedie'

    param = {
        'objective': objective_type,
        'metric': 'rmse', # RMSE is a common metric for regression
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']), # Include dart
        'verbosity': -1,
        'n_jobs': -1, # Use all available cores
        'seed': SEED,

        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000), # Rely on early stopping
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True),
        'max_bin': trial.suggest_int('max_bin', 128, 512),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0), # Corresponds to bagging_fraction
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0), # Corresponds to feature_fraction
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 5),

        # Optimize tweedie_variance_power specifically for the tweedie objective
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.0, 2.0) # Common range for Tweedie
    }

    # Perform stratified k-fold cross-validation on binned target
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores = []

    # Handle potential NaNs in binned target for StratifiedKFold
    non_nan_indices = y_binned.dropna().index
    if len(non_nan_indices) == 0:
        print("Warning: No non-NaN values in binned target. Cannot perform stratified CV.")
        return float('inf') # Return a large value if no valid data for CV

    X_non_nan = X.loc[non_nan_indices]
    y_non_nan = y.loc[non_nan_indices]
    y_binned_non_nan = y_binned.loc[non_nan_indices].astype(int)

    # Ensure there's more than one class in the binned target for stratification
    if len(np.unique(y_binned_non_nan)) < 2:
         print("Warning: Less than 2 unique classes in binned target for stratification. Falling back to KFold.")
         kf_fallback = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
         split_iterator = kf_fallback.split(X_non_nan)
    else:
         split_iterator = skf.split(X_non_nan, y_binned_non_nan)


    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        # Map back to original indices
        original_train_idx = non_nan_indices[train_idx]
        original_val_idx = non_nan_indices[val_idx]

        X_train, X_val = X.loc[original_train_idx], X.loc[original_val_idx]
        y_train, y_val = y.loc[original_train_idx], y.loc[original_val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(100, verbose=False)] # Increased early stopping rounds

        # Train model
        model = lgb.train(
            params=param,
            train_set=train_data,
            num_boost_round=param['n_estimators'], # Use n_estimators from trial
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    # Return the mean RMSE across all folds
    return np.mean(rmse_scores)

# Define objective function for Historically Adjusted Loss Cost (HALC) regression
def objective_halc(trial, X, y, y_binned, n_folds=5):
    """Objective function for HALC model optimization targeting RMSE with stratified CV"""
    # Objective is fixed to 'tweedie' as requested for HALC
    objective_type = 'tweedie'

    param = {
        'objective': objective_type,
        'metric': 'rmse', # RMSE is a common metric for regression
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']), # Include dart
        'verbosity': -1,
        'n_jobs': -1, # Use all available cores
        'seed': SEED,

        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000), # Rely on early stopping
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True),
        'max_bin': trial.suggest_int('max_bin', 128, 512),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0), # Corresponds to bagging_fraction
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0), # Corresponds to feature_fraction
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 5),

        # Optimize tweedie_variance_power specifically for the tweedie objective
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.0, 2.0) # Common range for Tweedie
    }

    # Perform stratified k-fold cross-validation on binned target
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores = []

    # Handle potential NaNs in binned target for StratifiedKFold
    non_nan_indices = y_binned.dropna().index
    if len(non_nan_indices) == 0:
        print("Warning: No non-NaN values in binned target. Cannot perform stratified CV.")
        return float('inf') # Return a large value if no valid data for CV

    X_non_nan = X.loc[non_nan_indices]
    y_non_nan = y.loc[non_nan_indices]
    y_binned_non_nan = y_binned.loc[non_nan_indices].astype(int)

    # Ensure there's more than one class in the binned target for stratification
    if len(np.unique(y_binned_non_nan)) < 2:
         print("Warning: Less than 2 unique classes in binned target for stratification. Falling back to KFold.")
         kf_fallback = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
         split_iterator = kf_fallback.split(X_non_nan)
    else:
         split_iterator = skf.split(X_non_nan, y_binned_non_nan)

    for fold, (train_idx, val_idx) in enumerate(split_iterator):
         # Map back to original indices
        original_train_idx = non_nan_indices[train_idx]
        original_val_idx = non_nan_indices[val_idx]

        X_train, X_val = X.loc[original_train_idx], X.loc[original_val_idx]
        y_train, y_val = y.loc[original_train_idx], y.loc[original_val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(100, verbose=False)] # Increased early stopping rounds

        # Train model
        model = lgb.train(
            params=param,
            train_set=train_data,
            num_boost_round=param['n_estimators'], # Use n_estimators from trial
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

    # Return the mean RMSE across all folds
    return np.mean(rmse_scores)


# Define objective function for Claim Status (CS) classification
# This objective targets AUC, which is appropriate for classification
def objective_cs(trial, X, y, n_folds=5):
    """Objective function for CS model optimization targeting AUC"""
    param = {
        'objective': 'binary',
        'metric': 'auc', # AUC is appropriate for binary classification
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'verbosity': -1,
        'n_jobs': -1, # Use all available cores
        'seed': SEED,

        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), # Use n_estimators
        'num_leaves': trial.suggest_int('num_leaves', 7, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'max_bin': trial.suggest_int('max_bin', 100, 300),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0) # Useful for imbalanced datasets
    }

    # Perform stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    auc_scores = []

    # Ensure there's more than one class in the target for stratification
    if len(np.unique(y)) < 2:
         print("Warning: Less than 2 unique classes in target for stratification. Cannot perform StratifiedKFold.")
         return float('inf') # Return a large value if stratification is not possible

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)] # Keep 50 for classification

        # Train model
        model = lgb.train(
            params=param,
            train_set=train_data,
            num_boost_round=param['n_estimators'], # Use n_estimators
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        # Calculate AUC
        auc = roc_auc_score(y_val, y_pred)
        auc_scores.append(auc)

    # Return the negative mean AUC across all folds (Optuna minimizes)
    return -np.mean(auc_scores)

# Run optimization
def run_optimization(objective, X, y, y_binned=None, n_trials=50, n_folds=5, study_name="study"):
    """Run Optuna optimization for a given objective function"""
    print(f"Starting optimization for {study_name} with {n_trials} trials...")
    start_time = time.time()

    # Create optimization study
    # Using an in-memory study. For parallel runs across multiple machines,
    # configure a shared storage backend (e.g., "sqlite:///optuna_study.db").
    study = optuna.create_study(direction="minimize", study_name=study_name)

    # Use partial to pass additional arguments to objective function
    if y_binned is not None:
         objective_with_params = partial(objective, X=X, y=y, y_binned=y_binned, n_folds=n_folds)
    else:
         objective_with_params = partial(objective, X=X, y=y, n_folds=n_folds)


    # Run optimization
    study.optimize(objective_with_params, n_trials=n_trials)

    # Print results
    print(f"Optimization for {study_name} completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best value: {study.best_value}")
    print(f"Best hyperparameters: {study.best_params}")

    # Create directory for saving parameters if it doesn't exist
    os.makedirs('params', exist_ok=True)

    # Prepare parameters for saving
    best_params = study.best_params.copy()
    # Rename num_boost_round to n_estimators for consistency with other scripts
    # Check if 'n_estimators' is already present (from trial.suggest_int)
    # If not, and 'num_boost_round' is (from a default or old trial), handle it.
    # Given the current objective functions use 'n_estimators', this might be redundant,
    # but kept for robustness.
    if 'num_boost_round' in best_params and 'n_estimators' not in best_params:
        best_params['n_estimators'] = best_params.pop('num_boost_round')

    # Save best parameters to file
    param_path = f'params/best_params_{study_name}.txt'
    with open(param_path, 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    print(f"Best parameters saved to '{param_path}'")

    return best_params

def main():
    """Main function to run hyperparameter optimization"""
    print("=" * 80)
    print("LIGHTGBM HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    args = parse_arguments()
    n_folds = args.n_folds
    n_bins_regression_stratification = args.n_bins_regression_stratification
    model_type = args.model_type
    n_trials = args.n_trials

    # --- Load data based on model type ---
    if args.optimize_meta:
        print("Loading data for meta-model optimization...")
        # Load meta features and targets
        X, y_lc, y_halc, y_cs, y_lc_binned, y_halc_binned = load_meta_features(n_bins_regression_stratification)
        if X is None:
             print("Meta-features could not be loaded. Exiting optimization.")
             return # Exit if meta-features are missing
    else:
        print("Loading data for base model optimization...")
        # Load base features and targets
        X, y_lc, y_halc, y_cs = load_data()
        # Bin regression targets for stratification for base models
        y_lc_binned = bin_regression_target(y_lc, n_bins_regression_stratification)
        y_halc_binned = bin_regression_target(y_halc, n_bins_regression_stratification)


    # --- Run optimization based on model_type argument ---
    if model_type == 'all':
        # Run optimization for all models (can be time consuming)
        if not args.optimize_meta:
             print("Optimizing all base models...")
             run_optimization(objective_lc, X, y_lc, y_lc_binned, n_trials=n_trials, n_folds=n_folds, study_name="lc")
             run_optimization(objective_halc, X, y_halc, y_halc_binned, n_trials=n_trials, n_folds=n_folds, study_name="halc")
             run_optimization(objective_cs, X, y_cs, n_trials=n_trials, n_folds=n_folds, study_name="cs")
        else:
             print("Optimizing all meta models...")
             run_optimization(objective_lc, X, y_lc, y_lc_binned, n_trials=n_trials, n_folds=n_folds, study_name="meta_lc")
             run_optimization(objective_halc, X, y_halc, y_halc_binned, n_trials=n_trials, n_folds=n_folds, study_name="meta_halc")
             run_optimization(objective_cs, X, y_cs, n_trials=n_trials, n_folds=n_folds, study_name="meta_cs")

    elif model_type == 'lc' and not args.optimize_meta:
        print("Optimizing base LC model...")
        run_optimization(objective_lc, X, y_lc, y_lc_binned, n_trials=n_trials, n_folds=n_folds, study_name="lc")

    elif model_type == 'halc' and not args.optimize_meta:
        print("Optimizing base HALC model...")
        run_optimization(objective_halc, X, y_halc, y_halc_binned, n_trials=n_trials, n_folds=n_folds, study_name="halc")

    elif model_type == 'cs' and not args.optimize_meta:
        print("Optimizing base CS model...")
        run_optimization(objective_cs, X, y_cs, n_trials=n_trials, n_folds=n_folds, study_name="cs")

    elif model_type == 'meta_lc' and args.optimize_meta:
        print("Optimizing meta LC model...")
        run_optimization(objective_lc, X, y_lc, y_lc_binned, n_trials=n_trials, n_folds=n_folds, study_name="meta_lc")

    elif model_type == 'meta_halc' and args.optimize_meta:
        print("Optimizing meta HALC model...")
        run_optimization(objective_halc, X, y_halc, y_halc_binned, n_trials=n_trials, n_folds=n_folds, study_name="meta_halc")

    elif model_type == 'meta_cs' and args.optimize_meta:
        print("Optimizing meta CS model...")
        run_optimization(objective_cs, X, y_cs, n_trials=n_trials, n_folds=n_folds, study_name="meta_cs")

    else:
        print(f"Invalid combination of --model_type ({model_type}) and --optimize_meta ({args.optimize_meta}).")
        print("Use --model_type [lc, halc, cs] for base models.")
        print("Use --model_type [meta_lc, meta_halc, meta_cs] with --optimize_meta for meta models.")
        print("Use --model_type all to optimize all base or all meta models (depending on --optimize_meta).")


    print("=" * 80)
    print("OPTIMIZATION PROCESS CONCLUDED")
    print("=" * 80)

if __name__ == "__main__":
    main()
