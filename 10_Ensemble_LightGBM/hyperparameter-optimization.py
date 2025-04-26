# =====================================================================
# UPDATED VERSION - Combined Stratification for Regression Targets
# Stratification for all models (base and meta) is now based on combined bins
# of Loss Cost and Historically Adjusted Loss Cost.
# Tweedie Objective is ALWAYS used for LC and HALC
# tweedie_variance_power is always optimized when using tweedie objective
# This script supports strategic optimization via command-line arguments
# Added saving of best parameters for meta models
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
import json # Import json for saving/loading parameters

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
    parser.add_argument('--optimize_meta', action='store_true', help='Flag to indicate if optimizing meta-models')
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
        clean_columns[col] = new_col
    return df.rename(columns=clean_columns)

def create_combined_stratification_bins(y_lc, y_halc, n_bins):
    """Create combined bins for stratification based on LC and HALC"""
    # Handle potential zero values before binning
    y_lc_nonzero = y_lc[y_lc > 0]
    y_halc_nonzero = y_halc[y_halc > 0]

    # Create bins for non-zero values
    lc_bins = pd.cut(y_lc_nonzero, bins=n_bins, labels=False, duplicates='drop') if len(y_lc_nonzero) > 0 else pd.Series(0, index=y_lc_nonzero.index)
    halc_bins = pd.cut(y_halc_nonzero, bins=n_bins, labels=False, duplicates='drop') if len(y_halc_nonzero) > 0 else pd.Series(0, index=y_halc_nonzero.index)

    # Initialize combined bins with a default value (e.g., -1 for zero values)
    combined_bins = pd.Series(-1, index=y_lc.index, dtype=str) # Use string type for combined bins

    # Assign combined bin labels for non-zero values
    if len(y_lc_nonzero) > 0 and len(y_halc_nonzero) > 0:
         combined_bins[y_lc_nonzero.index] = lc_bins.astype(str) + '_' + halc_bins.astype(str)
    elif len(y_lc_nonzero) > 0: # Only LC has non-zero values
         combined_bins[y_lc_nonzero.index] = lc_bins.astype(str) + '_NA' # Indicate missing HALC bin
    elif len(y_halc_nonzero) > 0: # Only HALC has non-zero values
         combined_bins[y_halc_nonzero.index] = 'NA_' + halc_bins.astype(str) # Indicate missing LC bin

    # Assign a unique bin for zero values (if any)
    zero_indices = y_lc[(y_lc == 0) | (y_halc == 0)].index # Consider zero in either LC or HALC
    if len(zero_indices) > 0:
        combined_bins[zero_indices] = 'ZERO' # Assign a distinct category for zeros

    # Convert to categorical for StratifiedKFold
    return combined_bins.astype('category')


# Define objective functions for Optuna optimization
def objective_lc(trial, X, y, y_combined_binned, n_folds):
    """Optuna objective function for Loss Cost (Regression with Tweedie)"""
    param = {
        'objective': 'tweedie',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1),
        'max_bin': trial.suggest_int('max_bin', 100, 512),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 0.1),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 5),
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.0, 2.0), # Tweedie parameter
        'n_jobs': -1,
        'verbose': -1,
        'seed': SEED,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
    }

    # Use the combined binned variable for stratification
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y_combined_binned)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse', # Use RMSE for evaluation during training
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # Add early stopping

        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False) # Calculate RMSE
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

def objective_halc(trial, X, y, y_combined_binned, n_folds):
    """Optuna objective function for Historically Adjusted Loss Cost (Regression with Tweedie)"""
    param = {
        'objective': 'tweedie',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1),
        'max_bin': trial.suggest_int('max_bin', 100, 512),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 0.1),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 5),
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.0, 2.0), # Tweedie parameter
        'n_jobs': -1,
        'verbose': -1,
        'seed': SEED,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
    }

    # Use the combined binned variable for stratification
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y_combined_binned)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse', # Use RMSE for evaluation during training
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # Add early stopping

        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False) # Calculate RMSE
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

def objective_cs(trial, X, y, y_combined_binned, n_folds):
    """Optuna objective function for Claim Status (Classification)"""
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'max_bin': trial.suggest_int('max_bin', 100, 512),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 0.1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0), # subsample
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10), # subsample_freq
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0), # colsample_bytree
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0), # Handle imbalance
        'n_jobs': -1,
        'verbose': -1,
        'seed': SEED,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
    }

    # Use the combined binned variable for stratification
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    auc_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y_combined_binned)): # Use combined bins for splitting
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # Add early stopping

        predictions_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, predictions_proba)
        auc_scores.append(auc)

    # Optuna aims to minimize the objective, so we return negative AUC
    return -np.mean(auc_scores)

def run_optimization(objective, X, y, y_combined_binned, n_trials=50, n_folds=5, study_name="lightgbm_optimization"):
    """Run Optuna optimization for a given objective function"""
    print(f"Starting optimization for {study_name} with {n_trials} trials...")
    start_time = time.time()

    # Use directional=minimize for regression (RMSE) and directional=maximize for classification (AUC)
    # Since objective_cs returns negative AUC, we still use minimize
    study = optuna.create_study(direction='minimize', study_name=study_name)

    # Pass the combined binned variable to the objective function
    study.optimize(partial(objective, X=X, y=y, y_combined_binned=y_combined_binned, n_folds=n_folds), n_trials=n_trials)


    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"Optimization for {study_name} completed in {duration:.2f} minutes")

    print(f"Best value: {study.best_value}")
    print("Best hyperparameters:")
    print(study.best_params)

    # Save best parameters to a file
    params_dir = 'params'
    os.makedirs(params_dir, exist_ok=True)
    param_filename = os.path.join(params_dir, f'best_params_{study_name}.txt')
    with open(param_filename, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best parameters saved to '{param_filename}'")


def main():
    args = parse_arguments()

    # Load data
    try:
        X = pd.read_csv('feature_selected_train.csv')
        y_train_df = pd.read_csv('feature_selected_y_train.csv')
        y_lc = y_train_df['Loss_Cost']
        y_halc = y_train_df['Historically_Adjusted_Loss_Cost']
        y_cs = y_train_df['Claim_Status']

        # Clean feature names
        X = clean_feature_names(X)

        print(f"Base Features shape: {X.shape}")
        print(f"Base Target shapes - Loss_Cost: {y_lc.shape}, HALC: {y_halc.shape}, Claim_Status: {y_cs.shape}")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure 'feature_selected_train.csv' and 'feature_selected_y_train.csv' are in the correct directory.")
        return

    # Create combined bins for stratified cross-validation for all models
    y_combined_binned = create_combined_stratification_bins(y_lc, y_halc, args.n_bins_regression_stratification)
    print(f"Created combined bins for stratification with {len(y_combined_binned.cat.categories)} categories.")


    if args.model_type == 'all':
        print("Optimizing all base models...")
        # Pass the combined binned variable to the optimization runs
        run_optimization(objective_lc, X, y_lc, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="lc")
        run_optimization(objective_halc, X, y_halc, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="halc")
        run_optimization(objective_cs, X, y_cs, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="cs")

        # For meta-model optimization, we need the base model predictions.
        # These would typically be generated by running the ensemble pipeline's training step first.
        # Assuming 'oof_predictions.csv' exists from a prior pipeline run for meta-optimization.
        print("Assuming 'oof_predictions.csv' exists for meta-model optimization.")
        try:
            oof_preds_df = pd.read_csv('oof_predictions.csv')
            # Clean feature names for meta features as well
            oof_preds_df = clean_feature_names(oof_preds_df)

            # Ensure the index aligns with the original training data
            meta_X = pd.concat([X, oof_preds_df], axis=1)

            print("Optimizing all meta models...")
            # Pass the combined binned variable to the meta optimization runs
            run_optimization(objective_lc, meta_X, y_lc, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="meta_lc")
            run_optimization(objective_halc, meta_X, y_halc, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="meta_halc")
            run_optimization(objective_cs, meta_X, y_cs, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="meta_cs")

        except FileNotFoundError:
             print("Warning: 'oof_predictions.csv' not found. Skipping meta-model optimization.")


    elif args.model_type in ['lc', 'halc', 'cs'] and not args.optimize_meta:
        print(f"Optimizing base {args.model_type.upper()} model...")
        # Pass the combined binned variable to the optimization run
        if args.model_type == 'lc':
            run_optimization(objective_lc, X, y_lc, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="lc")
        elif args.model_type == 'halc':
            run_optimization(objective_halc, X, y_halc, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="halc")
        elif args.model_type == 'cs':
            run_optimization(objective_cs, X, y_cs, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="cs")

    elif args.model_type in ['meta_lc', 'meta_halc', 'meta_cs'] and args.optimize_meta:
        print(f"Optimizing meta {args.model_type.upper()} model...")
        try:
            oof_preds_df = pd.read_csv('oof_predictions.csv')
            # Clean feature names for meta features
            oof_preds_df = clean_feature_names(oof_preds_df)

            # Ensure the index aligns with the original training data
            meta_X = pd.concat([X, oof_preds_df], axis=1)

            # Pass the combined binned variable to the meta optimization run
            if args.model_type == 'meta_lc':
                run_optimization(objective_lc, meta_X, y_lc, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="meta_lc")
            elif args.model_type == 'meta_halc':
                run_optimization(objective_halc, meta_X, y_halc, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="meta_halc")
            elif args.model_type == 'meta_cs':
                run_optimization(objective_cs, meta_X, y_cs, y_combined_binned, n_trials=args.n_trials, n_folds=args.n_folds, study_name="meta_cs")

        except FileNotFoundError:
            print("Error: 'oof_predictions.csv' not found. Cannot optimize meta-models without base model predictions.")

    else:
        print(f"Invalid combination of --model_type ({args.model_type}) and --optimize_meta ({args.optimize_meta}).")
        print("Use --model_type [lc, halc, cs, all] for base model optimization.")
        print("Use --model_type [meta_lc, meta_halc, meta_cs, all] with --optimize_meta for meta-model optimization.")


if __name__ == "__main__":
    print("=" * 80)
    print("LIGHTGBM HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    main()
    print("=" * 80)
    print("OPTIMIZATION PROCESS CONCLUDED")
    print("=" * 80)
