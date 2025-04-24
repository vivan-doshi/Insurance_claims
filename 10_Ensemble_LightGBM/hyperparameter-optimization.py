# =====================================================================
# FINAL VERSION - Fixes feature name issue + callbacks
# Save this file as hyperparameter-optimization.py
# This script includes feature name cleanup and uses callbacks for early stopping
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
    parser.add_argument('--n_trials', type=int, default=30, help='Number of optimization trials')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--model_type', type=str, default='all', 
                        help='Type of model to optimize (lc, halc, cs, meta_lc, meta_halc, meta_cs, all)')
    parser.add_argument('--optimize_meta', action='store_true', help='Optimize meta-models')
    
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

def load_data():
    """Load the base features and target variables"""
    print("Loading data...")
    X_train = pd.read_csv('feature_selected_train.csv', index_col=0)
    y_df = pd.read_csv('feature_selected_y_train.csv', index_col=0)
    
    # Clean feature names to avoid LightGBM errors
    X_train = clean_feature_names(X_train)
    
    # Extract target variables
    y_lc = y_df['Loss_Cost']
    y_halc = y_df['Historically_Adjusted_Loss_Cost']
    y_cs = y_df['Claim_Status']
    
    print(f"Features shape: {X_train.shape}")
    print(f"Target shapes - Loss_Cost: {y_lc.shape}, HALC: {y_halc.shape}, Claim_Status: {y_cs.shape}")
    
    return X_train, y_lc, y_halc, y_cs

def load_meta_features():
    """Load meta features (original features + OOF predictions)"""
    print("Loading meta features...")
    X_train, _, _, _ = load_data()
    
    if os.path.exists('oof_predictions.csv'):
        oof_df = pd.read_csv('oof_predictions.csv')
        
        # Create meta-features by combining original features with OOF predictions
        meta_X = X_train.copy()
        meta_X['oof_lc'] = oof_df['oof_lc']
        meta_X['oof_halc'] = oof_df['oof_halc']
        meta_X['oof_cs'] = oof_df['oof_cs']
        
        # Get target variables
        y_lc = oof_df['true_lc']
        y_halc = oof_df['true_halc']
        y_cs = oof_df['true_cs']
        
        print(f"Meta features shape: {meta_X.shape}")
        return meta_X, y_lc, y_halc, y_cs
    else:
        print("Error: 'oof_predictions.csv' file not found. Run base models training first.")
        return None, None, None, None

# Define objective function for Loss Cost (LC) regression
def objective_lc(trial, X, y, n_folds=5):
    """Objective function for LC model optimization"""
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': SEED,
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'max_bin': trial.suggest_int('max_bin', 100, 300),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
    }
    
    num_boost_round = trial.suggest_int('num_boost_round', 100, 1000)
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]
        
        # Train model
        model = lgb.train(
            params=param,
            train_set=train_data,
            num_boost_round=num_boost_round,
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
def objective_halc(trial, X, y, n_folds=5):
    """Objective function for HALC model optimization"""
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': SEED,
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'max_bin': trial.suggest_int('max_bin', 100, 300),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
    }
    
    num_boost_round = trial.suggest_int('num_boost_round', 100, 1000)
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]
        
        # Train model
        model = lgb.train(
            params=param,
            train_set=train_data,
            num_boost_round=num_boost_round,
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
def objective_cs(trial, X, y, n_folds=5):
    """Objective function for CS model optimization"""
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': SEED,
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
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
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
    }
    
    num_boost_round = trial.suggest_int('num_boost_round', 100, 1000)
    
    # Perform k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]
        
        # Train model
        model = lgb.train(
            params=param,
            train_set=train_data,
            num_boost_round=num_boost_round,
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
def run_optimization(objective, X, y, n_trials=30, n_folds=5, study_name="study"):
    """Run Optuna optimization for a given objective function"""
    print(f"Starting optimization for {study_name}...")
    start_time = time.time()
    
    # Create optimization study
    study = optuna.create_study(direction="minimize", study_name=study_name)
    
    # Use partial to pass additional arguments to objective function
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
    if 'num_boost_round' in best_params:
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
    args = parse_arguments()
    
    print("=" * 80)
    print("LIGHTGBM HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Number of trials: {args.n_trials}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Model type: {args.model_type}")
    print(f"Optimize meta-models: {args.optimize_meta}")
    print("=" * 80)
    
    # Create params directory if it doesn't exist
    os.makedirs('params', exist_ok=True)
    
    # Load data for base models
    if args.model_type in ['all', 'lc', 'halc', 'cs']:
        X, y_lc, y_halc, y_cs = load_data()
        
        if args.model_type == 'all' or args.model_type == 'lc':
            # Optimize LC model
            run_optimization(objective_lc, X, y_lc, args.n_trials, args.n_folds, "lc")
        
        if args.model_type == 'all' or args.model_type == 'halc':
            # Optimize HALC model
            run_optimization(objective_halc, X, y_halc, args.n_trials, args.n_folds, "halc")
        
        if args.model_type == 'all' or args.model_type == 'cs':
            # Optimize CS model
            run_optimization(objective_cs, X, y_cs, args.n_trials, args.n_folds, "cs")
    
    # Load data for meta-models
    if args.optimize_meta or args.model_type in ['meta_lc', 'meta_halc', 'meta_cs']:
        # Load meta features
        meta_X, meta_y_lc, meta_y_halc, meta_y_cs = load_meta_features()
        
        if meta_X is not None:
            if args.model_type == 'all' or args.model_type == 'meta_lc':
                # Optimize meta LC model
                run_optimization(objective_lc, meta_X, meta_y_lc, args.n_trials, args.n_folds, "meta_lc")
            
            if args.model_type == 'all' or args.model_type == 'meta_halc':
                # Optimize meta HALC model
                run_optimization(objective_halc, meta_X, meta_y_halc, args.n_trials, args.n_folds, "meta_halc")
            
            if args.model_type == 'all' or args.model_type == 'meta_cs':
                # Optimize meta CS model
                run_optimization(objective_cs, meta_X, meta_y_cs, args.n_trials, args.n_folds, "meta_cs")
        else:
            print("Skipping meta-model optimization due to missing out-of-bag predictions.")
            print("Run 'python ensemble-pipeline.py --train' first to generate OOF predictions.")
    
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()