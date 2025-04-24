import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import os
import re
from functools import partial # Import partial
import time # Import the time module
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Define file paths
X_TRAIN_PATH = 'feature_selected_train.csv'
Y_TRAIN_PATH = 'feature_selected_y_train.csv'

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


# Load data
def load_data():
    X = pd.read_csv(X_TRAIN_PATH, index_col=0)
    y = pd.read_csv(Y_TRAIN_PATH, index_col=0)

    # Clean feature names to avoid LightGBM errors
    X = clean_feature_names(X)

    # Extract target variables
    y_lc = y['Loss_Cost']
    y_halc = y['Historically_Adjusted_Loss_Cost']
    y_cs = y['Claim_Status']

    print(f"Features shape: {X.shape}")
    print(f"Target shapes - Loss_Cost: {y_lc.shape}, HALC: {y_halc.shape}, Claim_Status: {y_cs.shape}")

    return X, y_lc, y_halc, y_cs

# Create directory for model storage
os.makedirs('models', exist_ok=True)
os.makedirs('params', exist_ok=True)

# Load best parameters (This function is not directly used in the Optuna objective but is kept for completeness)
def load_best_params():
    best_params = {}

    # Path to parameter files
    param_files = {
        'lc': 'params/best_params_lc.txt',
        'halc': 'params/best_params_halc.txt',
        'cs': 'params/best_params_cs.txt'
    }

    # Default parameters if files don't exist
    default_params = {
        'lc': {
            'objective': 'regression_l2', # Changed to l2
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 1000, # Increased default estimators
            'max_depth': -1,
            'num_leaves': 64,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'min_child_weight': 1e-3, # Added default
            'subsample_freq': 0 # Added default
        },
        'halc': {
            'objective': 'regression_l2', # Changed to l2
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 1000, # Increased default estimators
            'max_depth': -1,
            'num_leaves': 64,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'min_child_weight': 1e-3, # Added default
            'subsample_freq': 0 # Added default
        },
        'cs': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 1000, # Increased default estimators
            'max_depth': -1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'scale_pos_weight': 3.0
        }
    }

    # Load parameters from files if they exist
    for model_type, file_path in param_files.items():
        if os.path.exists(file_path):
            params = {}
            with open(file_path, 'r') as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    try:
                        # Try to convert to numeric
                        params[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        # If not numeric, keep as string
                        params[key] = value
            best_params[model_type] = params
        else:
            print(f"Parameter file {file_path} not found. Using default parameters.")
            best_params[model_type] = default_params[model_type]

    return best_params

# Define objective function for Loss Cost (LC) regression
def objective_lc(trial, X, y, y_binned, n_folds=5):
    """Objective function for LC model optimization targeting RMSE with stratified CV"""
    # Include 'tweedie' in the objective search space
    objective_type = trial.suggest_categorical('objective', ['regression_l2', 'regression_l1', 'tweedie'])

    param = {
        'objective': objective_type,
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']), # Include dart
        'verbosity': -1,
        'n_jobs': -1,
        'seed': SEED,

        # Hyperparameters to optimize - expanded search space
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), # Narrowed range slightly
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000), # Increased range, rely on early stopping
        'num_leaves': trial.suggest_int('num_leaves', 16, 128), # Adjusted range
        'max_depth': trial.suggest_int('max_depth', -1, 15), # Increased max depth
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100), # Adjusted range
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True), # Added min_child_weight
        'max_bin': trial.suggest_int('max_bin', 128, 512), # Increased max_bin
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True), # Adjusted range
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True), # Adjusted range
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1), # Adjusted range
        'subsample': trial.suggest_float('subsample', 0.7, 1.0), # Added subsample (bagging_fraction)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0), # Added colsample_bytree (feature_fraction)
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 5), # Added subsample_freq
    }

    # Add tweedie_variance_power if objective is tweedie
    if objective_type == 'tweedie':
        param['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.0, 2.0) # Common range for Tweedie

    # Perform stratified k-fold cross-validation on binned target
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores = []

    # Note: StratifiedKFold requires integer labels. Handle potential NaNs in binned target.
    # We will split based on non-NaN binned values and apply the split to the original dataframes.
    non_nan_indices = y_binned.dropna().index
    X_non_nan = X.loc[non_nan_indices]
    y_non_nan = y.loc[non_nan_indices]
    y_binned_non_nan = y_binned.loc[non_nan_indices].astype(int) # StratifiedKFold needs int

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_non_nan, y_binned_non_nan)):
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
    # Include 'tweedie' in the objective search space
    objective_type = trial.suggest_categorical('objective', ['regression_l2', 'regression_l1', 'tweedie'])

    param = {
        'objective': objective_type,
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']), # Include dart
        'verbosity': -1,
        'n_jobs': -1,
        'seed': SEED,

        # Hyperparameters to optimize - expanded search space
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), # Narrowed range slightly
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000), # Increased range, rely on early stopping
        'num_leaves': trial.suggest_int('num_leaves', 16, 128), # Adjusted range
        'max_depth': trial.suggest_int('max_depth', -1, 15), # Increased max depth
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100), # Adjusted range
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True), # Added min_child_weight
        'max_bin': trial.suggest_int('max_bin', 128, 512), # Increased max_bin
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True), # Adjusted range
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True), # Adjusted range
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1), # Adjusted range
        'subsample': trial.suggest_float('subsample', 0.7, 1.0), # Added subsample (bagging_fraction)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0), # Added colsample_bytree (feature_fraction)
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 5), # Added subsample_freq
    }

    # Add tweedie_variance_power if objective is tweedie
    if objective_type == 'tweedie':
        param['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.0, 2.0) # Common range for Tweedie


    # Perform stratified k-fold cross-validation on binned target
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmse_scores = []

    # Note: StratifiedKFold requires integer labels. Handle potential NaNs in binned target.
    # We will split based on non-NaN binned values and apply the split to the original dataframes.
    non_nan_indices = y_binned.dropna().index
    X_non_nan = X.loc[non_nan_indices]
    y_non_nan = y.loc[non_nan_indices]
    y_binned_non_nan = y_binned.loc[non_nan_indices].astype(int) # StratifiedKFold needs int


    for fold, (train_idx, val_idx) in enumerate(skf.split(X_non_nan, y_binned_non_nan)):
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
        'metric': 'auc',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'verbosity': -1,
        'n_jobs': -1,
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
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
    }

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
    print(f"Starting optimization for {study_name}...")
    start_time = time.time()

    # Create optimization study
    # You can add a storage URL here to save/resume studies, e.g., "sqlite:///optuna_study.db"
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
    print("=" * 80)
    print("LIGHTGBM BASE MODEL HYPERPARAMETER OPTIMIZATION (using lightgbm-optuna-script.py)")
    print("=" * 80)

    # Load data for base models
    X, y_lc, y_halc, y_cs = load_data()

    # Define number of folds and bins for stratification (you can make these command line arguments if needed)
    n_folds = 5
    n_bins_regression_stratification = 10

    # Bin regression targets for stratification
    y_lc_binned = bin_regression_target(y_lc, n_bins_regression_stratification)
    y_halc_binned = bin_regression_target(y_halc, n_bins_regression_stratification)

    # Optimize LC model with stratified CV
    run_optimization(objective_lc, X, y_lc, y_lc_binned, n_trials=50, n_folds=n_folds, study_name="lc")

    # Optimize HALC model with stratified CV
    run_optimization(objective_halc, X, y_halc, y_halc_binned, n_trials=50, n_folds=n_folds, study_name="halc")

    # Optimize CS model (classification already uses StratifiedKFold)
    run_optimization(objective_cs, X, y_cs, n_trials=50, n_folds=n_folds, study_name="cs")


    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
