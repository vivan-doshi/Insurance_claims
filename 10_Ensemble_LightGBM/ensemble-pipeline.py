import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import argparse
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import re
import json # Import json for saving/loading scores

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def parse_arguments():
    parser = argparse.ArgumentParser(description='LightGBM Ensemble Pipeline')
    parser.add_argument('--train', action='store_true', help='Train base and meta models')
    parser.add_argument('--train_base', action='store_true', help='Train only base models')
    parser.add_argument('--train_meta', action='store_true', help='Train only meta models')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
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
    print("Loading data...")
    X_train = pd.read_csv('feature_selected_train.csv', index_col=0)
    y_df = pd.read_csv('feature_selected_y_train.csv', index_col=0)
    X_test = pd.read_csv('feature_selected_test.csv', index_col=0)

    # Clean feature names to avoid LightGBM errors
    X_train = clean_feature_names(X_train)
    X_test = clean_feature_names(X_test)

    # Extract target variables
    y_lc = y_df['Loss_Cost']
    y_halc = y_df['Historically_Adjusted_Loss_Cost']
    y_cs = y_df['Claim_Status']

    print(f"Features shape: {X_train.shape}")
    print(f"Target shapes - Loss_Cost: {y_lc.shape}, HALC: {y_halc.shape}, Claim_Status: {y_cs.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_lc, y_halc, y_cs, X_test

def load_best_params():
    best_params = {}

    # Ensure directories exist
    os.makedirs('params', exist_ok=True)

    # Path to parameter files
    param_files = {
        'lc': 'params/best_params_lc.txt',
        'halc': 'params/best_params_halc.txt',
        'cs': 'params/best_params_cs.txt',
        'meta_lc': 'params/best_params_meta_lc.txt',
        'meta_halc': 'params/best_params_meta_halc.txt',
        'meta_cs': 'params/best_params_meta_cs.txt'
    }

    # Default parameters if files don't exist (should be replaced by optimized params)
    # Note: These defaults should ideally be replaced by running hyperparameter optimization
    default_params = {
        'lc': {
            'objective': 'tweedie', # Using tweedie as requested
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'max_depth': -1,
            'num_leaves': 64,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'min_child_weight': 1e-3,
            'subsample_freq': 0,
            'tweedie_variance_power': 1.5 # Default tweedie power
        },
        'halc': {
            'objective': 'tweedie', # Using tweedie as requested
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'max_depth': -1,
            'num_leaves': 64,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'min_child_weight': 1e-3,
            'subsample_freq': 0,
            'tweedie_variance_power': 1.5 # Default tweedie power
        },
        'cs': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'max_depth': -1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'scale_pos_weight': 3.0
        },
        'meta_lc': {
            'objective': 'regression_l2', # Meta model can use L2 or Tweedie
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'n_estimators': 500,
            'max_depth': 5,
            'num_leaves': 16,
            'min_data_in_leaf': 10,
            'lambda_l1': 0.01,
            'lambda_l2': 0.01,
            'bagging_fraction': 0.9,
            'feature_fraction': 0.9,
            'min_child_weight': 1e-3,
            'subsample_freq': 0,
            'tweedie_variance_power': 1.5 # Default tweedie power for meta if used
        },
        'meta_halc': {
            'objective': 'regression_l2', # Meta model can use L2 or Tweedie
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'n_estimators': 500,
            'max_depth': 5,
            'num_leaves': 16,
            'min_data_in_leaf': 10,
            'lambda_l1': 0.01,
            'lambda_l2': 0.01,
            'bagging_fraction': 0.9,
            'feature_fraction': 0.9,
            'min_child_weight': 1e-3,
            'subsample_freq': 0,
            'tweedie_variance_power': 1.5 # Default tweedie power for meta if used
        },
        'meta_cs': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 8,
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
            print(f"Loaded parameters for {model_type} from {file_path}")
        else:
            print(f"Parameter file {file_path} not found. Using default parameters for {model_type}.")
            best_params[model_type] = default_params[model_type]

    return best_params

def train_base_models(X, y_lc, y_halc, y_cs, best_params, n_folds=5, n_bins_regression_stratification=10):
    print("Training base models...")

    # Create directories for models and scores
    os.makedirs('models', exist_ok=True)
    os.makedirs('scores', exist_ok=True)

    # Initialize k-fold and stratified k-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Initialize arrays to store OOF predictions
    oof_lc = np.zeros(len(X))
    oof_halc = np.zeros(len(X))
    oof_cs = np.zeros(len(X))

    # Initialize lists to store fold-wise scores
    lc_fold_scores = []
    halc_fold_scores = []
    cs_fold_scores = []

    # Initialize lists to store models (optional, models are saved to disk)
    # lc_models = []
    # halc_models = []
    # cs_models = []

    # Bin regression targets for stratification
    y_lc_binned = bin_regression_target(y_lc, n_bins_regression_stratification)
    y_halc_binned = bin_regression_target(y_halc, n_bins_regression_stratification)

    # Loss Cost models (Stratified CV)
    print("Training Loss Cost base models (Stratified CV)...")
    # Handle potential NaNs in binned target for StratifiedKFold
    non_nan_indices_lc = y_lc_binned.dropna().index
    X_lc_non_nan = X.loc[non_nan_indices_lc]
    y_lc_non_nan = y_lc.loc[non_nan_indices_lc]
    y_lc_binned_non_nan = y_lc_binned.loc[non_nan_indices_lc].astype(int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_lc_non_nan, y_lc_binned_non_nan)):
        # Map back to original indices
        original_train_idx = non_nan_indices_lc[train_idx]
        original_val_idx = non_nan_indices_lc[val_idx]

        X_train, X_val = X.loc[original_train_idx], X.loc[original_val_idx]
        y_train, y_val = y_lc.loc[original_train_idx], y_lc.loc[original_val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Get parameters for training
        params = best_params['lc'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 1000

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(100, verbose=False)]

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_lc[original_val_idx] = val_preds

        # Calculate and store fold score (RMSE)
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        lc_fold_scores.append(fold_rmse)

        # Save model
        # lc_models.append(model)
        model.save_model(f'models/lc_model_fold_{fold}.txt')

        print(f"LC Fold {fold+1} complete. RMSE: {fold_rmse:.4f}")

    # HALC models (Stratified CV)
    print("Training HALC base models (Stratified CV)...")
    # Handle potential NaNs in binned target for StratifiedKFold
    non_nan_indices_halc = y_halc_binned.dropna().index
    X_halc_non_nan = X.loc[non_nan_indices_halc]
    y_halc_non_nan = y_halc.loc[non_nan_indices_halc]
    y_halc_binned_non_nan = y_halc_binned.loc[non_nan_indices_halc].astype(int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_halc_non_nan, y_halc_binned_non_nan)):
        # Map back to original indices
        original_train_idx = non_nan_indices_halc[train_idx]
        original_val_idx = non_nan_indices_halc[val_idx]

        X_train, X_val = X.loc[original_train_idx], X.loc[original_val_idx]
        y_train, y_val = y_halc.loc[original_train_idx], y_halc.loc[original_val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Get parameters for training
        params = best_params['halc'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 1000

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(100, verbose=False)]

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_halc[original_val_idx] = val_preds

        # Calculate and store fold score (RMSE)
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        halc_fold_scores.append(fold_rmse)

        # Save model
        # halc_models.append(model)
        model.save_model(f'models/halc_model_fold_{fold}.txt')

        print(f"HALC Fold {fold+1} complete. RMSE: {fold_rmse:.4f}")

    # Claim Status models (Stratified CV)
    print("Training Claim Status base models (Stratified CV)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cs)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_cs.iloc[train_idx], y_cs.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Get parameters for training
        params = best_params['cs'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 1000

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_cs[val_idx] = val_preds

        # Calculate and store fold score (AUC)
        fold_auc = roc_auc_score(y_val, val_preds)
        cs_fold_scores.append(fold_auc)

        # Save model
        # cs_models.append(model)
        model.save_model(f'models/cs_model_fold_{fold}.txt')

        print(f"CS Fold {fold+1} complete. AUC: {fold_auc:.4f}")

    # Evaluate overall OOF predictions
    lc_rmse = np.sqrt(mean_squared_error(y_lc, oof_lc))
    halc_rmse = np.sqrt(mean_squared_error(y_halc, oof_halc))
    cs_auc = roc_auc_score(y_cs, oof_cs)

    print(f"Overall OOF Loss Cost RMSE: {lc_rmse:.4f}")
    print(f"Overall OOF HALC RMSE: {halc_rmse:.4f}")
    print(f"Overall OOF Claim Status AUC: {cs_auc:.4f}")

    # Save OOF predictions for meta-model training
    oof_df = pd.DataFrame({
        'oof_lc': oof_lc,
        'oof_halc': oof_halc,
        'oof_cs': oof_cs,
        'true_lc': y_lc,
        'true_halc': y_halc,
        'true_cs': y_cs
    })
    oof_df.to_csv('oof_predictions.csv', index=False)

    # Save fold-wise scores for weighted averaging
    base_model_scores = {
        'lc_rmse': lc_fold_scores,
        'halc_rmse': halc_fold_scores,
        'cs_auc': cs_fold_scores
    }
    with open('scores/base_model_fold_scores.json', 'w') as f:
        json.dump(base_model_scores, f)

    print("Base model fold scores saved to 'scores/base_model_fold_scores.json'")

    # Return None for models as they are saved to disk and reloaded for prediction/meta-training
    return None, None, None, oof_df


def create_meta_features(X, oof_df):
    print("Creating meta features...")

    # Create meta-features by combining original features with OOF predictions
    # The meta-model will learn how to combine these.
    meta_X = X.copy()
    meta_X['oof_lc'] = oof_df['oof_lc']
    meta_X['oof_halc'] = oof_df['oof_halc']
    meta_X['oof_cs'] = oof_df['oof_cs']

    return meta_X

def train_meta_models(meta_X, y_lc, y_halc, y_cs, best_params, n_folds=5, n_bins_regression_stratification=10):
    print("Training meta models...")

    # Create directories
    os.makedirs('meta_models', exist_ok=True)
    # os.makedirs('plots', exist_ok=True) # Assuming plots are handled elsewhere

    # Initialize k-fold and stratified k-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Initialize arrays to store meta OOF predictions
    meta_oof_lc = np.zeros(len(meta_X))
    meta_oof_halc = np.zeros(len(meta_X))
    meta_oof_cs = np.zeros(len(meta_X))

    # Initialize lists to store fold-wise meta scores (optional)
    # meta_lc_fold_scores = []
    # meta_halc_fold_scores = []
    # meta_cs_fold_scores = []

    # Initialize lists to store meta models (optional, models are saved to disk)
    # meta_lc_models = []
    # meta_halc_models = []
    # meta_cs_models = []

    # Bin regression targets for stratification
    y_lc_binned = bin_regression_target(y_lc, n_bins_regression_stratification)
    y_halc_binned = bin_regression_target(y_halc, n_bins_regression_stratification)


    # Meta Loss Cost models (Stratified CV)
    print("Training Meta Loss Cost models (Stratified CV)...")
    # Handle potential NaNs in binned target for StratifiedKFold
    non_nan_indices_lc = y_lc_binned.dropna().index
    X_lc_non_nan = meta_X.loc[non_nan_indices_lc]
    y_lc_non_nan = y_lc.loc[non_nan_indices_lc]
    y_lc_binned_non_nan = y_lc_binned.loc[non_nan_indices_lc].astype(int)


    for fold, (train_idx, val_idx) in enumerate(skf.split(X_lc_non_nan, y_lc_binned_non_nan)):
        # Map back to original indices
        original_train_idx = non_nan_indices_lc[train_idx]
        original_val_idx = non_nan_indices_lc[val_idx]

        X_train, X_val = meta_X.loc[original_train_idx], meta_X.loc[original_val_idx]
        y_train, y_val = y_lc.loc[original_train_idx], y_lc.loc[original_val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Get parameters for training
        params = best_params['meta_lc'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 500

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        meta_oof_lc[original_val_idx] = val_preds

        # Calculate and store fold score (RMSE) - optional for meta
        # fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        # meta_lc_fold_scores.append(fold_rmse)

        # Save model
        # meta_lc_models.append(model)
        model.save_model(f'meta_models/meta_lc_model_fold_{fold}.txt')

        print(f"Meta LC Fold {fold+1} complete.") # RMSE: {fold_rmse:.4f}")

    # Meta HALC models (Stratified CV)
    print("Training Meta HALC models (Stratified CV)...")
    # Handle potential NaNs in binned target for StratifiedKFold
    non_nan_indices_halc = y_halc_binned.dropna().index
    X_halc_non_nan = meta_X.loc[non_nan_indices_halc]
    y_halc_non_nan = y_halc.loc[non_nan_indices_halc]
    y_halc_binned_non_nan = y_halc_binned.loc[non_nan_indices_halc].astype(int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_halc_non_nan, y_halc_binned_non_nan)):
        # Map back to original indices
        original_train_idx = non_nan_indices_halc[train_idx]
        original_val_idx = non_nan_indices_halc[val_idx]

        X_train, X_val = meta_X.loc[original_train_idx], meta_X.loc[original_val_idx]
        y_train, y_val = y_halc.loc[original_train_idx], y_halc.loc[original_val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Get parameters for training
        params = best_params['meta_halc'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 500

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        meta_oof_halc[original_val_idx] = val_preds

        # Calculate and store fold score (RMSE) - optional for meta
        # fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        # meta_halc_fold_scores.append(fold_rmse)

        # Save model
        # meta_halc_models.append(model)
        model.save_model(f'meta_models/meta_halc_model_fold_{fold}.txt')

        print(f"Meta HALC Fold {fold+1} complete.") # RMSE: {fold_rmse:.4f}")

    # Meta Claim Status models (Stratified CV)
    print("Training Meta Claim Status models (Stratified CV)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y_cs)):
        X_train, X_val = meta_X.iloc[train_idx], meta_X.iloc[val_idx]
        y_train, y_val = y_cs.iloc[train_idx], y_cs.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Get parameters for training
        params = best_params['meta_cs'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 500

        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        meta_oof_cs[val_idx] = val_preds

        # Calculate and store fold score (AUC) - optional for meta
        # fold_auc = roc_auc_score(y_val, val_preds)
        # meta_cs_fold_scores.append(fold_auc)

        # Save model
        # meta_cs_models.append(model)
        model.save_model(f'meta_models/meta_cs_model_fold_{fold}.txt')

        print(f"Meta CS Fold {fold+1} complete.") # AUC: {fold_auc:.4f}")

    # Evaluate meta OOF predictions
    meta_lc_rmse = np.sqrt(mean_squared_error(y_lc, meta_oof_lc))
    meta_halc_rmse = np.sqrt(mean_squared_error(y_halc, meta_oof_halc))
    meta_cs_auc = roc_auc_score(y_cs, meta_oof_cs)

    print(f"Overall Meta OOF Loss Cost RMSE: {meta_lc_rmse:.4f}")
    print(f"Overall Meta OOF HALC RMSE: {meta_halc_rmse:.4f}")
    print(f"Overall Meta OOF Claim Status AUC: {meta_cs_auc:.4f}")

    # Save meta OOF predictions
    meta_oof_df = pd.DataFrame({
        'meta_oof_lc': meta_oof_lc,
        'meta_oof_halc': meta_oof_halc,
        'meta_oof_cs': meta_oof_cs,
        'true_lc': y_lc,
        'true_halc': y_halc,
        'true_cs': y_cs
    })

    meta_oof_df.to_csv('meta_oof_predictions.csv', index=False)

    # Return None for models as they are saved to disk
    return None, None, None

def load_models(n_folds=5):
    print("Loading trained models...")

    # Initialize lists to store models
    base_models_lc = []
    base_models_halc = []
    base_models_cs = []
    meta_models_lc = []
    meta_models_halc = []
    meta_models_cs = []

    # Load base models
    for fold in range(n_folds):
        try:
            lc_model = lgb.Booster(model_file=f'models/lc_model_fold_{fold}.txt')
            halc_model = lgb.Booster(model_file=f'models/halc_model_fold_{fold}.txt')
            cs_model = lgb.Booster(model_file=f'models/cs_model_fold_{fold}.txt')

            base_models_lc.append(lc_model)
            base_models_halc.append(halc_model)
            base_models_cs.append(cs_model)
        except Exception as e:
            print(f"Warning: Could not load base models for fold {fold}: {e}")
            # If models for a fold are missing, we cannot proceed with weighted averaging
            # Consider raising an error or skipping prediction if models are incomplete
            return [], [], [], [], [], [] # Return empty lists to indicate failure

    # Load meta models
    for fold in range(n_folds):
        try:
            meta_lc_model = lgb.Booster(model_file=f'meta_models/meta_lc_model_fold_{fold}.txt')
            meta_halc_model = lgb.Booster(model_file=f'meta_models/meta_halc_model_fold_{fold}.txt')
            meta_cs_model = lgb.Booster(model_file=f'meta_models/meta_cs_model_fold_{fold}.txt')

            meta_models_lc.append(meta_lc_model)
            meta_halc_models.append(meta_halc_model)
            meta_cs_models.append(meta_cs_model)
        except Exception as e:
            print(f"Warning: Could not load meta models for fold {fold}: {e}")
            # If meta models for a fold are missing, the meta prediction step will fail
            return base_models_lc, base_models_halc, base_models_cs, [], [], [] # Return empty meta lists


    return base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs

def calculate_weights(scores, score_type='rmse'):
    """Calculates weights based on fold scores."""
    if not scores:
        return [] # Return empty if no scores

    if score_type == 'rmse':
        # For RMSE, lower is better, so use inverse
        # Handle potential zero RMSE or very small values by adding a small epsilon
        epsilon = 1e-6
        inverse_scores = [1.0 / (s + epsilon) for s in scores]
        total_inverse = sum(inverse_scores)
        if total_inverse == 0:
             return [1.0 / len(scores)] * len(scores) # Equal weights if sum is zero
        weights = [s / total_inverse for s in inverse_scores]
    elif score_type == 'auc':
        # For AUC, higher is better, use scores directly
        total_score = sum(scores)
        if total_score == 0:
             return [1.0 / len(scores)] * len(scores) # Equal weights if sum is zero
        weights = [s / total_score for s in scores]
    else:
        # Default to equal weights if score_type is unknown
        print(f"Warning: Unknown score type '{score_type}'. Using equal weights.")
        weights = [1.0 / len(scores)] * len(scores)

    return weights


def predict(X_test, base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs, n_folds=5):
    print("Making predictions...")

    # Check if models were loaded successfully
    if not base_models_lc or not base_models_halc or not base_models_cs:
        print("Error: Base models not loaded. Cannot make predictions.")
        return None

    # Load base model fold scores for weighted averaging
    base_model_scores_path = 'scores/base_model_fold_scores.json'
    if not os.path.exists(base_model_scores_path):
        print(f"Error: Base model fold scores file '{base_model_scores_path}' not found. Cannot perform weighted averaging.")
        # Fallback to simple averaging if scores are missing? Or require scores?
        # For now, let's require scores for weighted averaging.
        return None

    with open(base_model_scores_path, 'r') as f:
        base_model_scores = json.load(f)

    lc_fold_scores = base_model_scores.get('lc_rmse', [])
    halc_fold_scores = base_model_scores.get('halc_rmse', [])
    cs_fold_scores = base_model_scores.get('cs_auc', [])

    # Check if we have scores for all folds
    if len(lc_fold_scores) != n_folds or len(halc_fold_scores) != n_folds or len(cs_fold_scores) != n_folds:
        print(f"Error: Incomplete fold scores found ({len(lc_fold_scores)}/{n_folds} for LC, {len(halc_fold_scores)}/{n_folds} for HALC, {len(cs_fold_scores)}/{n_folds} for CS). Cannot perform weighted averaging.")
        return None


    # Calculate weights based on fold scores
    lc_weights = calculate_weights(lc_fold_scores, score_type='rmse')
    halc_weights = calculate_weights(halc_fold_scores, score_type='rmse')
    cs_weights = calculate_weights(cs_fold_scores, score_type='auc')

    # Initialize arrays to store weighted base predictions
    weighted_test_pred_lc = np.zeros(len(X_test))
    weighted_test_pred_halc = np.zeros(len(X_test))
    weighted_test_pred_cs = np.zeros(len(X_test))

    # Make predictions with base models and apply weights
    print("Predicting with base models and applying weights...")

    for i in range(n_folds):
        lc_model = base_models_lc[i]
        halc_model = base_models_halc[i]
        cs_model = base_models_cs[i]

        weighted_test_pred_lc += lc_model.predict(X_test) * lc_weights[i]
        weighted_test_pred_halc += halc_model.predict(X_test) * halc_weights[i]
        weighted_test_pred_cs += cs_model.predict(X_test) * cs_weights[i]

    # Create meta features for test data using weighted base predictions
    meta_X_test = X_test.copy()
    meta_X_test['weighted_base_lc'] = weighted_test_pred_lc
    meta_X_test['weighted_base_halc'] = weighted_test_pred_halc
    meta_X_test['weighted_base_cs'] = weighted_test_pred_cs

    # Initialize arrays to store meta model predictions
    meta_test_pred_lc = np.zeros(len(X_test))
    meta_test_pred_halc = np.zeros(len(X_test))
    meta_test_pred_cs = np.zeros(len(X_test))

    # Make predictions with meta models
    print("Predicting with meta models...")
    # Check if meta models were loaded successfully
    if not meta_models_lc or not meta_models_halc or not meta_models_cs:
        print("Error: Meta models not loaded. Cannot make meta predictions.")
        # In this case, you might choose to return the weighted base predictions
        # or return None depending on your desired pipeline behavior.
        # Let's return None to indicate the full ensemble prediction failed.
        return None

    for i in range(n_folds):
        meta_lc_model = meta_models_lc[i]
        meta_halc_model = meta_models_halc[i]
        meta_cs_model = meta_models_cs[i]

        meta_test_pred_lc += meta_lc_model.predict(meta_X_test) / len(meta_models_lc)
        meta_test_pred_halc += meta_halc_model.predict(meta_X_test) / len(meta_halc_models)
        meta_test_pred_cs += meta_cs_model.predict(meta_X_test) / len(meta_models_cs)

    # Create final predictions DataFrame
    final_predictions = pd.DataFrame({
        'LC': meta_test_pred_lc,
        'HALC': meta_test_pred_halc,
        'CS': (meta_test_pred_cs > 0.5).astype(int)  # Convert probabilities to binary predictions
    })

    # Save final predictions
    final_predictions.to_csv('final_predictions.csv', index=False)

    return final_predictions

def main():
    args = parse_arguments()

    X_train, y_lc, y_halc, y_cs, X_test = load_data()
    best_params = load_best_params()

    # Initialize model lists to ensure they are defined in the scope
    # Models are loaded from disk, so these are not used directly for training
    # but are needed for loading in the predict step.
    base_models_lc, base_models_halc, base_models_cs = [], [], []
    meta_lc_models, meta_halc_models, meta_cs_models = [], [], []
    oof_df = None # Initialize oof_df as well


    if args.train or args.train_base:
        # Train base models
        # train_base_models now returns None for models, but saves them to disk
        _, _, _, oof_df = train_base_models(
            X_train, y_lc, y_halc, y_cs, best_params, args.n_folds, args.n_bins_regression_stratification
        )

        if args.train or args.train_meta:
            # Create meta features
            # oof_df is available from base training or loaded from file
            if oof_df is None and os.path.exists('oof_predictions.csv'):
                 oof_df = pd.read_csv('oof_predictions.csv')
            elif oof_df is None:
                 print("Error: OOF predictions not available. Cannot train meta models.")
                 return # Exit if OOF is missing

            meta_X = create_meta_features(X_train, oof_df)

            # Train meta models
            # train_meta_models now returns None for models, but saves them to disk
            _, _, _ = train_meta_models(
                meta_X, y_lc, y_halc, y_cs, best_params, args.n_folds, args.n_bins_regression_stratification
            )


    elif args.train_meta:
        # Load OOF predictions
        if os.path.exists('oof_predictions.csv'):
            oof_df = pd.read_csv('oof_predictions.csv')

            # Create meta features
            meta_X = create_meta_features(X_train, oof_df)

            # Train meta models
            # train_meta_models now returns None for models, but saves them to disk
            _, _, _ = train_meta_models(
                meta_X, y_lc, y_halc, y_cs, best_params, args.n_folds, args.n_bins_regression_stratification
            )
        else:
            print("Error: 'oof_predictions.csv' file not found. Run base models training first.")
            return

    if args.predict:
        # Load trained models - these will overwrite the initialized empty lists if successful
        # load_models now takes n_folds as argument
        base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs = load_models(args.n_folds)

        # Make predictions
        # predict now takes n_folds as argument
        final_predictions = predict(
            X_test,
            base_models_lc,
            base_models_halc,
            base_models_cs,
            meta_models_lc,
            meta_models_halc,
            meta_models_cs,
            args.n_folds # Pass n_folds to predict
        )

        # Check if predictions were generated successfully
        if final_predictions is not None:
            print("Final predictions saved to 'final_predictions.csv'")
        else:
            print("Failed to generate final predictions.")


if __name__ == "__main__":
    print("=" * 80)
    print("LIGHTGBM ENSEMBLE PIPELINE")
    print("=" * 80)
    main()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

