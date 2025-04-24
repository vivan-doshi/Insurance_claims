import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import argparse
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import re
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
    default_params = {
        'lc': {
            'objective': 'regression_l2', # Changed to l2 for consistency with tuning
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
            'subsample_freq': 0, # Added default
            'tweedie_variance_power': 1.5 # Added default for tweedie
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
            'subsample_freq': 0, # Added default
            'tweedie_variance_power': 1.5 # Added default for tweedie
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
        },
        'meta_lc': {
            'objective': 'regression_l2', # Changed to l2
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03, # Adjusted default
            'n_estimators': 500,
            'max_depth': 5, # Adjusted default
            'num_leaves': 16, # Adjusted default
            'min_data_in_leaf': 10, # Adjusted default
            'lambda_l1': 0.01, # Adjusted default
            'lambda_l2': 0.01, # Adjusted default
            'bagging_fraction': 0.9, # Adjusted default
            'feature_fraction': 0.9, # Adjusted default
            'min_child_weight': 1e-3, # Added default
            'subsample_freq': 0, # Added default
            'tweedie_variance_power': 1.5 # Added default for tweedie
        },
        'meta_halc': {
            'objective': 'regression_l2', # Changed to l2
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03, # Adjusted default
            'n_estimators': 500,
            'max_depth': 5, # Adjusted default
            'num_leaves': 16, # Adjusted default
            'min_data_in_leaf': 10, # Adjusted default
            'lambda_l1': 0.01, # Adjusted default
            'lambda_l2': 0.01, # Adjusted default
            'bagging_fraction': 0.9, # Adjusted default
            'feature_fraction': 0.9, # Adjusted default
            'min_child_weight': 1e-3, # Added default
            'subsample_freq': 0, # Added default
            'tweedie_variance_power': 1.5 # Added default for tweedie
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

    # Create directories for models
    os.makedirs('models', exist_ok=True)

    # Initialize k-fold and stratified k-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Initialize arrays to store OOF predictions
    oof_lc = np.zeros(len(X))
    oof_halc = np.zeros(len(X))
    oof_cs = np.zeros(len(X))

    # Initialize lists to store models
    lc_models = []
    halc_models = []
    cs_models = []

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
        callbacks = [lgb.early_stopping(100, verbose=False)] # Increased early stopping rounds

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        oof_lc[original_val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        # Save model
        lc_models.append(model)
        model.save_model(f'models/lc_model_fold_{fold}.txt')

        print(f"LC Fold {fold+1} complete")

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
        callbacks = [lgb.early_stopping(100, verbose=False)] # Increased early stopping rounds

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        oof_halc[original_val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        # Save model
        halc_models.append(model)
        model.save_model(f'models/halc_model_fold_{fold}.txt')

        print(f"HALC Fold {fold+1} complete")

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
        callbacks = [lgb.early_stopping(50, verbose=False)] # Keep 50 for classification

        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Make predictions
        oof_cs[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        # Save model
        cs_models.append(model)
        model.save_model(f'models/cs_model_fold_{fold}.txt')

        print(f"CS Fold {fold+1} complete")

    # Evaluate OOF predictions
    lc_rmse = np.sqrt(mean_squared_error(y_lc, oof_lc))
    halc_rmse = np.sqrt(mean_squared_error(y_halc, oof_halc))
    cs_auc = roc_auc_score(y_cs, oof_cs)

    print(f"OOF Loss Cost RMSE: {lc_rmse}")
    print(f"OOF HALC RMSE: {halc_rmse}")
    print(f"OOF Claim Status AUC: {cs_auc}")

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

    return lc_models, halc_models, cs_models, oof_df

def create_meta_features(X, oof_df):
    print("Creating meta features...")

    # Create meta-features by combining original features with OOF predictions
    meta_X = X.copy()
    meta_X['oof_lc'] = oof_df['oof_lc']
    meta_X['oof_halc'] = oof_df['oof_halc']
    meta_X['oof_cs'] = oof_df['oof_cs']

    return meta_X

def train_meta_models(meta_X, y_lc, y_halc, y_cs, best_params, n_folds=5, n_bins_regression_stratification=10):
    print("Training meta models...")

    # Create directories
    os.makedirs('meta_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Initialize k-fold and stratified k-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Initialize arrays to store meta OOF predictions
    meta_oof_lc = np.zeros(len(meta_X))
    meta_oof_halc = np.zeros(len(meta_X))
    meta_oof_cs = np.zeros(len(meta_X))

    # Initialize lists to store meta models
    meta_lc_models = []
    meta_halc_models = []
    meta_cs_models = []

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
        meta_oof_lc[original_val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        # Save model
        meta_lc_models.append(model)
        model.save_model(f'meta_models/meta_lc_model_fold_{fold}.txt')

        print(f"Meta LC Fold {fold+1} complete")

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
        meta_oof_halc[original_val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        # Save model
        meta_halc_models.append(model)
        model.save_model(f'meta_models/meta_halc_model_fold_{fold}.txt')

        print(f"Meta HALC Fold {fold+1} complete")

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
        meta_oof_cs[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

        # Save model
        meta_cs_models.append(model)
        model.save_model(f'meta_models/meta_cs_model_fold_{fold}.txt')

        print(f"Meta CS Fold {fold+1} complete")

    # Evaluate meta OOF predictions
    meta_lc_rmse = np.sqrt(mean_squared_error(y_lc, meta_oof_lc))
    meta_halc_rmse = np.sqrt(mean_squared_error(y_halc, meta_oof_halc))
    meta_cs_auc = roc_auc_score(y_cs, meta_oof_cs)

    print(f"Meta OOF Loss Cost RMSE: {meta_lc_rmse}")
    print(f"Meta OOF HALC RMSE: {meta_halc_rmse}")
    print(f"Meta OOF Claim Status AUC: {meta_cs_auc}")

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

    return meta_lc_models, meta_halc_models, meta_cs_models

def load_models():
    print("Loading trained models...")

    # Initialize lists to store models
    base_models_lc = []
    base_models_halc = []
    base_models_cs = []
    meta_models_lc = []
    meta_models_halc = []
    meta_models_cs = []

    # Load base models
    for fold in range(5):
        try:
            lc_model = lgb.Booster(model_file=f'models/lc_model_fold_{fold}.txt')
            halc_model = lgb.Booster(model_file=f'models/halc_model_fold_{fold}.txt')
            cs_model = lgb.Booster(model_file=f'models/cs_model_fold_{fold}.txt')

            base_models_lc.append(lc_model)
            base_models_halc.append(halc_model)
            base_models_cs.append(cs_model)
        except Exception as e:
            print(f"Warning: Could not load base models for fold {fold}: {e}")

    # Load meta models
    for fold in range(5):
        try:
            meta_lc_model = lgb.Booster(model_file=f'meta_models/meta_lc_model_fold_{fold}.txt')
            meta_halc_model = lgb.Booster(model_file=f'meta_models/meta_halc_model_fold_{fold}.txt')
            meta_cs_model = lgb.Booster(model_file=f'meta_models/meta_cs_model_fold_{fold}.txt')

            meta_models_lc.append(meta_lc_model)
            meta_halc_models.append(meta_halc_model)
            meta_cs_models.append(meta_cs_model)
        except Exception as e:
            print(f"Warning: Could not load meta models for fold {fold}: {e}")

    return base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs

def predict(X_test, base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs):
    print("Making predictions...")

    # Initialize arrays to store predictions from base models
    test_pred_lc = np.zeros(len(X_test))
    test_pred_halc = np.zeros(len(X_test))
    test_pred_cs = np.zeros(len(X_test))

    # Make predictions with base models
    print("Predicting with base models...")
    # Check if base models were loaded
    if not base_models_lc:
        print("Error: Base Loss Cost models not loaded. Cannot make base predictions.")
        return None, None, None # Return None for predictions if models are missing
    if not base_models_halc:
        print("Error: Base HALC models not loaded. Cannot make base predictions.")
        return None, None, None
    if not base_models_cs:
        print("Error: Base Claim Status models not loaded. Cannot make base predictions.")
        return None, None, None


    for i, (lc_model, halc_model, cs_model) in enumerate(zip(base_models_lc, base_models_halc, base_models_cs)):
        test_pred_lc += lc_model.predict(X_test) / len(base_models_lc)
        test_pred_halc += halc_model.predict(X_test) / len(base_models_halc)
        test_pred_cs += cs_model.predict(X_test) / len(base_models_cs)

    # Create meta features for test data
    meta_X_test = X_test.copy()
    meta_X_test['oof_lc'] = test_pred_lc
    meta_X_test['oof_halc'] = test_pred_halc
    meta_X_test['oof_cs'] = test_pred_cs

    # Initialize arrays to store meta model predictions
    meta_test_pred_lc = np.zeros(len(X_test))
    meta_test_pred_halc = np.zeros(len(X_test))
    meta_test_pred_cs = np.zeros(len(X_test))

    # Make predictions with meta models
    print("Predicting with meta models...")
    # Check if meta models were loaded
    if not meta_models_lc:
        print("Error: Meta Loss Cost models not loaded. Cannot make meta predictions.")
        return None, None, None # Return None for predictions if models are missing
    if not meta_models_halc:
        print("Error: Meta HALC models not loaded. Cannot make meta predictions.")
        return None, None, None
    if not meta_models_cs:
        print("Error: Meta Claim Status models not loaded. Cannot make meta predictions.")
        return None, None, None


    for i, (lc_model, halc_model, cs_model) in enumerate(zip(meta_models_lc, meta_models_halc, meta_models_cs)):
        meta_test_pred_lc += lc_model.predict(meta_X_test) / len(meta_models_lc)
        meta_test_pred_halc += halc_model.predict(meta_X_test) / len(meta_halc_models)
        meta_test_pred_cs += cs_model.predict(meta_X_test) / len(meta_models_cs)

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
    lc_models, halc_models, cs_models = [], [], []
    meta_lc_models, meta_halc_models, meta_cs_models = [], [], []
    oof_df = None # Initialize oof_df as well


    if args.train or args.train_base:
        # Train base models
        lc_models, halc_models, cs_models, oof_df = train_base_models(
            X_train, y_lc, y_halc, y_cs, best_params, args.n_folds, args.n_bins_regression_stratification
        )

        if args.train or args.train_meta:
            # Create meta features
            meta_X = create_meta_features(X_train, oof_df)

            # Train meta models
            meta_lc_models, meta_halc_models, meta_cs_models = train_meta_models(
                meta_X, y_lc, y_halc, y_cs, best_params, args.n_folds, args.n_bins_regression_stratification
            )


    elif args.train_meta:
        # Load OOF predictions
        if os.path.exists('oof_predictions.csv'):
            oof_df = pd.read_csv('oof_predictions.csv')

            # Create meta features
            meta_X = create_meta_features(X_train, oof_df)

            # Train meta models
            meta_lc_models, meta_halc_models, meta_cs_models = train_meta_models(
                meta_X, y_lc, y_halc, y_cs, best_params, args.n_folds, args.n_bins_regression_stratification
            )
        else:
            print("Error: 'oof_predictions.csv' file not found. Run base models training first.")
            return

    if args.predict:
        # Load trained models - these will overwrite the initialized empty lists if successful
        base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs = load_models()

        # Check if models were loaded successfully
        if not base_models_lc or not meta_models_lc:
            print("Error: Could not load required models. Make sure to train base and meta models first.")
            return

        # Make predictions
        final_predictions = predict(
            X_test,
            base_models_lc,
            base_models_halc,
            base_models_cs,
            meta_models_lc,
            meta_models_halc,
            meta_models_cs
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
