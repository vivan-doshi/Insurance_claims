# This script is primarily for training the meta-model and making final predictions.
# Hyperparameter tuning for the meta-model is handled in hyperparameter-optimization.py
# UPDATED VERSION - Added Tweedie Objective for Meta-Regression
# Included 'tweedie' objective and 'tweedie_variance_power' parameter
# Added tweedie_variance_power to default parameters in load_best_params
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Define file paths
OOF_PREDICTIONS_PATH = 'oof_predictions.csv'
X_TRAIN_PATH = 'feature_selected_train.csv'
X_TEST_PATH = 'feature_selected_test.csv'

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


# Create directories
os.makedirs('meta_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('params', exist_ok=True)

# Load data
def load_meta_data():
    """Loads meta features and true targets for meta-model training."""
    # Load original features
    X = pd.read_csv(X_TRAIN_PATH, index_col=0)
    X = clean_feature_names(X)

    # Load out-of-fold predictions
    if not os.path.exists(OOF_PREDICTIONS_PATH):
        print(f"Error: {OOF_PREDICTIONS_PATH} not found. Please run ensemble-pipeline.py --train_base first.")
        return None, None, None, None

    oof_df = pd.read_csv(OOF_PREDICTIONS_PATH)

    # Create meta-features by combining original features with OOF predictions
    meta_X = X.copy()
    meta_X['oof_lc'] = oof_df['oof_lc']
    meta_X['oof_halc'] = oof_df['oof_halc']
    meta_X['oof_cs'] = oof_df['oof_cs']

    # Get true target variables for meta-model training
    y_lc = oof_df['true_lc']
    y_halc = oof_df['true_halc']
    y_cs = oof_df['true_cs']

    print(f"Meta features shape: {meta_X.shape}")
    return meta_X, y_lc, y_halc, y_cs

# Load test data
def load_test_data():
    X_test = pd.read_csv(X_TEST_PATH, index_col=0)
    X_test = clean_feature_names(X_test)
    return X_test

# Create feature importance plots
def plot_feature_importance(model, feature_names, target_name):
    """Plots feature importance for a trained LightGBM model."""
    # Get feature importances
    importances = model.feature_importance(importance_type='gain')

    # Create DataFrame for visualization
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(30), palette='viridis')
    plt.title(f'Feature Importance for {target_name} Meta-Model')
    plt.tight_layout()
    plt.savefig(f'plots/{target_name}_feature_importance.png', dpi=300)
    plt.close()

    return feature_imp

# Load best parameters
def load_best_params():
    best_params = {}

    # Ensure directories exist
    os.makedirs('params', exist_ok=True)

    # Path to parameter files
    param_files = {
        'meta_lc': 'params/best_params_meta_lc.txt',
        'meta_halc': 'params/best_params_meta_halc.txt',
        'meta_cs': 'params/best_params_meta_cs.txt'
    }

    # Default parameters if files don't exist (should be replaced by optimized params)
    default_params = {
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
            print(f"Parameter file {file_path} not found. Using default parameters.")
            best_params[model_type] = default_params[model_type]

    return best_params

# Train meta-models with best hyperparameters
def train_meta_models(meta_X, y_lc, y_halc, y_cs, best_params, n_folds=5, n_bins_regression_stratification=10):
    print("Training meta models...")
    # Create out-of-fold predictions for meta-models
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Initialize arrays to store meta OOF predictions
    meta_oof_lc = np.zeros(len(meta_X))
    meta_oof_halc = np.zeros(len(meta_X))
    meta_oof_cs = np.zeros(len(meta_X))

    # Initialize lists to store meta-models
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

        # Generate feature importance plot
        # Only for the first fold to avoid multiple plots and ensure model is trained
        if fold == 0 and model:
            plot_feature_importance(model, X_train.columns, 'Loss_Cost')

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

        # Generate feature importance plot
        # Only for the first fold
        if fold == 0 and model:
            plot_feature_importance(model, X_train.columns, 'HALC')

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

        # Generate feature importance plot
        # Only for the first fold
        if fold == 0 and model:
            plot_feature_importance(model, X_train.columns, 'Claim_Status')

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

    return meta_lc_models, meta_halc_models, meta_cs_models, meta_oof_df

# Make predictions for test data
def predict_test_data(base_models_lc, base_models_halc, base_models_cs, meta_lc_models, meta_halc_models, meta_cs_models):
    # Load test data
    X_test = load_test_data()

    # Initialize arrays to store predictions from base models
    test_pred_lc = np.zeros(len(X_test))
    test_pred_halc = np.zeros(len(X_test))
    test_pred_cs = np.zeros(len(X_test))

    # Make predictions with base models
    print("Predicting with base models on test data...")
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

# Load base models
def load_base_models():
    base_models_lc = []
    base_models_halc = []
    base_models_cs = []

    for fold in range(5):
        try:
            lc_model = lgb.Booster(model_file=f'models/lc_model_fold_{fold}.txt')
            halc_model = lgb.Booster(model_file=f'models/halc_model_fold_{fold}.txt')
            cs_model = lgb.Booster(model_file=f'models/cs_model_fold_{fold}.txt')

            base_models_lc.append(lc_model)
            base_models_halc.append(halc_model)
            base_models_cs.append(cs_model)
        except Exception as e:
            print(f"Error loading base models for fold {fold}: {e}")

    return base_models_lc, base_models_halc, base_models_cs

# Main execution
if __name__ == "__main__":
    print("Starting LightGBM meta-model training...")

    # Load meta data
    meta_X, y_lc, y_halc, y_cs = load_meta_data()

    # Check if meta data loaded successfully
    if meta_X is None:
        exit() # Exit if OOF predictions were not found

    # Load best parameters
    best_params = load_best_params()

    # Train meta models
    print("Training meta models...")
    # Define number of folds and bins for stratification (you can make these command line arguments if needed)
    n_folds = 5
    n_bins_regression_stratification = 10

    meta_lc_models, meta_halc_models, meta_cs_models, meta_oof_df = train_meta_models(
        meta_X, y_lc, y_halc, y_cs, best_params, n_folds, n_bins_regression_stratification
    )

    print("Saved meta out-of-fold predictions to 'meta_oof_predictions.csv'")
    print("All meta models trained and saved in 'meta_models' directory")

    # Load base models for test prediction
    print("Loading base models for test prediction...")
    base_models_lc, base_models_halc, base_models_cs = load_base_models()

    # Check if base models loaded successfully
    if not base_models_lc or not base_models_halc or not base_models_cs:
         print("Error: Could not load base models. Cannot make test predictions.")
         exit()

    # Load meta models for test prediction (explicitly load them here for prediction)
    print("Loading meta models for test prediction...")
    meta_models_lc_predict = []
    meta_models_halc_predict = []
    meta_models_cs_predict = []

    for fold in range(n_folds):
        try:
            meta_lc_model = lgb.Booster(model_file=f'meta_models/meta_lc_model_fold_{fold}.txt')
            meta_halc_model = lgb.Booster(model_file=f'meta_models/meta_halc_model_fold_{fold}.txt')
            meta_cs_model = lgb.Booster(model_file=f'meta_models/meta_cs_model_fold_{fold}.txt')

            meta_models_lc_predict.append(meta_lc_model)
            meta_models_halc_predict.append(meta_halc_model)
            meta_models_cs_predict.append(meta_cs_model)
        except Exception as e:
            print(f"Warning: Could not load meta models for prediction for fold {fold}: {e}")


    # Check if meta models loaded successfully for prediction
    if not meta_models_lc_predict or not meta_models_halc_predict or not meta_models_cs_predict:
         print("Error: Could not load meta models for prediction. Cannot make test predictions.")
         exit()


    # Make predictions on test data
    print("Making final predictions for test data...")
    final_predictions = predict_test_data(
        base_models_lc,
        base_models_halc,
        base_models_cs,
        meta_models_lc_predict, # Use loaded meta models for prediction
        meta_models_halc_predict, # Use loaded meta models for prediction
        meta_models_cs_predict # Use loaded meta models for prediction
    )

    # Check if predictions were made successfully
    if final_predictions is not None:
        print("Final predictions saved to 'final_predictions.csv'")
        print("Meta-model training and prediction completed!")
    else:
        print("Failed to generate final predictions due to missing models.")
