# run_all_lgbm_tuning.py (v3 - robust binning, feature name sanitization)

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
# Use StratifiedKFold for both, but apply differently
from sklearn.metrics import mean_tweedie_deviance, roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score # Keep for classification
from sklearn.preprocessing import KBinsDiscretizer # Alternative binning
import logging
import sys
import os
import joblib
import time
import re # Import regex library for cleaning names

# --- Configuration ---
# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
# Assume the script is inside a folder like '05_ML_Notebooks', and data/output are siblings to that folder
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Get the parent directory of the script's directory

# Construct paths relative to the determined project root
DATA_DIR = os.path.join(PROJECT_ROOT, '01_Data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '011_Hyperparameters') # User's desired output dir name

TRAIN_FEATURES_FILE = os.path.join(DATA_DIR, 'feature_selected_train.csv')
TRAIN_TARGET_FILE = os.path.join(DATA_DIR, 'feature_selected_y_train.csv')

# --- Tuning Parameters ---
N_TRIALS = 50  # Adjust as needed
CV_FOLDS = 5   # Adjust as needed
RANDOM_STATE = 42
TWEEDIE_POWER = 1.5
N_BINS_FOR_STRATIFICATION = 4 # User's desired number of bins (adjust if needed)


# --- Define Targets and their Settings ---
TARGETS_TO_TUNE = [
    {
        'name': 'Loss_Cost',
        'task_type': 'regression',
        'direction': 'minimize',
        'metric_display': f'Mean Tweedie Deviance (Power={TWEEDIE_POWER})',
        'output_filename': f'lgbm_study_Loss_Cost_stratified_{N_TRIALS}trials.pkl'
    },
    {
        'name': 'Historically_Adjusted_Loss_Cost',
        'task_type': 'regression',
        'direction': 'minimize',
        'metric_display': f'Mean Tweedie Deviance (Power={TWEEDIE_POWER})',
        'output_filename': f'lgbm_study_HALC_stratified_{N_TRIALS}trials.pkl'
    },
    {
        'name': 'Claim_Status',
        'task_type': 'classification',
        'direction': 'maximize',
        'metric_display': 'ROC AUC',
        'output_filename': f'lgbm_study_Claim_Status_{N_TRIALS}trials.pkl'
    }
]

# --- Create output directory ---
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory for study files: {OUTPUT_DIR}")
except OSError as e:
    print(f"Error creating output directory {OUTPUT_DIR}: {e}")
    print("Please check write permissions or the path structure.")
    sys.exit(1)


# --- Logging Setup ---
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.INFO) # Set to WARNING to reduce output

# --- Global Variables ---
current_task_type = None
current_target_name = None
X_train_global = None
y_train_global = None
y_binned_global = None # For regression stratification
use_fallback_cv = False # Flag for regression fallback

# --- Define Scorer Functions ---
def tweedie_deviance_scorer_func(y_true, y_pred):
    """Calculates Mean Tweedie Deviance - standalone function."""
    y_pred = np.maximum(y_pred, 0) # Ensure predictions are non-negative
    return mean_tweedie_deviance(y_true, y_pred, power=TWEEDIE_POWER)

def roc_auc_scorer_func(y_true, y_pred_proba):
     """Calculates ROC AUC score - standalone function."""
     return roc_auc_score(y_true, y_pred_proba)

# --- Combined Objective Function ---
def objective(trial):
    """Optuna objective function - adapts based on global current_task_type."""
    global current_task_type, X_train_global, y_train_global, y_binned_global, use_fallback_cv

    # Common Hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'seed': RANDOM_STATE,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'verbose': -1 # Suppress LightGBM verbosity during training
    }

    # Task-Specific Parameters and Model Instantiation
    if current_task_type == 'regression':
        params['objective'] = 'tweedie'
        params['metric'] = 'None' # Using custom CV score
        params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.1, 1.9)
        model_class = lgb.LGBMRegressor
        scorer = tweedie_deviance_scorer_func
        metric_name_for_error = f"Mean Tweedie Deviance (Power={TWEEDIE_POWER})"
        direction_for_error = 'minimize'
        # CV strategy determined in main loop based on binning success/failure
        if use_fallback_cv or y_binned_global is None:
            cv_strategy = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        else:
            cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    elif current_task_type == 'classification':
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        model_class = lgb.LGBMClassifier
        cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        # Scorer for cross_val_score needs needs_proba=True for roc_auc
        scorer_cvs = make_scorer(roc_auc_score, needs_proba=True)
        metric_name_for_error = "ROC AUC"
        direction_for_error = 'maximize'
    else:
        raise ValueError(f"Invalid task type '{current_task_type}'")

    # --- Cross-Validation ---
    try:
        if current_task_type == 'regression':
            fold_scores = []
            # Determine split data based on CV strategy
            if use_fallback_cv or y_binned_global is None:
                split_iterator = cv_strategy.split(X_train_global) # KFold doesn't use y for split
            else:
                split_iterator = cv_strategy.split(X_train_global, y_binned_global) # StratifiedKFold uses binned y

            for fold, (train_idx, val_idx) in enumerate(split_iterator):
                X_train_fold, X_val_fold = X_train_global.iloc[train_idx], X_train_global.iloc[val_idx]
                y_train_fold, y_val_fold = y_train_global.iloc[train_idx], y_train_global.iloc[val_idx]

                model = model_class(**params)
                model.fit(X_train_fold, y_train_fold)
                y_pred_fold = model.predict(X_val_fold)
                fold_score = scorer(y_val_fold, y_pred_fold) # Use standalone scorer function
                fold_scores.append(fold_score)
            score = np.mean(fold_scores)

        elif current_task_type == 'classification':
            model = model_class(**params)
            scores = cross_val_score(model, X_train_global, y_train_global, cv=cv_strategy, scoring=scorer_cvs, n_jobs=-1)
            score = np.mean(scores)

    except ValueError as e:
         print(f"Trial {trial.number}: ValueError during CV ({metric_name_for_error}): {e}. Assigning {'low' if direction_for_error=='maximize' else 'high'} score.")
         return -np.inf if direction_for_error == 'maximize' else np.inf
    except Exception as e:
         # Catch LightGBM errors specifically if needed
         if "Do not support special JSON characters" in str(e):
              print(f"Trial {trial.number}: LightGBMError - Feature name issue detected: {e}. Assigning {'low' if direction_for_error=='maximize' else 'high'} score.")
         else:
              print(f"Trial {trial.number}: An unexpected error during CV ({metric_name_for_error}): {e}. Assigning {'low' if direction_for_error=='maximize' else 'high'} score.")
         return -np.inf if direction_for_error == 'maximize' else np.inf


    if np.isnan(score) or np.isinf(score):
        print(f"Trial {trial.number}: Score is NaN or Inf. Assigning {'low' if direction_for_error=='maximize' else 'high'} score.")
        return -np.inf if direction_for_error == 'maximize' else np.inf

    return score


# --- Main Loop ---
print("Starting hyperparameter tuning process...")

# Load Feature Data Once
try:
    print(f"\nLoading training features from: {TRAIN_FEATURES_FILE}")
    X_train_global = pd.read_csv(TRAIN_FEATURES_FILE, index_col=0)
    print(f"Feature dimensions: {X_train_global.shape}")

    # *** Sanitize Feature Names ***
    print("Sanitizing feature names...")
    original_cols = X_train_global.columns.tolist()
    # Replace characters forbidden by LightGBM/JSON with underscores
    X_train_global.columns = [re.sub(r'[\[\]{}:",\'<>]+', '_', col) for col in X_train_global.columns]
    new_cols = X_train_global.columns.tolist()
    changed_cols = {o: n for o, n in zip(original_cols, new_cols) if o != n}
    if changed_cols:
         print("Columns renamed:")
         # for o, n in changed_cols.items(): # Uncomment to see all changes
         #      print(f"  '{o}' -> '{n}'")
         print(f"  {len(changed_cols)} columns renamed.")
    else:
         print("  Feature names appear clean, no changes made.")
    # *** End Sanitization ***

except FileNotFoundError:
    print(f"Error: Feature file not found at {TRAIN_FEATURES_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading or sanitizing feature data: {e}")
    sys.exit(1)

# Load Full Target Data Once
try:
    print(f"Loading training targets from: {TRAIN_TARGET_FILE}")
    y_df_full = pd.read_csv(TRAIN_TARGET_FILE, index_col=0)
    # Check for and remove potential unnamed index column from CSV read
    if 'Unnamed: 0' in y_df_full.columns:
        print("Removing 'Unnamed: 0' column from target DataFrame.")
        y_df_full = y_df_full.drop(columns=['Unnamed: 0'])
    print(f"Available target columns: {list(y_df_full.columns)}")
except FileNotFoundError:
    print(f"Error: Target file not found at {TRAIN_TARGET_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading target data: {e}")
    sys.exit(1)


total_start_time = time.time()

for target_info in TARGETS_TO_TUNE:
    target_start_time = time.time()
    current_target_name = target_info['name']
    current_task_type = target_info['task_type']
    direction = target_info['direction']
    metric_display = target_info['metric_display']
    output_filename = target_info['output_filename']
    study_output_path = os.path.join(OUTPUT_DIR, output_filename)
    study_name = os.path.splitext(output_filename)[0]
    use_fallback_cv = False # Reset fallback flag for each target
    y_binned_global = None # Reset binned target

    print(f"\n{'='*20} Tuning for Target: {current_target_name} {'='*20}")
    print(f"Task Type: {current_task_type.capitalize()}, Metric: {metric_display}, Direction: {direction}")

    # Select Target Column
    try:
        if current_target_name not in y_df_full.columns:
             raise KeyError(f"Target column '{current_target_name}' not found in target file.")
        y_train_global = y_df_full[current_target_name].copy()
        print(f"Target '{current_target_name}' loaded successfully. Shape: {y_train_global.shape}")
    except KeyError as e:
        print(f"Error: {e}. Skipping this target.")
        continue
    except Exception as e:
        print(f"An unexpected error occurred preparing target '{current_target_name}': {e}. Skipping.")
        continue

    # Create Binned Target for Regression Stratification (Robust Method)
    if current_task_type == 'regression':
        print(f"Attempting Stratified K-Fold CV based on {N_BINS_FOR_STRATIFICATION} bins of target...")
        try:
            if not pd.api.types.is_numeric_dtype(y_train_global):
                raise TypeError("Target column is not numeric.")

            y_temp = y_train_global.copy()
            if y_temp.isnull().any():
                # Handle NaNs if necessary, e.g., fill or raise error
                # For binning, filling temporarily might be okay if it's rare
                print(f"Warning: Target contains {y_temp.isnull().sum()} NaN values. Check data integrity.")
                # Option: Skip target if NaNs are present
                # print("Skipping target due to NaN values.")
                # continue
                # Option: Fill NaNs for binning (use with caution)
                y_temp = y_temp.fillna(y_temp.median()) # Or another strategy

            zeros_mask = (y_temp == 0)
            positives = y_temp[~zeros_mask & (y_temp > 0)] # Ensure positive

            if N_BINS_FOR_STRATIFICATION < 2:
                 raise ValueError("N_BINS_FOR_STRATIFICATION must be at least 2.")

            bins = np.zeros(len(y_temp), dtype=int) # Bin 0 for non-positives/zeros

            # Check if enough positive, distinct values exist for binning
            if positives.empty or positives.nunique() < (N_BINS_FOR_STRATIFICATION - 1):
                 print(f"Warning: Insufficient distinct positive values ({positives.nunique()}) "
                       f"to create {N_BINS_FOR_STRATIFICATION - 1} positive bins. Falling back to KFold.")
                 use_fallback_cv = True
            else:
                # Try binning positive values into N_BINS_FOR_STRATIFICATION - 1 bins
                n_positive_bins = N_BINS_FOR_STRATIFICATION - 1
                try:
                    # Use qcut on positive values only, labels start from 1
                    pos_bins, bin_edges = pd.qcut(positives, q=n_positive_bins, labels=False, duplicates='drop', retbins=True)
                    bins[positives.index] = pos_bins + 1 # Assign bins 1, 2, ...
                    n_created_bins = len(np.unique(bins)) # Includes bin 0
                    print(f"Successfully created {n_created_bins} bins (incl. zero/non-positive bin) using custom qcut strategy.")
                    if n_created_bins < 2:
                         print("Warning: Binning resulted in < 2 distinct bins overall. Falling back to KFold.")
                         use_fallback_cv = True
                    else:
                         y_binned_global = pd.Series(bins, index=y_temp.index) # Store the successful bins

                except Exception as bin_err_inner:
                     print(f"Custom binning failed for positive values: {bin_err_inner}. Falling back to KFold.")
                     use_fallback_cv = True

            if use_fallback_cv:
                 print("Using standard KFold for cross-validation for this target.")
            elif y_binned_global is not None:
                 # Check bin distribution if successful stratification
                 print("Bin distribution for stratification:")
                 print(y_binned_global.value_counts(normalize=True).sort_index())

        except Exception as bin_err_outer:
            print(f"Error during binning setup for target '{current_target_name}': {bin_err_outer}")
            print("Falling back to standard KFold for this target.")
            use_fallback_cv = True
            y_binned_global = None


    elif current_task_type == 'classification':
        print("Using Stratified K-Fold CV.")
        # Check classification target validity
        unique_values = y_train_global.nunique()
        print(f"Value counts:\n{y_train_global.value_counts(normalize=True)}")
        if not set(y_train_global.unique()).issubset({0, 1}):
            print(f"Warning: Classification target '{current_target_name}' contains values other than 0 and 1 ({y_train_global.unique()}). Check data.")


    # --- Create and Run Optuna Study ---
    # Safety check before running study for regression
    if current_task_type == 'regression' and not use_fallback_cv and y_binned_global is None:
         print(f"Skipping Optuna study for {current_target_name} because required binned data for stratification is missing or failed.")
         continue

    print(f"Output study file: {study_output_path}")
    study = optuna.create_study(direction=direction, study_name=study_name)
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None)
    except Exception as e:
        print(f"An error occurred during the Optuna study for {current_target_name}: {e}")
        continue # Continue with the next target

    # --- Output and Save Results ---
    target_end_time = time.time()
    print(f"\nOptuna study finished for {current_target_name}. Duration: {target_end_time - target_start_time:.2f} seconds.")

    if not study.trials:
         print(f"No trials were completed for {current_target_name}.")
         continue

    try:
        best_trial = study.best_trial
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial finished with {metric_display} score: {best_trial.value}")
        print("Best parameters found:")
        best_params = best_trial.params
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        print(f"\nSaving full Optuna study object to: {study_output_path}")
        try:
            joblib.dump(study, study_output_path)
            print("Optuna study saved successfully.")
        except Exception as e:
            print(f"Error saving Optuna study object for {current_target_name}: {e}")

    except optuna.exceptions.InvalidTrialStateError:
         print(f"Could not find the best trial for {current_target_name}.")
    except Exception as e:
         print(f"An error occurred retrieving or saving results for {current_target_name}: {e}")


total_end_time = time.time()
print(f"\n{'='*50}")
print(f"Completed tuning for all targets. Total script duration: {total_end_time - total_start_time:.2f} seconds.")
print(f"Study files saved in: {OUTPUT_DIR}")
print("Hyperparameter tuning script completed.")