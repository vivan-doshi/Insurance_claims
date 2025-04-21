import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
# Added accuracy_score and classification_report
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import optuna
import warnings
import shap # Import SHAP library
import io # To capture print output for saving

warnings.filterwarnings('ignore')

# --- Restore original file path handling ---
head, tail = os.path.split(os.getcwd())
data_dir = os.path.join(head, '01_Data')
if os.path.isdir(data_dir):
    os.chdir(data_dir)
    print(f"Changed directory to: {data_dir}")
else:
    print(f"Warning: Directory {data_dir} not found. Assuming data files are in the script's current directory: {os.getcwd()}")
    data_dir = os.getcwd()

# --- Utility Functions ---
def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def select_features(X, y, top_n=30, is_classifier=False):
    """Select top features based on LightGBM feature importance"""
    print(f"\nSelecting top {top_n} features...")
    if is_classifier:
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    else:
        model = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1)
    model.fit(X, y)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    top_features = importances.head(top_n)['feature'].tolist()
    print(f"Selected {len(top_features)} features. Top 5: {', '.join(top_features[:5])}")
    return top_features

# --- Optuna Objective Functions ---
def objective_cs(trial, X_train_feat, y_train_cs):
    # Objective function for Claim Status (Classification) using AUC maximization
    param = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'verbosity': -1, 'n_jobs': -1, 'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 8, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    cv_scores = []
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train_feat, y_train_cs):
        X_fold_train, X_fold_val = X_train_feat.iloc[train_idx], X_train_feat.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_cs.iloc[train_idx], y_train_cs.iloc[val_idx]
        model = lgb.LGBMClassifier(**param)
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(auc)
    return np.mean(cv_scores)

# *** MODIFIED: Objective Function for LC Severity using Tweedie ***
def objective_lc_regression(trial, X_train_feat, y_train_lc):
    # Objective function for LC Regression severity using Tweedie objective
    non_zero_mask_train = y_train_lc > 0
    X_train_nz = X_train_feat[non_zero_mask_train]
    y_train_nz = y_train_lc[non_zero_mask_train] # Use original scale LC
    if len(y_train_nz) == 0: return 1e6

    param = {
        'objective': 'tweedie', # *** Use Tweedie objective ***
        'metric': 'rmse', # Evaluate with RMSE
        'boosting_type': 'gbdt',
        'verbosity': -1, 'n_jobs': -1, 'random_state': 42,
        # *** Tune Tweedie variance power ***
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 8, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    cv_scores = []
    n_folds = 5
    zero_indicator = (y_train_lc > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train_feat, zero_indicator):
        y_fold_train_full = y_train_lc.iloc[train_idx]
        y_fold_val_full = y_train_lc.iloc[val_idx]
        train_nz_mask_fold = y_fold_train_full > 0
        val_nz_mask_fold = y_fold_val_full > 0
        X_fold_train_nz = X_train_feat.iloc[train_idx][train_nz_mask_fold]
        y_fold_train_nz = y_fold_train_full[train_nz_mask_fold] # Original scale
        X_fold_val_nz = X_train_feat.iloc[val_idx][val_nz_mask_fold]
        y_fold_val_nz = y_fold_val_full[val_nz_mask_fold] # Original scale
        if len(y_fold_val_nz) == 0: continue
        if len(y_fold_train_nz) == 0: continue

        # *** No log transform needed ***
        model = lgb.LGBMRegressor(**param)
        model.fit(X_fold_train_nz, y_fold_train_nz, # Fit on original scale non-zero
                  eval_set=[(X_fold_val_nz, y_fold_val_nz)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        y_pred = model.predict(X_fold_val_nz) # Predict directly on original scale
        y_pred = np.maximum(y_pred, 0)
        fold_rmse = rmse(y_fold_val_nz, y_pred) # Calculate RMSE on original scale
        cv_scores.append(fold_rmse)
    if not cv_scores: return 1e6
    return np.mean(cv_scores)

def objective_halc_regression(trial, X_train_feat, y_train_halc):
    # Objective function for HALC Regression severity using Tweedie objective
    non_zero_mask_train = y_train_halc > 0
    X_train_nz = X_train_feat[non_zero_mask_train]
    y_train_nz = y_train_halc[non_zero_mask_train] # Use original scale HALC
    if len(y_train_nz) == 0: return 1e6

    param = {
        'objective': 'tweedie', # Use Tweedie objective
        'metric': 'rmse', # Evaluate with RMSE
        'boosting_type': 'gbdt',
        'verbosity': -1, 'n_jobs': -1, 'random_state': 42,
        'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 8, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    cv_scores = []
    n_folds = 5
    zero_indicator = (y_train_halc > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train_feat, zero_indicator):
        y_fold_train_full = y_train_halc.iloc[train_idx]
        y_fold_val_full = y_train_halc.iloc[val_idx]
        train_nz_mask_fold = y_fold_train_full > 0
        val_nz_mask_fold = y_fold_val_full > 0
        X_fold_train_nz = X_train_feat.iloc[train_idx][train_nz_mask_fold]
        y_fold_train_nz = y_fold_train_full[train_nz_mask_fold] # Original scale
        X_fold_val_nz = X_train_feat.iloc[val_idx][val_nz_mask_fold]
        y_fold_val_nz = y_fold_val_full[val_nz_mask_fold] # Original scale
        if len(y_fold_val_nz) == 0: continue
        if len(y_fold_train_nz) == 0: continue

        model = lgb.LGBMRegressor(**param)
        model.fit(X_fold_train_nz, y_fold_train_nz, # Fit on original scale non-zero
                  eval_set=[(X_fold_val_nz, y_fold_val_nz)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        y_pred = model.predict(X_fold_val_nz) # Predict directly on original scale
        y_pred = np.maximum(y_pred, 0)
        fold_rmse = rmse(y_fold_val_nz, y_pred) # Calculate RMSE on original scale
        cv_scores.append(fold_rmse)
    if not cv_scores: return 1e6
    return np.mean(cv_scores)


# --- Load Data ---
print("Loading data...")
train_file = 'cleaned_data.csv'
test_file = 'cleaned_test.csv'
try:
    train_data = pd.read_csv(os.path.join(data_dir, train_file), index_col=0)
    test_data = pd.read_csv(os.path.join(data_dir, test_file), index_col=0)
    print(f"Data loaded successfully from: {data_dir}")
except FileNotFoundError:
    print(f"Error: {train_file} or {test_file} not found in {data_dir}. Please check the path and file names.")
    exit()
except KeyError as e:
    print(f"Error: Index column {e} not found. Please check the CSV files or index_col parameter.")
    exit()

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# --- Preprocessing & Feature Engineering ---
print("\nPreprocessing data...")
def preprocess_cleaned(df, fit_mode=True):
    # Preprocessing function (remains the same)
    X = df.copy()
    Y = {}
    target_cols_train = ['Loss_Cost', 'Historically_Adjusted_Loss_Cost', 'Claim_Status']
    cols_to_drop_base = [
        'ID', 'Total_Cost_Claims_Current_Yr', 'Total_Number_Claims_Current_Yr',
        'Total_Number_Claims_Entire_Duration', 'Ratio_Claims_Total_Duration_Force',
        'Start_Date_Contract', 'Date_Last_Renewal', 'Date_Next_Renewal',
        'Date_Of_Birth', 'Date_Of_DL_Issuance'
    ]
    if fit_mode:
        missing_targets = [col for col in target_cols_train if col not in X.columns]
        if missing_targets:
            print(f"Warning: Expected target columns missing in training data: {missing_targets}")
            for col in missing_targets: X[col] = 0
        Y['LC'] = X['Loss_Cost'].fillna(0)
        Y['HALC'] = X['Historically_Adjusted_Loss_Cost'].fillna(0)
        Y['CS'] = X['Claim_Status'].fillna(0).astype(int)
        cols_to_drop = cols_to_drop_base + target_cols_train
    else:
        cols_to_drop = cols_to_drop_base
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')
    if 'Energy_Source' in X.columns:
        X['Energy_Source'] = X['Energy_Source'].fillna('Other')
    cols_to_encode = [col for col in ['Car_Age_Cat', 'Energy_Source'] if col in X.columns]
    if cols_to_encode:
         X = pd.get_dummies(X, columns=cols_to_encode, dtype=int, drop_first=True)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        print(f"Warning: Found unexpected object columns: {list(categorical_cols)}. Attempting one-hot encoding.")
        X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False, drop_first=True)
    numeric_cols = X.select_dtypes(include=np.number).columns
    for col in numeric_cols:
         if X[col].isnull().any():
              X[col] = X[col].fillna(0)
    if fit_mode:
        return X, Y
    else:
        return X

X_train_full, Y_train_full = preprocess_cleaned(train_data, fit_mode=True)
Y_LC = Y_train_full['LC']
Y_HALC = Y_train_full['HALC']
Y_CS = Y_train_full['CS']
X_test_full = preprocess_cleaned(test_data, fit_mode=False)

print("Aligning columns between training and testing sets...")
train_cols = set(X_train_full.columns)
test_cols = set(X_test_full.columns)
missing_in_test = list(train_cols - test_cols)
missing_in_train = list(test_cols - train_cols)
if missing_in_test:
    print(f"Adding {len(missing_in_test)} columns missing in test set (filled with 0)...")
    for col in missing_in_test:
        X_test_full[col] = 0
if missing_in_train:
    print(f"Warning: Dropping {len(missing_in_train)} columns from test set not seen during training...")
    X_test_full = X_test_full.drop(columns=missing_in_train)
X_test_full = X_test_full[X_train_full.columns]
print(f"Final training features shape: {X_train_full.shape}")
print(f"Final test features shape: {X_test_full.shape}")

# --- Feature Selection ---
N_FEATURES = 30
cs_features = select_features(X_train_full, Y_CS, is_classifier=True)
lc_features = select_features(X_train_full, Y_LC, is_classifier=False) # Still select based on LC target
halc_features = select_features(X_train_full, Y_HALC, is_classifier=False) # Still select based on HALC target

# --- Hyperparameter Tuning (Optuna) ---
N_TRIALS = 100 # Keep increased trials
print(f"\nStarting Optuna hyperparameter tuning ({N_TRIALS} trials per target)...")
best_params_cs = {}
best_params_lc_reg = {}
best_params_halc_reg = {}

study_cs = optuna.create_study(direction='maximize')
try:
    study_cs.optimize(lambda trial: objective_cs(trial, X_train_full[cs_features], Y_CS), n_trials=N_TRIALS)
    best_params_cs = study_cs.best_params
    print(f"Best AUC for CS: {study_cs.best_value:.4f}")
except Exception as e:
    print(f"Optuna optimization for CS failed: {e}. Using default parameters.")

study_lc = optuna.create_study(direction='minimize')
try:
    # *** Use LC objective with Tweedie ***
    study_lc.optimize(lambda trial: objective_lc_regression(trial, X_train_full[lc_features], Y_LC), n_trials=N_TRIALS)
    best_params_lc_reg = study_lc.best_params
    print(f"\nBest CV RMSE for LC (non-zero, original scale, Tweedie tuned): {study_lc.best_value:.4f}")
except Exception as e:
    print(f"Optuna optimization for LC failed: {e}. Using default parameters.")

study_halc = optuna.create_study(direction='minimize')
try:
    # *** Use HALC objective with Tweedie ***
    study_halc.optimize(lambda trial: objective_halc_regression(trial, X_train_full[halc_features], Y_HALC), n_trials=N_TRIALS)
    best_params_halc_reg = study_halc.best_params
    print(f"\nBest CV RMSE for HALC (non-zero, original scale, Tweedie tuned): {study_halc.best_value:.4f}")
except Exception as e:
    print(f"Optuna optimization for HALC failed: {e}. Using default parameters.")


# --- Final Model Training ---
print("\nTraining final models using best hyperparameters on full training data...")
# 1. Claim Status Model
final_params_cs = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'n_estimators': 500, **best_params_cs
}
if 'n_estimators' not in final_params_cs: final_params_cs['n_estimators'] = 500
cs_model = lgb.LGBMClassifier(**final_params_cs)
cs_model.fit(X_train_full[cs_features], Y_CS)
print("Claim Status model trained.")

# 2. Loss Cost Severity Model (Uses Tweedie objective)
final_params_lc_reg = {
    'objective': 'tweedie', # *** Set Tweedie objective ***
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'n_estimators': 500,
    'tweedie_variance_power': 1.5, # Default if tuning failed
    **best_params_lc_reg # Use LC-tuned params (includes tweedie_variance_power)
}
if 'n_estimators' not in final_params_lc_reg: final_params_lc_reg['n_estimators'] = 500
if 'tweedie_variance_power' not in final_params_lc_reg: final_params_lc_reg['tweedie_variance_power'] = 1.5

lc_reg_model = lgb.LGBMRegressor(**final_params_lc_reg)
lc_train_mask_nz = Y_LC > 0
X_lc_train_nz = X_train_full.loc[lc_train_mask_nz, lc_features]
Y_lc_train_nz = Y_LC[lc_train_mask_nz] # Use original scale LC for training
if len(X_lc_train_nz) > 0:
    # *** Train on original scale non-zero LC ***
    lc_reg_model.fit(X_lc_train_nz, Y_lc_train_nz)
    print("LC Severity model trained (using Tweedie).")
else:
    lc_reg_model = None
    print("Warning: No non-zero LC data to train severity model.")

# 3. HALC Severity Model (Uses Tweedie objective)
final_params_halc_reg = {
    'objective': 'tweedie', # Use Tweedie objective
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'n_estimators': 500,
    'tweedie_variance_power': 1.5, # Default if tuning failed
    **best_params_halc_reg # Use HALC-tuned params (includes tweedie_variance_power)
}
if 'n_estimators' not in final_params_halc_reg: final_params_halc_reg['n_estimators'] = 500
if 'tweedie_variance_power' not in final_params_halc_reg: final_params_halc_reg['tweedie_variance_power'] = 1.5

halc_reg_model = lgb.LGBMRegressor(**final_params_halc_reg)
halc_train_mask_nz = Y_HALC > 0
X_halc_train_nz = X_train_full.loc[halc_train_mask_nz, halc_features]
Y_halc_train_nz = Y_HALC[halc_train_mask_nz] # Use original scale HALC for training
if len(X_halc_train_nz) > 0:
    halc_reg_model.fit(X_halc_train_nz, Y_halc_train_nz)
    print("HALC Severity model trained (using Tweedie).")
else:
    halc_reg_model = None
    print("Warning: No non-zero HALC data to train severity model.")


# --- Find Optimal Threshold for CS Model based on Training Accuracy ---
print("\nFinding optimal probability threshold for Claim Status model...")
optimal_threshold = 0.5
best_accuracy = 0
if 'cs_model' in locals() and cs_model is not None:
    X_train_cs_eval = X_train_full[cs_features]
    cs_pred_proba_train_eval = cs_model.predict_proba(X_train_cs_eval)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)
    accuracies = []
    for thresh in thresholds:
        cs_pred_binary_eval = (cs_pred_proba_train_eval >= thresh).astype(int)
        acc = accuracy_score(Y_CS, cs_pred_binary_eval)
        accuracies.append(acc)
    best_accuracy_index = np.argmax(accuracies)
    optimal_threshold = thresholds[best_accuracy_index]
    best_accuracy = accuracies[best_accuracy_index]
    print(f"Optimal threshold based on training accuracy: {optimal_threshold:.2f} (Accuracy: {best_accuracy:.4f})")
else:
    print("Claim Status model not available, using default threshold 0.5.")


# ***********************************************************
# ***** START: Evaluation Metrics and SHAP Analysis Block *****
# ***********************************************************
print("\n--- Evaluating Models on Training Data (Saving Metrics, Showing SHAP Graphs) ---")
try:
    script_start_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    script_start_dir = os.getcwd()
metrics_output_file = os.path.join(script_start_dir, 'evaluation_metrics.txt')
models_available = 'cs_model' in locals() and cs_model is not None
lc_model_trained = 'lc_reg_model' in locals() and lc_reg_model is not None # Check specifically for LC model
halc_model_trained = 'halc_reg_model' in locals() and halc_reg_model is not None
train_rmse_lc = np.nan
train_rmse_halc = np.nan
cm = np.array([['N/A', 'N/A'], ['N/A', 'N/A']])

if models_available:
    print("Generating predictions on training data for evaluation...")
    X_train_cs = X_train_full[cs_features]
    cs_pred_proba_train = cs_model.predict_proba(X_train_cs)[:, 1]
    cs_pred_binary_train = (cs_pred_proba_train >= optimal_threshold).astype(int)

    # Use the LC model (Tweedie) for prediction - no inverse transform needed
    if lc_model_trained:
        X_train_lc = X_train_full[lc_features]
        # *** Predict directly with Tweedie model ***
        lc_pred_severity_train = lc_reg_model.predict(X_train_lc)
        lc_pred_severity_train = np.maximum(lc_pred_severity_train, 0) # Ensure non-negative
    else:
        lc_pred_severity_train = np.zeros(len(X_train_full))

    # Use the HALC model (Tweedie) for prediction - no inverse transform needed
    if halc_model_trained:
        X_train_halc = X_train_full[halc_features]
        # *** Predict directly with Tweedie model ***
        halc_pred_severity_train = halc_reg_model.predict(X_train_halc)
        halc_pred_severity_train = np.maximum(halc_pred_severity_train, 0) # Ensure non-negative
    else:
        halc_pred_severity_train = np.zeros(len(X_train_full))

    lc_pred_final_train = cs_pred_binary_train * lc_pred_severity_train
    halc_pred_final_train = cs_pred_binary_train * halc_pred_severity_train

    train_rmse_lc = np.sqrt(mean_squared_error(Y_LC, lc_pred_final_train))
    train_rmse_halc = np.sqrt(mean_squared_error(Y_HALC, halc_pred_final_train))
    print(f"\nTraining Data RMSE (LC - using threshold {optimal_threshold:.2f}, Tweedie): {train_rmse_lc:.4f}") # Note Tweedie
    print(f"Training Data RMSE (HALC - using threshold {optimal_threshold:.2f}, Tweedie): {train_rmse_halc:.4f}")
    print("\nTraining Data Confusion Matrix (Claim Status - using threshold {optimal_threshold:.2f}):")
    cm = confusion_matrix(Y_CS, cs_pred_binary_train)
    print(cm)
    print("\nClassification Report (Training Data - using threshold {optimal_threshold:.2f}):")
    print(classification_report(Y_CS, cs_pred_binary_train))

    print(f"\n--- Saving Evaluation Metrics to {metrics_output_file} ---")
    try:
        with open(metrics_output_file, 'w') as f:
            f.write("Evaluation Metrics on Training Data\n")
            f.write("="*35 + "\n")
            f.write(f"Optimal CS Threshold (Max Accuracy): {optimal_threshold:.2f}\n")
            f.write(f"Training Accuracy at Optimal Threshold: {best_accuracy:.4f}\n\n")
            f.write(f"RMSE (Loss Cost - using threshold {optimal_threshold:.2f}, Tweedie): {train_rmse_lc:.4f}\n") # Note Tweedie
            f.write(f"RMSE (Historically Adjusted Loss Cost - using threshold {optimal_threshold:.2f}, Tweedie): {train_rmse_halc:.4f}\n\n")
            f.write(f"Confusion Matrix (Claim Status - using threshold {optimal_threshold:.2f}):\n")
            s_cm = io.StringIO()
            np.savetxt(s_cm, cm, fmt='%d', delimiter='\t')
            f.write(s_cm.getvalue())
            f.write("\nClassification Report (Claim Status - using threshold {optimal_threshold:.2f}):\n")
            f.write(classification_report(Y_CS, cs_pred_binary_train))
        print(f"Metrics saved successfully.")
    except Exception as e:
        print(f"Error saving metrics file: {e}")
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Claim Status Confusion Matrix (Training Data, Threshold={optimal_threshold:.2f})')
        plt.show()
    except Exception as e:
        print(f"Could not display Confusion Matrix plot: {e}")
    print("\n--- Calculating SHAP Values (on Sample) and Plotting ---")
    sample_size = min(1000, len(X_train_full))
    if isinstance(X_train_full, pd.DataFrame) and sample_size > 0:
        X_train_sample = X_train_full.sample(sample_size, random_state=42)
        print("Calculating and plotting SHAP for Claim Status model...")
        # ... [SHAP CS code remains same] ...
        try:
            explainer_cs = shap.TreeExplainer(cs_model)
            shap_values_cs = explainer_cs.shap_values(X_train_sample[cs_features], check_additivity=False)
            if isinstance(shap_values_cs, list): shap_values_cs_plot = shap_values_cs[1]
            else: shap_values_cs_plot = shap_values_cs
            shap.summary_plot(shap_values_cs_plot, X_train_sample[cs_features], plot_type="dot", show=False)
            plt.title('SHAP Summary Plot (Claim Status)')
            plt.show()
        except Exception as e:
            print(f"Could not generate SHAP plot for Claim Status: {e}")

        # SHAP for LC Severity Model (Tweedie)
        if lc_model_trained:
            print("\nCalculating and plotting SHAP for LC Severity model (Tweedie)...")
            try:
                lc_train_mask_nz_sample = Y_LC.loc[X_train_sample.index] > 0
                X_lc_train_nz_sample = X_train_sample.loc[lc_train_mask_nz_sample, lc_features]
                if not X_lc_train_nz_sample.empty:
                   explainer_lc = shap.TreeExplainer(lc_reg_model) # Use lc_reg_model
                   shap_values_lc = explainer_lc.shap_values(X_lc_train_nz_sample, check_additivity=False)
                   shap.summary_plot(shap_values_lc, X_lc_train_nz_sample, plot_type="dot", show=False)
                   # Updated title
                   plt.title('SHAP Summary Plot (LC Severity - Tweedie on non-zero sample)')
                   plt.show()
                else:
                    print("Not enough non-zero LC data in the sample for SHAP plot.")
            except Exception as e:
                print(f"Could not generate SHAP plot for LC Severity: {e}")

        # SHAP for HALC Severity Model (Tweedie)
        if halc_model_trained:
            print("\nCalculating and plotting SHAP for HALC Severity model (Tweedie)...")
            try:
                halc_train_mask_nz_sample = Y_HALC.loc[X_train_sample.index] > 0
                X_halc_train_nz_sample = X_train_sample.loc[halc_train_mask_nz_sample, halc_features]
                if not X_halc_train_nz_sample.empty:
                    explainer_halc = shap.TreeExplainer(halc_reg_model)
                    shap_values_halc = explainer_halc.shap_values(X_halc_train_nz_sample, check_additivity=False)
                    shap.summary_plot(shap_values_halc, X_halc_train_nz_sample, plot_type="dot", show=False)
                    plt.title('SHAP Summary Plot (HALC Severity - Tweedie on non-zero sample)')
                    plt.show()
                else:
                    print("Not enough non-zero HALC data in the sample for SHAP plot.")
            except Exception as e:
                print(f"Could not generate SHAP plot for HALC Severity: {e}")
    else:
         print("Error: X_train_full not found, not a DataFrame, or sample size is zero. Cannot generate SHAP plots.")
else:
    print("Core models were not available or trained successfully. Skipping evaluation and SHAP analysis.")

# ***********************************************************
# ****** END: Evaluation Metrics and SHAP Analysis Block ******
# ***********************************************************


# --- Prediction on Test Set ---
print("\nGenerating predictions on the test set...")
if models_available:
    X_test_cs = X_test_full[cs_features]
    cs_pred_proba = cs_model.predict_proba(X_test_cs)[:, 1]
    cs_pred_binary = (cs_pred_proba >= optimal_threshold).astype(int)

    # Use the LC model (Tweedie) for prediction - no inverse transform needed
    if lc_model_trained and lc_reg_model is not None:
        X_test_lc = X_test_full[lc_features]
        # *** Predict directly with Tweedie model ***
        lc_pred_severity = lc_reg_model.predict(X_test_lc)
        lc_pred_severity = np.maximum(lc_pred_severity, 0) # Ensure non-negative
    else:
        print("LC Severity model not available for test prediction. Predicting 0.")
        lc_pred_severity = np.zeros(len(X_test_full))

    # Use the HALC model (Tweedie) for prediction - no inverse transform needed
    if halc_model_trained and halc_reg_model is not None:
        X_test_halc = X_test_full[halc_features]
        # *** Predict directly with Tweedie model ***
        halc_pred_severity = halc_reg_model.predict(X_test_halc)
        halc_pred_severity = np.maximum(halc_pred_severity, 0) # Ensure non-negative
    else:
        print("HALC Severity model not available for test prediction. Predicting 0.")
        halc_pred_severity = np.zeros(len(X_test_full))

    lc_pred_final = cs_pred_binary * lc_pred_severity
    halc_pred_final = cs_pred_binary * halc_pred_severity

    # --- Create Submission File ---
    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'LC': lc_pred_final,
        'HALC': halc_pred_final,
        'CS': cs_pred_binary
    }, index=X_test_full.index)
    submission_df = submission_df[['LC', 'HALC', 'CS']]
    group_number = 'x' # <<< CHANGE THIS TO YOUR GROUP NUMBER
    output_filename = f'group_{group_number}_prediction.csv'
    output_path = os.path.join(script_start_dir, output_filename)
    try:
        submission_df.to_csv(output_path, index=True)
        print(f"Submission file saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving submission file: {e}")
else:
    print("Models were not trained successfully. Cannot generate test predictions or submission file.")

print("\nScript finished.")
