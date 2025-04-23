import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
# Import train_test_split
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
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
        # *** MODIFIED: Use 'regression' objective for feature selection to avoid Tweedie error ***
        # Tweedie objective requires sum of labels > 0, which might not hold in all scenarios during selection.
        # Standard regression is sufficient for ranking features here.
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
# These functions now operate on the training split passed to them
def objective_cs(trial, X_train_cv, y_train_cs_cv):
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
    # Perform CV within the training data provided to this function
    for train_idx, val_idx in skf.split(X_train_cv, y_train_cs_cv):
        X_fold_train, X_fold_val = X_train_cv.iloc[train_idx], X_train_cv.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_cs_cv.iloc[train_idx], y_train_cs_cv.iloc[val_idx]
        model = lgb.LGBMClassifier(**param)
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(auc)
    return np.mean(cv_scores)

# *** Unified Objective Function for Tweedie Regression with Quantile Stratification ***
def objective_tweedie_regression(trial, X_train_cv, y_train_target_cv):
    # Objective function for Tweedie Regression using RMSE minimization
    param = {
        'objective': 'tweedie', 'metric': 'rmse', 'boosting_type': 'gbdt',
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

    # Create bins for stratification using quantiles for non-zero values
    n_quantiles = 4
    non_zero_mask = y_train_target_cv > 0
    stratify_bins = pd.Series(0, index=y_train_target_cv.index, dtype=int)
    if non_zero_mask.sum() > n_quantiles and y_train_target_cv[non_zero_mask].nunique() >= n_quantiles :
        try:
            non_zero_bins = pd.qcut(y_train_target_cv[non_zero_mask], q=n_quantiles, labels=False, duplicates='drop') + 1
            stratify_bins.loc[non_zero_mask] = non_zero_bins
        except ValueError as e:
             print(f"Warning: Quantile binning failed ({e}). Falling back to 0/1 stratification.")
             stratify_bins = (y_train_target_cv > 0).astype(int)
    else:
        stratify_bins = (y_train_target_cv > 0).astype(int)

    # Use StratifiedKFold with the created bins
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train_cv, stratify_bins):
        X_fold_train, X_fold_val = X_train_cv.iloc[train_idx], X_train_cv.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_target_cv.iloc[train_idx], y_train_target_cv.iloc[val_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        y_pred = model.predict(X_fold_val)
        y_pred = np.maximum(y_pred, 0)
        fold_rmse = rmse(y_fold_val, y_pred)
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

print(f"Original training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# --- Preprocessing & Feature Engineering ---
print("\nPreprocessing data...")
def preprocess_cleaned(df, fit_mode=True):
    # Preprocessing function
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
    cols_to_encode = [col for col in ['Car_Age_Cat', 'Energy_Source', 'Car_Age_Contract_Cat'] if col in X.columns]
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
Y_LC_full = Y_train_full['LC']
Y_HALC_full = Y_train_full['HALC']
Y_CS_full = Y_train_full['CS']
X_test_processed = preprocess_cleaned(test_data, fit_mode=False)

print("Aligning columns between (full) training and testing sets...")
train_cols = set(X_train_full.columns)
test_cols = set(X_test_processed.columns)
missing_in_test = list(train_cols - test_cols)
missing_in_train = list(test_cols - train_cols)
if missing_in_test:
    print(f"Adding {len(missing_in_test)} columns missing in test set (filled with 0)...")
    for col in missing_in_test:
        X_test_processed[col] = 0
if missing_in_train:
    print(f"Warning: Dropping {len(missing_in_train)} columns from test set not seen during training...")
    X_test_processed = X_test_processed.drop(columns=missing_in_train)
X_test_processed = X_test_processed[X_train_full.columns]
print(f"Processed full training features shape: {X_train_full.shape}")
print(f"Processed test features shape: {X_test_processed.shape}")

# --- Feature Selection (Performed on Full Training Data) ---
N_FEATURES = 30
print("\n--- Performing Feature Selection on Full Training Data ---")
cs_features = select_features(X_train_full, Y_CS_full, is_classifier=True)
lc_features = select_features(X_train_full, Y_LC_full, is_classifier=False)
halc_features = select_features(X_train_full, Y_HALC_full, is_classifier=False)

# Select features in the dataframes
all_selected_features = list(set(cs_features + lc_features + halc_features))
X_train_full_fs = X_train_full[all_selected_features]
X_test_fs = X_test_processed[all_selected_features]

# --- Train/Validation Split (Stratified by CS and Binned LC/HALC) ---
print("\n--- Splitting Data into Training and Validation Sets (Stratified by CS & Binned Costs) ---")
test_size = 0.2
n_quantiles_split = 4
lc_bins = pd.Series(0, index=Y_LC_full.index, dtype=int)
lc_non_zero_mask = Y_LC_full > 0
if lc_non_zero_mask.sum() > n_quantiles_split and Y_LC_full[lc_non_zero_mask].nunique() >= n_quantiles_split:
    try:
        lc_bins.loc[lc_non_zero_mask] = pd.qcut(Y_LC_full[lc_non_zero_mask], q=n_quantiles_split, labels=False, duplicates='drop') + 1
    except ValueError:
        lc_bins.loc[lc_non_zero_mask] = 1
else:
    lc_bins.loc[lc_non_zero_mask] = 1
halc_bins = pd.Series(0, index=Y_HALC_full.index, dtype=int)
halc_non_zero_mask = Y_HALC_full > 0
if halc_non_zero_mask.sum() > n_quantiles_split and Y_HALC_full[halc_non_zero_mask].nunique() >= n_quantiles_split:
    try:
        halc_bins.loc[halc_non_zero_mask] = pd.qcut(Y_HALC_full[halc_non_zero_mask], q=n_quantiles_split, labels=False, duplicates='drop') + 1
    except ValueError:
        halc_bins.loc[halc_non_zero_mask] = 1
else:
    halc_bins.loc[halc_non_zero_mask] = 1
combined_stratify_key = Y_CS_full.astype(str) + "_LC" + lc_bins.astype(str) + "_HALC" + halc_bins.astype(str)
print("Example combined stratification keys:", combined_stratify_key.unique()[:10])
try:
    X_train, X_val, y_cs_train, y_cs_val, y_lc_train, y_lc_val, y_halc_train, y_halc_val = train_test_split(
        X_train_full_fs,
        Y_CS_full,
        Y_LC_full,
        Y_HALC_full,
        test_size=test_size,
        random_state=42,
        stratify=combined_stratify_key
    )
    print("Successfully split data using combined stratification.")
except ValueError as e:
    print(f"Warning: Combined stratification failed ({e}). Falling back to stratifying by Claim Status only.")
    X_train, X_val, y_cs_train, y_cs_val, y_lc_train, y_lc_val, y_halc_train, y_halc_val = train_test_split(
        X_train_full_fs,
        Y_CS_full,
        Y_LC_full,
        Y_HALC_full,
        test_size=test_size,
        random_state=42,
        stratify=Y_CS_full
    )
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# --- Hyperparameter Tuning (Optuna) ---
N_TRIALS = 100
print(f"\nStarting Optuna hyperparameter tuning ({N_TRIALS} trials per target, using CV within training split)...")
best_params_cs = {}
best_params_lc_tweedie = {}
best_params_halc_tweedie = {}

study_cs = optuna.create_study(direction='maximize')
try:
    study_cs.optimize(lambda trial: objective_cs(trial, X_train[cs_features], y_cs_train), n_trials=N_TRIALS)
    best_params_cs = study_cs.best_params
    print(f"Best AUC for CS: {study_cs.best_value:.4f}")
except Exception as e:
    print(f"Optuna optimization for CS failed: {e}. Using default parameters.")

study_lc_tweedie = optuna.create_study(direction='minimize')
try:
    study_lc_tweedie.optimize(lambda trial: objective_tweedie_regression(trial, X_train[lc_features], y_lc_train), n_trials=N_TRIALS)
    best_params_lc_tweedie = study_lc_tweedie.best_params
    print(f"\nBest CV RMSE for LC (Tweedie tuned): {study_lc_tweedie.best_value:.4f}")
except Exception as e:
    print(f"Optuna optimization for LC failed: {e}. Using default parameters.")

study_halc_tweedie = optuna.create_study(direction='minimize')
try:
    study_halc_tweedie.optimize(lambda trial: objective_tweedie_regression(trial, X_train[halc_features], y_halc_train), n_trials=N_TRIALS)
    best_params_halc_tweedie = study_halc_tweedie.best_params
    print(f"\nBest CV RMSE for HALC (Tweedie tuned): {study_halc_tweedie.best_value:.4f}")
except Exception as e:
    print(f"Optuna optimization for HALC failed: {e}. Using default parameters.")


# --- Final Model Training ---
print("\nTraining final models using best hyperparameters on the TRAINING SPLIT...")
# 1. Claim Status Model
final_params_cs = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'n_estimators': 500, **best_params_cs
}
if 'n_estimators' not in final_params_cs: final_params_cs['n_estimators'] = 500
cs_model = lgb.LGBMClassifier(**final_params_cs)
cs_model.fit(X_train[cs_features], y_cs_train)
print("Claim Status model trained.")

# 2. Loss Cost Model (Direct Tweedie)
final_params_lc_tweedie = {
    'objective': 'tweedie', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'n_estimators': 500,
    'tweedie_variance_power': 1.5, **best_params_lc_tweedie
}
if 'n_estimators' not in final_params_lc_tweedie: final_params_lc_tweedie['n_estimators'] = 500
if 'tweedie_variance_power' not in final_params_lc_tweedie: final_params_lc_tweedie['tweedie_variance_power'] = 1.5
lc_model = lgb.LGBMRegressor(**final_params_lc_tweedie)
lc_model.fit(X_train[lc_features], y_lc_train)
print("LC model trained (using Tweedie).")

# 3. HALC Model (Direct Tweedie)
final_params_halc_tweedie = {
    'objective': 'tweedie', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'n_estimators': 500,
    'tweedie_variance_power': 1.5, **best_params_halc_tweedie
}
if 'n_estimators' not in final_params_halc_tweedie: final_params_halc_tweedie['n_estimators'] = 500
if 'tweedie_variance_power' not in final_params_halc_tweedie: final_params_halc_tweedie['tweedie_variance_power'] = 1.5
halc_model = lgb.LGBMRegressor(**final_params_halc_tweedie)
halc_model.fit(X_train[halc_features], y_halc_train)
print("HALC model trained (using Tweedie).")


# --- Find Optimal Threshold for CS Model based on VALIDATION Accuracy ---
print("\nFinding optimal probability threshold for Claim Status model using VALIDATION set...")
optimal_threshold = 0.5
best_accuracy = 0
if 'cs_model' in locals() and cs_model is not None:
    X_val_cs = X_val[cs_features]
    cs_pred_proba_val = cs_model.predict_proba(X_val_cs)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)
    accuracies = []
    for thresh in thresholds:
        cs_pred_binary_val = (cs_pred_proba_val >= thresh).astype(int)
        acc = accuracy_score(y_cs_val, cs_pred_binary_val)
        accuracies.append(acc)
    best_accuracy_index = np.argmax(accuracies)
    optimal_threshold = thresholds[best_accuracy_index]
    best_accuracy = accuracies[best_accuracy_index]
    print(f"Optimal threshold based on validation accuracy: {optimal_threshold:.2f} (Accuracy: {best_accuracy:.4f})")
else:
    print("Claim Status model not available, using default threshold 0.5.")


# ***********************************************************
# ***** START: Evaluation Metrics and SHAP Analysis Block *****
# ***********************************************************
print("\n--- Evaluating Models on Validation Set (Saving Metrics, Showing SHAP Graphs) ---")
try:
    script_start_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    script_start_dir = os.getcwd()
metrics_output_file = os.path.join(script_start_dir, 'evaluation_metrics.txt')
models_available = 'cs_model' in locals() and cs_model is not None
lc_model_available = 'lc_model' in locals() and lc_model is not None
halc_model_available = 'halc_model' in locals() and halc_model is not None
val_rmse_lc = np.nan
val_rmse_halc = np.nan
cm_val = np.array([['N/A', 'N/A'], ['N/A', 'N/A']])

if models_available and lc_model_available and halc_model_available:
    print("Generating predictions on validation data for evaluation...")
    # CS Predictions on Validation Set
    X_val_cs = X_val[cs_features]
    cs_pred_proba_val = cs_model.predict_proba(X_val_cs)[:, 1]
    cs_pred_binary_val = (cs_pred_proba_val >= optimal_threshold).astype(int)

    # LC Predictions on Validation Set (Direct Tweedie)
    X_val_lc = X_val[lc_features]
    lc_pred_final_val = lc_model.predict(X_val_lc)
    lc_pred_final_val = np.maximum(lc_pred_final_val, 0)

    # HALC Predictions on Validation Set (Direct Tweedie)
    X_val_halc = X_val[halc_features]
    halc_pred_final_val = halc_model.predict(X_val_halc)
    halc_pred_final_val = np.maximum(halc_pred_final_val, 0)

    # --- Calculate Metrics on Validation Set ---
    val_rmse_lc = np.sqrt(mean_squared_error(y_lc_val, lc_pred_final_val))
    val_rmse_halc = np.sqrt(mean_squared_error(y_halc_val, halc_pred_final_val))
    print(f"\nValidation Data RMSE (LC - Direct Tweedie): {val_rmse_lc:.4f}")
    print(f"Validation Data RMSE (HALC - Direct Tweedie): {val_rmse_halc:.4f}")
    print("\nValidation Data Confusion Matrix (Claim Status - using threshold {optimal_threshold:.2f}):")
    cm_val = confusion_matrix(y_cs_val, cs_pred_binary_val)
    print(cm_val)
    print("\nClassification Report (Validation Data - using threshold {optimal_threshold:.2f}):")
    print(classification_report(y_cs_val, cs_pred_binary_val))

    print(f"\n--- Saving Evaluation Metrics to {metrics_output_file} ---")
    try:
        with open(metrics_output_file, 'w') as f:
            f.write("Evaluation Metrics on Validation Set (Direct Tweedie Models)\n")
            f.write("="*60 + "\n")
            f.write(f"Optimal CS Threshold (Max Validation Accuracy): {optimal_threshold:.2f}\n")
            f.write(f"Validation Accuracy at Optimal Threshold: {best_accuracy:.4f}\n\n")
            f.write(f"RMSE (Loss Cost - Direct Tweedie): {val_rmse_lc:.4f}\n")
            f.write(f"RMSE (Historically Adjusted Loss Cost - Direct Tweedie): {val_rmse_halc:.4f}\n\n")
            f.write(f"Confusion Matrix (Claim Status - using threshold {optimal_threshold:.2f}):\n")
            s_cm = io.StringIO()
            np.savetxt(s_cm, cm_val, fmt='%d', delimiter='\t')
            f.write(s_cm.getvalue())
            f.write("\nClassification Report (Claim Status - using threshold {optimal_threshold:.2f}):\n")
            f.write(classification_report(y_cs_val, cs_pred_binary_val))
        print(f"Metrics saved successfully.")
    except Exception as e:
        print(f"Error saving metrics file: {e}")
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_val)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Claim Status Confusion Matrix (Validation Data, Threshold={optimal_threshold:.2f})')
        plt.show()
    except Exception as e:
        print(f"Could not display Confusion Matrix plot: {e}")
    print("\n--- Calculating SHAP Values (on Validation Sample) and Plotting ---")
    sample_size = min(1000, len(X_val))
    if isinstance(X_val, pd.DataFrame) and sample_size > 0:
        X_val_sample = X_val.sample(sample_size, random_state=42)
        print("Calculating and plotting SHAP for Claim Status model...")
        try:
            explainer_cs = shap.TreeExplainer(cs_model)
            shap_values_cs = explainer_cs.shap_values(X_val_sample[cs_features], check_additivity=False)
            if isinstance(shap_values_cs, list): shap_values_cs_plot = shap_values_cs[1]
            else: shap_values_cs_plot = shap_values_cs
            shap.summary_plot(shap_values_cs_plot, X_val_sample[cs_features], plot_type="dot", show=False)
            plt.title('SHAP Summary Plot (Claim Status - Validation Sample)')
            plt.show()
        except Exception as e:
            print(f"Could not generate SHAP plot for Claim Status: {e}")

        print("\nCalculating and plotting SHAP for LC model (Tweedie)...")
        try:
            explainer_lc = shap.TreeExplainer(lc_model)
            shap_values_lc = explainer_lc.shap_values(X_val_sample[lc_features], check_additivity=False)
            shap.summary_plot(shap_values_lc, X_val_sample[lc_features], plot_type="dot", show=False)
            plt.title('SHAP Summary Plot (LC Model - Tweedie - Validation Sample)')
            plt.show()
        except Exception as e:
            print(f"Could not generate SHAP plot for LC Model: {e}")

        print("\nCalculating and plotting SHAP for HALC model (Tweedie)...")
        try:
            explainer_halc = shap.TreeExplainer(halc_model)
            shap_values_halc = explainer_halc.shap_values(X_val_sample[halc_features], check_additivity=False)
            shap.summary_plot(shap_values_halc, X_val_sample[halc_features], plot_type="dot", show=False)
            plt.title('SHAP Summary Plot (HALC Model - Tweedie - Validation Sample)')
            plt.show()
        except Exception as e:
            print(f"Could not generate SHAP plot for HALC Model: {e}")
    else:
         print("Error: X_val not found, not a DataFrame, or sample size is zero. Cannot generate SHAP plots.")
else:
    print("Core models were not available or trained successfully. Skipping evaluation and SHAP analysis.")

# ***********************************************************
# ****** END: Evaluation Metrics and SHAP Analysis Block ******
# ***********************************************************


# --- Prediction on Test Set ---
print("\nGenerating predictions on the test set...")
if models_available and lc_model_available and halc_model_available:
    # CS Prediction
    X_test_cs = X_test_fs[cs_features]
    cs_pred_proba = cs_model.predict_proba(X_test_cs)[:, 1]
    cs_pred_binary = (cs_pred_proba >= optimal_threshold).astype(int)

    # LC Prediction (Direct Tweedie)
    X_test_lc = X_test_fs[lc_features]
    lc_pred_final = lc_model.predict(X_test_lc)
    lc_pred_final = np.maximum(lc_pred_final, 0)

    # HALC Prediction (Direct Tweedie)
    X_test_halc = X_test_fs[halc_features]
    halc_pred_final = halc_model.predict(X_test_halc)
    halc_pred_final = np.maximum(halc_pred_final, 0)

    # --- Create Submission File ---
    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'LC': lc_pred_final,
        'HALC': halc_pred_final,
        'CS': cs_pred_binary
    }, index=X_test_fs.index)
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
    print("One or more models were not trained successfully. Cannot generate test predictions or submission file.")

print("\nScript finished.")
