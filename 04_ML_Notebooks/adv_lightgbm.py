import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold # Keep both KFold imports for now
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
        # Use Tweedie for feature selection for regression targets as well
        model = lgb.LGBMRegressor(objective='tweedie', random_state=42, n_jobs=-1)
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
    # Trains on ALL data passed to it (within CV folds)
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
    # Use StratifiedKFold for classification
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

# *** MODIFIED: Unified Objective Function for Tweedie Regression (LC & HALC) with Stratification ***
def objective_tweedie_regression(trial, X_train_feat, y_train_target):
    # Objective function for Tweedie Regression using RMSE minimization
    # Trains on ALL data passed to it (within CV folds)
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

    # *** Create bins for stratification based on target value ***
    # Simple approach: Bin 0 vs non-zero
    stratify_bins = (y_train_target > 0).astype(int)
    # Alternative: Quantile bins for non-zero values + zero bin
    # n_quantiles = 4 # Example number of quantile bins
    # non_zero_mask = y_train_target > 0
    # if non_zero_mask.sum() > n_quantiles : # Check if enough non-zero values for qcut
    #     # Create quantile bins for non-zero values, assigning bin number (e.g., 1 to n_quantiles)
    #     non_zero_bins = pd.qcut(y_train_target[non_zero_mask], q=n_quantiles, labels=False, duplicates='drop') + 1
    #     # Initialize bins with 0 for zero values
    #     stratify_bins = pd.Series(0, index=y_train_target.index)
    #     # Assign the calculated quantile bin numbers to the non-zero entries
    #     stratify_bins.loc[non_zero_mask] = non_zero_bins
    # else: # Fallback to simple 0 vs non-zero if not enough unique non-zero values
    #     print("Warning: Not enough unique non-zero values for quantile binning, using 0/1 stratification.")
    #     stratify_bins = (y_train_target > 0).astype(int)

    # *** Use StratifiedKFold with the created bins ***
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train_feat, stratify_bins): # Pass bins to split
        X_fold_train, X_fold_val = X_train_feat.iloc[train_idx], X_train_feat.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_target.iloc[train_idx], y_train_target.iloc[val_idx] # Original scale

        model = lgb.LGBMRegressor(**param)
        # Fit directly on the fold data (original scale)
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        y_pred = model.predict(X_fold_val) # Predict directly
        y_pred = np.maximum(y_pred, 0) # Ensure non-negative
        fold_rmse = rmse(y_fold_val, y_pred) # Calculate RMSE
        cv_scores.append(fold_rmse)
    # Handle case where CV might fail
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
lc_features = select_features(X_train_full, Y_LC, is_classifier=False)
halc_features = select_features(X_train_full, Y_HALC, is_classifier=False)

# --- Train Test Split ---





# --- Hyperparameter Tuning (Optuna) ---
N_TRIALS = 100 # Keep increased trials
print(f"\nStarting Optuna hyperparameter tuning ({N_TRIALS} trials per target)...")
best_params_cs = {}
best_params_lc_tweedie = {}
best_params_halc_tweedie = {}

study_cs = optuna.create_study(direction='maximize')
try:
    study_cs.optimize(lambda trial: objective_cs(trial, X_train_full[cs_features], Y_CS), n_trials=N_TRIALS)
    best_params_cs = study_cs.best_params
    print(f"Best AUC for CS: {study_cs.best_value:.4f}")
except Exception as e:
    print(f"Optuna optimization for CS failed: {e}. Using default parameters.")

study_lc_tweedie = optuna.create_study(direction='minimize')
try:
    study_lc_tweedie.optimize(lambda trial: objective_tweedie_regression(trial, X_train_full[lc_features], Y_LC), n_trials=N_TRIALS)
    best_params_lc_tweedie = study_lc_tweedie.best_params
    print(f"\nBest CV RMSE for LC (Tweedie tuned): {study_lc_tweedie.best_value:.4f}")
except Exception as e:
    print(f"Optuna optimization for LC failed: {e}. Using default parameters.")

study_halc_tweedie = optuna.create_study(direction='minimize')
try:
    study_halc_tweedie.optimize(lambda trial: objective_tweedie_regression(trial, X_train_full[halc_features], Y_HALC), n_trials=N_TRIALS)
    best_params_halc_tweedie = study_halc_tweedie.best_params
    print(f"\nBest CV RMSE for HALC (Tweedie tuned): {study_halc_tweedie.best_value:.4f}")
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

# 2. Loss Cost Model (Direct Tweedie)
final_params_lc_tweedie = {
    'objective': 'tweedie',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'n_estimators': 500,
    'tweedie_variance_power': 1.5, # Default if tuning failed
    **best_params_lc_tweedie
}
if 'n_estimators' not in final_params_lc_tweedie: final_params_lc_tweedie['n_estimators'] = 500
if 'tweedie_variance_power' not in final_params_lc_tweedie: final_params_lc_tweedie['tweedie_variance_power'] = 1.5
lc_model = lgb.LGBMRegressor(**final_params_lc_tweedie)
lc_model.fit(X_train_full[lc_features], Y_LC)
print("LC model trained (using Tweedie).")

# 3. HALC Model (Direct Tweedie)
final_params_halc_tweedie = {
    'objective': 'tweedie',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1, 'random_state': 42, 'n_jobs': -1,
    'learning_rate': 0.05, 'n_estimators': 500,
    'tweedie_variance_power': 1.5, # Default if tuning failed
    **best_params_halc_tweedie
}
if 'n_estimators' not in final_params_halc_tweedie: final_params_halc_tweedie['n_estimators'] = 500
if 'tweedie_variance_power' not in final_params_halc_tweedie: final_params_halc_tweedie['tweedie_variance_power'] = 1.5
halc_model = lgb.LGBMRegressor(**final_params_halc_tweedie)
halc_model.fit(X_train_full[halc_features], Y_HALC)
print("HALC model trained (using Tweedie).")


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
lc_model_available = 'lc_model' in locals() and lc_model is not None
halc_model_available = 'halc_model' in locals() and halc_model is not None
train_rmse_lc = np.nan
train_rmse_halc = np.nan
cm = np.array([['N/A', 'N/A'], ['N/A', 'N/A']])

if models_available and lc_model_available and halc_model_available:
    print("Generating predictions on training data for evaluation...")
    # CS Predictions
    X_train_cs = X_train_full[cs_features]
    cs_pred_proba_train = cs_model.predict_proba(X_train_cs)[:, 1]
    cs_pred_binary_train = (cs_pred_proba_train >= optimal_threshold).astype(int)

    # LC Predictions (Direct Tweedie)
    X_train_lc = X_train_full[lc_features]
    lc_pred_final_train = lc_model.predict(X_train_lc)
    lc_pred_final_train = np.maximum(lc_pred_final_train, 0)

    # HALC Predictions (Direct Tweedie)
    X_train_halc = X_train_full[halc_features]
    halc_pred_final_train = halc_model.predict(X_train_halc)
    halc_pred_final_train = np.maximum(halc_pred_final_train, 0)

    # --- Calculate Metrics ---
    train_rmse_lc = np.sqrt(mean_squared_error(Y_LC, lc_pred_final_train))
    train_rmse_halc = np.sqrt(mean_squared_error(Y_HALC, halc_pred_final_train))
    print(f"\nTraining Data RMSE (LC - Direct Tweedie): {train_rmse_lc:.4f}")
    print(f"Training Data RMSE (HALC - Direct Tweedie): {train_rmse_halc:.4f}")
    print("\nTraining Data Confusion Matrix (Claim Status - using threshold {optimal_threshold:.2f}):")
    cm = confusion_matrix(Y_CS, cs_pred_binary_train)
    print(cm)
    print("\nClassification Report (Training Data - using threshold {optimal_threshold:.2f}):")
    print(classification_report(Y_CS, cs_pred_binary_train))

    print(f"\n--- Saving Evaluation Metrics to {metrics_output_file} ---")
    try:
        with open(metrics_output_file, 'w') as f:
            f.write("Evaluation Metrics on Training Data (Direct Tweedie Models)\n")
            f.write("="*55 + "\n")
            f.write(f"Optimal CS Threshold (Max Accuracy): {optimal_threshold:.2f}\n")
            f.write(f"Training Accuracy at Optimal Threshold: {best_accuracy:.4f}\n\n")
            f.write(f"RMSE (Loss Cost - Direct Tweedie): {train_rmse_lc:.4f}\n")
            f.write(f"RMSE (Historically Adjusted Loss Cost - Direct Tweedie): {train_rmse_halc:.4f}\n\n")
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

        print("\nCalculating and plotting SHAP for LC model (Tweedie)...")
        try:
            explainer_lc = shap.TreeExplainer(lc_model)
            shap_values_lc = explainer_lc.shap_values(X_train_sample[lc_features], check_additivity=False)
            shap.summary_plot(shap_values_lc, X_train_sample[lc_features], plot_type="dot", show=False)
            plt.title('SHAP Summary Plot (LC Model - Tweedie)')
            plt.show()
        except Exception as e:
            print(f"Could not generate SHAP plot for LC Model: {e}")

        print("\nCalculating and plotting SHAP for HALC model (Tweedie)...")
        try:
            explainer_halc = shap.TreeExplainer(halc_model)
            shap_values_halc = explainer_halc.shap_values(X_train_sample[halc_features], check_additivity=False)
            shap.summary_plot(shap_values_halc, X_train_sample[halc_features], plot_type="dot", show=False)
            plt.title('SHAP Summary Plot (HALC Model - Tweedie)')
            plt.show()
        except Exception as e:
            print(f"Could not generate SHAP plot for HALC Model: {e}")
    else:
         print("Error: X_train_full not found, not a DataFrame, or sample size is zero. Cannot generate SHAP plots.")
else:
    print("Core models were not available or trained successfully. Skipping evaluation and SHAP analysis.")

# ***********************************************************
# ****** END: Evaluation Metrics and SHAP Analysis Block ******
# ***********************************************************


# --- Prediction on Test Set ---
print("\nGenerating predictions on the test set...")
if models_available and lc_model_available and halc_model_available:
    # CS Prediction
    X_test_cs = X_test_full[cs_features]
    cs_pred_proba = cs_model.predict_proba(X_test_cs)[:, 1]
    cs_pred_binary = (cs_pred_proba >= optimal_threshold).astype(int)

    # LC Prediction (Direct Tweedie)
    X_test_lc = X_test_full[lc_features]
    lc_pred_final = lc_model.predict(X_test_lc)
    lc_pred_final = np.maximum(lc_pred_final, 0)

    # HALC Prediction (Direct Tweedie)
    X_test_halc = X_test_full[halc_features]
    halc_pred_final = halc_model.predict(X_test_halc)
    halc_pred_final = np.maximum(halc_pred_final, 0)

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
    print("One or more models were not trained successfully. Cannot generate test predictions or submission file.")

print("\nScript finished.")
