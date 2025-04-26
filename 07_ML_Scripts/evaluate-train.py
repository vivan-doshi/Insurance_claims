# --- Imports ---
import pandas as pd
import numpy as np
import pickle
import os
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, mean_tweedie_deviance
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer # For binning continuous variables
import joblib # Import joblib to load Optuna study files

# --- Configuration ---
# User-specified Path Configuration: Use parent of the Current Working Directory
print(f"Current Working Directory: {os.getcwd()}")
# head will be the parent directory of the CWD
head, _ = os.path.split(os.getcwd())
BASE_DIR = head # Use the parent directory as the base
print(f"Using parent of CWD as BASE_DIR: {BASE_DIR}")
print("(ASSUMPTION: This script MUST be run from a directory *inside* the main 'Insurance_claims' directory, e.g., '06_ML_Notebooks')")

# Construct paths using os.path.join relative to the determined BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "01_Data")
MODELS_DIR = os.path.join(BASE_DIR, "012_Models") # Folder to save final models
CV_RESULTS_DIR = os.path.join(BASE_DIR, "013_CV_Results") # Folder to save CV results (metrics, plots, SHAPs)
FINAL_DIR = os.path.join(BASE_DIR, "99_Final_File") # Directory for final output
# Directory where hyperparameter tuning results are stored
HYPERPARAMS_DIR = os.path.join(BASE_DIR, "011_Hyperparameters")

# Input files
TRAIN_FEATURES_FILE = os.path.join(DATA_DIR, "feature_selected_train.csv") # Contains training features
TRAIN_TARGETS_FILE = os.path.join(DATA_DIR, "feature_selected_y_train.csv") # Contains training targets
TEST_FEATURES_FILE = os.path.join(DATA_DIR, "feature_selected_test.csv") # For final prediction features

# Output file: Final predictions (using group number 29)
PREDICTION_FILENAME = "group_29_prediction.csv"
PREDICTIONS_OUTPUT_FILE = os.path.join(FINAL_DIR, PREDICTION_FILENAME)

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CV_RESULTS_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# Define target column names (must match those used in tuning script)
TARGET_LOSS_COST = 'Loss_Cost'
TARGET_HIST_LOSS_COST = 'Historically_Adjusted_Loss_Cost'
TARGET_CLAIM_STATUS = 'Claim_Status'

# Define filenames for the saved Optuna study files (must match tuning script output)
# Adjust these filenames if they are different in your tuning script
STUDY_FILE_LC = os.path.join(HYPERPARAMS_DIR, 'lgbm_study_Loss_Cost_stratified_RMSE_TweedieTune_50trials.pkl') # Example filename, update if needed
STUDY_FILE_HLC = os.path.join(HYPERPARAMS_DIR, 'lgbm_study_HALC_stratified_RMSE_TweedieTune_50trials.pkl') # Example filename, update if needed
STUDY_FILE_CS = os.path.join(HYPERPARAMS_DIR, 'lgbm_study_Claim_Status_50trials.pkl') # Example filename, update if needed

# --- Load Data ---
print(f"\nLoading training features from: {TRAIN_FEATURES_FILE}")
try:
    if not os.path.exists(TRAIN_FEATURES_FILE):
        print(f"Error: Training features file not found at '{TRAIN_FEATURES_FILE}'")
        exit()
    # Load training features
    X = pd.read_csv(TRAIN_FEATURES_FILE, index_col=0)
    print(f"Training features loaded successfully. Shape: {X.shape}")

    print(f"Loading training targets from: {TRAIN_TARGETS_FILE}")
    if not os.path.exists(TRAIN_TARGETS_FILE):
        print(f"Error: Training targets file not found at '{TRAIN_TARGETS_FILE}'")
        exit()
    # Load training targets
    y = pd.read_csv(TRAIN_TARGETS_FILE, index_col=0)
    # Check for and remove potential unnamed index column from CSV read
    if 'Unnamed: 0' in y.columns:
        print("Removing 'Unnamed: 0' column from target DataFrame.")
        y = y.drop(columns=['Unnamed: 0'])
    print(f"Training targets loaded successfully. Shape: {y.shape}")

    print(f"Loading test features from: {TEST_FEATURES_FILE}")
    if not os.path.exists(TEST_FEATURES_FILE):
        print(f"Error: Test features file not found at '{TEST_FEATURES_FILE}'")
        exit()
    X_test_final = pd.read_csv(TEST_FEATURES_FILE, index_col=0)
    print(f"Test features loaded successfully. Shape: {X_test_final.shape}")

except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# Ensure target columns exist in the loaded training targets
if not all(col in y.columns for col in [TARGET_LOSS_COST, TARGET_HIST_LOSS_COST, TARGET_CLAIM_STATUS]):
    print("Error: Training targets file does not contain expected target columns.")
    print(f"Expected: {[TARGET_LOSS_COST, TARGET_HIST_LOSS_COST, TARGET_CLAIM_STATUS]}")
    print(f"Found: {y.columns.tolist()}")
    exit()


print(f"\nTraining Features shape: {X.shape}")
print(f"Training Targets shape: {y.shape}")
print(f"Test Features shape: {X_test_final.shape}")

# --- Load Tuned Hyperparameters ---
print("\nLoading tuned hyperparameters...")
tuned_params = {}

try:
    # Load Loss Cost Hyperparameters
    if os.path.exists(STUDY_FILE_LC):
        study_lc = joblib.load(STUDY_FILE_LC)
        tuned_params[TARGET_LOSS_COST] = study_lc.best_params
        print(f"Loaded tuned hyperparameters for {TARGET_LOSS_COST}.")
    else:
        print(f"Warning: Tuned hyperparameters file not found for {TARGET_LOSS_COST} at '{STUDY_FILE_LC}'. Using default/hardcoded parameters.")
        # Define default/fallback parameters if the tuned file is missing
        tuned_params[TARGET_LOSS_COST] = {'objective': 'tweedie', 'metric': 'rmse', 'tweedie_variance_power': 1.5, 'random_state': 42}

    # Load Historically Adjusted Loss Cost Hyperparameters
    if os.path.exists(STUDY_FILE_HLC):
        study_hlc = joblib.load(STUDY_FILE_HLC)
        tuned_params[TARGET_HIST_LOSS_COST] = study_hlc.best_params
        print(f"Loaded tuned hyperparameters for {TARGET_HIST_LOSS_COST}.")
    else:
        print(f"Warning: Tuned hyperparameters file not found for {TARGET_HIST_LOSS_COST} at '{STUDY_FILE_HLC}'. Using default/hardcoded parameters.")
        # Define default/fallback parameters if the tuned file is missing
        tuned_params[TARGET_HIST_LOSS_COST] = {'objective': 'tweedie', 'metric': 'rmse', 'tweedie_variance_power': 1.5, 'random_state': 42}

    # Load Claim Status Hyperparameters
    if os.path.exists(STUDY_FILE_CS):
        study_cs = joblib.load(STUDY_FILE_CS)
        tuned_params[TARGET_CLAIM_STATUS] = study_cs.best_params
        print(f"Loaded tuned hyperparameters for {TARGET_CLAIM_STATUS}.")
    else:
        print(f"Warning: Tuned hyperparameters file not found for {TARGET_CLAIM_STATUS} at '{STUDY_FILE_CS}'. Using default/hardcoded parameters.")
        # Define default/fallback parameters if the tuned file is missing
        tuned_params[TARGET_CLAIM_STATUS] = {'objective': 'binary', 'metric': 'auc', 'random_state': 42}

except Exception as e:
    print(f"Error loading tuned hyperparameters: {e}. Using default/hardcoded parameters for all models.")
    # Fallback to default parameters for all if any error occurs during loading
    tuned_params = {
        TARGET_LOSS_COST: {'objective': 'tweedie', 'metric': 'rmse', 'tweedie_variance_power': 1.5, 'random_state': 42},
        TARGET_HIST_LOSS_COST: {'objective': 'tweedie', 'metric': 'rmse', 'tweedie_variance_power': 1.5, 'random_state': 42},
        TARGET_CLAIM_STATUS: {'objective': 'binary', 'metric': 'auc', 'random_state': 42}
    }

# Add random_state to loaded params if not present (important for reproducibility)
for target in tuned_params:
    if 'random_state' not in tuned_params[target]:
        tuned_params[target]['random_state'] = 42 # Use a consistent random state
    # Ensure verbosity is suppressed during CV and final training
    tuned_params[target]['verbose'] = -1
    tuned_params[target]['n_jobs'] = -1 # Use all available cores

print("\nHyperparameters to be used:")
for target, params in tuned_params.items():
    print(f"- {target}: {params}")


# --- Preprocessing ---

# Handle categorical target for Claim_Status (using training data for fitting)
print("\nEncoding Claim_Status target...")
label_encoder = LabelEncoder()
y_claim_status_encoded = label_encoder.fit_transform(y[TARGET_CLAIM_STATUS])
print("Claim_Status encoded.")

# Save the fitted label encoder
le_filename = "lgbm_claim_status_label_encoder_model.pkl"
le_filepath = os.path.join(MODELS_DIR, le_filename)
try:
    with open(le_filepath, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder to: {le_filepath}")
except Exception as e:
    print(f"Error saving label encoder: {e}")

# Bin continuous targets for stratified cross-validation (using training data for fitting)
# Using QuantileTransformer for binning
n_bins = 10 # You can adjust the number of bins
print(f"\nBinning continuous targets ({TARGET_LOSS_COST}, {TARGET_HIST_LOSS_COST}) for stratification using {n_bins} bins...")

qt_lc = QuantileTransformer(n_quantiles=n_bins, output_distribution='uniform', random_state=42)
y_lc_binned = qt_lc.fit_transform(y[[TARGET_LOSS_COST]]).flatten()
y_lc_binned_labels = pd.cut(y_lc_binned, bins=n_bins, labels=False, include_lowest=True)

qt_hlc = QuantileTransformer(n_quantiles=n_bins, output_distribution='uniform', random_state=42)
y_hlc_binned = qt_hlc.fit_transform(y[[TARGET_HIST_LOSS_COST]]).flatten()
y_hlc_binned_labels = pd.cut(y_hlc_binned, bins=n_bins, labels=False, include_lowest=True)

print("Continuous targets binned.")

# Combine binned continuous targets and encoded claim status for stratification key
# This creates a composite key for stratification
y_stratify = pd.DataFrame({
    'lc_bin': y_lc_binned_labels,
    'hlc_bin': y_hlc_binned_labels,
    'claim_status': y_claim_status_encoded
})

# Convert the composite key to a single string or integer representation for StratifiedKFold
# A simple way is to create a unique string for each combination of bins and status
y_stratify_key = y_stratify.astype(str).agg('_'.join, axis=1)


# --- Cross-Validation Setup ---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_results = {
    'loss_cost': [],
    'hist_adj_loss_cost': [],
    'claim_status': []
}
shap_values_list = {
    'loss_cost': [],
    'hist_adj_loss_cost': [],
    'claim_status': []
}
metrics_list = []

print(f"\nStarting {n_splits}-fold stratified cross-validation...")

# --- Cross-Validation Loop ---
for fold, (train_index, val_index) in enumerate(skf.split(X, y_stratify_key)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index] # Correctly using y loaded from TRAIN_TARGETS_FILE
    y_train_cs_encoded, y_val_cs_encoded = y_claim_status_encoded[train_index], y_claim_status_encoded[val_index]

    fold_metrics = {'fold': fold + 1}

    # --- Train and Evaluate Loss Cost Model ---
    print("Training Loss Cost model...")
    # Use loaded tuned parameters for the model
    lgbm_lc = lgb.LGBMRegressor(**tuned_params[TARGET_LOSS_COST])
    lgbm_lc.fit(X_train, y_train[TARGET_LOSS_COST])
    y_pred_lc = lgbm_lc.predict(X_val)
    y_pred_lc = np.maximum(0, y_pred_lc) # Ensure non-negative predictions

    # Calculate Loss Cost metrics
    rmse_lc = mean_squared_error(y_val[TARGET_LOSS_COST], y_pred_lc, squared=False)
    # Use the tweedie_variance_power from the tuned parameters for deviance calculation
    # Ensure actual values are non-negative and predicted values are strictly positive for tweedie deviance calculation
    tweedie_dev_lc = mean_tweedie_deviance(np.maximum(0, y_val[TARGET_LOSS_COST]), np.maximum(0, y_pred_lc) + np.finfo(float).eps, power=tuned_params[TARGET_LOSS_COST].get('tweedie_variance_power', 1.5))
    fold_metrics['lc_rmse'] = rmse_lc
    fold_metrics['lc_tweedie_deviance'] = tweedie_dev_lc
    print(f"Loss Cost - RMSE: {rmse_lc:.4f}, Tweedie Deviance: {tweedie_dev_lc:.4f}")

    # Generate SHAP values for Loss Cost
    print("Generating SHAP values for Loss Cost model...")
    explainer_lc = shap.TreeExplainer(lgbm_lc)
    shap_values_lc = explainer_lc.shap_values(X_val)
    shap_values_list['loss_cost'].append(shap_values_lc)
    print("SHAP values generated for Loss Cost.")

    # Plot Actual vs. Predicted for Loss Cost
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_val[TARGET_LOSS_COST], y=y_pred_lc, alpha=0.6)
    plt.plot([y_val[TARGET_LOSS_COST].min(), y_val[TARGET_LOSS_COST].max()], [y_val[TARGET_LOSS_COST].min(), y_val[TARGET_LOSS_COST].max()], 'k--', lw=2)
    plt.xlabel("Actual Loss Cost")
    plt.ylabel("Predicted Loss Cost")
    plt.title(f"Fold {fold + 1} - Actual vs. Predicted Loss Cost")
    plt.savefig(os.path.join(CV_RESULTS_DIR, f"fold_{fold+1}_lc_actual_vs_predicted.png"))
    plt.close()
    print("Actual vs. Predicted plot saved for Loss Cost.")


    # --- Train and Evaluate Historically Adjusted Loss Cost Model ---
    print("Training Historically Adjusted Loss Cost model...")
    # Use loaded tuned parameters for the model
    lgbm_hlc = lgb.LGBMRegressor(**tuned_params[TARGET_HIST_LOSS_COST])
    lgbm_hlc.fit(X_train, y_train[TARGET_HIST_LOSS_COST])
    y_pred_hlc = lgbm_hlc.predict(X_val)
    y_pred_hlc = np.maximum(0, y_pred_hlc) # Ensure non-negative predictions

    # Calculate Historically Adjusted Loss Cost metrics
    rmse_hlc = mean_squared_error(y_val[TARGET_HIST_LOSS_COST], y_pred_hlc, squared=False)
    # Use the tweedie_variance_power from the tuned parameters for deviance calculation
    # Ensure actual values are non-negative and predicted values are strictly positive for tweedie deviance calculation
    tweedie_dev_hlc = mean_tweedie_deviance(np.maximum(0, y_val[TARGET_HIST_LOSS_COST]), np.maximum(0, y_pred_hlc) + np.finfo(float).eps, power=tuned_params[TARGET_HIST_LOSS_COST].get('tweedie_variance_power', 1.5))
    fold_metrics['hlc_rmse'] = rmse_hlc
    fold_metrics['hlc_tweedie_deviance'] = tweedie_dev_hlc
    print(f"Hist. Adj. Loss Cost - RMSE: {rmse_hlc:.4f}, Tweedie Deviance: {tweedie_dev_hlc:.4f}")

    # Generate SHAP values for Historically Adjusted Loss Cost
    print("Generating SHAP values for Historically Adjusted Loss Cost model...")
    explainer_hlc = shap.TreeExplainer(lgbm_hlc)
    shap_values_hlc = explainer_hlc.shap_values(X_val)
    shap_values_list['hist_adj_loss_cost'].append(shap_values_hlc)
    print("SHAP values generated for Historically Adjusted Loss Cost.")

    # Plot Actual vs. Predicted for Historically Adjusted Loss Cost
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_val[TARGET_HIST_LOSS_COST], y=y_pred_hlc, alpha=0.6)
    plt.plot([y_val[TARGET_HIST_LOSS_COST].min(), y_val[TARGET_HIST_LOSS_COST].max()], [y_val[TARGET_HIST_LOSS_COST].min(), y_val[TARGET_HIST_LOSS_COST].max()], 'k--', lw=2)
    plt.xlabel("Actual Historically Adjusted Loss Cost")
    plt.ylabel("Predicted Historically Adjusted Loss Cost")
    plt.title(f"Fold {fold + 1} - Actual vs. Predicted Historically Adjusted Loss Cost")
    plt.savefig(os.path.join(CV_RESULTS_DIR, f"fold_{fold+1}_hlc_actual_vs_predicted.png"))
    plt.close()
    print("Actual vs. Predicted plot saved for Historically Adjusted Loss Cost.")


    # --- Train and Evaluate Claim Status Model ---
    print("Training Claim Status model...")
    # Use loaded tuned parameters for the model
    lgbm_cs = lgb.LGBMClassifier(**tuned_params[TARGET_CLAIM_STATUS])
    lgbm_cs.fit(X_train, y_train_cs_encoded)
    y_pred_cs_proba = lgbm_cs.predict_proba(X_val)[:, 1] # Probability of the positive class
    y_pred_cs_class = lgbm_cs.predict(X_val) # Predicted class (0 or 1)

    # Calculate Claim Status metrics
    auc_roc_cs = roc_auc_score(y_val_cs_encoded, y_pred_cs_proba)
    cm_cs = confusion_matrix(y_val_cs_encoded, y_pred_cs_class)
    precision_cs = precision_score(y_val_cs_encoded, y_pred_cs_class)
    recall_cs = recall_score(y_val_cs_encoded, y_pred_cs_class)
    f1_cs = f1_score(y_val_cs_encoded, y_pred_cs_class)

    fold_metrics['cs_auc_roc'] = auc_roc_cs
    fold_metrics['cs_confusion_matrix'] = cm_cs.tolist() # Store as list
    fold_metrics['cs_precision'] = precision_cs
    fold_metrics['cs_recall'] = recall_cs
    fold_metrics['cs_f1_score'] = f1_cs

    print(f"Claim Status - AUC-ROC: {auc_roc_cs:.4f}")
    print("Confusion Matrix:")
    print(cm_cs)
    print(f"Precision: {precision_cs:.4f}, Recall: {recall_cs:.4f}, F1 Score: {f1_cs:.4f}")

    # Generate SHAP values for Claim Status
    print("Generating SHAP values for Claim Status model...")
    # For classification, shap_values can return a list of arrays (one for each class)
    explainer_cs = shap.TreeExplainer(lgbm_cs)
    shap_values_cs = explainer_cs.shap_values(X_val)
    # For binary classification, often the SHAP values for the positive class are used
    if isinstance(shap_values_cs, list) and len(shap_values_cs) > 1:
        shap_values_cs = shap_values_cs[1] # Take SHAP values for the positive class
    shap_values_list['claim_status'].append(shap_values_cs)
    print("SHAP values generated for Claim Status.")

    metrics_list.append(fold_metrics)

print("\nCross-validation completed.")

# --- Aggregate and Display CV Results ---
print("\nAggregated Cross-Validation Results:")
cv_metrics_df = pd.DataFrame(metrics_list)

# Exclude 'cs_confusion_matrix' column before calculating mean
numeric_cv_metrics_df = cv_metrics_df.drop(columns=['cs_confusion_matrix'])

print(numeric_cv_metrics_df.mean())

# Save the aggregated numeric metrics to a file
numeric_cv_metrics_df.mean().to_csv(os.path.join(CV_RESULTS_DIR, "aggregated_cv_metrics.csv"), header=True)
print(f"\nAggregated metrics saved to: {os.path.join(CV_RESULTS_DIR, 'aggregated_cv_metrics.csv')}")

# --- Generate and Save SHAP Summary Plots ---
print("\nGenerating and saving SHAP summary plots...")

# Concatenate SHAP values from all folds for plotting
# Note: This assumes the order of features is consistent across folds, which it should be.
# For large datasets, concatenating all SHAP values might require significant memory.
try:
    # Loss Cost SHAP Summary Plot
    if shap_values_list['loss_cost']:
        all_shap_lc = np.concatenate(shap_values_list['loss_cost'], axis=0)
        # Need a background dataset for shap.summary_plot
        # Using a sample of the training data features for the background
        # background_sample_lc = X.sample(min(1000, X.shape[0]), random_state=42) # Sample up to 1000 rows
        print("Generating SHAP summary plot for Loss Cost...")
        plt.figure(figsize=(10, 8))
        # Use X_val from the last fold as background for plotting, or a sample of X
        shap.summary_plot(all_shap_lc, X.iloc[skf.split(X, y_stratify_key).__next__()[1]].columns, show=False) # Use columns from one validation set
        plt.title("SHAP Summary Plot - Loss Cost")
        plt.tight_layout()
        plt.savefig(os.path.join(CV_RESULTS_DIR, "shap_summary_loss_cost.png"))
        plt.close()
        print(f"SHAP summary plot saved for Loss Cost to: {os.path.join(CV_RESULTS_DIR, 'shap_summary_loss_cost.png')}")
    else:
        print("No SHAP values collected for Loss Cost to plot.")


    # Historically Adjusted Loss Cost SHAP Summary Plot
    if shap_values_list['hist_adj_loss_cost']:
        all_shap_hlc = np.concatenate(shap_values_list['hist_adj_loss_cost'], axis=0)
        # background_sample_hlc = X.sample(min(1000, X.shape[0]), random_state=42) # Sample up to 1000 rows
        print("Generating SHAP summary plot for Historically Adjusted Loss Cost...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(all_shap_hlc, X.iloc[skf.split(X, y_stratify_key).__next__()[1]].columns, show=False) # Use columns from one validation set
        plt.title("SHAP Summary Plot - Historically Adjusted Loss Cost")
        plt.tight_layout()
        plt.savefig(os.path.join(CV_RESULTS_DIR, "shap_summary_hist_adj_loss_cost.png"))
        plt.close()
        print(f"SHAP summary plot saved for Historically Adjusted Loss Cost to: {os.path.join(CV_RESULTS_DIR, 'shap_summary_hist_adj_loss_cost.png')}")
    else:
        print("No SHAP values collected for Historically Adjusted Loss Cost to plot.")

    # Claim Status SHAP Summary Plot
    if shap_values_list['claim_status']:
         # For binary classification, shap_values_list['claim_status'] contains SHAP values for the positive class (index 1)
        all_shap_cs = np.concatenate(shap_values_list['claim_status'], axis=0)
        # background_sample_cs = X.sample(min(1000, X.shape[0]), random_state=42) # Sample up to 1000 rows
        print("Generating SHAP summary plot for Claim Status...")
        plt.figure(figsize=(10, 8))
        # For classification, the expected_value is often a list, need to handle this for plotting
        # A common approach is to use the mean of the expected values from the explainers
        # However, shap.summary_plot often works directly with the SHAP values array and feature names
        shap.summary_plot(all_shap_cs, X.iloc[skf.split(X, y_stratify_key).__next__()[1]].columns, show=False) # Use columns from one validation set
        plt.title("SHAP Summary Plot - Claim Status")
        plt.tight_layout()
        plt.savefig(os.path.join(CV_RESULTS_DIR, "shap_summary_claim_status.png"))
        plt.close()
        print(f"SHAP summary plot saved for Claim Status to: {os.path.join(CV_RESULTS_DIR, 'shap_summary_claim_status.png')}")
    else:
        print("No SHAP values collected for Claim Status to plot.")

except Exception as e:
    print(f"Error generating SHAP summary plots: {e}")


# --- Train Final Models on 100% Data ---
print("\nTraining final models on 100% training data using tuned hyperparameters...")

# Final Loss Cost Model
print("Training final Loss Cost model...")
# Use loaded tuned parameters for the final model
final_lgbm_lc = lgb.LGBMRegressor(**tuned_params[TARGET_LOSS_COST])
final_lgbm_lc.fit(X, y[TARGET_LOSS_COST])
print("Final Loss Cost model trained.")

# Final Historically Adjusted Loss Cost Model
print("Training final Historically Adjusted Loss Cost model...")
# Use loaded tuned parameters for the final model
final_lgbm_hlc = lgb.LGBMRegressor(**tuned_params[TARGET_HIST_LOSS_COST])
final_lgbm_hlc.fit(X, y[TARGET_HIST_LOSS_COST])
print("Final Historically Adjusted Loss Cost model trained.")

# Final Claim Status Model
print("Training final Claim Status model...")
# Use loaded tuned parameters for the final model
final_lgbm_cs = lgb.LGBMClassifier(**tuned_params[TARGET_CLAIM_STATUS])
final_lgbm_cs.fit(X, y_claim_status_encoded) # Use encoded target for training
print("Final Claim Status model trained.")

# --- Save Final Models ---
print("\nSaving final models...")
try:
    with open(os.path.join(MODELS_DIR, "lgbm_loss_cost_model.pkl"), 'wb') as f:
        pickle.dump(final_lgbm_lc, f)
    print("Saved final Loss Cost model.")

    with open(os.path.join(MODELS_DIR, "lgbm_hist_adj_loss_cost_model.pkl"), 'wb') as f:
        pickle.dump(final_lgbm_hlc, f)
    print("Saved final Historically Adjusted Loss Cost model.")

    with open(os.path.join(MODELS_DIR, "lgbm_claim_status_model.pkl"), 'wb') as f:
        pickle.dump(final_lgbm_cs, f)
    print("Saved final Claim Status model.")

except Exception as e:
    print(f"Error saving final models: {e}")

# --- Generate Final Predictions on Test Data ---
print("\nGenerating final predictions on test data...")
final_predictions = pd.DataFrame(index=X_test_final.index)

try:
    # Predict Loss Cost on test data
    pred_lc_final = final_lgbm_lc.predict(X_test_final)
    final_predictions[TARGET_LOSS_COST] = np.maximum(0, pred_lc_final) # Ensure non-negative
    print(f"Predicted {TARGET_LOSS_COST} on test data.")

    # Predict Historically Adjusted Loss Cost on test data
    pred_hlc_final = final_lgbm_hlc.predict(X_test_final)
    final_predictions[TARGET_HIST_LOSS_COST] = np.maximum(0, pred_hlc_final) # Ensure non-negative
    print(f"Predicted {TARGET_HIST_LOSS_COST} on test data.")

    # Predict Claim Status on test data
    pred_cs_encoded_final = final_lgbm_cs.predict(X_test_final)
    # Decode using the loaded label encoder
    pred_cs_decoded_final = label_encoder.inverse_transform(pred_cs_encoded_final)
    final_predictions[TARGET_CLAIM_STATUS] = pred_cs_decoded_final
    print(f"Predicted and decoded {TARGET_CLAIM_STATUS} on test data.")

except Exception as e:
    print(f"Error during final prediction generation: {e}")
    exit()

# --- Save Final Predictions ---
print(f"\nSaving final predictions to: {PREDICTIONS_OUTPUT_FILE}")
try:
    # Save predictions to CSV, including the index
    final_predictions.to_csv(PREDICTIONS_OUTPUT_FILE, index=True)
    print("Final predictions saved successfully.")
except Exception as e:
    print(f"Error saving final predictions to CSV: {e}")

print("\n--- Script Finished ---")
