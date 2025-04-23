# --- Imports ---
import pandas as pd
import numpy as np
import pickle
import os
import lightgbm as lgb # Required for model loading if custom objects used
import shap # Import SHAP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Needed if re-encoding target
import matplotlib.pyplot as plt

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
MODELS_DIR = os.path.join(BASE_DIR, "012_Models") # Folder where models were saved
HYPERPARAMS_DIR = os.path.join(BASE_DIR, "011_Hyperparameters") # Needed only if re-loading params (not done here)

FEATURES_FILE = os.path.join(DATA_DIR, "feature_selected_train.csv")
TARGETS_FILE = os.path.join(DATA_DIR, "feature_selected_y_train.csv")

# Target column names (Must match names used for training)
TARGET_LOSS_COST = 'Loss_Cost'
TARGET_HIST_LOSS_COST = 'Historically_Adjusted_Loss_Cost'
TARGET_CLAIM_STATUS = 'Claim_Status'

TEST_SIZE = 0.20 # Same split ratio as training
RANDOM_STATE = 42 # Same random state as training
N_BINS = 10 # Same number of bins as training

# --- Load Data (Same as training script) ---
print("\nLoading data for splitting...")
try:
    if not os.path.exists(FEATURES_FILE) or not os.path.exists(TARGETS_FILE):
        print(f"Error: Feature ({FEATURES_FILE}) or Target ({TARGETS_FILE}) file not found.")
        exit()
    # Load CSVs using the first column as index
    X = pd.read_csv(FEATURES_FILE, index_col=0)
    y = pd.read_csv(TARGETS_FILE, index_col=0)
    print("Data loaded successfully.")
    # Verify target columns
    required_targets = [TARGET_LOSS_COST, TARGET_HIST_LOSS_COST, TARGET_CLAIM_STATUS]
    if not all(col in y.columns for col in required_targets):
         print(f"Error: One or more target columns not found in {TARGETS_FILE}.")
         exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# Align data just in case indices mismatch after loading
common_index = X.index.intersection(y.index)
if len(common_index) < len(X) or len(common_index) < len(y):
    print("Warning: Indices between X and y do not perfectly match. Aligning by common index.")
    X = X.loc[common_index]
    y = y.loc[common_index]
print(f"Aligned data shape: X={X.shape}, y={y.shape}")


# --- Prepare for Stratified Splitting (Same as training script) ---
print("\nPreparing data for stratified splitting...")
y_stratify = y.copy()
try:
    if y_stratify[TARGET_LOSS_COST].nunique() >= N_BINS:
        y_stratify[f'{TARGET_LOSS_COST}_binned'] = pd.qcut(y_stratify[TARGET_LOSS_COST], q=N_BINS, labels=False, duplicates='drop')
    else: y_stratify[f'{TARGET_LOSS_COST}_binned'] = 0
    if y_stratify[TARGET_HIST_LOSS_COST].nunique() >= N_BINS:
        y_stratify[f'{TARGET_HIST_LOSS_COST}_binned'] = pd.qcut(y_stratify[TARGET_HIST_LOSS_COST], q=N_BINS, labels=False, duplicates='drop')
    else: y_stratify[f'{TARGET_HIST_LOSS_COST}_binned'] = 0
    stratify_cols = [f'{TARGET_LOSS_COST}_binned', f'{TARGET_HIST_LOSS_COST}_binned', TARGET_CLAIM_STATUS]
    y_stratify['stratify_key'] = y_stratify[stratify_cols].astype(str).agg('_'.join, axis=1)
except Exception as e:
    print(f"Error during binning/stratification prep: {e}")
    exit()

# --- Split Data (Same as training script) ---
print(f"\nSplitting data ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, # Use aligned X
        y, # Use aligned y
        test_size=TEST_SIZE,
        stratify=y_stratify['stratify_key'], # Use aligned stratify key source
        random_state=RANDOM_STATE
    )
    print("Stratified split successful.")
except ValueError as e:
    print(f"Warning: Stratified split failed ('{e}'). Using non-stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
print(f"Train shapes: X={X_train.shape}")
print(f"Test shapes: X={X_test.shape}") # We'll explain predictions on X_test

# --- Load Saved Models ---
print("\nLoading saved models...")
models = {}
model_names = ['loss_cost', 'hist_adj_loss_cost', 'claim_status']
try:
    for name in model_names:
        # Assume models are saved with lgbm_ prefix and _model.pkl suffix
        model_filename = f"lgbm_{name}_model.pkl"
        model_filepath = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(model_filepath):
             with open(model_filepath, 'rb') as f:
                 models[name] = pickle.load(f)
             print(f"Loaded model: {model_filename}")
        else:
            print(f"Error: Model file not found at '{model_filepath}'")
            exit()
except Exception as e:
    print(f"Error loading model file: {e}")
    exit()

# --- Calculate and Plot SHAP Values ---
print("\nCalculating SHAP values (this may take a while)...")

# Use X_test for calculating SHAP values for explanation
# Using a sample for speed if X_test is large:
# X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
X_test_sample = X_test # Use full X_test for now

# 1. Loss Cost Model
try:
    print(f"\n--- Explaining {TARGET_LOSS_COST} Model ---")
    explainer_lc = shap.TreeExplainer(models['loss_cost'])
    shap_values_lc = explainer_lc.shap_values(X_test_sample)

    print("Generating SHAP Summary Plot for Loss Cost...")
    shap.summary_plot(shap_values_lc, X_test_sample, show=False)
    plt.title(f"SHAP Summary Plot - {TARGET_LOSS_COST}")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error generating SHAP values/plot for Loss Cost: {e}")

# 2. Historically Adjusted Loss Cost Model
try:
    print(f"\n--- Explaining {TARGET_HIST_LOSS_COST} Model ---")
    explainer_hlc = shap.TreeExplainer(models['hist_adj_loss_cost'])
    shap_values_hlc = explainer_hlc.shap_values(X_test_sample)

    print("Generating SHAP Summary Plot for Historically Adjusted Loss Cost...")
    shap.summary_plot(shap_values_hlc, X_test_sample, show=False)
    plt.title(f"SHAP Summary Plot - {TARGET_HIST_LOSS_COST}")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error generating SHAP values/plot for Hist Adj Loss Cost: {e}")

# 3. Claim Status Model
try:
    print(f"\n--- Explaining {TARGET_CLAIM_STATUS} Model ---")
    explainer_cs = shap.TreeExplainer(models['claim_status'])
    # SHAP values for classifiers often have shape (n_classes, n_samples, n_features)
    shap_values_cs = explainer_cs.shap_values(X_test_sample)

    print("Generating SHAP Summary Plot for Claim Status...")
    # For binary classification, shap_values_cs is often a list [shap_for_class_0, shap_for_class_1]
    # We typically plot for the positive class (usually class 1)
    # For multiclass, choose the class index of interest or use plot_type='bar'
    if isinstance(shap_values_cs, list) and len(shap_values_cs) == 2: # Binary classification
        print("Plotting SHAP values for the positive class (class 1)")
        class_index_to_plot = 1 # Index for the positive class
        shap.summary_plot(shap_values_cs[class_index_to_plot], X_test_sample, show=False)
    else: # Multiclass or unexpected format, use bar plot
         print("Using bar plot for multi-class or non-standard SHAP output.")
         shap.summary_plot(shap_values_cs, X_test_sample, plot_type="bar", show=False)

    plt.title(f"SHAP Summary Plot - {TARGET_CLAIM_STATUS}")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error generating SHAP values/plot for Claim Status: {e}")


print("\n--- SHAP Analysis Script Finished ---")