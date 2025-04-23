# --- Imports ---
import pandas as pd
import numpy as np # Ensure numpy is imported
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle # Import pickle for loading/saving parameters and models
import os

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
HYPERPARAMS_DIR = os.path.join(BASE_DIR, "011_Hyperparameters") # Folder containing .pkl files
MODELS_DIR = os.path.join(BASE_DIR, "012_Models") # Folder to save trained models

FEATURES_FILE = os.path.join(DATA_DIR, "feature_selected_train.csv")
TARGETS_FILE = os.path.join(DATA_DIR, "feature_selected_y_train.csv")

# Target column names (UPDATED based on user input)
TARGET_LOSS_COST = 'Loss_Cost'
TARGET_HIST_LOSS_COST = 'Historically_Adjusted_Loss_Cost'
TARGET_CLAIM_STATUS = 'Claim_Status'

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_BINS = 10 # Bins for stratification

# --- Load Data ---
print("\nLoading data...")
try:
    # Basic check if data directory exists relative to BASE_DIR
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at the expected location: '{DATA_DIR}'")
        print(f"Please ensure BASE_DIR '{BASE_DIR}' is correct and contains '01_Data'.")
        print("(Check that you are running the script from a folder located directly inside the main project folder)")
        exit()
    # Check if files exist before loading
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: Features file not found at '{FEATURES_FILE}'")
        exit()
    if not os.path.exists(TARGETS_FILE):
         print(f"Error: Targets file not found at '{TARGETS_FILE}'")
         exit()

    # Load CSVs using the first column as index
    X = pd.read_csv(FEATURES_FILE, index_col=0) # Assuming first column is index
    y = pd.read_csv(TARGETS_FILE, index_col=0) # Assuming first column is index

    print(f"Features loaded: {FEATURES_FILE}")
    print(f"Targets loaded: {TARGETS_FILE}")
    print("Data loaded successfully.")

    # Verify that the specified target columns exist in the loaded 'y' dataframe
    required_targets = [TARGET_LOSS_COST, TARGET_HIST_LOSS_COST, TARGET_CLAIM_STATUS]
    missing_targets = [col for col in required_targets if col not in y.columns]
    if missing_targets:
        print(f"\nError: The following target columns were NOT found in {TARGETS_FILE}: {missing_targets}")
        print(f"Available columns: {y.columns.tolist()}")
        print("Please check the column names provided.")
        exit()
    else:
        print(f"Confirmed target columns exist: {required_targets}")

except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print(f"Please ensure the required CSV files exist in '{DATA_DIR}'.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# Basic check on data shapes and index alignment
print(f"Feature data shape: {X.shape}")
print(f"Target data shape: {y.shape}")
if len(X) != len(y):
    print(f"Warning: Feature rows ({len(X)}) != Target rows ({len(y)}). Check data row count alignment.")
    # Simple alignment by minimum length (might not be appropriate if indices don't match)
    min_len = min(len(X), len(y))
    # Consider aligning by index intersection instead if necessary
    common_index = X.index.intersection(y.index)
    if len(common_index) < len(X) or len(common_index) < len(y):
         print("Warning: Indices between X and y do not perfectly match. Aligning by common index.")
         X = X.loc[common_index]
         y = y.loc[common_index]
    elif len(X) != len(y): # If indices match but length differs (shouldn't happen with index_col=0 unless source files differ)
         print("Warning: Row count mismatch despite matching index type. Check source files.")
         X = X.iloc[:min_len]
         y = y.iloc[:min_len]
    print(f"Aligned data shape: X={X.shape}, y={y.shape}")

# --- Prepare for Stratified Splitting ---
print("\nPreparing data for stratified splitting...")
y_stratify = y.copy()

# Bin continuous variables simply
try:
    # Only bin if enough unique values exist
    if y_stratify[TARGET_LOSS_COST].nunique() >= N_BINS:
        y_stratify[f'{TARGET_LOSS_COST}_binned'] = pd.qcut(y_stratify[TARGET_LOSS_COST], q=N_BINS, labels=False, duplicates='drop')
    else:
        y_stratify[f'{TARGET_LOSS_COST}_binned'] = 0 # Assign a single bin if cannot quantile bin
        print(f"Warning: Not enough unique values to bin {TARGET_LOSS_COST}")

    if y_stratify[TARGET_HIST_LOSS_COST].nunique() >= N_BINS:
        y_stratify[f'{TARGET_HIST_LOSS_COST}_binned'] = pd.qcut(y_stratify[TARGET_HIST_LOSS_COST], q=N_BINS, labels=False, duplicates='drop')
    else:
         y_stratify[f'{TARGET_HIST_LOSS_COST}_binned'] = 0 # Assign a single bin
         print(f"Warning: Not enough unique values to bin {TARGET_HIST_LOSS_COST}")

    # Combine stratification keys
    stratify_cols = [f'{TARGET_LOSS_COST}_binned', f'{TARGET_HIST_LOSS_COST}_binned', TARGET_CLAIM_STATUS]
    y_stratify['stratify_key'] = y_stratify[stratify_cols].astype(str).agg('_'.join, axis=1)
    print("Stratification key created.")
except KeyError as e:
    print(f"Error: Target column '{e}' not found in {TARGETS_FILE} during stratification prep.")
    print("Please double-check the column names used (TARGET_LOSS_COST, etc.) match the file header.")
    exit()
except Exception as e:
    print(f"Error during binning/stratification prep: {e}")
    exit()


# --- Split Data ---
print(f"\nSplitting data ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")
try:
    # Use index alignment in split just in case, although should be aligned already
    common_index = X.index.intersection(y.index)
    X_aligned = X.loc[common_index]
    y_aligned = y.loc[common_index]
    y_stratify_aligned = y_stratify.loc[common_index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_aligned,
        y_aligned,
        test_size=TEST_SIZE,
        stratify=y_stratify_aligned['stratify_key'],
        random_state=RANDOM_STATE
    )
    print("Stratified split successful.")
except ValueError as e:
    print(f"Warning: Stratified split failed ('{e}'). Using non-stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE # Fallback uses original X, y
    )
print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")


# --- Hyperparameter Loading Function (Using Pickle) ---
def load_hyperparameters(model_name):
    """Loads hyperparameters from a pickle file (.pkl)."""
    # Assume files are named like 'lgbm_loss_cost_params.pkl'
    param_file = os.path.join(HYPERPARAMS_DIR, f"lgbm_{model_name}_params.pkl") # Expect .pkl extension
    default_params = {}
    try:
        if os.path.exists(param_file):
            with open(param_file, 'rb') as f: # Open in read binary ('rb') mode for pickle
                params = pickle.load(f) # Use pickle.load()
            print(f"Loaded hyperparameters for {model_name} from {param_file}")
            # Ensure consistency for estimators/iterations
            if 'num_iterations' in params and 'n_estimators' not in params:
                params['n_estimators'] = params['num_iterations']
            # Ensure random state is set for reproducibility
            params['random_state'] = RANDOM_STATE
            return params # Return loaded params immediately
        else:
             print(f"Warning: Hyperparameter file not found: {param_file}. Using defaults.")

    except (pickle.UnpicklingError, FileNotFoundError, EOFError, Exception) as e: # Added pickle/EOF errors
        print(f"Warning: Error loading {param_file} ({e}). Using defaults.")

    # Define simple defaults if loading failed or file not found
    print(f"Generating default parameters for {model_name}.")
    if model_name in ['loss_cost', 'hist_adj_loss_cost']:
        # Naming convention check for model name in function call vs filename:
        # Assume function calls use 'loss_cost', 'hist_adj_loss_cost'
        default_params = {'objective': 'tweedie', 'metric': 'rmse', 'n_estimators': 100, 'tweedie_variance_power': 1.5}
    elif model_name == 'claim_status':
        # Need y_train in scope to determine num_classes for defaults
        try:
             num_classes = y_train[TARGET_CLAIM_STATUS].nunique()
        except NameError:
             print("Warning: Cannot determine num_classes for defaults (y_train not available here). Assuming binary.")
             num_classes = 2 # Default assumption if y_train isn't available

        if num_classes == 2:
             default_params = {'objective': 'binary', 'metric': 'auc', 'n_estimators': 100}
        else:
             default_params = {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': num_classes, 'n_estimators': 100}
    else: # Generic fallback
        default_params = {'n_estimators': 100}

    default_params['random_state'] = RANDOM_STATE # Ensure random state is set
    return default_params

# --- Model Saving Function ---
def save_model(model, model_name, save_dir):
    """Saves a trained model to a pickle file."""
    try:
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Construct filename
        filename = f"lgbm_{model_name}_model.pkl"
        filepath = os.path.join(save_dir, filename)
        # Save the model using pickle
        with open(filepath, 'wb') as f: # Open in write binary ('wb') mode
            pickle.dump(model, f)
        print(f"Model for '{model_name}' saved successfully to: {filepath}")
    except Exception as e:
        print(f"Error saving model '{model_name}' to {save_dir}: {e}")


# --- Helper function for evaluation plots (Simplified) ---
def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(7, 7)) # Slightly smaller default size
    plt.scatter(y_true, y_pred, alpha=0.3, s=20) # Smaller points
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', lw=2, color='red', label='Ideal')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True, linestyle=':') # Lighter grid
    plt.legend()
    plt.tight_layout() # Adjust layout
    plt.show()

# --- Model Training & Evaluation (Simplified Structure) ---

# Model 1: Loss Cost (Tweedie Regression)
print(f"\n--- {TARGET_LOSS_COST} Model ---")
lc_params = load_hyperparameters('loss_cost') # Assumes param filename uses 'loss_cost'
lc_params['objective'] = 'tweedie' # Ensure objective
y_train_lc = y_train[TARGET_LOSS_COST].clip(lower=0) # Ensure non-negative target
y_test_lc = y_test[TARGET_LOSS_COST].clip(lower=0)

lgbm_lc = lgb.LGBMRegressor(**lc_params)
print(f"Training with params: {lc_params}")
lgbm_lc.fit(X_train, y_train_lc)
# Save the trained model
save_model(lgbm_lc, 'loss_cost', MODELS_DIR) # Pass simplified name for saving convention

y_pred_lc = np.maximum(0, lgbm_lc.predict(X_test)) # Ensure non-negative predictions using np.maximum
rmse_lc = np.sqrt(mean_squared_error(y_test_lc, y_pred_lc))
print(f"RMSE: {rmse_lc:.4f}")
plot_actual_vs_predicted(y_test_lc, y_pred_lc, f'Actual vs Predicted - {TARGET_LOSS_COST}')

# Model 2: Hist Adj Loss Cost (Tweedie Regression)
print(f"\n--- {TARGET_HIST_LOSS_COST} Model ---")
hlc_params = load_hyperparameters('hist_adj_loss_cost') # Assumes param filename uses 'hist_adj_loss_cost'
hlc_params['objective'] = 'tweedie' # Ensure objective
y_train_hlc = y_train[TARGET_HIST_LOSS_COST].clip(lower=0) # Ensure non-negative target
y_test_hlc = y_test[TARGET_HIST_LOSS_COST].clip(lower=0)

lgbm_hlc = lgb.LGBMRegressor(**hlc_params)
print(f"Training with params: {hlc_params}")
lgbm_hlc.fit(X_train, y_train_hlc)
# Save the trained model
save_model(lgbm_hlc, 'hist_adj_loss_cost', MODELS_DIR) # Pass simplified name for saving convention

y_pred_hlc = np.maximum(0, lgbm_hlc.predict(X_test)) # Ensure non-negative predictions using np.maximum
rmse_hlc = np.sqrt(mean_squared_error(y_test_hlc, y_pred_hlc))
print(f"RMSE: {rmse_hlc:.4f}")
plot_actual_vs_predicted(y_test_hlc, y_pred_hlc, f'Actual vs Predicted - {TARGET_HIST_LOSS_COST}')

# Model 3: Claim Status (Classification)
print(f"\n--- {TARGET_CLAIM_STATUS} Model ---")
cs_params = load_hyperparameters('claim_status') # Assumes param filename uses 'claim_status'
# Encode target labels simply
le = LabelEncoder()
try:
    y_train_cs = le.fit_transform(y_train[TARGET_CLAIM_STATUS])
    y_test_cs = le.transform(y_test[TARGET_CLAIM_STATUS])
    print(f"Labels encoded. Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    is_binary = len(le.classes_) == 2
    if not is_binary and 'num_class' not in cs_params: # Add num_class if missing for multiclass
         cs_params['num_class'] = len(le.classes_)
    # Ensure objective matches data (binary/multiclass) if not loaded correctly
    if is_binary and cs_params.get('objective') == 'multiclass':
        print("Warning: Forcing objective to 'binary' based on data.")
        cs_params['objective'] = 'binary'
    elif not is_binary and cs_params.get('objective') == 'binary':
        print("Warning: Forcing objective to 'multiclass' based on data.")
        cs_params['objective'] = 'multiclass'
        cs_params['num_class'] = len(le.classes_)

except Exception as e:
    print(f"Error encoding labels for {TARGET_CLAIM_STATUS}: {e}")
    exit()

lgbm_cs = lgb.LGBMClassifier(**cs_params)
print(f"Training with params: {cs_params}")
lgbm_cs.fit(X_train, y_train_cs) # Use encoded target
# Save the trained model
save_model(lgbm_cs, 'claim_status', MODELS_DIR) # Pass simplified name for saving convention
# Save the label encoder as well, as it's needed for decoding predictions later
save_model(le, 'claim_status_label_encoder', MODELS_DIR)


y_pred_class_cs = lgbm_cs.predict(X_test)
y_pred_proba_cs = lgbm_cs.predict_proba(X_test)

# Evaluation (Simpler Output)
print("\nEvaluation Metrics:")
# Confusion Matrix
cm = confusion_matrix(y_test_cs, y_pred_class_cs)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_cs, y_pred_class_cs, target_names=[str(c) for c in le.classes_], zero_division=0))

# AUC-ROC (only if binary)
if is_binary:
    try:
        # Ensure positive class probability is selected correctly (usually index 1 after encoding 0, 1)
        positive_class_index = 1
        auc_score = roc_auc_score(y_test_cs, y_pred_proba_cs[:, positive_class_index])
        print(f"\nAUC-ROC Score: {auc_score:.4f}")
        RocCurveDisplay.from_estimator(lgbm_cs, X_test, y_test_cs, pos_label=1) # pos_label=1 assumes '1' is the positive class
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.title('ROC Curve')
        plt.show()
    except IndexError:
         print("Error plotting ROC: Could not access positive class probability. Check model output shape.")
    except Exception as e:
        print(f"Could not calculate/plot AUC: {e}")
elif len(le.classes_) > 2 : # If multiclass
     try:
         auc_score_ovr = roc_auc_score(y_test_cs, y_pred_proba_cs, multi_class='ovr', average='macro')
         print(f"\nAUC-ROC Score (Macro Avg - OvR): {auc_score_ovr:.4f}")
     except Exception as e:
         print(f"Could not calculate multiclass AUC: {e}")

print("\n--- Script Finished ---")