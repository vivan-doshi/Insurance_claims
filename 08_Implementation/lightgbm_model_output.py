# --- Imports ---
import pandas as pd
import numpy as np # Ensure numpy is imported
import pickle # Import pickle for loading parameters and models
import os
import lightgbm as lgb # Import lgb here as well
from sklearn.preprocessing import LabelEncoder
# Import other necessary modules if you add more functionality
# (Plotting libraries like matplotlib/seaborn are not strictly needed for prediction only)
# from sklearn.model_selection import train_test_split # Not needed for prediction
# from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay # Not needed for prediction
# import matplotlib.pyplot as plt
# import seaborn as sns


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
FINAL_DIR = os.path.join(BASE_DIR, "99_Final_File") # Directory for final output

# Input file: Test set features
TEST_FEATURES_FILE = os.path.join(DATA_DIR, "feature_selected_test.csv")

# Output file: Final predictions (using group number 29)
PREDICTION_FILENAME = "group_29_prediction.csv" # Updated filename
PREDICTIONS_OUTPUT_FILE = os.path.join(FINAL_DIR, PREDICTION_FILENAME)


# --- Load Models and Label Encoder ---
print("\nLoading saved models and label encoder...")
models = {}
label_encoder = None
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

    # --- Correction: Use the filename confirmed by the user ---
    le_filename = "lgbm_claim_status_label_encoder_model.pkl" # Correct filename based on user confirmation
    # --- End Correction ---
    le_filepath = os.path.join(MODELS_DIR, le_filename)
    if os.path.exists(le_filepath):
        with open(le_filepath, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Loaded label encoder: {le_filename}")
    else:
        # If the confirmed filename is still not found, error out.
        print(f"Error: Label encoder file not found at '{le_filepath}'")
        print("Please ensure the training script saved the label encoder correctly with this exact name.")
        exit() # Exit if encoder isn't found

except (pickle.UnpicklingError, FileNotFoundError, EOFError, Exception) as e:
    print(f"Error loading model or encoder file: {e}")
    exit()

# Check if all necessary components were loaded
if len(models) != len(model_names):
     print("Error: Not all models were loaded successfully.")
     exit()
# This check might be redundant now since we exit above if LE isn't found
if 'claim_status' in models and label_encoder is None:
     print("Error: Claim status model loaded, but label encoder is missing. Cannot proceed.")
     exit()

# --- Load Test Data ---
print(f"\nLoading test features from: {TEST_FEATURES_FILE}")
try:
    if not os.path.exists(TEST_FEATURES_FILE):
        print(f"Error: Test features file not found at '{TEST_FEATURES_FILE}'")
        exit()

    # Load test features, using the first column as index
    X_test_final = pd.read_csv(TEST_FEATURES_FILE, index_col=0)
    print(f"Test features loaded successfully. Shape: {X_test_final.shape}")

except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during test data loading: {e}")
    exit()

# --- Generate Predictions ---
print("\nGenerating predictions...")
predictions = pd.DataFrame(index=X_test_final.index) # Use the index from test data

# Target column names for output (matching original request)
TARGET_LOSS_COST = 'Loss_Cost'
TARGET_HIST_LOSS_COST = 'Historically_Adjusted_Loss_Cost'
TARGET_CLAIM_STATUS = 'Claim_Status'

try:
    # Predict Loss Cost
    pred_lc = models['loss_cost'].predict(X_test_final)
    predictions[TARGET_LOSS_COST] = np.maximum(0, pred_lc) # Ensure non-negative
    print(f"Predicted {TARGET_LOSS_COST}.")

    # Predict Historically Adjusted Loss Cost
    pred_hlc = models['hist_adj_loss_cost'].predict(X_test_final)
    predictions[TARGET_HIST_LOSS_COST] = np.maximum(0, pred_hlc) # Ensure non-negative
    print(f"Predicted {TARGET_HIST_LOSS_COST}.")

    # Predict Claim Status
    pred_cs_encoded = models['claim_status'].predict(X_test_final)
    # Decode using the loaded label encoder
    pred_cs_decoded = label_encoder.inverse_transform(pred_cs_encoded)
    predictions[TARGET_CLAIM_STATUS] = pred_cs_decoded
    print(f"Predicted and decoded {TARGET_CLAIM_STATUS}.")

except Exception as e:
    print(f"Error during prediction generation: {e}")
    exit()

# --- Save Predictions ---
print(f"\nSaving final predictions to: {PREDICTIONS_OUTPUT_FILE}")
try:
    # Ensure the output directory exists
    os.makedirs(FINAL_DIR, exist_ok=True)
    print(f"Ensured output directory exists: {FINAL_DIR}")

    # Save predictions to CSV, including the index
    predictions.to_csv(PREDICTIONS_OUTPUT_FILE, index=True)
    print("Predictions saved successfully.")
except Exception as e:
    print(f"Error saving predictions to CSV: {e}")

print("\n--- Prediction Script Finished ---")