import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import json
import re
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def clean_feature_names(df):
    """Clean feature names to avoid LightGBM errors"""
    # Replace characters that cause LightGBM errors
    clean_columns = {}
    for col in df.columns:
        # Replace special characters with underscores
        new_col = re.sub(r'[^A-Za-z0-9_]', '_', col)
        clean_columns[col] = new_col
    return df.rename(columns=clean_columns)

def tweedie_deviance(y_true, y_pred, p=1.5):
    """
    Calculate Tweedie Deviance.
    p=1: Poisson distribution (equivalent to Poisson Deviance)
    p=2: Gamma distribution (equivalent to Gamma Deviance)
    1 < p < 2: Compound Poisson-Gamma distribution
    """
    # Ensure predictions are non-negative
    y_pred = np.maximum(y_pred, 1e-9) # Avoid log(0) or division by zero

    if p == 0: # Normal distribution
        deviance = np.mean((y_true - y_pred)**2)
    elif p == 1: # Poisson distribution
        deviance = 2 * np.mean(y_true * np.log(y_true / y_pred) - (y_true - y_pred))
    elif p == 2: # Gamma distribution
        deviance = 2 * np.mean(np.log(y_pred / y_true) + y_true / y_pred - 1)
    else: # Compound Poisson-Gamma
        deviance = 2 * np.mean(y_true**(2-p) / ((1-p)*(2-p)) - y_true * y_pred**(1-p) / (1-p) + y_pred**(2-p) / (2-p))

    # Handle cases where y_true is zero for p=1 (log(0) issue)
    if p == 1:
         zero_mask = (y_true == 0)
         deviance = 2 * np.mean(y_true[~zero_mask] * np.log(y_true[~zero_mask] / y_pred[~zero_mask]) - (y_true[~zero_mask] - y_pred[~zero_mask]))
         # For y_true == 0, the term y_true * log(y_true / y_pred) is 0, and y_true - y_pred is -y_pred
         deviance += 2 * np.mean(-(y_true[zero_mask] - y_pred[zero_mask]))


    return deviance


def create_combined_stratification_bins(y_lc, y_halc, n_bins):
    """Create combined bins for stratification based on LC and HALC"""
    # Handle potential zero values before binning
    y_lc_nonzero = y_lc[y_lc > 0]
    y_halc_nonzero = y_halc[y_halc > 0]

    # Create bins for non-zero values
    lc_bins = pd.cut(y_lc_nonzero, bins=n_bins, labels=False, duplicates='drop') if len(y_lc_nonzero) > 0 else pd.Series(-1, index=y_lc_nonzero.index) # Use -1 for no bin if empty
    halc_bins = pd.cut(y_halc_nonzero, bins=n_bins, labels=False, duplicates='drop') if len(y_halc_nonzero) > 0 else pd.Series(-1, index=y_halc_nonzero.index) # Use -1 for no bin if empty


    # Initialize combined bins with a default value (e.g., 'ZERO' for zero values)
    combined_bins = pd.Series('ZERO', index=y_lc.index, dtype=str) # Use string type for combined bins


    # Assign combined bin labels for non-zero values
    # Ensure indices are aligned before combining
    nonzero_indices = y_lc[(y_lc > 0) | (y_halc > 0)].index
    if len(nonzero_indices) > 0:
        # Align the bin series to the original index
        lc_bins_aligned = lc_bins.reindex(nonzero_indices).fillna(-1).astype(int) # Fill NaN with -1 for missing bins
        halc_bins_aligned = halc_bins.reindex(nonzero_indices).fillna(-1).astype(int) # Fill NaN with -1 for missing bins
        combined_bins[nonzero_indices] = lc_bins_aligned.astype(str) + '_' + halc_bins_aligned.astype(str)


    # Convert to categorical for StratifiedKFold
    return combined_bins.astype('category')


def load_data():
    """Load training data"""
    try:
        X_train = pd.read_csv('feature_selected_train.csv', index_col=0)
        y_train_df = pd.read_csv('feature_selected_y_train.csv', index_col=0)

        # Clean feature names immediately after loading
        X_train = clean_feature_names(X_train)
        y_train_df.columns = [clean_feature_names(pd.DataFrame(columns=[col])).columns[0] for col in y_train_df.columns]

        y_lc = y_train_df['Loss_Cost']
        y_halc = y_train_df['Historically_Adjusted_Loss_Cost']
        y_cs = y_train_df['Claim_Status']

        print(f"Training features shape: {X_train.shape}")
        print(f"Training target shapes - Loss_Cost: {y_lc.shape}, HALC: {y_halc.shape}, Claim_Status: {y_cs.shape}")

        return X_train, y_lc, y_halc, y_cs

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure 'feature_selected_train.csv' and 'feature_selected_y_train.csv' are in the correct directory.")
        return None, None, None, None

def load_params(model_name):
    """Load hyperparameters from a JSON file"""
    params_dir = 'params'
    param_filename = os.path.join(params_dir, f'best_params_{model_name}.txt')
    try:
        with open(param_filename, 'r') as f:
            params = json.load(f)
        print(f"Loaded parameters for {model_name} from {param_filename}")
        return params
    except FileNotFoundError:
        print(f"Parameters file not found for {model_name}: {param_filename}")
        print("Please ensure hyperparameter optimization has been run and parameter files exist.")
        return None

def evaluate_fold(fold, X_train_fold, X_val_fold, y_train_lc, y_val_lc, y_train_halc, y_val_halc, y_train_cs, y_val_cs, params_base_lc, params_base_halc, params_base_cs, params_meta_lc, params_meta_halc, params_meta_cs):
    """Train and evaluate models for a single CV fold"""
    print(f"\n--- Evaluating Fold {fold + 1} ---")

    fold_scores = {}

    # --- Train and Evaluate Base Models ---
    print("  - Training Base Models...")

    # Loss Cost Base Model
    model_base_lc = lgb.LGBMRegressor(**params_base_lc)
    model_base_lc.fit(X_train_fold, y_train_lc)
    base_lc_pred_val = model_base_lc.predict(X_val_fold)
    fold_scores['base_lc_rmse'] = mean_squared_error(y_val_lc, base_lc_pred_val, squared=False)
    # Assume p=1.5 for Tweedie Deviance for regression tasks unless specified otherwise
    fold_scores['base_lc_tweedie_deviance'] = tweedie_deviance(y_val_lc, base_lc_pred_val, p=1.5)


    # HALC Base Model
    model_base_halc = lgb.LGBMRegressor(**params_base_halc)
    model_base_halc.fit(X_train_fold, y_train_halc)
    base_halc_pred_val = model_base_halc.predict(X_val_fold)
    fold_scores['base_halc_rmse'] = mean_squared_error(y_val_halc, base_halc_pred_val, squared=False)
    fold_scores['base_halc_tweedie_deviance'] = tweedie_deviance(y_val_halc, base_halc_pred_val, p=1.5)


    # Claim Status Base Model
    model_base_cs = lgb.LGBMClassifier(**params_base_cs)
    model_base_cs.fit(X_train_fold, y_train_cs)
    base_cs_pred_proba_val = model_base_cs.predict_proba(X_val_fold)[:, 1]
    base_cs_pred_class_val = (base_cs_pred_proba_val > 0.5).astype(int) # Assuming 0.5 threshold

    fold_scores['base_cs_auc'] = roc_auc_score(y_val_cs, base_cs_pred_proba_val)
    fold_scores['base_cs_f1'] = f1_score(y_val_cs, base_cs_pred_class_val)
    fold_scores['base_cs_precision'] = precision_score(y_val_cs, base_cs_pred_class_val)
    fold_scores['base_cs_recall'] = recall_score(y_val_cs, base_cs_pred_class_val)
    # Store confusion matrix for this fold
    fold_scores['base_cs_confusion_matrix'] = confusion_matrix(y_val_cs, base_cs_pred_class_val).tolist() # Convert to list for JSON saving


    print(f"    Base LC RMSE: {fold_scores['base_lc_rmse']:.4f}, Tweedie Deviance: {fold_scores['base_lc_tweedie_deviance']:.4f}")
    print(f"    Base HALC RMSE: {fold_scores['base_halc_rmse']:.4f}, Tweedie Deviance: {fold_scores['base_halc_tweedie_deviance']:.4f}")
    print(f"    Base CS AUC: {fold_scores['base_cs_auc']:.4f}, F1: {fold_scores['base_cs_f1']:.4f}, Precision: {fold_scores['base_cs_precision']:.4f}, Recall: {fold_scores['base_cs_recall']:.4f}")


    # --- Train and Evaluate Meta Models ---
    print("  - Training Meta Models...")

    # Create meta features for the validation set using base model predictions
    meta_X_val = pd.concat([
        X_val_fold.reset_index(drop=True), # Reset index for concat
        pd.DataFrame({
            'base_lc_pred': base_lc_pred_val,
            'base_halc_pred': base_halc_pred_val,
            'base_cs_pred': base_cs_pred_proba_val
        })
    ], axis=1)

    # Create meta features for the training set using base model predictions (requires OOF from base models on the full train set)
    # For simplicity in this CV evaluation script, we will retrain base models on the train_fold to get predictions on train_fold
    # A more robust approach for meta-training would use OOF predictions from base models trained on the *other* folds.
    # However, for evaluating the meta-model structure within a fold, training on the same fold's base predictions is acceptable.
    base_lc_pred_train = model_base_lc.predict(X_train_fold)
    base_halc_pred_train = model_base_halc.predict(X_train_fold)
    base_cs_pred_proba_train = model_base_cs.predict_proba(X_train_fold)[:, 1]

    meta_X_train = pd.concat([
        X_train_fold.reset_index(drop=True), # Reset index for concat
        pd.DataFrame({
            'base_lc_pred': base_lc_pred_train,
            'base_halc_pred': base_halc_pred_train,
            'base_cs_pred': base_cs_pred_proba_train
        })
    ], axis=1)


    # Loss Cost Meta Model
    model_meta_lc = lgb.LGBMRegressor(**params_meta_lc)
    model_meta_lc.fit(meta_X_train, y_train_lc)
    meta_lc_pred_val = model_meta_lc.predict(meta_X_val)
    fold_scores['meta_lc_rmse'] = mean_squared_error(y_val_lc, meta_lc_pred_val, squared=False)
    fold_scores['meta_lc_tweedie_deviance'] = tweedie_deviance(y_val_lc, meta_lc_pred_val, p=1.5)


    # HALC Meta Model
    model_meta_halc = lgb.LGBMRegressor(**params_meta_halc)
    model_meta_halc.fit(meta_X_train, y_train_halc)
    meta_halc_pred_val = model_meta_halc.predict(meta_X_val)
    fold_scores['meta_halc_rmse'] = mean_squared_error(y_val_halc, meta_halc_pred_val, squared=False)
    fold_scores['meta_halc_tweedie_deviance'] = tweedie_deviance(y_val_halc, meta_halc_pred_val, p=1.5)


    # Claim Status Meta Model
    model_meta_cs = lgb.LGBMClassifier(**params_meta_cs)
    model_meta_cs.fit(meta_X_train, y_train_cs)
    meta_cs_pred_proba_val = model_meta_cs.predict_proba(meta_X_val)[:, 1]
    meta_cs_pred_class_val = (meta_cs_pred_proba_val > 0.5).astype(int) # Assuming 0.5 threshold

    fold_scores['meta_cs_auc'] = roc_auc_score(y_val_cs, meta_cs_pred_proba_val)
    fold_scores['meta_cs_f1'] = f1_score(y_val_cs, meta_cs_pred_class_val)
    fold_scores['meta_cs_precision'] = precision_score(y_val_cs, meta_cs_pred_class_val)
    fold_scores['meta_cs_recall'] = recall_score(y_val_cs, meta_cs_pred_class_val)
    # Store confusion matrix for this fold
    fold_scores['meta_cs_confusion_matrix'] = confusion_matrix(y_val_cs, meta_cs_pred_class_val).tolist() # Convert to list for JSON saving


    print(f"    Meta LC RMSE: {fold_scores['meta_lc_rmse']:.4f}, Tweedie Deviance: {fold_scores['meta_lc_tweedie_deviance']:.4f}")
    print(f"    Meta HALC RMSE: {fold_scores['meta_halc_rmse']:.4f}, Tweedie Deviance: {fold_scores['meta_halc_tweedie_deviance']:.4f}")
    print(f"    Meta CS AUC: {fold_scores['meta_cs_auc']:.4f}, F1: {fold_scores['meta_cs_f1']:.4f}, Precision: {fold_scores['meta_cs_precision']:.4f}, Recall: {fold_scores['meta_cs_recall']:.4f}")


    # Return predictions for plotting later (using OOF predictions from the last fold for simplicity in plotting)
    if fold == 4: # Assuming 5 folds, use the last fold's validation predictions for plots
        return fold_scores, {
            'lc_actual': y_val_lc, 'lc_base_pred': base_lc_pred_val, 'lc_meta_pred': meta_lc_pred_val,
            'halc_actual': y_val_halc, 'halc_base_pred': base_halc_pred_val, 'halc_meta_pred': meta_halc_pred_val,
            'cs_actual': y_val_cs, 'cs_base_pred_proba': base_cs_pred_proba_val, 'cs_meta_pred_proba': meta_cs_pred_proba_val
        }
    else:
        return fold_scores, None


def train_full_models(X, y_lc, y_halc, y_cs, params_base_lc, params_base_halc, params_base_cs, params_meta_lc, params_meta_halc, params_meta_cs):
    """Train base and meta models on the full training data"""
    print("\n--- Training Models on Full Training Data for SHAP ---")

    full_models = {}

    # Train Base Models on full data
    print("  - Training Base Models on full data...")
    model_base_lc = lgb.LGBMRegressor(**params_base_lc)
    model_base_lc.fit(X, y_lc)
    full_models['base_lc'] = model_base_lc

    model_base_halc = lgb.LGBMRegressor(**params_base_halc)
    model_base_halc.fit(X, y_halc)
    full_models['base_halc'] = model_base_halc

    model_base_cs = lgb.LGBMClassifier(**params_base_cs)
    model_base_cs.fit(X, y_cs)
    full_models['base_cs'] = model_base_cs
    print("    Base models trained on full data.")

    # Create meta features using predictions from base models trained on full data
    base_lc_pred_full = model_base_lc.predict(X)
    base_halc_pred_full = model_base_halc.predict(X)
    base_cs_pred_proba_full = model_base_cs.predict_proba(X)[:, 1]

    meta_X_full = pd.concat([
        X.reset_index(drop=True), # Reset index for concat
        pd.DataFrame({
            'base_lc_pred': base_lc_pred_full,
            'base_halc_pred': base_halc_pred_full,
            'base_cs_pred': base_cs_pred_proba_full
        })
    ], axis=1)

    # Train Meta Models on full data
    print("  - Training Meta Models on full data...")
    model_meta_lc = lgb.LGBMRegressor(**params_meta_lc)
    model_meta_lc.fit(meta_X_full, y_lc)
    full_models['meta_lc'] = model_meta_lc

    model_meta_halc = lgb.LGBMRegressor(**params_meta_halc)
    model_meta_halc.fit(meta_X_full, y_halc)
    full_models['meta_halc'] = model_meta_halc

    model_meta_cs = lgb.LGBMClassifier(**params_meta_cs)
    model_meta_cs.fit(meta_X_full, y_cs)
    full_models['meta_cs'] = model_meta_cs
    print("    Meta models trained on full data.")


    return full_models, meta_X_full


def calculate_shap_values(models, X, model_type):
    """Calculate SHAP values for each model and target"""
    print(f"\n--- Calculating SHAP values for {model_type} Models ---")
    shap_values_dict = {}

    for name, model in models.items():
        if model_type == 'Base' and name.startswith('base_'):
            target = name.split('_')[1]
            print(f"  - Calculating SHAP for {target}...")
            explainer = shap.TreeExplainer(model)
            # Use a sample of the data for SHAP calculation for performance
            shap_data = X.sample(min(1000, X.shape[0]), random_state=SEED)
            shap_values = explainer.shap_values(shap_data)
            shap_values_dict[target] = shap_values
        elif model_type == 'Meta' and name.startswith('meta_'):
             target = name.split('_')[1]
             print(f"  - Calculating SHAP for {target}...")
             explainer = shap.TreeExplainer(model)
             # Use a sample of the meta data for SHAP calculation
             shap_data = X.sample(min(1000, X.shape[0]), random_state=SEED)
             shap_values = explainer.shap_values(shap_data)
             shap_values_dict[target] = shap_values

    return shap_values_dict


def main():
    n_folds = 5
    n_bins_regression_stratification = 10 # Consistent with ensemble-pipeline.py

    X, y_lc, y_halc, y_cs = load_data()

    if X is None:
        return # Exit if data loading failed

    # Load hyperparameters for base and meta models
    params_base_lc = load_params('lc')
    params_base_halc = load_params('halc')
    params_base_cs = load_params('cs')
    params_meta_lc = load_params('meta_lc')
    params_meta_halc = load_params('meta_halc')
    params_meta_cs = load_params('meta_cs')

    if any(p is None for p in [params_base_lc, params_base_halc, params_base_cs, params_meta_lc, params_meta_halc, params_meta_cs]):
        print("\nCannot proceed without all parameter files. Please run hyperparameter optimization first.")
        return

    # Create combined bins for stratified cross-validation
    y_combined_binned = create_combined_stratification_bins(y_lc, y_halc, n_bins_regression_stratification)
    print(f"\nCreated combined bins for stratification with {len(y_combined_binned.cat.categories)} categories.")


    # --- Cross-Validation Evaluation ---
    print("\n--- Running 5-Fold Cross-Validation Evaluation ---")
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_scores_list = []
    plot_data = None # To store data for plotting from the last fold

    # Reset index of X and y for consistent splitting with StratifiedKFold
    X_reset = X.reset_index(drop=True)
    y_lc_reset = y_lc.reset_index(drop=True)
    y_halc_reset = y_halc.reset_index(drop=True)
    y_cs_reset = y_cs.reset_index(drop=True)
    y_combined_binned_reset = y_combined_binned.reset_index(drop=True)


    for fold, (train_index, val_index) in enumerate(kf.split(X_reset, y_combined_binned_reset)):
        X_train_fold, X_val_fold = X_reset.iloc[train_index], X_reset.iloc[val_index]
        y_train_lc, y_val_lc = y_lc_reset.iloc[train_index], y_lc_reset.iloc[val_index]
        y_train_halc, y_val_halc = y_halc_reset.iloc[train_index], y_halc_reset.iloc[val_index]
        y_train_cs, y_val_cs = y_cs_reset.iloc[train_index], y_cs_reset.iloc[val_index]


        fold_scores, current_plot_data = evaluate_fold(
            fold,
            X_train_fold, X_val_fold,
            y_train_lc, y_val_lc,
            y_train_halc, y_val_halc,
            y_train_cs, y_val_cs,
            params_base_lc, params_base_halc, params_base_cs,
            params_meta_lc, params_meta_halc, params_meta_cs
        )
        fold_scores_list.append(fold_scores)

        if current_plot_data is not None:
            plot_data = current_plot_data # Store data from the last fold


    # Calculate average scores across folds
    avg_scores = {}
    for key in fold_scores_list[0].keys():
        # Handle confusion matrices separately - cannot average matrices directly
        if 'confusion_matrix' in key:
            # Sum confusion matrices
            sum_cm = np.sum([np.array(score[key]) for score in fold_scores_list], axis=0)
            avg_scores[key.replace('_confusion_matrix', '_avg_confusion_matrix')] = sum_cm.tolist()
        else:
            avg_scores[key] = np.mean([score[key] for score in fold_scores_list])

    print("\n--- Average Cross-Validation Scores ---")
    for key, value in avg_scores.items():
        if 'confusion_matrix' in key:
            print(f"{key}: \n{np.array(value)}")
        else:
            print(f"{key}: {value:.4f}")


    # --- Train Models on Full Data and Calculate SHAP ---
    full_models, meta_X_full = train_full_models(
        X, y_lc, y_halc, y_cs,
        params_base_lc, params_base_halc, params_base_cs,
        params_meta_lc, params_meta_halc, params_meta_cs
    )

    # Calculate SHAP values
    base_shap_values = calculate_shap_values({k: full_models[k] for k in full_models if k.startswith('base_')}, X, "Base")
    meta_shap_values = calculate_shap_values({k: full_models[k] for k in full_models if k.startswith('meta_')}, meta_X_full, "Meta")


    # --- Generate and Save Plots ---
    print("\n--- Generating Plots ---")
    plots_dir = 'train_evaluation_plots'
    os.makedirs(plots_dir, exist_ok=True)

    if plot_data is not None:
        # Plot Actual vs. Predicted for regression tasks (using last fold's data)
        plt.figure(figsize=(18, 6))

        # Loss Cost
        plt.subplot(1, 3, 1)
        plt.scatter(plot_data['lc_actual'], plot_data['lc_base_pred'], alpha=0.5, label='Base Model')
        plt.scatter(plot_data['lc_actual'], plot_data['lc_meta_pred'], alpha=0.5, label='Meta Model')
        plt.title('Loss Cost: Actual vs. Predicted (Last Fold)')
        plt.xlabel('Actual Loss Cost')
        plt.ylabel('Predicted Loss Cost')
        min_val = min(plot_data['lc_actual'].min(), plot_data['lc_base_pred'].min(), plot_data['lc_meta_pred'].min())
        max_val = max(plot_data['lc_actual'].max(), plot_data['lc_base_pred'].max(), plot_data['lc_meta_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2) # Diagonal line
        plt.legend()
        plt.grid(True)

        # Historically Adjusted Loss Cost
        plt.subplot(1, 3, 2)
        plt.scatter(plot_data['halc_actual'], plot_data['halc_base_pred'], alpha=0.5, label='Base Model', color='orange')
        plt.scatter(plot_data['halc_actual'], plot_data['halc_meta_pred'], alpha=0.5, label='Meta Model', color='green')
        plt.title('HALC: Actual vs. Predicted (Last Fold)')
        plt.xlabel('Actual HALC')
        plt.ylabel('Predicted HALC')
        min_val = min(plot_data['halc_actual'].min(), plot_data['halc_base_pred'].min(), plot_data['halc_meta_pred'].min())
        max_val = max(plot_data['halc_actual'].max(), plot_data['halc_base_pred'].max(), plot_data['halc_meta_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2) # Diagonal line
        plt.legend()
        plt.grid(True)

        # Confusion Matrix for Claim Status (using average matrix)
        plt.subplot(1, 3, 3)
        # Use the average confusion matrix calculated earlier
        avg_cm_cs = np.array(avg_scores['base_cs_avg_confusion_matrix']) # Using base model average CM for plot
        disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm_cs, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca()) # Plot on the current subplot axis
        plt.title('Base Model Avg Confusion Matrix (Claim Status)')

        plt.tight_layout()
        actual_vs_predicted_filepath = os.path.join(plots_dir, 'train_actual_vs_predicted_and_cm.png')
        plt.savefig(actual_vs_predicted_filepath)
        plt.close() # Close the plot
        print(f"Actual vs. Predicted plots and Base CM saved to '{actual_vs_predicted_filepath}'")

        # Plot Meta Model Average Confusion Matrix separately
        plt.figure(figsize=(8, 6))
        avg_cm_meta_cs = np.array(avg_scores['meta_cs_avg_confusion_matrix'])
        disp_meta = ConfusionMatrixDisplay(confusion_matrix=avg_cm_meta_cs, display_labels=[0, 1])
        disp_meta.plot(cmap=plt.cm.Blues)
        plt.title('Meta Model Avg Confusion Matrix (Claim Status)')
        meta_cm_filepath = os.path.join(plots_dir, 'train_meta_cm.png')
        plt.savefig(meta_cm_filepath)
        plt.close()
        print(f"Meta Model Avg CM saved to '{meta_cm_filepath}'")


    # Generate and save SHAP summary plots for models trained on full data
    print("Generating SHAP summary plots for models trained on full data...")

    # Base Model SHAP plots
    for target, shap_values in base_shap_values.items():
        plt.figure()
        # For classification (cs), shap_values is a list of arrays. Use the positive class (index 1).
        if target == 'cs' and isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X.sample(min(1000, X.shape[0]), random_state=SEED), show=False)
        else:
            shap.summary_plot(shap_values, X.sample(min(1000, X.shape[0]), random_state=SEED), show=False)
        plt.title(f'Base Model SHAP Summary Plot ({target}) - Full Data')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'base_full_shap_summary_{target}.png'))
        plt.close()

    # Meta Model SHAP plots
    for target, shap_values in meta_shap_values.items():
        plt.figure()
        # For classification (cs), shap_values is a list of arrays. Use the positive class (index 1).
        if target == 'cs' and isinstance(shap_values, list):
             # Use the meta_X_full sample for plotting
             shap.summary_plot(shap_values[1], meta_X_full.sample(min(1000, meta_X_full.shape[0]), random_state=SEED), show=False)
        else:
             # Use the meta_X_full sample for plotting
             shap.summary_plot(shap_values, meta_X_full.sample(min(1000, meta_X_full.shape[0]), random_state=SEED), show=False)

        plt.title(f'Meta Model SHAP Summary Plot ({target}) - Full Data')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'meta_full_shap_summary_{target}.png'))
        plt.close()

    print("SHAP summary plots for models trained on full data saved.")


    # --- Save Results to File ---
    print("\n--- Saving Results ---")
    output_dir = 'train_evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    # Save average scores
    avg_scores_filepath = os.path.join(output_dir, 'average_cv_scores.json')
    with open(avg_scores_filepath, 'w') as f:
        json.dump(avg_scores, f, indent=4)
    print(f"Average cross-validation scores saved to '{avg_scores_filepath}'")

    # Save fold-wise scores
    fold_scores_filepath = os.path.join(output_dir, 'fold_cv_scores.json')
    with open(fold_scores_filepath, 'w') as f:
        json.dump(fold_scores_list, f, indent=4)
    print(f"Fold-wise cross-validation scores saved to '{fold_scores_filepath}'")

    # Save SHAP values (can be large, consider saving in a compressed format or subsets)
    base_shap_filepath = os.path.join(output_dir, 'base_full_shap_values.joblib')
    joblib.dump(base_shap_values, base_shap_filepath)
    print(f"Base model SHAP values (full data) saved to '{base_shap_filepath}'")

    meta_shap_filepath = os.path.join(output_dir, 'meta_full_shap_values.joblib')
    joblib.dump(meta_shap_values, meta_shap_filepath)
    print(f"Meta model SHAP values (full data) saved to '{meta_shap_filepath}'")

    # Save a summary of the evaluation
    summary_filepath = os.path.join(output_dir, 'train_evaluation_summary.txt')
    with open(summary_filepath, 'w') as f:
        f.write("Ensemble Model Training Evaluation Summary (5-Fold Cross-Validation)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of folds: {n_folds}\n\n")

        f.write("Average Cross-Validation Scores:\n")
        f.write("--------------------------------\n")
        for key, value in avg_scores.items():
             if 'confusion_matrix' in key:
                  f.write(f"{key}: \n{np.array(value)}\n")
             else:
                  f.write(f"{key}: {value:.4f}\n")
        f.write("\n")

        f.write(f"Fold-wise scores saved to '{fold_scores_filepath}'.\n\n")

        f.write("Plots Generated:\n")
        f.write("----------------\n")
        f.write(f"- Actual vs. Predicted for LC and HALC, and Base Model Confusion Matrix for CS (using last CV fold data): '{actual_vs_predicted_filepath}'\n")
        f.write(f"- Meta Model Confusion Matrix for CS (using average CV matrix): '{meta_cm_filepath}'\n")
        f.write(f"- SHAP Summary Plots for Base Models (trained on full data) saved in '{plots_dir}/base_full_shap_summary_[target].png'\n")
        f.write(f"- SHAP Summary Plots for Meta Models (trained on full data) saved in '{plots_dir}/meta_full_shap_summary_[target].png'\n\n")

        f.write("SHAP Values Saved:\n")
        f.write("------------------\n")
        f.write(f"- Base Model SHAP values (full data): '{base_shap_filepath}'\n")
        f.write(f"- Meta Model SHAP values (full data): '{meta_shap_filepath}'\n\n")

        f.write("Evaluation complete.")


    print(f"Evaluation summary saved to '{summary_filepath}'")
    print("\nTraining evaluation process complete.")


if __name__ == "__main__":
    main()
