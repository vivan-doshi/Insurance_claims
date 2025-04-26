import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import argparse
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import re
import json
import joblib # Import joblib for saving and loading models

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LightGBM Ensemble Pipeline')
    parser.add_argument('--train', action='store_true', help='Train base and meta models')
    parser.add_argument('--train_base', action='store_true', help='Train only base models')
    parser.add_argument('--train_meta', action='store_true', help='Train only meta models')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--n_bins_regression_stratification', type=int, default=10,
                        help='Number of bins for stratifying regression targets')


    return parser.parse_args()

def clean_feature_names(df):
    """Clean feature names to avoid LightGBM errors"""
    # Replace characters that cause LightGBM errors
    clean_columns = {}
    for col in df.columns:
        # Replace special characters with underscores
        new_col = re.sub(r'[^A-Za-z0-9_]', '_', col)
        clean_columns[col] = new_col
    return df.rename(columns=clean_columns)

def create_combined_stratification_bins(y_lc, y_halc, n_bins):
    """Create combined bins for stratification based on LC and HALC"""
    # Handle potential zero values before binning
    y_lc_nonzero = y_lc[y_lc > 0]
    y_halc_nonzero = y_halc[y_halc > 0]

    # Create bins for non-zero values
    lc_bins = pd.cut(y_lc_nonzero, bins=n_bins, labels=False, duplicates='drop') if len(y_lc_nonzero) > 0 else pd.Series(0, index=y_lc_nonzero.index)
    halc_bins = pd.cut(y_halc_nonzero, bins=n_bins, labels=False, duplicates='drop') if len(y_halc_nonzero) > 0 else pd.Series(0, index=y_halc_nonzero.index)

    # Initialize combined bins with a default value (e.g., -1 for zero values)
    combined_bins = pd.Series(-1, index=y_lc.index, dtype=str) # Use string type for combined bins

    # Assign combined bin labels for non-zero values
    if len(y_lc_nonzero) > 0 and len(y_halc_nonzero) > 0:
         combined_bins[y_lc_nonzero.index] = lc_bins.astype(str) + '_' + halc_bins.astype(str)
    elif len(y_lc_nonzero) > 0: # Only LC has non-zero values
         combined_bins[y_lc_nonzero.index] = lc_bins.astype(str) + '_NA' # Indicate missing HALC bin
    elif len(y_halc_nonzero) > 0: # Only HALC has non-zero values
         combined_bins[y_halc_nonzero.index] = 'NA_' + halc_bins.astype(str) # Indicate missing LC bin


    # Assign a unique bin for zero values (if any)
    zero_indices = y_lc[(y_lc == 0) | (y_halc == 0)].index # Consider zero in either LC or HALC
    if len(zero_indices) > 0:
        combined_bins[zero_indices] = 'ZERO' # Assign a distinct category for zeros


    # Convert to categorical for StratifiedKFold
    return combined_bins.astype('category')

def load_data():
    """Load training and test data"""
    try:
        X_train = pd.read_csv('feature_selected_train.csv')
        y_train_df = pd.read_csv('feature_selected_y_train.csv')
        X_test = pd.read_csv('feature_selected_test.csv')

        # Clean feature names immediately after loading
        X_train = clean_feature_names(X_train)
        X_test = clean_feature_names(X_test)
        y_train_df.columns = [clean_feature_names(pd.DataFrame(columns=[col])).columns[0] for col in y_train_df.columns]


        y_lc = y_train_df['Loss_Cost']
        y_halc = y_train_df['Historically_Adjusted_Loss_Cost']
        y_cs = y_train_df['Claim_Status']

        print(f"Features shape: {X_train.shape}")
        print(f"Target shapes - Loss_Cost: {y_lc.shape}, HALC: {y_halc.shape}, Claim_Status: {y_cs.shape}")
        print(f"Test shape: {X_test.shape}")

        return X_train, y_lc, y_halc, y_cs, X_test

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure data files are in the correct directory.")
        return None, None, None, None, None

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
        print("Run hyperparameter optimization first.")
        return None

def train_base_models(X, y_lc, y_halc, y_cs, y_combined_binned, n_folds):
    """Train base models using cross-validation and generate OOF predictions"""
    print("Training base models...")

    # Use the combined binned variable for consistent stratification across all base models
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_preds_lc = np.zeros(X.shape[0])
    oof_preds_halc = np.zeros(X.shape[0])
    oof_preds_cs = np.zeros(X.shape[0])

    base_models_lc = []
    base_models_halc = []
    base_models_cs = []

    scores = {'lc_rmse': [], 'halc_rmse': [], 'cs_auc': []}

    # Load best parameters for base models
    params_lc = load_params('lc')
    params_halc = load_params('halc')
    params_cs = load_params('cs')

    if params_lc is None or params_halc is None or params_cs is None:
        print("Cannot train base models without hyperparameters. Run optimization first.")
        return None, None, None, None


    # Train base models using the same stratified splits
    for fold, (train_index, val_index) in enumerate(kf.split(X, y_combined_binned)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_lc, y_val_lc = y_lc.iloc[train_index], y_lc.iloc[val_index]
        y_train_halc, y_val_halc = y_halc.iloc[train_index], y_halc.iloc[val_index]
        y_train_cs, y_val_cs = y_cs.iloc[train_index], y_cs.iloc[val_index]


        # Loss Cost model
        model_lc = lgb.LGBMRegressor(**params_lc)
        model_lc.fit(X_train, y_train_lc,
                     eval_set=[(X_val, y_val_lc)],
                     eval_metric='rmse',
                     callbacks=[lgb.early_stopping(100, verbose=False)])

        oof_preds_lc[val_index] = model_lc.predict(X_val)
        rmse_lc = mean_squared_error(y_val_lc, oof_preds_lc[val_index], squared=False)
        scores['lc_rmse'].append(rmse_lc)
        base_models_lc.append(model_lc)
        print(f"LC Fold {fold + 1} complete. RMSE: {rmse_lc:.4f}")

        # HALC model
        model_halc = lgb.LGBMRegressor(**params_halc)
        model_halc.fit(X_train, y_train_halc,
                      eval_set=[(X_val, y_val_halc)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(100, verbose=False)])

        oof_preds_halc[val_index] = model_halc.predict(X_val)
        rmse_halc = mean_squared_error(y_val_halc, oof_preds_halc[val_index], squared=False)
        scores['halc_rmse'].append(rmse_halc)
        base_models_halc.append(model_halc)
        print(f"HALC Fold {fold + 1} complete. RMSE: {rmse_halc:.4f}")


        # Claim Status model
        model_cs = lgb.LGBMClassifier(**params_cs)
        model_cs.fit(X_train, y_train_cs,
                    eval_set=[(X_val, y_val_cs)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(100, verbose=False)])

        oof_preds_cs[val_index] = model_cs.predict_proba(X_val)[:, 1]
        auc_cs = roc_auc_score(y_val_cs, oof_preds_cs[val_index])
        scores['cs_auc'].append(auc_cs)
        base_models_cs.append(model_cs)
        print(f"CS Fold {fold + 1} complete. AUC: {auc_cs:.4f}")


    print(f"Overall OOF Loss Cost RMSE: {np.mean(scores['lc_rmse']):.4f}")
    print(f"Overall OOF HALC RMSE: {np.mean(scores['halc_rmse']):.4f}")
    print(f"Overall OOF Claim Status AUC: {np.mean(scores['cs_auc']):.4f}")

    # Save fold scores
    scores_dir = 'scores'
    os.makedirs(scores_dir, exist_ok=True)
    scores_filename = os.path.join(scores_dir, 'base_model_fold_scores.json')
    with open(scores_filename, 'w') as f:
        json.dump(scores, f, indent=4)
    print(f"Base model fold scores saved to '{scores_filename}'")

    # Save OOF predictions for meta-training
    oof_preds_df = pd.DataFrame({
        'lc_pred': oof_preds_lc,
        'halc_pred': oof_preds_halc,
        'cs_pred': oof_preds_cs
    })
    oof_preds_df.to_csv('oof_predictions.csv', index=False)
    print("OOF predictions saved to 'oof_predictions.csv'")

    # Save base models
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    for i, model in enumerate(base_models_lc):
        joblib.dump(model, os.path.join(models_dir, f'base_lc_fold_{i}.pkl'))
    for i, model in enumerate(base_models_halc):
        joblib.dump(model, os.path.join(models_dir, f'base_halc_fold_{i}.pkl'))
    for i, model in enumerate(base_models_cs):
        joblib.dump(model, os.path.join(models_dir, f'base_cs_fold_{i}.pkl'))
    print("Base models saved.")


    return oof_preds_df # Return OOF predictions for meta-training


def train_meta_models(meta_X, y_lc, y_halc, y_cs, y_combined_binned, n_folds):
    """Train meta models using cross-validation"""
    print("Training meta models...")

    # Use the combined binned variable for consistent stratification across all meta models
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    meta_models_lc = []
    meta_models_halc = []
    meta_models_cs = []

    # Load best parameters for meta models
    params_meta_lc = load_params('meta_lc')
    params_meta_halc = load_params('meta_halc')
    params_meta_cs = load_params('meta_cs')

    if params_meta_lc is None or params_meta_halc is None or params_meta_cs is None:
        print("Cannot train meta models without hyperparameters. Run meta optimization first.")
        return None, None, None


    # Train meta models using the same stratified splits
    for fold, (train_index, val_index) in enumerate(kf.split(meta_X, y_combined_binned)): # Use combined bins for splitting
        X_train, X_val = meta_X.iloc[train_index], meta_X.iloc[val_index]
        y_train_lc, y_val_lc = y_lc.iloc[train_index], y_lc.iloc[val_index]
        y_train_halc, y_val_halc = y_halc.iloc[train_index], y_halc.iloc[val_index]
        y_train_cs, y_val_cs = y_cs.iloc[train_index], y_cs.iloc[val_index]


        # Meta Loss Cost model
        model_meta_lc = lgb.LGBMRegressor(**params_meta_lc)
        model_meta_lc.fit(X_train, y_train_lc,
                          eval_set=[(X_val, y_val_lc)],
                          eval_metric='rmse',
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        meta_models_lc.append(model_meta_lc)
        print(f"Meta LC Fold {fold + 1} complete.")

        # Meta HALC model
        model_meta_halc = lgb.LGBMRegressor(**params_meta_halc)
        model_meta_halc.fit(X_train, y_train_halc,
                           eval_set=[(X_val, y_val_halc)],
                           eval_metric='rmse',
                           callbacks=[lgb.early_stopping(100, verbose=False)])
        meta_models_halc.append(model_meta_halc)
        print(f"Meta HALC Fold {fold + 1} complete.")


        # Meta Claim Status model
        model_meta_cs = lgb.LGBMClassifier(**params_meta_cs)
        model_meta_cs.fit(X_train, y_train_cs,
                         eval_set=[(X_val, y_val_cs)],
                         eval_metric='auc',
                         callbacks=[lgb.early_stopping(100, verbose=False)])
        meta_models_cs.append(model_meta_cs)
        print(f"Meta CS Fold {fold + 1} complete.")

    # Calculate and print overall OOF scores for meta models
    # Need to regenerate OOF predictions from the trained meta models
    meta_oof_preds_lc = np.zeros(meta_X.shape[0])
    meta_oof_preds_halc = np.zeros(meta_X.shape[0])
    meta_oof_preds_cs = np.zeros(meta_X.shape[0])

    # Use the same splits as base models for consistency in OOF calculation
    for fold, (train_index, val_index) in enumerate(kf.split(meta_X, y_combined_binned)): # Use combined bins for splitting
         X_val = meta_X.iloc[val_index]
         meta_oof_preds_lc[val_index] = meta_models_lc[fold].predict(X_val)
         meta_oof_preds_halc[val_index] = meta_models_halc[fold].predict(X_val)
         meta_oof_preds_cs[val_index] = meta_models_cs[fold].predict_proba(X_val)[:, 1]


    overall_meta_lc_rmse = mean_squared_error(y_lc, meta_oof_preds_lc, squared=False)
    overall_meta_halc_rmse = mean_squared_error(y_halc, meta_oof_preds_halc, squared=False)
    overall_meta_cs_auc = roc_auc_score(y_cs, meta_oof_preds_cs)

    print(f"Overall Meta OOF Loss Cost RMSE: {overall_meta_lc_rmse:.4f}")
    print(f"Overall Meta OOF HALC RMSE: {overall_meta_halc_rmse:.4f}")
    print(f"Overall Meta OOF Claim Status AUC: {overall_meta_cs_auc:.4f}")

    # Save meta models
    meta_models_dir = 'meta_models'
    os.makedirs(meta_models_dir, exist_ok=True)
    for i, model in enumerate(meta_models_lc):
        joblib.dump(model, os.path.join(meta_models_dir, f'meta_lc_fold_{i}.pkl'))
    for i, model in enumerate(meta_models_halc):
        joblib.dump(model, os.path.join(meta_models_dir, f'meta_halc_fold_{i}.pkl'))
    for i, model in enumerate(meta_models_cs):
        joblib.dump(model, os.path.join(meta_models_dir, f'meta_cs_fold_{i}.pkl'))
    print("Meta models saved.")

    # Return the trained meta models
    return meta_models_lc, meta_models_halc, meta_models_cs


def load_models(n_folds):
    """Load trained base and meta models"""
    print("Loading trained models...")
    base_models_lc = []
    base_models_halc = []
    base_models_cs = []
    meta_models_lc = []
    meta_models_halc = []
    meta_models_cs = []

    models_dir = 'models'
    meta_models_dir = 'meta_models'

    try:
        for i in range(n_folds):
            base_models_lc.append(joblib.load(os.path.join(models_dir, f'base_lc_fold_{i}.pkl')))
            base_models_halc.append(joblib.load(os.path.join(models_dir, f'base_halc_fold_{i}.pkl')))
            base_models_cs.append(joblib.load(os.path.join(models_dir, f'base_cs_fold_{i}.pkl')))
        print("Base models loaded successfully.")
    except FileNotFoundError:
        print("Error loading base models. Make sure they are trained and saved in the 'models' directory.")
        base_models_lc, base_models_halc, base_models_cs = None, None, None


    try:
        for i in range(n_folds):
            meta_models_lc.append(joblib.load(os.path.join(meta_models_dir, f'meta_lc_fold_{i}.pkl')))
            meta_models_halc.append(joblib.load(os.path.join(meta_models_dir, f'meta_halc_fold_{i}.pkl')))
            meta_models_cs.append(joblib.load(os.path.join(meta_models_dir, f'meta_cs_fold_{i}.pkl')))
        print("Meta models loaded successfully.")
    except FileNotFoundError:
        print("Error loading meta models. Make sure they are trained and saved in the 'meta_models' directory.")
        meta_models_lc, meta_models_halc, meta_models_cs = None, None, None

    return base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs


def predict(X_test, base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs, n_folds):
    """Make predictions on the test set using the ensemble"""
    print("Making predictions...")

    if base_models_lc is None or base_models_halc is None or base_models_cs is None:
        print("Base models not loaded. Cannot make predictions.")
        return None

    # Predict with base models
    base_test_preds_lc = np.zeros(X_test.shape[0])
    base_test_preds_halc = np.zeros(X_test.shape[0])
    base_test_preds_cs = np.zeros(X_test.shape[0])

    for model in base_models_lc:
        base_test_preds_lc += model.predict(X_test) / n_folds
    for model in base_models_halc:
        base_test_preds_halc += model.predict(X_test) / n_folds
    for model in base_models_cs:
        base_test_preds_cs += model.predict_proba(X_test)[:, 1] / n_folds

    # Create meta features for test set
    meta_test_X = pd.concat([
        X_test,
        pd.DataFrame({
            'base_lc_pred': base_test_preds_lc,
            'base_halc_pred': base_test_preds_halc,
            'base_cs_pred': base_test_preds_cs
        }, index=X_test.index) # Ensure index alignment
    ], axis=1)

    if meta_models_lc is None or meta_models_halc is None or meta_models_cs is None:
        print("Meta models not loaded. Predicting only with base models.")
        final_predictions = pd.DataFrame({
            'Loss_Cost': base_test_preds_lc,
            'Historically_Adjusted_Loss_Cost': base_test_preds_halc,
            'Claim_Status': base_test_preds_cs
        })
    else:
        # Predict with meta models
        final_preds_lc = np.zeros(X_test.shape[0])
        final_preds_halc = np.zeros(X_test.shape[0])
        final_preds_cs = np.zeros(X_test.shape[0])

        for model in meta_models_lc:
            final_preds_lc += model.predict(meta_test_X) / n_folds
        for model in meta_models_halc:
            final_preds_halc += model.predict(meta_test_X) / n_folds
        for model in meta_models_cs:
            final_preds_cs += model.predict_proba(meta_test_X)[:, 1] / n_folds

        final_predictions = pd.DataFrame({
            'Loss_Cost': final_preds_lc,
            'Historically_Adjusted_Loss_Cost': final_preds_halc,
            'Claim_Status': final_preds_cs
        })

    # Save final predictions
    final_predictions.to_csv('final_predictions.csv', index=False)
    return final_predictions


def main():
    args = parse_arguments()

    X, y_lc, y_halc, y_cs, X_test = load_data()

    if X is None:
        return # Exit if data loading failed

    # Create combined bins for stratified cross-validation for all models
    y_combined_binned = create_combined_stratification_bins(y_lc, y_halc, args.n_bins_regression_stratification)
    print(f"Created combined bins for stratification with {len(y_combined_binned.cat.categories)} categories.")


    if args.train or args.train_base:
        # Train base models and get OOF predictions
        oof_preds_df = train_base_models(X, y_lc, y_halc, y_cs, y_combined_binned, args.n_folds)

        if args.train or args.train_meta:
             # Create meta features using OOF predictions
             # Check if oof_preds_df was successfully generated
             if oof_preds_df is not None:
                 meta_X = pd.concat([X, oof_preds_df], axis=1)
                 print(f"Meta features shape: {meta_X.shape}")
                 # Train meta models
                 train_meta_models(meta_X, y_lc, y_halc, y_cs, y_combined_binned, args.n_folds)
             else:
                 print("Error: Base model training failed. Cannot train meta models.")


    if args.predict:
        # Load trained models - these will overwrite the initialized empty lists if successful
        # load_models now takes n_folds as argument
        base_models_lc, base_models_halc, base_models_cs, meta_models_lc, meta_models_halc, meta_models_cs = load_models(args.n_folds)

        # Make predictions
        # predict now takes n_folds as argument
        final_predictions = predict(
            X_test,
            base_models_lc,
            base_models_halc,
            base_models_cs,
            meta_models_lc,
            meta_models_halc,
            meta_models_cs,
            args.n_folds
        )

        # Check if predictions were generated successfully
        if final_predictions is not None:
            print("Final predictions saved to 'final_predictions.csv'")
        else:
            print("Failed to generate final predictions.")


if __name__ == "__main__":
    print("=" * 80)
    print("LIGHTGBM ENSEMBLE PIPELINE")
    print("=" * 80)
    main()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
