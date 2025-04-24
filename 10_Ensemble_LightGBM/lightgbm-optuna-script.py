import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import warnings
import os
import re
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Define file paths
X_TRAIN_PATH = 'feature_selected_train.csv'
Y_TRAIN_PATH = 'feature_selected_y_train.csv'

def clean_feature_names(df):
    """Clean feature names to avoid LightGBM errors"""
    # Replace characters that cause LightGBM errors
    clean_columns = {}
    for col in df.columns:
        # Replace special characters with underscores
        new_col = re.sub(r'[^A-Za-z0-9_]', '_', col)
        # Ensure name starts with a letter or underscore
        if not new_col[0].isalpha() and new_col[0] != '_':
            new_col = 'f_' + new_col
        clean_columns[col] = new_col
    
    # Rename columns
    df = df.rename(columns=clean_columns)
    return df

# Load data
def load_data():
    X = pd.read_csv(X_TRAIN_PATH, index_col=0)
    y = pd.read_csv(Y_TRAIN_PATH, index_col=0)
    
    # Clean feature names to avoid LightGBM errors
    X = clean_feature_names(X)
    
    # Extract target variables
    y_lc = y['Loss_Cost']
    y_halc = y['Historically_Adjusted_Loss_Cost']
    y_cs = y['Claim_Status']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shapes - Loss_Cost: {y_lc.shape}, HALC: {y_halc.shape}, Claim_Status: {y_cs.shape}")
    
    return X, y_lc, y_halc, y_cs

# Create directory for model storage
os.makedirs('models', exist_ok=True)
os.makedirs('params', exist_ok=True)

# Load best parameters
def load_best_params():
    best_params = {}
    
    # Path to parameter files
    param_files = {
        'lc': 'params/best_params_lc.txt',
        'halc': 'params/best_params_halc.txt',
        'cs': 'params/best_params_cs.txt'
    }
    
    # Default parameters if files don't exist
    default_params = {
        'lc': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 8,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8
        },
        'halc': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 8,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8
        },
        'cs': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 8,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'scale_pos_weight': 3.0
        }
    }
    
    # Load parameters from files if they exist
    for model_type, file_path in param_files.items():
        if os.path.exists(file_path):
            params = {}
            with open(file_path, 'r') as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    try:
                        # Try to convert to numeric
                        params[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        # If not numeric, keep as string
                        params[key] = value
            best_params[model_type] = params
        else:
            print(f"Parameter file {file_path} not found. Using default parameters.")
            best_params[model_type] = default_params[model_type]
    
    return best_params

# Train base models
def train_base_models(X, y_lc, y_halc, y_cs, best_params, n_folds=5):
    # Initialize k-fold and stratified k-fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize arrays to store OOF predictions
    oof_lc = np.zeros(len(X))
    oof_halc = np.zeros(len(X))
    oof_cs = np.zeros(len(X))
    
    # Initialize lists to store models
    lc_models = []
    halc_models = []
    cs_models = []
    
    # Loss Cost models
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_lc.iloc[train_idx], y_lc.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Get parameters for training
        params = best_params['lc'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 500
        
        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]
        
        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        # Make predictions
        oof_lc[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Save model
        lc_models.append(model)
        model.save_model(f'models/lc_model_fold_{fold}.txt')
        
        print(f"LC Fold {fold+1} complete")
    
    # HALC models
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_halc.iloc[train_idx], y_halc.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Get parameters for training
        params = best_params['halc'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 500
        
        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]
        
        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        # Make predictions
        oof_halc[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Save model
        halc_models.append(model)
        model.save_model(f'models/halc_model_fold_{fold}.txt')
        
        print(f"HALC Fold {fold+1} complete")
    
    # Claim Status models
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cs)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_cs.iloc[train_idx], y_cs.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Get parameters for training
        params = best_params['cs'].copy()
        n_estimators = params.pop('n_estimators') if 'n_estimators' in params else 500
        
        # Use callbacks for early stopping
        callbacks = [lgb.early_stopping(50, verbose=False)]
        
        # Train model
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        # Make predictions
        oof_cs[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Save model
        cs_models.append(model)
        model.save_model(f'models/cs_model_fold_{fold}.txt')
        
        print(f"CS Fold {fold+1} complete")
    
    # Evaluate OOF predictions
    lc_rmse = np.sqrt(mean_squared_error(y_lc, oof_lc))
    halc_rmse = np.sqrt(mean_squared_error(y_halc, oof_halc))
    cs_auc = roc_auc_score(y_cs, oof_cs)
    
    print(f"OOF Loss Cost RMSE: {lc_rmse}")
    print(f"OOF HALC RMSE: {halc_rmse}")
    print(f"OOF Claim Status AUC: {cs_auc}")
    
    # Save OOF predictions for meta-model training
    oof_df = pd.DataFrame({
        'oof_lc': oof_lc,
        'oof_halc': oof_halc,
        'oof_cs': oof_cs,
        'true_lc': y_lc,
        'true_halc': y_halc,
        'true_cs': y_cs
    })
    
    oof_df.to_csv('oof_predictions.csv', index=False)
    
    return lc_models, halc_models, cs_models, oof_df

# Main execution
if __name__ == "__main__":
    print("Starting LightGBM base model training...")
    
    # Load data
    X, y_lc, y_halc, y_cs = load_data()
    
    # Load best parameters
    best_params = load_best_params()
    
    print("Training base models...")
    lc_models, halc_models, cs_models, oof_df = train_base_models(
        X, y_lc, y_halc, y_cs, best_params
    )
    
    print("Saved out-of-fold predictions to 'oof_predictions.csv'")
    print("All models trained and saved in 'models' directory")
    print("Base model training completed!")