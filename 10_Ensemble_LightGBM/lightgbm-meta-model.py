import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Define file paths
OOF_PREDICTIONS_PATH = 'oof_predictions.csv'
X_TRAIN_PATH = 'feature_selected_train.csv'
X_TEST_PATH = 'feature_selected_test.csv'

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

# Create directories
os.makedirs('meta_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('params', exist_ok=True)

# Load data
def load_meta_data():
    # Load original features
    X = pd.read_csv(X_TRAIN_PATH, index_col=0)
    X = clean_feature_names(X)
    
    # Load out-of-fold predictions
    oof_df = pd.read_csv(OOF_PREDICTIONS_PATH)
    
    # Create meta-features by combining original features with OOF predictions
    meta_X = X.copy()
    meta_X['oof_lc'] = oof_df['oof_lc']
    meta_X['oof_halc'] = oof_df['oof_halc']
    meta_X['oof_cs'] = oof_df['oof_cs']
    
    # Create target variables
    y_lc = oof_df['true_lc']
    y_halc = oof_df['true_halc']
    y_cs = oof_df['true_cs']
    
    return meta_X, y_lc, y_halc, y_cs

# Load test data
def load_test_data():
    X_test = pd.read_csv(X_TEST_PATH, index_col=0)
    X_test = clean_feature_names(X_test)
    return X_test

# Create feature importance plots
def plot_feature_importance(model, feature_names, target_name):
    # Get feature importances
    importances = model.feature_importance(importance_type='gain')
    
    # Create DataFrame for visualization
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(30), palette='viridis')
    plt.title(f'Feature Importance for {target_name} Meta-Model')
    plt.tight_layout()
    plt.savefig(f'plots/{target_name}_feature_importance.png', dpi=300)
    plt.close()
    
    return feature_imp

# Load best parameters
def load_best_params():
    best_params = {}
    
    # Path to parameter files
    param_files = {
        'meta_lc': 'params/best_params_meta_lc.txt',
        'meta_halc': 'params/best_params_meta_halc.txt',
        'meta_cs': 'params/best_params_meta_cs.txt'
    }
    
    # Default parameters if files don't exist
    default_params = {
        'meta_lc': {
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
        'meta_halc': {
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
        'meta_cs': {
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

# Train meta-models with best hyperparameters
def train_meta_models(meta_X, y_lc, y_halc, y_cs, best_params, n_folds=5):
    # Create out-of-fold predictions for meta-models
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize arrays to store OOF predictions from meta-models
    meta_oof_lc = np.zeros(len(meta_X))
    meta_oof_halc = np.zeros(len(meta_X))
    meta_oof_cs = np.zeros(len(meta_X))
    
    # Initialize lists to store meta-models
    meta_lc_models = []
    meta_halc_models = []
    meta_cs_models = []
    
    # Meta Loss Cost models
    for fold, (train_idx, val_idx) in enumerate(kf.split(meta_X)):
        X_train, X_val = meta_X.iloc[train_idx], meta_X.iloc[val_idx]
        y_train, y_val = y_lc.iloc[train_idx], y_lc.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Get parameters for training
        params = best_params['meta_lc'].copy()
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
        meta_oof_lc[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Save model
        meta_lc_models.append(model)
        model.save_model(f'meta_models/meta_lc_model_fold_{fold}.txt')
        
        # Generate feature importance plot
        if fold == 0:  # Only for the first fold to avoid multiple plots
            plot_feature_importance(model, X_train.columns, 'Loss_Cost')
        
        print(f"Meta LC Fold {fold+1} complete")
    
    # Meta HALC models
    for fold, (train_idx, val_idx) in enumerate(kf.split(meta_X)):
        X_train, X_val = meta_X.iloc[train_idx], meta_X.iloc[val_idx]
        y_train, y_val = y_halc.iloc[train_idx], y_halc.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Get parameters for training
        params = best_params['meta_halc'].copy()
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
        meta_oof_halc[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Save model
        meta_halc_models.append(model)
        model.save_model(f'meta_models/meta_halc_model_fold_{fold}.txt')
        
        # Generate feature importance plot
        if fold == 0:  # Only for the first fold
            plot_feature_importance(model, X_train.columns, 'HALC')
        
        print(f"Meta HALC Fold {fold+1} complete")
    
    # Meta Claim Status models
    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, y_cs)):
        X_train, X_val = meta_X.iloc[train_idx], meta_X.iloc[val_idx]
        y_train, y_val = y_cs.iloc[train_idx], y_cs.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Get parameters for training
        params = best_params['meta_cs'].copy()
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
        meta_oof_cs[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Save model
        meta_cs_models.append(model)
        model.save_model(f'meta_models/meta_cs_model_fold_{fold}.txt')
        
        # Generate feature importance plot
        if fold == 0:  # Only for the first fold
            plot_feature_importance(model, X_train.columns, 'Claim_Status')
        
        print(f"Meta CS Fold {fold+1} complete")
    
    # Evaluate meta OOF predictions
    meta_lc_rmse = np.sqrt(mean_squared_error(y_lc, meta_oof_lc))
    meta_halc_rmse = np.sqrt(mean_squared_error(y_halc, meta_oof_halc))
    meta_cs_auc = roc_auc_score(y_cs, meta_oof_cs)
    
    print(f"Meta OOF Loss Cost RMSE: {meta_lc_rmse}")
    print(f"Meta OOF HALC RMSE: {meta_halc_rmse}")
    print(f"Meta OOF Claim Status AUC: {meta_cs_auc}")
    
    # Save meta OOF predictions
    meta_oof_df = pd.DataFrame({
        'meta_oof_lc': meta_oof_lc,
        'meta_oof_halc': meta_oof_halc,
        'meta_oof_cs': meta_oof_cs,
        'true_lc': y_lc,
        'true_halc': y_halc,
        'true_cs': y_cs
    })
    
    meta_oof_df.to_csv('meta_oof_predictions.csv', index=False)
    
    return meta_lc_models, meta_halc_models, meta_cs_models, meta_oof_df

# Make predictions for test data
def predict_test_data(base_models_lc, base_models_halc, base_models_cs, meta_lc_models, meta_halc_models, meta_cs_models):
    # Load test data
    X_test = load_test_data()
    
    # Initialize arrays to store predictions from base models
    test_pred_lc = np.zeros(len(X_test))
    test_pred_halc = np.zeros(len(X_test))
    test_pred_cs = np.zeros(len(X_test))
    
    # Make predictions with base models
    for i, (lc_model, halc_model, cs_model) in enumerate(zip(base_models_lc, base_models_halc, base_models_cs)):
        test_pred_lc += lc_model.predict(X_test) / len(base_models_lc)
        test_pred_halc += halc_model.predict(X_test) / len(base_models_halc)
        test_pred_cs += cs_model.predict(X_test) / len(base_models_cs)
    
    # Create meta features for test data
    meta_X_test = X_test.copy()
    meta_X_test['oof_lc'] = test_pred_lc
    meta_X_test['oof_halc'] = test_pred_halc
    meta_X_test['oof_cs'] = test_pred_cs
    
    # Initialize arrays to store meta model predictions
    meta_test_pred_lc = np.zeros(len(X_test))
    meta_test_pred_halc = np.zeros(len(X_test))
    meta_test_pred_cs = np.zeros(len(X_test))
    
    # Make predictions with meta models
    for i, (lc_model, halc_model, cs_model) in enumerate(zip(meta_lc_models, meta_halc_models, meta_cs_models)):
        meta_test_pred_lc += lc_model.predict(meta_X_test) / len(meta_lc_models)
        meta_test_pred_halc += halc_model.predict(meta_X_test) / len(meta_halc_models)
        meta_test_pred_cs += cs_model.predict(meta_X_test) / len(meta_cs_models)
    
    # Create final predictions DataFrame
    final_predictions = pd.DataFrame({
        'LC': meta_test_pred_lc,
        'HALC': meta_test_pred_halc,
        'CS': (meta_test_pred_cs > 0.5).astype(int)  # Convert probabilities to binary predictions
    })
    
    # Save final predictions
    final_predictions.to_csv('final_predictions.csv', index=False)
    
    return final_predictions

# Load base models
def load_base_models():
    base_models_lc = []
    base_models_halc = []
    base_models_cs = []
    
    for fold in range(5):
        try:
            lc_model = lgb.Booster(model_file=f'models/lc_model_fold_{fold}.txt')
            halc_model = lgb.Booster(model_file=f'models/halc_model_fold_{fold}.txt')
            cs_model = lgb.Booster(model_file=f'models/cs_model_fold_{fold}.txt')
            
            base_models_lc.append(lc_model)
            base_models_halc.append(halc_model)
            base_models_cs.append(cs_model)
        except Exception as e:
            print(f"Error loading base models for fold {fold}: {e}")
    
    return base_models_lc, base_models_halc, base_models_cs

# Main execution
if __name__ == "__main__":
    print("Starting LightGBM meta-model training...")
    
    # Load meta data
    meta_X, y_lc, y_halc, y_cs = load_meta_data()
    
    # Load best parameters
    best_params = load_best_params()
    
    # Train meta models
    print("Training meta models...")
    meta_lc_models, meta_halc_models, meta_cs_models, meta_oof_df = train_meta_models(
        meta_X, y_lc, y_halc, y_cs, best_params
    )
    
    print("Saved meta out-of-fold predictions to 'meta_oof_predictions.csv'")
    print("All meta models trained and saved in 'meta_models' directory")
    
    # Load base models for test prediction
    print("Loading base models for test prediction...")
    base_models_lc, base_models_halc, base_models_cs = load_base_models()
    
    # Make predictions on test data
    print("Making final predictions for test data...")
    final_predictions = predict_test_data(
        base_models_lc,
        base_models_halc,
        base_models_cs,
        meta_lc_models,
        meta_halc_models,
        meta_cs_models
    )
    
    print("Final predictions saved to 'final_predictions.csv'")
    print("Meta-model training and prediction completed!")