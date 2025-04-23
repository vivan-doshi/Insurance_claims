## =========== Import ===========
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import RFECV
from lightgbm import LGBMRegressor, LGBMClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

## =========== Change file location ===========
head, tail = os.path.split(os.getcwd())
os.chdir(os.path.join(head,'01_Data'))

## =========== Reading data ===========
data = pd.read_parquet('cleaned_data.parquet')
X = data.copy()

## =========== Creating X and Y ===========
# drop the columns
X = X.drop(columns=['ID', 'Total_Cost_Claims_Current_Yr', 'Total_Number_Claims_Current_Yr',
                    'Total_Number_Claims_Entire_Duration', 'Ratio_Claims_Total_Duration_Force',
                    'Loss_Cost', 'Historically_Adjusted_Loss_Cost', 'Claim_Status'])

# creating Y reg and class
Y_reg = data[['Loss_Cost','Historically_Adjusted_Loss_Cost']]
Y_class = data[['Claim_Status']]

## =========== Fixing X Data ===========
# fixing energy source
X['Energy_Source'] = X['Energy_Source'].fillna('Other')

# representing categorical better
X = pd.get_dummies(X, columns=['Car_Age_Cat', 'Energy_Source'], dtype=int, drop_first=False)
X = X.drop(columns=['Car_Age_Cat_New', 'Energy_Source_Other'])

# dropping times columns
X = X.drop(columns=['Start_Date_Contract','Date_Last_Renewal','Date_Next_Renewal','Date_Of_Birth','Date_Of_DL_Issuance'])

## =========== Fixing Y Data ===========
Y_reg = Y_reg.fillna(0)

## =========== Splitting Train and Test ===========
## ----------- For class -----------
X_class_train, X_class_test, Y_class_train, Y_class_test = train_test_split(
    X, Y_class,
    test_size=0.2,
    stratify=Y_class,
    random_state=42
)

## ----------- For reg -----------
# create a zero-flag
zero_flag = (Y_reg['Loss_Cost'] == 0).astype(int)
# split it
X_reg_train, X_reg_test, Y_reg_train, Y_reg_test = train_test_split(
    X, Y_reg,
    test_size=0.2,
    stratify=zero_flag,
    random_state=42
)

## =========== Creating helper function ===========

## ----------- Making Stratified Bins -----------

def make_stratify_bins(y, n_bins=10):
    return pd.qcut(y, q=n_bins, duplicates='drop', labels=False)

## ----------- RMSE -----------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

## =========== Variable Selection ===========
## ----------- Select features with RFECV -----------
def select_features_with_rfecv(X, y, objective="regression"):
    estimator = (
        LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
        if objective == "regression"
        else LGBMClassifier(random_state=42)
    )
    scoring = "neg_root_mean_squared_error" if objective == "regression" else "roc_auc"

    if objective == "regression":
        y_strat = make_stratify_bins(y, n_bins=5)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rfecv = RFECV(estimator=estimator, step=1, cv=cv.split(X, y_strat), scoring=scoring, min_features_to_select=5)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring=scoring, min_features_to_select=5)

    rfecv.fit(X, y)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross-Validation Score")
    plt.title("RFECV Performance by Number of Features")
    plt.tight_layout()
    plt.show()

    selected_features = X.columns[rfecv.support_].tolist()
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    return X[selected_features]

## =========== Hyperparameter tuning ===========
## ----------- Optuna tuning -----------
def tune_lightgbm(X, y, folds, objective, scorer, use_gpu=False, show_plots=False):
    """
    X : predictors (DataFrame)
    y : target (Series)
    folds : a StratifiedKFold object
    objective : "tweedie" or "binary"
    scorer : "rmse" or "auc"
    use_gpu : Set to True to enable GPU (default False)
    show_plots : If True, displays Optuna visualizations (default False)
    """

    direction = "minimize" if scorer == "rmse" else "maximize"

    if scorer == "rmse":
        y_strat = make_stratify_bins(y)
    else:
        y_strat = y

    def objective_fn(trial):
        params = {
            "objective": objective,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("leaves", 16, 255),
            "feature_fraction": trial.suggest_float("ff", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bf", 0.5, 1.0),
            "bagging_freq": 1,
            "max_depth": -1,
            "lambda_l1": trial.suggest_float("l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("l2", 0.0, 5.0),
            "random_state": 42,
            "verbosity": -1
        }

        if use_gpu:
            params["device"] = "gpu"
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0

        if objective == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float("p", 1.1, 1.9)
        if objective == "binary":
            params["is_unbalance"] = True

        scores = []

        for tr_idx, val_idx in folds.split(X, y_strat):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = (
                LGBMRegressor(**params, n_estimators=5000)
                if objective == "tweedie"
                else LGBMClassifier(**params, n_estimators=5000)
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=scorer
            )

            preds = model.predict(X_val)

            if scorer == "rmse":
                scores.append(rmse(y_val, preds))
            else:
                scores.append(roc_auc_score(y_val, preds))

        return np.mean(scores)

    study = optuna.create_study(direction=direction)
    study.optimize(objective_fn, n_trials=60, timeout=1800, show_progress_bar=True)

    if show_plots:
        try:
            from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

            plot_optimization_history(study)
            plt.title("Optuna Optimization History")
            plt.tight_layout()
            plt.show()

            plot_param_importances(study)
            plt.title("Hyperparameter Importance")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Install matplotlib and optuna[visualization] to enable plots.")

    return study.best_params

## =========== Pipelines ===========
## ----------- For LC (Loss Cost) -----------
X_lc_selected = select_features_with_rfecv(X_reg_train, Y_reg_train["Loss_Cost"], objective="regression")
skf_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_lc_params = tune_lightgbm(X_lc_selected, Y_reg_train["Loss_Cost"], skf_lc, "tweedie", "rmse")
X_lc_test_selected = X_reg_test[X_lc_selected.columns]

## ----------- For HALC (Historically Adjusted Loss Cost) -----------
X_halc_selected = select_features_with_rfecv(X_reg_train, Y_reg_train["Historically_Adjusted_Loss_Cost"], objective="regression")
skf_halc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_halc_params = tune_lightgbm(X_halc_selected, Y_reg_train["Historically_Adjusted_Loss_Cost"], skf_halc, "tweedie", "rmse")
X_halc_test_selected = X_reg_test[X_halc_selected.columns]

## ----------- For CS (Claim Status) -----------
X_cs_selected = select_features_with_rfecv(X_class_train, Y_class_train["Claim_Status"], objective="binary")
skf_cs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_cs_params = tune_lightgbm(X_cs_selected, Y_class_train["Claim_Status"], skf_cs, "binary", "auc")
X_cs_test_selected = X_class_test[X_cs_selected.columns]

## =========== Finding RMSE ===========
## ----------- For LC (Loss Cost) -----------
model_lc = LGBMRegressor(**best_lc_params, n_estimators=5000)
model_lc.fit(X_lc_selected, Y_reg_train["Loss_Cost"])

preds_lc_train = model_lc.predict(X_lc_selected)
rmse_lc_train = rmse(Y_reg_train["Loss_Cost"], preds_lc_train)
print(f"Train RMSE for Loss_Cost: {rmse_lc_train:.4f}")

preds_lc = model_lc.predict(X_lc_test_selected)
rmse_lc = rmse(Y_reg_test["Loss_Cost"], preds_lc)
print(f"Test RMSE for Loss_Cost: {rmse_lc:.4f}")

## ----------- For HALC (Historically Adjusted Loss Cost) -----------
model_halc = LGBMRegressor(**best_halc_params, n_estimators=5000)
model_halc.fit(X_halc_selected, Y_reg_train["Historically_Adjusted_Loss_Cost"])

preds_halc_train = model_halc.predict(X_halc_selected)
rmse_halc_train = rmse(Y_reg_train["Historically_Adjusted_Loss_Cost"], preds_halc_train)
print(f"Train RMSE for Historically_Adjusted_Loss_Cost: {rmse_halc_train:.4f}")

preds_halc = model_halc.predict(X_halc_test_selected)
rmse_halc = rmse(Y_reg_test["Historically_Adjusted_Loss_Cost"], preds_halc)
print(f"Test RMSE for Historically_Adjusted_Loss_Cost: {rmse_halc:.4f}")

## =========== Finding RUC-AUC ===========
## ----------- For CS (Claim Status) -----------
model_cs = LGBMClassifier(**best_cs_params, n_estimators=5000)
model_cs.fit(X_cs_selected, Y_class_train["Claim_Status"])

preds_cs_train = model_cs.predict_proba(X_cs_selected)[:, 1]
auc_cs_train = roc_auc_score(Y_class_train["Claim_Status"], preds_cs_train)
print(f"Train AUC for Claim_Status: {auc_cs_train:.4f}")

preds_cs = model_cs.predict_proba(X_cs_test_selected)[:, 1]
auc_cs = roc_auc_score(Y_class_test["Claim_Status"], preds_cs)
print(f"Test AUC for Claim_Status: {auc_cs:.4f}")