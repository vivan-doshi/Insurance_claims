import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, mean_tweedie_deviance, roc_auc_score
import matplotlib.pyplot as plt # Added for charts
import warnings
import os
import pickle # Added for saving Y_train_dict

warnings.filterwarnings('ignore')

# --- Configuration ---
# Determine data directory relative to script location or CWD
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback for interactive environments
# Assuming '01_Data' is one level up from script dir or relative to CWD
data_dir_guess1 = os.path.join(os.path.dirname(script_dir), '01_Data')
data_dir_guess2 = os.path.join(script_dir, '01_Data')
data_dir_cwd = os.getcwd()

if os.path.isdir(data_dir_guess1):
    data_dir = data_dir_guess1
elif os.path.isdir(data_dir_guess2):
    data_dir = data_dir_guess2
else:
    print(f"Warning: Standard '01_Data' directories not found relative to script.")
    print(f"Assuming data files are in the current directory: {data_dir_cwd}")
    data_dir = data_dir_cwd # Use current working directory if others aren't found

print(f"Using data directory: {data_dir}")

train_data_path = os.path.join(data_dir, 'cleaned_data.csv')
test_data_path = os.path.join(data_dir, 'cleaned_test.csv')
output_train_path = os.path.join(data_dir, 'feature_selected_train.csv')
output_test_path = os.path.join(data_dir, 'feature_selected_test.csv')
output_ytrain_path = os.path.join(data_dir, 'feature_selected_y_train.csv') # Changed to save all targets

# Define all target variables
target_variables = ['Loss_Cost', 'Historically_Adjusted_Loss_Cost', 'Claim_Status']

# --- Feature Selection Method Toggle ---
# Choose which method's feature list to apply for filtering and saving.
# Options: 'union_top_n', 'union_threshold',
#          'rfecv_Loss_Cost', 'rfecv_Historically_Adjusted_Loss_Cost', 'rfecv_Claim_Status',
#          'none' (to only run selection and plots, without saving filtered data)
final_selection_method_to_apply = 'rfecv_Loss_Cost' # <<< SET YOUR CHOICE HERE

# --- Other Parameters ---
n_splits_cv = 5
n_bins_stratify = 5
n_top_features_per_target = 30 # Increased for potentially better union/plots
importance_threshold_relative = 0.01
min_features_rfe = 10
tweedie_power = 1.5
n_features_to_plot = 40 # How many features to show in the importance bar chart

# --- Preprocessing Function (reuse) ---
def preprocess_data(df, target_cols=None, fit_mode=True):
    # (Same preprocessing function as before - kept concise here)
    X = df.copy(); Y_dict = {}
    if fit_mode and target_cols:
        for target_col in target_cols:
            if target_col in X.columns:
                Y_dict[target_col] = X[target_col].copy()
                if target_col == 'Claim_Status': Y_dict[target_col] = Y_dict[target_col].fillna(0).astype(int)
                else: Y_dict[target_col] = Y_dict[target_col].fillna(0)
            else: print(f"Warning: Target '{target_col}' not found."); Y_dict[target_col] = None
        X = X.drop(columns=[col for col in target_cols if col in X.columns])
    else:
         cols_to_drop_targets = [col for col in target_cols if col in X.columns]
         if cols_to_drop_targets: X = X.drop(columns=cols_to_drop_targets)
    other_ids = ['ID']; cols_to_drop_ids = [col for col in other_ids if col in X.columns]
    if cols_to_drop_ids: X = X.drop(columns=cols_to_drop_ids)
    object_cols = X.select_dtypes(include=['object', 'category']).columns
    if object_cols.any(): X = pd.get_dummies(X, columns=object_cols, dummy_na=False, drop_first=True, dtype=int)
    for col in X.columns:
        if X[col].dtype not in ['int64', 'float64', 'int32', 'float32', 'uint8']:
             try: X[col] = pd.to_numeric(X[col])
             except ValueError: print(f"Warning: Could not convert '{col}'. Dropping."); X = X.drop(columns=[col])
    if X.isnull().sum().sum() > 0: X = X.fillna(0); print("Warning: NaNs found, filled with 0.")
    if fit_mode:
        if any(y is None for y in Y_dict.values()): print("Error: Not all targets found.")
        return X, Y_dict
    else: return X

# --- Load Data ---
print(f"Loading training data from: {train_data_path}")
try: train_df = pd.read_csv(train_data_path, index_col=0); print("Training data loaded.")
except Exception as e: print(f"Error loading training data: {e}"); exit()
print(f"Loading test data from: {test_data_path}")
try: test_df = pd.read_csv(test_data_path, index_col=0); print("Test data loaded.")
except Exception as e: print(f"Error loading test data: {e}"); exit()

# --- Preprocess Train and Test Data ---
X_train_processed, Y_train_dict = preprocess_data(train_df, target_cols=target_variables, fit_mode=True)
X_test_processed = preprocess_data(test_df, target_cols=target_variables, fit_mode=False)
if any(y is None for y in Y_train_dict.values()): print("Exiting due to missing targets."); exit()

# --- Align Columns After Preprocessing ---
print("\nAligning columns between processed train and test sets...")
train_cols = X_train_processed.columns; test_cols = X_test_processed.columns
missing_in_test = list(set(train_cols) - set(test_cols))
if missing_in_test:
    for col in missing_in_test: X_test_processed[col] = 0
missing_in_train = list(set(test_cols) - set(train_cols))
if missing_in_train:
    cols_to_drop_in_test = [col for col in missing_in_train if col in X_test_processed.columns]
    X_test_processed = X_test_processed.drop(columns=cols_to_drop_in_test)
    # print(f"Warning: Dropped {len(cols_to_drop_in_test)} columns from test set.") # Less verbose
X_test_processed = X_test_processed[train_cols]
print(f"Processed training features shape: {X_train_processed.shape}")
print(f"Processed test features shape: {X_test_processed.shape}")
if X_train_processed.shape[1] != X_test_processed.shape[1]: print("Error: Column alignment failed!"); exit()


# --- Feature Selection Methods ---
skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
feature_names = X_train_processed.columns
all_top_features_per_target = {}
all_threshold_features_per_target = {}
rfecv_features_per_target = {}
all_importance_dfs = {} # Store importance dfs for combined plot

# == Methods 1 & 2: Importance Calculation Loop ==
print(f"\n--- Methods 1 & 2: Calculating Feature Importance per Target ---")
for target_name in target_variables:
    print(f"\n-- Calculating Importance for Target: {target_name} --")
    y_target = Y_train_dict[target_name]; cv_iterator = None; model = None
    if target_name == 'Claim_Status':
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1); stratify_values = y_target
        if stratify_values.nunique() < 2: print(f"Warning: Cannot stratify {target_name}. Using KFold."); cv_iterator = KFold(n_splits=n_splits_cv, shuffle=True, random_state=42).split(X_train_processed)
        else: cv_iterator = skf.split(X_train_processed, stratify_values)
    else: # LC or HALC
        model = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1); stratify_values = (y_target > 0).astype(int)
        if stratify_values.nunique() < 2: print(f"Warning: Cannot stratify {target_name}. Using KFold."); cv_iterator = KFold(n_splits=n_splits_cv, shuffle=True, random_state=42).split(X_train_processed)
        else: cv_iterator = skf.split(X_train_processed, stratify_values)
    fold_importances = pd.DataFrame(index=feature_names)
    try:
        for fold, (train_index, val_index) in enumerate(cv_iterator):
            X_fold_train, y_fold_train = X_train_processed.iloc[train_index], y_target.iloc[train_index]
            model.fit(X_fold_train, y_fold_train)
            fold_importances[f'fold_{fold+1}'] = model.feature_importances_
        fold_importances['average_importance'] = fold_importances.mean(axis=1)
        importance_df = fold_importances[['average_importance']].reset_index().rename(columns={'index': 'feature'})
        importance_df = importance_df.sort_values(by='average_importance', ascending=False).reset_index(drop=True)
        all_importance_dfs[target_name] = importance_df # Store for combined plot
        top_n = importance_df['feature'].head(n_top_features_per_target).tolist()
        all_top_features_per_target[target_name] = top_n
        total_avg_importance = importance_df['average_importance'].sum()
        abs_thresh = importance_threshold_relative * total_avg_importance if total_avg_importance > 0 else 0
        threshold_list = importance_df[importance_df['average_importance'] > abs_thresh]['feature'].tolist()
        all_threshold_features_per_target[target_name] = threshold_list
        print(f"  Importance calculated. Top 5: {top_n[:5]}")
        print(f"  Found {len(threshold_list)} features above threshold {importance_threshold_relative:.1%}")
    except Exception as e:
         print(f"Error calculating importance for {target_name}: {e}")
         all_top_features_per_target[target_name] = []; all_threshold_features_per_target[target_name] = []; all_importance_dfs[target_name] = pd.DataFrame()

# == Combine features & Plot for Methods 1 & 2 ==
# Method 1: Union of Top-N
combined_top_n_set = set().union(*all_top_features_per_target.values())
final_union_top_n_features = sorted(list(combined_top_n_set))
# Method 2: Union of Threshold
combined_threshold_set = set().union(*all_threshold_features_per_target.values())
final_union_threshold_features = sorted(list(combined_threshold_set))

# Create combined importance plot
print("\n-- Generating Combined Feature Importance Plot --")
if all_importance_dfs:
    # Combine importance scores across targets
    combined_importance = pd.DataFrame(index=feature_names)
    for target_name, imp_df in all_importance_dfs.items():
         if not imp_df.empty:
              combined_importance = combined_importance.merge(
                  imp_df.set_index('feature'),
                  left_index=True, right_index=True, how='left', suffixes=('', f'_{target_name}')
              )
              combined_importance.rename(columns={'average_importance': f'importance_{target_name}'}, inplace=True)

    combined_importance = combined_importance.fillna(0)
    # Calculate overall average importance (simple mean across targets)
    importance_cols = [col for col in combined_importance.columns if 'importance_' in col]
    if importance_cols:
        combined_importance['overall_avg_importance'] = combined_importance[importance_cols].mean(axis=1)
        combined_importance = combined_importance.sort_values(by='overall_avg_importance', ascending=False)

        # Plotting
        plt.figure(figsize=(10, max(6, n_features_to_plot // 3)))
        top_importances = combined_importance.head(n_features_to_plot)
        plt.barh(top_importances.index, top_importances['overall_avg_importance'], align='center')
        plt.xlabel('Overall Average Importance (Gain)')
        plt.ylabel('Feature')
        plt.title(f'Top {n_features_to_plot} Features by Overall Average Importance')
        plt.gca().invert_yaxis() # Display highest importance at the top
        plt.tight_layout()
        plt.show()
    else:
        print("Could not calculate overall importance.")
else:
    print("No importance dataframes available to generate plot.")


# == Method 3: RFECV Loop for Each Target ==
print(f"\n--- Method 3: Running RFECV for Each Target Variable ---")
rfecv_selectors = {} # Store fitted selectors for plotting
for target_name in target_variables:
    print(f"\n-- Running RFECV for Target: {target_name} --")
    y_target = Y_train_dict[target_name]; cv_iterator_rfecv = None; lgbm_rfecv_estimator = None; scorer = None; selector_rfecv = None
    if target_name == 'Claim_Status':
        stratify_values_rfecv = y_target; lgbm_rfecv_estimator = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42, n_jobs=-1); scorer = 'roc_auc'
        print("Using LGBMClassifier (Scoring: roc_auc)")
        if stratify_values_rfecv.nunique() < 2: print(f"Warning: Cannot stratify RFECV. Using KFold."); cv_iterator_rfecv = KFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
        else: cv_iterator_rfecv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
    else: # LC or HALC
        stratify_values_rfecv = (y_target > 0).astype(int); lgbm_rfecv_estimator = lgb.LGBMRegressor(objective='tweedie', tweedie_variance_power=tweedie_power, n_estimators=50, learning_rate=0.1, n_jobs=-1, random_state=42); scorer = make_scorer(mean_tweedie_deviance, greater_is_better=False, power=tweedie_power)
        print(f"Using LGBMRegressor (Tweedie P={tweedie_power}, Scoring: neg_mean_tweedie_deviance)")
        if stratify_values_rfecv.nunique() < 2: print(f"Warning: Cannot stratify RFECV. Using KFold."); cv_iterator_rfecv = KFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
        else: cv_iterator_rfecv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)

    selector_rfecv = RFECV(estimator=lgbm_rfecv_estimator, step=1, cv=cv_iterator_rfecv, scoring=scorer, min_features_to_select=min_features_rfe, n_jobs=-1, verbose=1)
    print(f"Running RFECV fit...")
    try:
        # RFECV fit needs correct y and potentially groups for stratified splitting
        if isinstance(cv_iterator_rfecv, StratifiedKFold):
             # Need groups=stratify_values_rfecv if cv is a generator requiring groups
             selector_rfecv.fit(X_train_processed, y_target, groups=stratify_values_rfecv)
        else: # KFold doesn't use groups
             selector_rfecv.fit(X_train_processed, y_target)

        print("RFECV fitting complete.")
        n_opt_features = selector_rfecv.n_features_
        selected_list = feature_names[selector_rfecv.support_].tolist()
        rfecv_features_per_target[target_name] = selected_list
        rfecv_selectors[target_name] = selector_rfecv # Store selector for plotting
        print(f"Optimal features found for {target_name}: {n_opt_features}")
    except Exception as e:
         print(f"Error during RFECV execution for {target_name}: {e}")
         rfecv_features_per_target[target_name] = []

# Plot RFECV results
print("\n-- Generating RFECV Performance Plots --")
for target_name, selector in rfecv_selectors.items():
    try:
        if hasattr(selector, 'cv_results_'):
            cv_scores = selector.cv_results_['mean_test_score']
            # Determine the correct range for x-axis (number of features)
            # It depends on initial features, step, and min_features_to_select
            n_features_tested = len(cv_scores)
            # Assuming step=1, range goes from n_features_in_ down to n_features_in_ - n_features_tested + 1
            # Correct range starts from min_features_to_select if specified range is hit
            max_features = selector.n_features_in_
            min_features = selector.min_features_to_select
            # Range typically starts from max_features down to min_features if step=1
            # The cv_results_ are often ordered from fewest features to most, need checking
            # Let's assume the array corresponds to features from min_features up to min_features + len(cv_scores) - 1
            # This might require adjustment based on sklearn version / specific RFECV behavior
            # A safer way might be len(cv_scores) on x-axis if exact feature count is tricky
            # Let's plot vs index assuming it corresponds to features added/removed

            plt.figure(figsize=(8, 5))
            plt.xlabel("Number of features evaluated (check order)") # Label needs careful interpretation
            plt.ylabel(f"CV Score ({selector.scoring})")
            plt.title(f"RFECV Performance for {target_name}")
            plt.plot(range(1, n_features_tested + 1), cv_scores) # Plot vs index
            # Highlight optimal point
            optimal_idx = selector.n_features_ - min_features # Approximate index if ordered descending; needs verification
            # Safer: find index of max score (or min if greater_is_better=False)
            # optimal_score_idx = np.argmax(cv_scores) # Assuming higher is better - adjust if needed!
            # plt.plot(optimal_score_idx + 1, cv_scores[optimal_score_idx], 'ro', markersize=8, label=f'Optimal ({selector.n_features_} features)')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
             print(f"No cv_results_ found for {target_name} RFECV selector.")
    except Exception as e:
        print(f"Could not plot RFECV results for {target_name}: {e}")


# --- Summary of Feature Lists ---
print("\n--- Feature Selection Summary ---")
print(f"\nMethod 1 (Union of Top-{n_top_features_per_target} per target): {len(final_union_top_n_features)} features")
print(f"\nMethod 2 (Union of Threshold {importance_threshold_relative:.1%} per target): {len(final_union_threshold_features)} features")
print(f"\nMethod 3 (RFECV optimized per target):")
for target_name, feature_list in rfecv_features_per_target.items():
    n_features = len(feature_list)
    optimal_n = rfecv_selectors.get(target_name, None)
    optimal_n_str = f"(Optimal: {optimal_n.n_features_})" if optimal_n and hasattr(optimal_n, 'n_features_') else ""
    print(f"  - For '{target_name}': {n_features} features {optimal_n_str}")


# --- Apply Selection and Save Data (Based on Toggle) ---
print(f"\n--- Applying Final Selection and Saving (Method: {final_selection_method_to_apply}) ---")

chosen_features = []
if final_selection_method_to_apply == 'union_top_n':
    chosen_features = final_union_top_n_features
elif final_selection_method_to_apply == 'union_threshold':
    chosen_features = final_union_threshold_features
elif final_selection_method_to_apply.startswith('rfecv_'):
    target_for_rfecv_apply = final_selection_method_to_apply.replace('rfecv_', '')
    if target_for_rfecv_apply in rfecv_features_per_target:
        chosen_features = rfecv_features_per_target[target_for_rfecv_apply]
    else:
        print(f"Warning: RFECV results for '{target_for_rfecv_apply}' not found. Cannot apply this selection.")
        final_selection_method_to_apply = 'none' # Prevent saving attempt

if final_selection_method_to_apply != 'none' and chosen_features:
    print(f"Applying selected features ({len(chosen_features)}) from method '{final_selection_method_to_apply}'...")
    X_train_final = X_train_processed[chosen_features]
    X_test_final = X_test_processed[chosen_features]

    # Save filtered data
    print(f"Saving filtered training features to: {output_train_path}")
    X_train_final.to_csv(output_train_path, index=True)

    print(f"Saving filtered test features to: {output_test_path}")
    X_test_final.to_csv(output_test_path, index=True)

    # Save corresponding Y_train dictionary (all targets, aligned index)
    print(f"Saving corresponding training targets to: {output_ytrain_path}")
    # Save as CSV - index alignment is key
    Y_train_df_tosave = pd.DataFrame(Y_train_dict)
    Y_train_df_tosave = Y_train_df_tosave.loc[X_train_final.index] # Ensure index matches X_train_final
    Y_train_df_tosave.to_csv(output_ytrain_path, index=True)

    print("\nFiltered data saved successfully.")
    print(f"Final Training Features Shape: {X_train_final.shape}")
    print(f"Final Test Features Shape:     {X_test_final.shape}")
    print(f"Final Training Targets Shape:  {Y_train_df_tosave.shape}")

elif not chosen_features and final_selection_method_to_apply != 'none':
     print(f"Warning: The chosen selection method '{final_selection_method_to_apply}' resulted in an empty feature list. No data saved.")
else:
    print("Selection method set to 'none'. No filtered data saved.")

print("\n--- Feature Selection Script Finished ---")