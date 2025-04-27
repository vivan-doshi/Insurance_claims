# --- Important Imports ---
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import make_scorer, mean_tweedie_deviance, roc_auc_score
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# --- Configure Variables ---
# data path -- update if needed
head, _ = os.path.split(os.getcwd())
data_dir = os.path.join(head + '/01_Data')

# tweedie power for regression models (LC, HALC)
tweedie_power = 1.5

# number of cross validation folds
n_splits_cv = 5

# number of bins we are creating for stratification of continuous targets (LC, HALC)
# Note: This applies to the non-zero values of each target before combining bins.
n_bins_stratify = 4

# top feature count per target (for Method 1: LGBM Importance - Top N)
n_top_features_per_target = 26

# feature importance threshold (relative) (for Method 2: LGBM Importance - Threshold)
importance_threshold_relative = 0.1 # (interpreted as relative)

# min features for RFECV
min_features_rfe = 5 # User specified

# Define all target variables
target_variables = ['Loss_Cost', 'Historically_Adjusted_Loss_Cost', 'Claim_Status']

# How many features to show in the importance bar chart
n_features_to_plot = 40

# --- File Names ---
train_data_path = os.path.join(data_dir, 'cleaned_data.csv')
test_data_path = os.path.join(data_dir, 'cleaned_test.csv')
output_train_path = os.path.join(data_dir, 'feature_selected_train.csv')
output_test_path = os.path.join(data_dir, 'feature_selected_test.csv')
output_ytrain_path = os.path.join(data_dir, 'feature_selected_y_train.csv')

# --- Feature Selection Method Toggle ---
# Choose which method's feature list to apply for filtering and saving.
# Options represent methods based on LightGBM Importance or RFECV results:
#   'lgbm_union_top_n':      Union of top N features based on LGBM importance per target.
#   'lgbm_union_threshold':  Union of features above threshold based on LGBM importance per target.
#   'rfecv_Loss_Cost':       Features selected by RFECV optimizing for Loss_Cost.
#   'rfecv_Historically_Adjusted_Loss_Cost': Features selected by RFECV optimizing for HALC.
#   'rfecv_Claim_Status':    Features selected by RFECV optimizing for Claim_Status.
#   'none':                  Run selection/plots but do not save filtered data.
final_selection_method_to_apply = 'none' # <<< SET YOUR CHOICE HERE

# --- Preprocessing Function ---
def preprocess_data(df, target_cols=None, fit_mode=True):
    """Applies consistent preprocessing steps, extracts multiple targets, and drops specified cols."""
    print(f"Preprocessing data... Fit mode: {fit_mode}")
    X = df.copy()
    Y_dict = {}

    # Define columns like IDs and known date columns to always drop
    cols_to_always_drop = [
        'ID', # ID column
        # Potential Date columns (adjust list as needed)
        'Start_Date_Contract', 'Date_Last_Renewal', 'Date_Next_Renewal',
        'Date_Of_Birth', 'Date_Of_DL_Issuance', 'Total_Cost_Claims_Current_Yr', 'Total_Number_Claims_Current_Yr',
        'Total_Number_Claims_Entire_Duration', 'Ratio_Claims_Total_Duration_Force'
        # Add any other known non-feature columns here
    ]

    # Separate targets if in fit_mode (training data)
    if fit_mode and target_cols:
        for target_col in target_cols:
            if target_col in X.columns:
                Y_dict[target_col] = X[target_col].copy()
                # Ensure Claim_Status is integer, fill others with 0
                if target_col == 'Claim_Status':
                    Y_dict[target_col] = Y_dict[target_col].fillna(0).astype(int)
                else:
                    Y_dict[target_col] = Y_dict[target_col].fillna(0)
            else:
                print(f"Warning: Target '{target_col}' not found.")
                Y_dict[target_col] = None
        # Add target columns to the drop list for training data
        cols_to_always_drop.extend([col for col in target_cols if col in X.columns])
    else:
         # If not fit_mode, ensure potential target columns are also dropped from test set
         cols_to_always_drop.extend([col for col in target_cols if col in X.columns])

    # Drop the combined list of columns (IDs, Dates, Targets)
    actual_cols_to_drop = [col for col in cols_to_always_drop if col in X.columns]
    if actual_cols_to_drop:
        print(f"Dropping specified columns: {actual_cols_to_drop}")
        X = X.drop(columns=actual_cols_to_drop)

    # --- Remaining preprocessing steps ---
    # Handle non-numeric (Objects/Categories)
    object_cols = X.select_dtypes(include=['object', 'category']).columns
    if object_cols.any():
        print(f"One-hot encoding: {object_cols.tolist()}")
        X = pd.get_dummies(X, columns=object_cols, dummy_na=False, drop_first=True, dtype=int)

    # Ensure all remaining columns are numeric
    for col in X.columns:
        # Check if column is already numeric
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        # Attempt conversion if not numeric
        try:
            X[col] = pd.to_numeric(X[col])
            # print(f"Converted column '{col}' to numeric.") # Less verbose
        except ValueError:
            print(f"Warning: Could not convert column '{col}' to numeric after OHE. Dropping it.")
            X = X.drop(columns=[col])

    # Fill NaNs as a safeguard
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs found in features after processing. Filling with 0.")
        X = X.fillna(0)

    if fit_mode:
        if any(y is None for y in Y_dict.values()):
            print("Error: Not all targets found.")
        return X, Y_dict
    else:
        return X

# --- Load Data ---
print(f"Loading training data from: {train_data_path}")
try:
    train_df = pd.read_csv(train_data_path, index_col=0)
    print("Training data loaded.")
except Exception as e:
    print(f"Error loading training data: {e}")
    exit()

print(f"Loading test data from: {test_data_path}")
try:
    test_df = pd.read_csv(test_data_path, index_col=0)
    print("Test data loaded.")
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

# --- Preprocess Train and Test Data ---
X_train_processed, Y_train_dict = preprocess_data(train_df, target_cols=target_variables, fit_mode=True)
X_test_processed = preprocess_data(test_df, target_cols=target_variables, fit_mode=False)

if any(y is None for y in Y_train_dict.values()):
    print("Exiting due to missing targets.")
    exit()

# --- Align Columns After Preprocessing ---
print("\nAligning columns between processed train and test sets...")
train_cols = X_train_processed.columns
test_cols = X_test_processed.columns

missing_in_test = list(set(train_cols) - set(test_cols))
if missing_in_test:
    for col in missing_in_test:
        X_test_processed[col] = 0

missing_in_train = list(set(test_cols) - set(train_cols))
if missing_in_train:
    cols_to_drop_in_test = [col for col in missing_in_train if col in X_test_processed.columns]
    X_test_processed = X_test_processed.drop(columns=cols_to_drop_in_test)

# Ensure test set columns are in the same order as the training set
X_test_processed = X_test_processed[train_cols]

print(f"Processed training features shape: {X_train_processed.shape}")
print(f"Processed test features shape: {X_test_processed.shape}")

if X_train_processed.shape[1] != X_test_processed.shape[1]:
    print("Error: Column alignment failed!")
    exit()

# --- Feature Selection Methods ---
feature_names = X_train_processed.columns
all_top_features_per_target = {}
all_threshold_features_per_target = {}
rfecv_features_per_target = {}
all_importance_dfs = {}
rfecv_selectors = {}

# Helper function to create combined LC/HALC bins
def create_combined_lc_halc_bins(y_lc, y_halc, n_bins):
    """Creates a combined stratification key based on bins of LC and HALC."""
    if y_lc is None or y_halc is None or len(y_lc) != len(y_halc):
        print("Warning: LC or HALC data missing or mismatched lengths. Cannot create combined bins.")
        return None

    # Handle zero values separately for each target
    lc_non_zero_mask = y_lc > 0
    halc_non_zero_mask = y_halc > 0

    lc_bins = pd.Series('LC_Bin_Zero', index=y_lc.index, dtype=object)
    halc_bins = pd.Series('HALC_Bin_Zero', index=y_halc.index, dtype=object)

    try:
        # Bin non-zero LC values
        if lc_non_zero_mask.sum() > n_bins and y_lc[lc_non_zero_mask].nunique() >= n_bins:
            lc_bins[lc_non_zero_mask] = pd.qcut(y_lc[lc_non_zero_mask], q=n_bins, labels=[f'LC_Bin_{i+1}' for i in range(n_bins)], duplicates='drop')
        elif lc_non_zero_mask.sum() > 0:
             # Fallback for LC if not enough unique non-zero values for n_bins
             lc_bins[lc_non_zero_mask] = 'LC_Bin_NonZero'


        # Bin non-zero HALC values
        if halc_non_zero_mask.sum() > n_bins and y_halc[halc_non_zero_mask].nunique() >= n_bins:
             halc_bins[halc_non_zero_mask] = pd.qcut(y_halc[halc_non_zero_mask], q=n_bins, labels=[f'HALC_Bin_{i+1}' for i in range(n_bins)], duplicates='drop')
        elif halc_non_zero_mask.sum() > 0:
             # Fallback for HALC if not enough unique non-zero values for n_bins
             halc_bins[halc_non_zero_mask] = 'HALC_Bin_NonZero'

        # Combine the bin labels into a single string key
        combined_bins = lc_bins.astype(str) + '_' + halc_bins.astype(str)
        return combined_bins

    except Exception as e:
        print(f"Error during combined LC/HALC binning: {e}")
        # Fallback to a simple 0 vs >0 stratification based on LC+HALC sum if binning fails
        if y_lc is not None and y_halc is not None:
            print("Falling back to 0 vs >0 stratification based on LC + HALC sum.")
            return (y_lc + y_halc > 0).astype(int)
        else:
            return None


# == Methods 1 & 2: Importance Calculation Loop ==
print(f"\n--- Methods 1 & 2: Calculating Feature Importance per Target (based on LGBM) ---")
for target_name in target_variables:
    print(f"\n-- Calculating Importance for Target: {target_name} --")
    y_target = Y_train_dict[target_name]
    cv_iterator = None
    model = None
    stratify_values = None

    # --- Determine model and stratification values for importance calculation ---
    if target_name == 'Claim_Status':
        # For classification (Claim_Status), stratify directly on the target values
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
        stratify_values = y_target
        print("Using LGBMClassifier for importance.")
    elif target_name in ['Loss_Cost', 'Historically_Adjusted_Loss_Cost']:
        # For regression (LC, HALC), use LGBMRegressor and use COMBINED bins for stratification
        model = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1)
        print("Using LGBMRegressor for importance.")
        # Use the combined LC/HALC bins for stratification
        stratify_values = create_combined_lc_halc_bins(
            Y_train_dict.get('Loss_Cost'),
            Y_train_dict.get('Historically_Adjusted_Loss_Cost'),
            n_bins_stratify
        )
        if stratify_values is not None:
             print(f"Using combined LC/HALC bins for importance stratification ({stratify_values.nunique()} unique bins).")
        else:
             print("Warning: Could not create combined LC/HALC bins. Falling back to KFold for importance.")


    else:
        # Should not happen with defined target_variables, but as a safeguard
        print(f"Warning: Unexpected target '{target_name}'. Skipping importance calculation.")
        continue


    # --- Generate CV splits based on determined stratification values ---
    if stratify_values is not None and stratify_values.nunique() >= 2:
        # Use StratifiedKFold if stratification values are available and have at least 2 unique values
        print(f"Generating StratifiedKFold splits based on computed bins/values for importance.")
        skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
        # Ensure stratify_values index aligns with X_train_processed index
        cv_iterator = skf.split(X_train_processed, stratify_values.loc[X_train_processed.index])
    else:
        # Fallback to KFold if stratification is not possible or not enough unique values
        print(f"Warning: Cannot stratify importance for {target_name}. Using KFold splits.")
        kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
        cv_iterator = kf.split(X_train_processed)

    fold_importances = pd.DataFrame(index=feature_names)

    try:
        # Fit the model for each fold and collect feature importances
        for fold, (train_index, val_index) in enumerate(cv_iterator):
            X_fold_train, y_fold_train = X_train_processed.iloc[train_index], y_target.iloc[train_index]
            model.fit(X_fold_train, y_fold_train)
            fold_importances[f'fold_{fold+1}'] = model.feature_importances_

        # Calculate average importance across folds
        fold_importances['average_importance'] = fold_importances.mean(axis=1)
        importance_df = fold_importances[['average_importance']].reset_index().rename(columns={'index': 'feature'})
        importance_df = importance_df.sort_values(by='average_importance', ascending=False).reset_index(drop=True)
        all_importance_dfs[target_name] = importance_df

        # Method 1: Select top N features
        top_n = importance_df['feature'].head(n_top_features_per_target).tolist()
        all_top_features_per_target[target_name] = top_n
        print(f"  Importance calculated. Top {n_top_features_per_target}: {top_n[:5]}...") # Show only first 5

        # Method 2: Select features above a relative threshold
        total_avg_importance = importance_df['average_importance'].sum()
        abs_thresh = importance_threshold_relative * total_avg_importance if total_avg_importance > 0 else 0
        threshold_list = importance_df[importance_df['average_importance'] > abs_thresh]['feature'].tolist()
        all_threshold_features_per_target[target_name] = threshold_list
        print(f"  Found {len(threshold_list)} features above relative threshold {importance_threshold_relative:.1%}")

    except Exception as e:
        print(f"Error calculating importance for {target_name}: {e}")
        all_top_features_per_target[target_name] = []
        all_threshold_features_per_target[target_name] = []
        all_importance_dfs[target_name] = pd.DataFrame()

# == Combine features & Plot for Methods 1 & 2 ==
combined_top_n_set = set().union(*all_top_features_per_target.values())
final_union_top_n_features = sorted(list(combined_top_n_set))

combined_threshold_set = set().union(*all_threshold_features_per_target.values())
final_union_threshold_features = sorted(list(combined_threshold_set))

print("\n-- Generating Combined Feature Importance Plot --")
if all_importance_dfs:
    combined_importance = pd.DataFrame(index=feature_names)
    for target_name, imp_df in all_importance_dfs.items():
         if not imp_df.empty:
             combined_importance = combined_importance.merge(imp_df.set_index('feature'), left_index=True, right_index=True, how='left')
             combined_importance.rename(columns={'average_importance': f'importance_{target_name}'}, inplace=True)

    combined_importance = combined_importance.fillna(0)
    importance_cols = [col for col in combined_importance.columns if 'importance_' in col]

    if importance_cols:
        # Calculate overall average importance across all targets
        combined_importance['overall_avg_importance'] = combined_importance[importance_cols].mean(axis=1)
        combined_importance = combined_importance.sort_values(by='overall_avg_importance', ascending=False)

        # Plot top N overall features
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
    y_target = Y_train_dict[target_name]
    lgbm_rfecv_estimator = None
    scorer = None
    stratify_values_rfecv = None
    cv_splits = None # To store the pre-generated splits

    # --- Determine estimator, scorer, and stratification values for RFECV ---
    if target_name == 'Claim_Status':
        # For classification (Claim_Status), stratify directly on the target values
        stratify_values_rfecv = y_target
        lgbm_rfecv_estimator = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42, n_jobs=-1)
        scorer = 'roc_auc' # Use AUC for classification
        print("Using LGBMClassifier (Scoring: roc_auc) for RFECV.")
    elif target_name in ['Loss_Cost', 'Historically_Adjusted_Loss_Cost']:
        # For regression (LC, HALC), use LGBMRegressor and use COMBINED bins for stratification
        lgbm_rfecv_estimator = lgb.LGBMRegressor(objective='tweedie', tweedie_variance_power=tweedie_power, n_estimators=50, learning_rate=0.1, n_jobs=-1, random_state=42)
        # Use negative mean Tweedie deviance as the scorer for regression
        scorer = make_scorer(mean_tweedie_deviance, greater_is_better=False, power=tweedie_power)
        print(f"Using LGBMRegressor (Tweedie P={tweedie_power}, Scoring: neg_mean_tweedie_deviance) for RFECV.")

        # Use the combined LC/HALC bins for stratification
        stratify_values_rfecv = create_combined_lc_halc_bins(
            Y_train_dict.get('Loss_Cost'),
            Y_train_dict.get('Historically_Adjusted_Loss_Cost'),
            n_bins_stratify
        )
        if stratify_values_rfecv is not None:
             print(f"Using combined LC/HALC bins for RFECV stratification ({stratify_values_rfecv.nunique()} unique bins).")
        else:
             print("Warning: Could not create combined LC/HALC bins. Falling back to KFold for RFECV.")

    else:
         # Should not happen with defined target_variables, but as a safeguard
         print(f"Warning: Unexpected target '{target_name}'. Skipping RFECV.")
         continue


    # --- Generate CV Splits ---
    if stratify_values_rfecv is not None and stratify_values_rfecv.nunique() >= 2:
        # Use StratifiedKFold if stratification values are available and have at least 2 unique values
        print(f"Generating StratifiedKFold splits based on computed bins/values for RFECV.")
        skf_rfecv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
        # Ensure stratify_values_rfecv index aligns with X_train_processed index
        cv_splits = list(skf_rfecv.split(X_train_processed, stratify_values_rfecv.loc[X_train_processed.index]))
    else:
        # Fallback to KFold if stratification is not possible or not enough unique values
        print(f"Warning: Cannot stratify RFECV for {target_name}. Using KFold splits.")
        kf_rfecv = KFold(n_splits=n_splits_cv, shuffle=True, random_state=42)
        cv_splits = list(kf_rfecv.split(X_train_processed))

    # --- Instantiate and run RFECV ---
    # Use min_features_rfe here
    selector_rfecv = RFECV(
        estimator=lgbm_rfecv_estimator,
        step=1,
        cv=cv_splits, # Pass the pre-generated list of splits
        scoring=scorer,
        min_features_to_select=min_features_rfe,
        n_jobs=-1, # Use all available cores
        verbose=1 # Print progress
    )
    print(f"Running RFECV fit...")
    try:
        # Fit using the original y_target; groups argument is not needed when cv is a list of splits
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

# --- Plot RFECV results ---
print("\n-- Generating RFECV Performance Plots --")
for target_name, selector in rfecv_selectors.items():
    try:
        # Check if RFECV ran successfully and has results
        if hasattr(selector, 'cv_results_') and 'mean_test_score' in selector.cv_results_:
            cv_scores = selector.cv_results_['mean_test_score']
            n_features_tested = len(cv_scores)
            # Determine the range of features tested in RFECV
            features_range = range(selector.min_features_to_select, selector.min_features_to_select + n_features_tested)

            plt.figure(figsize=(8, 5))
            plt.xlabel("Number of Features Selected")
            plt.ylabel(f"CV Score ({selector.scoring})")
            plt.title(f"RFECV Performance for {target_name}")
            plt.plot(features_range, cv_scores)
            plt.grid(True)

            # Mark the optimal number of features found by RFECV
            if hasattr(selector, 'n_features_') and (selector.n_features_ - selector.min_features_to_select) >= 0 and (selector.n_features_ - selector.min_features_to_select) < len(cv_scores):
                 optimal_score_index = selector.n_features_ - selector.min_features_to_select
                 plt.plot(selector.n_features_, selector.cv_results_['mean_test_score'][optimal_score_index], 'ro', markersize=8, label=f'Optimal ({selector.n_features_})')
                 plt.legend()

            plt.show()
        else:
            print(f"No valid cv_results_ found for {target_name} RFECV.")
    except Exception as e:
        print(f"Could not plot RFECV results for {target_name}: {e}")

# --- Summary of Feature Lists ---
print("\n--- Feature Selection Summary ---")
print("\nMethods based on LGBM Importance (yielding unified feature sets):")
print(f"  Method 'lgbm_union_top_n':      {len(final_union_top_n_features)} features (Union of Top-{n_top_features_per_target} per target)")
print(f"  Method 'lgbm_union_threshold':  {len(final_union_threshold_features)} features (Union of Threshold {importance_threshold_relative:.1%} per target)")
print(f"\nMethod based on RFECV (yielding target-specific feature sets):")
for target_name, feature_list in rfecv_features_per_target.items():
    n_features = len(feature_list)
    # Display the optimal number of features found by RFECV if available
    optimal_n_str = f"(Optimal: {rfecv_selectors.get(target_name).n_features_})" if target_name in rfecv_selectors and hasattr(rfecv_selectors[target_name], 'n_features_') else ""
    print(f"  Method 'rfecv_{target_name}': {n_features} features {optimal_n_str}")


# --- Apply Selection and Save Data (Based on Toggle) ---
print(f"\n--- Applying Final Selection and Saving (Chosen Method: {final_selection_method_to_apply}) ---")
chosen_features = []
method_applied = "None"

# Select the feature list based on the chosen method
if final_selection_method_to_apply == 'lgbm_union_top_n':
    chosen_features = final_union_top_n_features
    method_applied = final_selection_method_to_apply
elif final_selection_method_to_apply == 'lgbm_union_threshold':
    chosen_features = final_union_threshold_features
    method_applied = final_selection_method_to_apply
elif final_selection_method_to_apply.startswith('rfecv_'):
    target_for_rfecv_apply = final_selection_method_to_apply.replace('rfecv_', '')
    if target_for_rfecv_apply in rfecv_features_per_target:
        chosen_features = rfecv_features_per_target[target_for_rfecv_apply]
        method_applied = final_selection_method_to_apply
    else:
        print(f"Warning: RFECV results for '{target_for_rfecv_apply}' not found.")
        method_applied = "None"
elif final_selection_method_to_apply == 'none':
    method_applied = "None"
else:
    print(f"Warning: Unknown selection method '{final_selection_method_to_apply}'.")
    method_applied = "None"

# Save the filtered data if a valid method was applied and features were selected
if method_applied != "None" and chosen_features:
    print(f"Applying selected features ({len(chosen_features)}) from method '{method_applied}'...")

    # Filter the training and test dataframes to keep only the chosen features
    X_train_final = X_train_processed[chosen_features]
    X_test_final = X_test_processed[chosen_features]

    # Save the filtered feature data
    print(f"Saving filtered training features to: {output_train_path}")
    X_train_final.to_csv(output_train_path, index=True)

    print(f"Saving filtered test features to: {output_test_path}")
    X_test_final.to_csv(output_test_path, index=True)

    # Save the corresponding training targets (ensuring index alignment)
    print(f"Saving corresponding training targets to: {output_ytrain_path}")
    Y_train_df_tosave = pd.DataFrame(Y_train_dict).loc[X_train_final.index]
    Y_train_df_tosave.to_csv(output_ytrain_path, index=True)

    print("\nFiltered data saved successfully.")
    print(f"Final Training Features Shape: {X_train_final.shape}")
    print(f"Final Test Features Shape:     {X_test_final.shape}")
    print(f"Final Training Targets Shape:  {Y_train_df_tosave.shape}")

elif method_applied != "None" and not chosen_features:
    print(f"Warning: Chosen method '{method_applied}' resulted in empty list. No data saved.")
else:
    print("Selection method set to 'none' or chosen method failed. No filtered data saved.")

print("\n--- Feature Selection Script Finished ---")
