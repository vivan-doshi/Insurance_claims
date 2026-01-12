# Insurance Claims Analysis & Prediction

## Project Overview
This project involves a comprehensive analysis of insurance claims data to predict key metrics: **Claim Status (CS)**, **Loss Cost (LC)**, and **Historically Adjusted Loss Cost (HALC)**. The solution employs a robust pipeline consisting of data processing, advanced feature engineering, specific feature selection strategies, and LightGBM modeling with Tweedie regression.

## Directory Structure
- **`01_Data`**: Contains the raw input files (`insurance_train.csv`, `insurance_test.csv`) and processed artifacts.
- **`02_Exploration_notebook`**: Jupyter notebooks (`clean_eda.ipynb`, `exploring.ipynb`) used for initial Exploratory Data Analysis (EDA).
- **`03_Data_Processing`**: Scripts for data cleaning and feature engineering (`data_process.py`).
- **`04_Feature_Selection`**: Implementation of feature selection strategies (`feature_selection_model.py`).
- **`06_ML_Notebooks`**: Advanced machine learning modeling scripts (`adv_lightgbm.py`, etc.).
- **`99_Final_File`**: Contains the final prediction output (e.g., `group_29_prediction.csv`).

## Methodology

### 1. Data Processing & Feature Engineering
**Script:** `03_Data_Processing/data_process.py`

The raw data undergoes significant preprocessing to create a clean, feature-rich dataset:
- **Date Parsing**: Columns like `Start_Date_Contract`, `Date_Of_Birth`, and `Date_Last_Renewal` are converted to datetime objects.
- **Target Creation**:
    - **`Loss_Cost`**: Calculated as `Total_Cost_Claims_Current_Yr / Total_Number_Claims_Current_Yr`.
    - **`Historically_Adjusted_Loss_Cost`**: `Loss_Cost` adjusted by `Ratio_Claims_Total_Duration_Force`.
    - **`Claim_Status`**: Binary flag (1 if total number of claims > 0, else 0).
- **Feature Engineering**:
    - **Demographics & Tenure**: `Age`, `Years_Driving`, `Age_at_contract`, `Time_Since_Last_Renewal`.
    - **Vehicle Info**: `Car_Age`, `Car_Age_Cat` (binned), `Power_Wt_Ratio`.
    - **Risk Factors**: `New_License`, `Young_Driver` (<25yo), `Non_Payment_Termination`.
    - **Complex Interactions**: 
        - `Customer_Loyalty`: Weighted score based on years associated, policies held, etc.
        - `New_Bhp_Risk` & `Young_Bhp_Risk`: Interaction between driver experience/age and high vehicle power (>250 HP).

### 2. Feature Selection
**Script:** `04_Feature_Selection/feature_selection_model.py`

To improve model performance and interpretability, we employ a two-stage feature selection process:
- **Importance-Based Selection**: Uses a LightGBM model to rank features by split gain.
- **RFECV (Recursive Feature Elimination)**: 
    - **Claim Status**: Optimizes for **AUC**.
    - **LC / HALC**: Optimizes for **Negative Mean Tweedie Deviance** (p=1.5).
- **Stratification**: A unique "Combined Binning" strategy is used for regression targets, stratifying folds based on quantiles of `Loss_Cost` and `HALC` to ensure representative training splits.

### 3. Modeling
**Script:** `06_ML_Notebooks/adv_lightgbm.py`

The final models are built using **LightGBM** with **Optuna** for hyperparameter tuning.

#### Models
1.  **Claim Status (Classification)**
    - **Objective**: `binary`
    - **Metric**: `AUC`
    - **Optimization**: Maximizes accuracy and AUC to correctly identify policyholders likely to file a claim.
2.  **Loss Cost & HALC (Regression)**
    - **Objective**: `tweedie` (Variance Power p=1.5)
    - **Metric**: `RMSE`
    - **Reasoning**: The Tweedie distribution usually handles zero-inflated continuous data (like insurance losses) better than standard regression objectives.

#### Evaluation
- **Stratified Validation**: Models are evaluated on a 20% holdout set, stratified by the complex `Claim_Status + LC + HALC` bins.
- **Metrics**: RMSE for regression targets, Confusion Matrix and Classification Report for Claim Status.
- **Interpretability**: SHAP (SHapley Additive exPlanations) summary plots are generated to visualize feature contributions.

## Results & Key Findings

Based on the validation set performance:

### 1. Claim Status Prediction
- **Accuracy**: ~90%
- **Performance**: The model achieves high overall accuracy, largely driven by the correct identification of non-claimants (Precision ~91%).
- **Challenges**: Identifying positive claims remains difficult (Recall ~19% at 0.45 threshold), indicating a need for techniques to address class imbalance (e.g., lower thresholds, resampling).

### 2. Loss Cost Prediction
- **Loss Cost (LC) RMSE**: ~451.37
- **Historically Adjusted Loss Cost (HALC) RMSE**: ~1096.88
- **Insight**: The Tweedie objective function successfully captures the variance in loss costs, though the high RMSE for HALC suggests it is a harder target to predict due to the historical adjustment factors.

## Usage Instructions

To replicate the analysis and generate predictions:

1.  **Install Dependencies**:
    Ensure you have `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `optuna`, `matplotlib`, and `shap` installed.

2.  **Run Data Processing**:
    ```bash
    python 03_Data_Processing/data_process.py
    ```
    *Input: `insurance_train.csv` | Output: `cleaned_data.csv`*

3.  **Run Feature Selection**:
    ```bash
    python 04_Feature_Selection/feature_selection_model.py
    ```
    *Input: `cleaned_data.csv` | Output: `feature_selected_train.csv`*

4.  **Run Modeling**:
    ```bash
    python 06_ML_Notebooks/adv_lightgbm.py
    ```
    *Input: `feature_selected_train.csv` | Output: `group_29_prediction.csv`*

## Final Output
The comprehensive prediction file is located at `99_Final_File/group_29_prediction.csv`.