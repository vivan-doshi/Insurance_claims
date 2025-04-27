``` mermaid
flowchart LR
    TD["Training Data"]:::box --> EDA["EDA(Null Filling, Visualizations, Sanity Checks)"]:::box
    EDA --> CD["Cleaned Data"]:::box --> FE["Feature Engineering"]:::box --> FS["Feature Selection"]:::box
    FS --> XGB["XGBoost"]:::model
    FS --> LGBM["LGBM"]:::model
    FS --> NN["Neural Network"]:::model
    FS --> TS["Two-Stage"]:::model
    FS --> ENS["Ensemble"]:::model
    XGB --> TUNE
    LGBM --> TUNE
    NN --> TUNE
    TS --> TUNE
    ENS --> TUNE
    TUNE["Hyperparameter Tuning"]:::box --> EVAL["Model Evaluation"]:::box --> SELECT["Model Selection"]:::box --> PRED["Prediction / Classification\n(on Test Data)"]:::box
    classDef box fill:#f3e8ff,stroke:#9c88ff,stroke-width:1px,color:#2d2155;
    classDef model fill:#e0d4fc,stroke:#a18ee3,stroke-width:1px,color:#2d2155;
```