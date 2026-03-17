# Experiment Run Log: codex

## Step: 0. Setup
- completion status: Completed
- key actions: Set global random seed to 42, initialized run log and evidence folder, read variable dictionary file only.
- key outputs: run_log_codex.md, evidence_codex/
- important warnings or errors: None

## Step: 1.1 Dataset ingestion and schema checks
- completion status: Completed
- key actions: Loaded dataset into participation_raw, validated required variables, checked duplicates, dtypes, and null counts.
- key outputs: participation_raw shape=(34378, 11); schema checks passed.
- important warnings or errors: None

## Step: 1.2 Problem definition
- completion status: Completed
- key actions: Defined prediction objective, declared target/features, documented target missing-value rule and variable table.
- key outputs: Markdown definition completed in notebook.
- important warnings or errors: None

## Step: 2. EDA
- completion status: Completed
- key actions: Dropped target rows coded -3/3, created participation_eda with new target column, generated and saved EDA figures.
- key outputs: participation_eda shape=(34338, 11); EDA figures saved to evidence_codex/EDA_codex_Pics
- important warnings or errors: None

## Step: 3. Missingness handling
- completion status: Completed
- key actions: Applied variable-specific non-informative code handling and imputation to feature variables.
- key outputs: participation_clean shape=(34338, 11); no feature missing values; summary saved to evidence_codex/missingness_summary_codex.csv
- important warnings or errors: None

## Step: 4.1 Prepare modeling data
- completion status: Completed
- key actions: Defined X/y, created separate preprocessing pipelines for LR and XGBoost, and produced stratified 70/15/15 splits.
- key outputs: X_train=(24036, 10), X_val=(5151, 10), X_test=(5151, 10)
- important warnings or errors: None

## Step: 4.2 Evaluation harness
- completion status: Completed
- key actions: Implemented shared metric functions and validation-threshold tuning rule for both model families.
- key outputs: evaluate_predictions, find_best_threshold, evaluate_model functions defined.
- important warnings or errors: None

## Step: 4.3 Baseline LR
- completion status: Completed
- key actions: Trained baseline Logistic Regression and evaluated on validation set at threshold 0.50.
- key outputs: Validation F2=0.3710, PR-AUC=0.1915
- important warnings or errors: None

## Step: 5.1 Improve LR
- completion status: Completed
- key actions: Completed grid search over LR hyperparameters and validation threshold tuning.
- key outputs: Best LR params={'model__C': 3.0, 'model__penalty': 'l2', 'model__class_weight': 'balanced'}, best threshold=0.45, best validation F2=0.3783
- important warnings or errors: None

## Step: 5.2 Tune XGBoost
- completion status: Completed
- key actions: Completed grid search over XGBoost hyperparameters and validation threshold tuning.
- key outputs: Best XGB params={'model__n_estimators': 400, 'model__max_depth': 3, 'model__learning_rate': 0.1, 'model__subsample': 1.0}, best threshold=0.10, best validation F2=0.3783
- important warnings or errors: None

## Step: 5.3 Model comparison
- completion status: Completed
- key actions: Evaluated baseline LR, tuned LR, and tuned XGBoost on the test set only.
- key outputs: Comparison table saved to evidence_codex/model_comparison_test_codex.csv
- important warnings or errors: None

## Step: 5.4 Final model decision
- completion status: Completed
- key actions: Applied weighted quantitative framework to tuned LR and tuned XGBoost and selected final model.
- key outputs: Final model=Tuned XGBoost; selection table saved to evidence_codex/final_model_selection_table_codex.csv
- important warnings or errors: None

