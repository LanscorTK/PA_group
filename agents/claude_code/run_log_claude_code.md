# Run Log — claude_code

### Step 0 — Setup
- **Status:** SUCCESS
- **Actions:** Imported libraries; set RANDOM_STATE=42; created evidence directories; read data dictionary
- **Outputs:** run_log_claude_code.md, evidence_claude_code/
- **Warnings/Errors:** None

### Step 1.1 — Dataset Ingestion & Schema Checks
- **Status:** SUCCESS
- **Actions:** Loaded 34378 rows x 11 cols; verified all 11 required variables present; subset to required columns
- **Outputs:** participation_raw DataFrame
- **Warnings/Errors:** Duplicate rows: 8602; Extra columns dropped: []

### Step 1.2 — Problem Definition
- **Status:** SUCCESS
- **Actions:** Defined binary classification task (under-engagement identification); documented target and feature variables; created variable dictionary table
- **Outputs:** Markdown documentation in notebook
- **Warnings/Errors:** None

### Step 2 — EDA
- **Status:** SUCCESS
- **Actions:** Removed 40 rows with CARTS_NET in [-3, 3]; created binary target; generated 4 visualisations
- **Outputs:** participation_eda DataFrame; target_distribution.png, feature_distributions.png, feature_target_relationships.png, correlation_heatmap.png
- **Warnings/Errors:** None

### Step 3 — Missingness Handling
- **Status:** SUCCESS
- **Actions:** Applied tiered missingness handling; Tier 1 (drop): ['AGEBAND', 'SEX', 'QWORK', 'FINHARD', 'CHILDHH']; Tier 2 (recode): ['EDUCAT3', 'CINTOFT', 'COHAB']
- **Outputs:** participation_clean (31351 rows, 91.3% retention); missingness_handling_summary.csv
- **Warnings/Errors:** None

### Step 4.1 — Prepare Modelling Data
- **Status:** SUCCESS
- **Actions:** Defined X (10 features) and y; created LR and XGBoost preprocessors; stratified 70/15/15 split
- **Outputs:** Train=21945, Val=4703, Test=4703; Encoded dims: LR=49, XGB=59
- **Warnings/Errors:** None

### Step 4.3 — Baseline Logistic Regression
- **Status:** SUCCESS
- **Actions:** Trained baseline LR (C=1.0, L2, balanced); evaluated on validation set
- **Outputs:** Recall=0.6704, Balanced_Acc=0.6926, ROC_AUC=0.7418; baseline_lr_validation_metrics.csv
- **Warnings/Errors:** None

### Step 5.1 — Tune Logistic Regression
- **Status:** SUCCESS
- **Actions:** Grid searched 24 configs; best: C=0.1, l2, wt=balanced; threshold=0.51
- **Outputs:** Recall=0.6620, Balanced_Acc=0.6942; lr_tuning_results.csv
- **Warnings/Errors:** None

### Step 5.2 — Tune XGBoost
- **Status:** SUCCESS
- **Actions:** Grid searched 54 configs; best: depth=3, est=100, lr=0.1, spw=12.16; threshold=0.50
- **Outputs:** Recall=0.6704, Balanced_Acc=0.6926; xgb_tuning_results.csv
- **Warnings/Errors:** None

### Step 5.3 — Model Comparison on Test Set
- **Status:** SUCCESS
- **Actions:** Evaluated baseline LR, tuned LR, and tuned XGBoost on held-out test set
- **Outputs:** test_model_comparison.csv
- **Warnings/Errors:** None

### Step 5.4 — Final Model Decision
- **Status:** SUCCESS
- **Actions:** Applied weighted scoring framework; selected Test — Baseline LR
- **Outputs:** model_selection_framework.csv; tuning summaries printed
- **Warnings/Errors:** None

### Step 7 — Documentation
- **Status:** SUCCESS
- **Actions:** Generated non-technical policy report from actual experiment results
- **Outputs:** Report_claude_code.md
- **Warnings/Errors:** None

### EXPERIMENT COMPLETE
- **Status:** SUCCESS
- **Actions:** All steps (0-7) completed successfully
- **Outputs:** experiment_claude_code.ipynb, run_log_claude_code.md, Report_claude_code.md, requirements.txt, README.md, evidence_claude_code/
- **Warnings/Errors:** None

