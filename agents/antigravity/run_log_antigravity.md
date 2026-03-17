# Experiment Run Log: antigravity

### Step: 0. Setup
- **Completion Status:** SUCCESS
- **Key Actions:** Imported libraries, set random seed to 42, initialized experiment tracking structure.
- **Key Outputs:** evidence_antigravity/, run_log_antigravity.md
- **Warnings/Errors:** None

### Step: 1.1 Dataset ingestion and schema checks
- **Completion Status:** SUCCESS
- **Key Actions:** Loaded raw data and performed schema checks for required columns.
- **Key Outputs:** participation_raw shape: (34378, 11)
- **Warnings/Errors:** None

### Step: 1.2 Problem Definition
- **Completion Status:** SUCCESS
- **Key Actions:** Defined prediction task and variables in markdown.
- **Key Outputs:** Markdown cell with problem definition and variable dictionary table.
- **Warnings/Errors:** None

### Step: 2. EDA
- **Completion Status:** SUCCESS
- **Key Actions:** Filtered invalid target rows and generated EDA visualizations.
- **Key Outputs:** evidence_antigravity/EDA_antigravity_Pics with .png figures
- **Warnings/Errors:** None

### Step: 3. Missingness handling
- **Completion Status:** SUCCESS
- **Key Actions:** Replaced negative missing codes with the mode of valid values for each feature.
- **Key Outputs:** participation_clean dataframe created.
- **Warnings/Errors:** None

### Step: 4.1 Prepare Modeling Data
- **Completion Status:** SUCCESS
- **Key Actions:** Defined X and y, set up preprocessor, performed 70/15/15 stratified split.
- **Key Outputs:** X_train, X_val, X_test splits
- **Warnings/Errors:** None

### Step: 4.2 Create evaluation harness
- **Completion Status:** SUCCESS
- **Key Actions:** Defined evaluation function incorporating classification report, PR-AUC, and confusion matrix.
- **Key Outputs:** evaluate_model function
- **Warnings/Errors:** None

### Step: 4.3 Baseline Model (LR)
- **Completion Status:** SUCCESS
- **Key Actions:** Trained baseline Logistic Regression model and evaluated on validation set.
- **Key Outputs:** PR-AUC: 0.2093
- **Warnings/Errors:** None

### Step: 5.1 Improve LR
- **Completion Status:** SUCCESS
- **Key Actions:** Tuned LR C parameter and decision threshold over validation set.
- **Key Outputs:** Best F1: 0.2766, PR-AUC: 0.2092
- **Warnings/Errors:** None

### Step: 5.2 Tune XGBoost
- **Completion Status:** SUCCESS
- **Key Actions:** Tuned XGB parameters and decision threshold over validation set.
- **Key Outputs:** Best F1: 0.2780, PR-AUC: 0.2131
- **Warnings/Errors:** None

### Step: 5.3 Model Comparison
- **Completion Status:** SUCCESS
- **Key Actions:** Evaluated Baseline LR, Tuned LR, and Tuned XGBoost on the unseen Test set.
- **Key Outputs:** Comparison DataFrame Output generated.
- **Warnings/Errors:** None

### Step: 5.4 Final Model Decision
- **Completion Status:** SUCCESS
- **Key Actions:** Summarized tuning logs and stated final model selection criterion and decision.
- **Key Outputs:** Tuning text summary output.
- **Warnings/Errors:** None

