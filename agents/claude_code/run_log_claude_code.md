# Run Log: claude_code

If any failure prevents reliable continuation of the pipeline, it will be marked as `EARLY FAILURE` with the step that failed, the reason for failure, and which later steps were affected.

---

## Step 0: Setup

- **Status**: COMPLETE
- **Key actions**:
  - Created Jupyter notebook `experiment_claude_code.ipynb`
  - Read variable dictionary to understand variable meanings and coded values
  - Set global random seed `RANDOM_STATE = 42`
  - Created evidence folder `evidence_claude_code/` and EDA subfolder
- **Key outputs**:
  - `experiment_claude_code.ipynb`
  - `evidence_claude_code/`
  - `evidence_claude_code/EDA_claude_code_Pics/`
- **Warnings/errors**: None

## Step 1: Dataset Ingestion, Schema Checks, and Problem Definition

- **Status**: COMPLETE
- **Key actions**:
  - Loaded `participation_2024-25_experiment.tab` into `participation_raw` (34,378 rows x 11 columns)
  - Verified all 11 required variables present
  - Checked data types (all int64), value ranges, and duplicates (8,602 duplicate rows found)
  - Defined binary classification task: predict under-engagement in physical arts participation
  - Documented target variable (CARTS_NET) and 10 feature variables
  - Stated that CARTS_NET values -3 and 3 will be dropped as missing
  - Included variable dictionary table
- **Key outputs**:
  - `participation_raw` DataFrame (34,378 x 11)
- **Warnings/errors**: 8,602 duplicate rows detected but retained (common in survey data with categorical features)

## Step 2: Exploratory Data Analysis

- **Status**: COMPLETE
- **Key actions**:
  - Removed 40 rows with CARTS_NET = -3 or 3 (34,338 rows remaining)
  - Created binary target: 1 = under-engaged (CARTS_NET=2), 0 = engaged (CARTS_NET=1)
  - Target distribution: 91.1% engaged, 8.9% under-engaged (severe class imbalance)
  - Generated 4 EDA visualisations saved as PNG
  - Identified high non-response rates in COHAB (71%) and EDUCAT3 (24%)
- **Key outputs**:
  - `participation_eda` DataFrame (34,338 x 11)
  - `evidence_claude_code/EDA_claude_code_Pics/target_distribution.png`
  - `evidence_claude_code/EDA_claude_code_Pics/feature_distributions.png`
  - `evidence_claude_code/EDA_claude_code_Pics/feature_target_relationships.png`
  - `evidence_claude_code/EDA_claude_code_Pics/correlation_heatmap.png`
- **Warnings/errors**: Severe class imbalance noted; high missingness in COHAB and EDUCAT3 flagged for tiered handling

## Step 3: Missingness Handling

- **Status**: COMPLETE
- **Key actions**:
  - Reviewed non-informative codes per variable from data dictionary
  - Applied tiered handling strategy:
    - Tier 1 (rate < 5%): Dropped rows for AGEBAND, SEX, QWORK, FINHARD, CHILDHH
    - Tier 2 (rate >= 5%): Recoded non-informative codes to 0 (Unknown) for EDUCAT3, CINTOFT, COHAB
  - COHAB (71.2% non-informative) and EDUCAT3 (24.0%) recoded rather than dropped to preserve data
  - CINTOFT (5.9%) also recoded due to threshold
  - Final dataset: 29,848 rows (86.9% data retention rate)
- **Key outputs**:
  - `participation_clean` DataFrame (29,848 x 11)
  - `evidence_claude_code/missingness_handling_summary.csv`
- **Warnings/errors**: None

## Step 4: Modelling Setup and Baseline

- **Status**: COMPLETE
- **Key actions**:
  - Defined X (10 features) and y (binary target)
  - Created OneHotEncoder preprocessing pipelines for both LR and XGBoost
  - Performed stratified 70/15/15 train/val/test split preserving class proportions
  - Defined evaluation harness with metrics: Accuracy, Precision, Recall, F1, Balanced Accuracy, ROC-AUC, PR-AUC, Specificity, Confusion Matrix
  - Trained baseline Logistic Regression (C=1.0, L2, class_weight='balanced')
  - Baseline LR validation: Recall=0.69, Balanced Accuracy=0.70, ROC-AUC=0.76
- **Key outputs**:
  - Train/Val/Test splits (stratified)
  - `evidence_claude_code/baseline_lr_validation_metrics.csv`
- **Warnings/errors**: None

## Step 5: Improving Performance

- **Status**: COMPLETE
- **Key actions**:
  - Tuned Logistic Regression via grid search (24 configurations): C, penalty (l1/l2), class_weight
  - Best LR: C=1.0, L2, balanced; validation balanced accuracy=0.7027
  - Tuned XGBoost via grid search (54 configurations): max_depth, n_estimators, learning_rate, scale_pos_weight
  - Best XGBoost: max_depth=5, n_estimators=300, learning_rate=0.01, scale_pos_weight=13.32; validation balanced accuracy=0.7009
  - Test set comparison (used only once):
    - Baseline LR: Recall=0.6709, ROC-AUC=0.7509, Balanced Acc=0.6912
    - Tuned LR: Recall=0.6709, ROC-AUC=0.7508, Balanced Acc=0.6908
    - Tuned XGBoost: Recall=0.6550, ROC-AUC=0.7464, Balanced Acc=0.6820
  - Applied multi-dimensional model selection framework (weighted scoring)
  - **Final model selected: Baseline Logistic Regression** (weighted score: 0.6030)
- **Key outputs**:
  - `evidence_claude_code/lr_tuning_results.csv`
  - `evidence_claude_code/xgb_tuning_results.csv`
  - `evidence_claude_code/test_model_comparison.csv`
  - `evidence_claude_code/model_selection_framework.csv`
- **Warnings/errors**: None

## Step 6: Reproducible Packaging

- **Status**: COMPLETE
- **Key actions**:
  - Created `requirements.txt` with pinned package versions
  - Created `README.md` with input files, run instructions, outputs, and reproducibility measures
- **Key outputs**:
  - `requirements.txt`
  - `README.md`
- **Warnings/errors**: None

## Step 7: Writing Documentation

- **Status**: COMPLETE
- **Key actions**:
  - Wrote non-technical report for a government arts department (~400 words)
  - Used actual results from notebook
  - Covered: purpose, data/approach, findings, model choice, practical implications, limitations
- **Key outputs**:
  - `Report_claude_code.md`
- **Warnings/errors**: None
