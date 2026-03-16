# Run Log: claude_code

## EARLY FAILURE Protocol

If any failure prevents reliable continuation of the pipeline, it will be recorded here with:
- Which step failed
- The reason for failure
- Which later steps were affected

No early failures occurred during this experiment.

---

## Step 0: Setup

- **Status**: Complete
- **Key actions**:
  - Created Jupyter Notebook `experiment_claude_code.ipynb`
  - Read and reviewed the variable dictionary
  - Set global random seed to 42
  - Confirmed both required data files are present
  - Created run log and evidence folder
- **Key outputs**: `experiment_claude_code.ipynb`, `run_log_claude_code.md`, `evidence_claude_code/`
- **Warnings/errors**: None

## Step 1: Dataset Ingestion + Schema Checks + Problem Definition

- **Status**: Complete
- **Key actions**:
  - Loaded `participation_2024-25_experiment.tab` into `participation_raw`
  - Verified dataset shape: 34,378 rows x 11 columns
  - Schema checks: column presence, data types, value ranges, NaN counts
  - Defined binary classification task and variable summary table
- **Key outputs**: `participation_raw` DataFrame; markdown documentation of schema checks and problem definition
- **Warnings/errors**: None

## Step 2: EDA and Insight Generation

- **Status**: Complete
- **Key actions**:
  - Removed 40 rows with CARTS_NET values -3 and 3
  - Created `participation_eda` with binary target column (34,338 rows)
  - Performed EDA: target distribution, feature distributions, feature-target relationships, Spearman correlation analysis, missing value overview
  - Saved all figures as .png
- **Key outputs**: `participation_eda` DataFrame; 5 EDA figures in `evidence_claude_code/EDA_claude_code_Pics/`
- **Warnings/errors**: Severe class imbalance noted (91.1% engaged vs 8.9% not engaged)

## Step 3: Missingness Handling

- **Status**: Complete
- **Key actions**:
  - Classified features by missing rate into tiers
  - Tier 1 (low missing: AGEBAND, SEX, QWORK, FINHARD, CINTOFT, CHILDHH): recoded to NaN, dropped rows
  - Tier 2/3 (high missing: EDUCAT3 24%, COHAB 71%): recoded to "Unknown" category (code 0)
  - Dropped 4,490 rows (13.1%) from Tier 1 NaN
- **Key outputs**: `participation_clean` (29,848 rows, 0 missing values)
- **Warnings/errors**: None

## Step 4: Baseline Model Training + Evaluation Harness

- **Status**: Complete
- **Key actions**:
  - Defined X (10 features) and y (binary target, 1=engaged, 0=not engaged)
  - Created preprocessing pipelines (OneHotEncoder for all features)
  - Split data: 20,893 train / 4,477 val / 4,478 test (stratified)
  - Defined evaluation harness: Accuracy, Precision/Recall/F1 (macro), ROC-AUC, PR-AUC, Confusion Matrix
  - Trained baseline Logistic Regression (C=1.0, l2, balanced class weights)
  - Evaluated on validation set only
- **Key outputs**: `lr_baseline` model; validation metrics
- **Warnings/errors**: None

## Step 5: Improving Performance

- **Status**: Complete
- **Key actions**:
  - Tuned LR via GridSearchCV (24 configurations): best C, penalty, class_weight found
  - Tuned XGBoost via GridSearchCV (108 configurations): best hyperparameters found
  - Evaluated all 3 models on test set (first and only test set use)
  - Applied multi-dimensional weighted scoring framework for model selection
- **Key outputs**:
  - Test set results: Baseline LR (F1=0.5355, AUC=0.7617), Tuned LR (F1=0.5161, AUC=0.7331), Tuned XGBoost (F1=0.5505, AUC=0.7342)
  - Selected model: Baseline LR (weighted score 0.7495)
  - `evidence_claude_code/test_set_comparison.csv`, `evidence_claude_code/model_selection_scores.csv`
- **Warnings/errors**: Some convergence warnings during LR tuning with saga solver (non-critical)

## Step 6: Reproducible Packaging

- **Status**: Complete
- **Key actions**:
  - Created `requirements.txt` with all Python packages
  - Created `README.md` with input files, execution instructions, outputs, and reproducibility measures
- **Key outputs**: `requirements.txt`, `README.md`
- **Warnings/errors**: None

## Step 7: Writing Documentation

- **Status**: Complete
- **Key actions**:
  - Wrote non-technical report (~400 words) for government arts department
  - Grounded in actual notebook results
- **Key outputs**: `Report_claude_code.md`
- **Warnings/errors**: None
