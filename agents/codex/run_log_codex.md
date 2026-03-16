# Run Log: codex

## EARLY FAILURE Protocol

If any failure prevents reliable continuation of the pipeline, record it here with:
- which step failed
- the reason for failure
- which later steps are affected

---

## Step 0: Setup

- **Status**: Complete
- **Key actions**:
  - Created notebook `experiment_codex.ipynb`
  - Read the variable dictionary (`../../data/participation_2024-25_data_dictionary_cleaned.txt`)
  - Set global random seed to 42
  - Confirmed required input files are present via relative paths
  - Created evidence folder `evidence_codex/`
- **Key outputs**:
  - `experiment_codex.ipynb`
  - `run_log_codex.md`
  - `evidence_codex/`
- **Warnings/errors**: None

## Step 1: Dataset Ingestion + Schema Checks + Problem Definition

- **Status**: Complete
- **Key actions**:
  - Loaded dataset into `participation_raw`
  - Reported row/column shape
  - Checked required variable presence and duplicate columns
  - Checked observed coded values against dictionary-based expected ranges
  - Added task definition markdown and full variable summary table
- **Key outputs**:
  - Step 1 notebook cells (schema summary table + problem-definition markdown)
- **Warnings/errors**: None

## Step 2: EDA and Insight Generation

- **Status**: Complete
- **Key actions**:
  - Removed rows where `CARTS_NET` in `{-3, 3}`
  - Created `participation_eda` by dropping original `CARTS_NET` and adding `target_binary`
  - Created figure folder `evidence_codex/EDA_codex_Pics/`
  - Generated and saved EDA visualisations and insights
- **Key outputs**:
  - `evidence_codex/EDA_codex_Pics/01_target_distribution.png`
  - `evidence_codex/EDA_codex_Pics/02_feature_distributions.png`
  - `evidence_codex/EDA_codex_Pics/03_non_engagement_rates.png`
- **Warnings/errors**: None

## Step 3: Missingness Handling

- **Status**: Complete
- **Key actions**:
  - Defined variable-specific non-informative code rules from dictionary meanings
  - Applied handling on `participation_eda` features and created `participation_clean`
  - Preserved rows and encoded non-informative responses as `Unknown`
  - Verified no missing values remain in feature variables
- **Key outputs**:
  - `participation_clean` (in notebook memory)
  - `evidence_codex/missingness_handling_summary.csv`
- **Warnings/errors**: None

## Step 4: Baseline Model Training + Evaluation Harness

- **Status**: Complete
- **Key actions**:
  - Defined `X` and `y` from `participation_clean` (`y=1` means under-engagement)
  - Created preprocessing pipelines for Logistic Regression and XGBoost
  - Performed stratified 0.7/0.15/0.15 split
  - Built common evaluation harness for imbalanced classification
  - Trained baseline Logistic Regression and evaluated on validation set only
- **Key outputs**:
  - `evidence_codex/baseline_lr_validation_metrics.csv`
- **Warnings/errors**: None

## Step 5: Improving Performance

- **Status**: Complete
- **Key actions**:
  - Tuned Logistic Regression on validation set only
  - Trained and tuned XGBoost on validation set only
  - Printed structured tuning summaries for both models
  - Used test set only after tuning to compare baseline LR, tuned LR, and tuned XGBoost
  - Applied quantitative multi-dimensional model-selection framework
  - Selected final model: **Baseline Logistic Regression**
- **Key outputs**:
  - `evidence_codex/lr_tuning_results.csv`
  - `evidence_codex/xgb_tuning_results.csv`
  - `evidence_codex/test_model_comparison.csv`
  - `evidence_codex/model_selection_framework.csv`
- **Warnings/errors**: None

## Step 6: Producing Reproducible Packaging

- **Status**: Complete
- **Key actions**:
  - Created agent-level dependency file
  - Created concise run/readme documentation aligned with notebook outputs
- **Key outputs**:
  - `requirements.txt`
  - `README.md`
- **Warnings/errors**: None

## Step 7: Writing Documentation

- **Status**: Complete
- **Key actions**:
  - Wrote non-technical policy-facing report based on actual notebook results
  - Included purpose, data/method overview, key findings, final model choice, implications, and limitations
- **Key outputs**:
  - `Report_codex.md`
- **Warnings/errors**: None
