# Run Log: Antigravity

This run log records progress step-by-step throughout the experiment. 
If any failure prevents reliable continuation of the pipeline, it will be marked clearly as `EARLY FAILURE` and state:
- which step failed
- the reason for failure
- which later steps were affected

## 0 Setup
- **step name**: 0 Setup (section 0/step 0)
- **completion status**: COMPLETE
- **key actions**:
  - Acknowledged data and dictionary files via symlinks to the core `data/` directory.
  - Created `experiment_antigravity.ipynb` with initial markdown and code cells for seed initialization.
  - Created `evidence_antigravity` folder for outputs.
  - Set global random seed (`random_state=42`) for reproducibility.
  - Ensured only relative paths will be used in the notebook.
  - Verified variables based on data dictionary mapping without loading data.
- **key outputs**:
  - `experiment_antigravity.ipynb`
  - `run_log_antigravity.md`
  - `evidence_antigravity/` (directory)
- **important warnings or errors**: None

## 1 Dataset Ingestion + Schema Checks + Problem Definition
- **step name**: 1 Dataset Ingestion + Schema Checks + Problem Definition (section 1/step 1)
- **completion status**: COMPLETE
- **key actions**:
  - Ingested participation_2024-25_experiment.tab into pandas dataframe `participation_raw`.
  - Verified dataset shape (rows & columns).
  - Verified that all required 11 variables are present in the dataframe.
  - Checked for duplicate columns and non-numeric datatypes. Expected all to be numeric based on categorical encoding.
  - Defined the problem as binary classification targeting `CARTS_NET` and listing 10 features.
  - Explicity stated in notebook that target rows with `-3` and `3` will be dropped downstream.
  - Produced summary table of variables referencing dictionary labels.
- **key outputs**:
  - Notebook appended with Section 1 markdown and code cells.
- **important warnings or errors**: None

## 2 EDA and Insight Generation
- **step name**: 2 EDA and Insight Generation (section 2/step 2)
- **completion status**: COMPLETE
- **key actions**:
  - Filtered `participation_raw` to remove target values `-3` and `3`. Copied to `participation_eda`.
  - Created a new binary target variable `arts_engaged` strictly copying from `CARTS_NET` without recoding to 0/1 yet.
  - Dropped the original `CARTS_NET` column in `participation_eda`.
  - Created an `EDA_antigravity_Pics` output folder.
  - Generated bar plots using matplotlib/seaborn to explore target imbalance and relationship between engagement and socio-demographic features (AGEBAND, FINHARD, rur11cat). Added insights markdown drawing modeling implications.
- **key outputs**:
  - Notebook appended with Section 2 markdown and code cells.
  - Visualisations saved as PNG files into `EDA_antigravity_Pics/`.
- **important warnings or errors**: None

## 3 Missingness Handling
- **step name**: 3 Missingness Handling (section 3/step 3)
- **completion status**: COMPLETE
- **key actions**:
  - Detailed variable-specific handling rules based on documentation.
  - Designed missingness logic: dropped rows with missing ordinal/geographic anchor variables (`AGEBAND`, `CHILDHH`, `gor`, `rur11cat`). Recoded nominals (`SEX`, `QWORK`, etc.) to a new `999.0` ('Unknown' or 'Prefer not to say' category).
  - Showed rows before and after filtering logic (from `participation_eda`).
  - Created finalized frame `participation_clean` containing no missing data, ready for model training.
- **key outputs**:
  - Notebook appended with Section 3 Markdown defining rules and python cell implementing it.
- **important warnings or errors**: None

## 4 Baseline Model Training + Evaluation Harness
- **step name**: 4 Baseline Model Training + Evaluation Harness (section 4/step 4)
- **completion status**: COMPLETE
- **key actions**:
  - Defined target `y` as under-engagement (Class 1) and `X` as the 10 features.
  - Created Sklearn `ColumnTransformer` pipelines with scaling and OHE for LR, and just OHE for XGBoost.
  - Implemented a 0.7 / 0.15 / 0.15 train/val/test split, stratified by the target.
  - Established Evaluation Harness focusing on F1, Precision, Recall, PR-AUC, ROC-AUC, and Confusion Matrix to suit the imbalanced target.
  - Trained baseline LogisticRegression model and evaluated strictly on the Validation set.
- **key outputs**:
  - Notebook appended with Section 4 Markdown establishing the harness and Python code implementing the baseline pipeline.
- **important warnings or errors**: None

## 5 Improving Performance
- **step name**: 5 Improving Performance (section 5/step 5)
- **completion status**: COMPLETE
- **key actions**:
  - Tuned Logistic Regression and XGBoost strictly on the Validation set using manual grid search.
  - Printed structured tuning summaries for both models per protocol.
  - Re-fit the optimal hyperparameters to train data and evaluated the three models (Baseline LR, Tuned LR, Tuned XGB) strictly on the Test Set.
  - Specified a multi-dimensional framework evaluating Precision/Recall, Interpretability, and FNR.
  - Recommended Tuned XGBoost as the final decision due to superior non-linear capabilities and strong recall.
- **key outputs**:
  - Notebook appended with Section 5 tuning scripts, summaries, test set evaluation loop, and final selection framework.
- **important warnings or errors**: None

## 6 Producing Reproducible Packaging
- **step name**: 6 Producing Reproducible Packaging
- **completion status**: COMPLETE
- **key actions**:
  - Validated there was no new notebook content added in this step.
  - Authored a `requirements.txt` listing `pandas`, `xgboost`, `scikit-learn`, `nbformat` and others.
  - Drafted a `README.md` containing all strict criteria: input files needed, instructions on execution, outputs generated/folder locations, and explicit references to the `random_state=42` control ensuring reproducibility.
- **key outputs**:
  - `requirements.txt`
  - `README.md`
- **important warnings or errors**: None

## 7 Writing documentation
- **step name**: 7 Writing documentation
- **completion status**: COMPLETE
- **key actions**:
  - Validated there was no new notebook content added.
  - Extracted results from the executed notebook (including Baseline LR, Tuned XGBoost recall and precision metrics).
  - Authored a ~400 word non-technical policy report targeting government arts departments.
  - Structured the report precisely to cover: purpose, data approach, main findings, final model choice (XGBoost), practical implications, and causal/predictive limitations.
- **key outputs**:
  - `Report_antigravity.md`
- **important warnings or errors**: None
