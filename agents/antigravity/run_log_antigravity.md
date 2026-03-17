# Run Log: antigravity

## Step 0: Setup
- **Completion status**: Completed
- **Key actions**:
  - Created directory structure (`./agents/antigravity`, `./agents/antigravity/evidence_antigravity`, `./agents/antigravity/data`).
  - Copied `participation_2024-25_data_dictionary_cleaned.txt` and `participation_2024-25_experiment.tab` to `data` dir.
  - Initialized Jupyter Notebook `experiment_antigravity.ipynb` with title and global random seed (42).
- **Key outputs**:
  - `experiment_antigravity.ipynb`
  - `evidence_antigravity/` folder
  - `data/` folder containing data files
- **Important warnings or errors**: None

## Step 1.1: Dataset Ingestion & Schema Checks
- **Completion status**: Completed
- **Key actions**:
  - Loaded `participation_2024-25_experiment.tab` into `participation_raw`.
  - Added code to notebook checking dataset dimensions (34,378 rows, 11 columns).
  - Added schema checks verifying expected variables mapping to data dictionary type.
- **Key outputs**:
  - Added markdown and code cells to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 1.2: Problem Definition
- **Completion status**: Completed
- **Key actions**:
  - Added markdown cell with problem definition, variable listing, and data dictionary table.
- **Key outputs**:
  - Added markdown cell to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 2: Exploratory Data Analysis
- **Completion status**: Completed
- **Key actions**:
  - Filtered `CARTS_NET` missing values (-3, 3) to create `participation_eda`.
  - Added code to generate EDA charts (Target distribution, Age, Financial Hardship, Internet usage).
  - Saved EDA charts as `.png` under `EDA_antigravity_Pics`.
- **Key outputs**:
  - Added EDA cells to `experiment_antigravity.ipynb`.
  - `EDA_antigravity_Pics/` folder inside `evidence_antigravity`.
- **Important warnings or errors**: None

## Step 3: Missingness Handling
- **Completion status**: Completed
- **Key actions**:
  - Defined non-informative values as `< 0` and `>= 997`.
  - Applied boolean mask across all feature variables to drop rows with these values, resulting in `participation_clean`.
- **Key outputs**:
  - Added missingness handling cells to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 4.1: Prepare Modeling Data
- **Completion status**: Completed
- **Key actions**:
  - Defined `X` and `y`.
  - Recoded `y` (`CARTS_NET`): 1 -> 1, 2 -> 0.
  - Created Logistic Regression pipeline (`OneHotEncoder`) and XGBoost pipeline (`passthrough`).
  - Split data into 70% train, 15% validation, 15% test, stratified by target.
- **Key outputs**:
  - Added modeling data setup cells to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 4.2: Create Evaluation Harness
- **Completion status**: Completed
- **Key actions**:
  - Created markdown cell explaining the rationale for chosen metrics (ROC-AUC, Precision/Recall/F1, Confusion Matrix).
  - Defined `evaluate_model` function to output these metrics consistently.
- **Key outputs**:
  - Added evaluation harness cells to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 4.3: Baseline Model LR
- **Completion status**: Completed
- **Key actions**:
  - Trained baseline Logistic Regression mapping standard parameters.
  - Evaluated on validation set using the evaluation harness.
- **Key outputs**:
  - Added Model evaluation to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 5.1: Tune Logistic Regression
- **Completion status**: Completed
- **Key actions**:
  - Trained LR models over a grid of `C` values.
  - Identified best hyperparameters by checking ROC-AUC solely on the validation set.
  - Evaluated the best LR model on the validation set.
- **Key outputs**:
  - Added LR tuning code to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 5.2: Tune XGBoost
- **Completion status**: Completed
- **Key actions**:
  - Trained XGBoost models over a grid of `max_depth` and `learning_rate`.
  - Identified best hyperparameters by checking ROC-AUC solely on the validation set.
  - Evaluated the best XGBoost model on the validation set.
- **Key outputs**:
  - Added XGBoost tuning code to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 5.3: Model Comparison
- **Completion status**: Completed
- **Key actions**:
  - Evaluated all three models (Baseline LR, Tuned LR, Tuned XGBoost) on the test set.
- **Key outputs**:
  - Added final comparison code block to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 5.4: Final Model Decision
- **Completion status**: Completed
- **Key actions**:
  - Provided a model selection framework based on ROC-AUC and Class 0 Recall/F1.
  - Outputted the tuning summary for both LR and XGBoost.
  - Made the final model decision favoring Tuned XGBoost.
- **Key outputs**:
  - Added final model decision cells to `experiment_antigravity.ipynb`.
- **Important warnings or errors**: None

## Step 6: Producing Reproducible Packaging
- **Completion status**: Completed
- **Key actions**:
  - Created `requirements.txt` with required python dependencies.
  - Created `README.md` defining how to load inputs, run the experiment, output location, and reproducibility measures.
- **Key outputs**:
  - `requirements.txt`
  - `README.md`
- **Important warnings or errors**: None

## Step 7: Writing Documentation
- **Completion status**: Completed
- **Key actions**:
  - Extracted output metrics from notebook execution.
  - Drafted a non-technical (~400 words) policy report summarizing purpose, data approach, model choice, main findings, and limitations.
  - Saved the report as `Report_antigravity.md` in the working directory.
- **Key outputs**:
  - `Report_antigravity.md`
- **Important warnings or errors**: None
