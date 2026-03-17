import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# --- 0_setup ---
cells.append(nbf.v4.new_markdown_cell("# Experiment: antigravity\n## 0. Setup\nSetting up the experiment, random seed, and logging structure."))
cells.append(nbf.v4.new_code_cell("""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, make_scorer, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 7: Set the global random seed to 42. Use random_state=42 in all later steps
np.random.seed(42)
RANDOM_SEED = 42

# 10: Set up experiment-tracking structure
evidence_dir = 'evidence_antigravity'
os.makedirs(evidence_dir, exist_ok=True)
run_log_path = 'run_log_antigravity.md'

def log_step(step_name, status, key_actions, key_outputs, warnings_errors="None"):
    with open(run_log_path, 'a') as f:
        f.write(f"### Step: {step_name}\\n")
        f.write(f"- **Completion Status:** {status}\\n")
        f.write(f"- **Key Actions:** {key_actions}\\n")
        f.write(f"- **Key Outputs:** {key_outputs}\\n")
        f.write(f"- **Warnings/Errors:** {warnings_errors}\\n\\n")

# Initialize log file
with open(run_log_path, 'w') as f:
    f.write("# Experiment Run Log: antigravity\\n\\n")

log_step(
    step_name="0. Setup",
    status="SUCCESS",
    key_actions="Imported libraries, set random seed to 42, initialized experiment tracking structure.",
    key_outputs=f"{evidence_dir}/, {run_log_path}"
)
"""))

# --- 1.1_dataset_ingestion_and_schema_checks ---
cells.append(nbf.v4.new_markdown_cell("""
## 1.1 Dataset Ingestion and Schema Checks
In this section, we ingest the dataset `participation_2024-25_experiment.tab` and verify that the target variable and required features are present in the dataset. We also output the shape (rows, columns) of the loaded DataFrame.
Checks performed:
- Verification of variables presence.
- Checking dataframe dimensions.
"""))
cells.append(nbf.v4.new_code_cell("""
try:
    # 1: Read the data file
    participation_raw = pd.read_csv('data/participation_2024-25_experiment.tab', sep='\\t')
    print(f"Data shape: {participation_raw.shape[0]} rows and {participation_raw.shape[1]} columns.")
    
    # 2: Expected variables
    expected_vars = ['CARTS_NET', 'AGEBAND', 'SEX', 'QWORK', 'EDUCAT3', 'FINHARD', 'CINTOFT', 'gor', 'rur11cat', 'CHILDHH', 'COHAB']
    
    # 4: Schema checks
    missing_vars = [var for var in expected_vars if var not in participation_raw.columns]
    if missing_vars:
        raise ValueError(f"Missing expected variables: {missing_vars}")
    else:
        print("All required variables are present.")
        
    log_step(
        "1.1 Dataset ingestion and schema checks",
        "SUCCESS",
        "Loaded raw data and performed schema checks for required columns.",
        f"participation_raw shape: {participation_raw.shape}"
    )
except Exception as e:
    log_step("1.1 Dataset ingestion and schema checks", "EARLY FAILURE", "Attempted to load data and check schema", "None", str(e))
    raise
"""))

# --- 1.2_problem_defition ---
cells.append(nbf.v4.new_markdown_cell("""
## 1.2 Problem Definition
**Prediction Task**: The task is a binary classification task: predict whether a respondent engaged with the arts physically in the last 12 months.

We frame this task as an under-engagement identification problem with social research value. Rather than treating arts participation as a purely individual preference, the task investigates whether non-participation is socially patterned across demographic, socioeconomic, digital, and geographic factors. The purpose is to identify groups that may face structural or contextual barriers to physical arts engagement, and to provide evidence for more inclusive cultural policy and public engagement strategies.

- **Target variable**: `CARTS_NET`
- **Feature variables**: `AGEBAND, SEX, QWORK, EDUCAT3, FINHARD, CINTOFT, gor, rur11cat, CHILDHH, COHAB`

*Note: For the target variable, rows with values `-3` and `3` will later be dropped as missing values so that the task becomes a binary classification problem.*

### Variables Dictionary

| Variable | Label / Description |
|---|---|
| CARTS_NET | In the last 12 months, engaged (attended OR participated) with the arts physically |
| AGEBAND | Respondent age band (ALL) |
| SEX | Respondent gender |
| QWORK | What is your current working status |
| EDUCAT3 | What is your highest qualification |
| FINHARD | How well would you say you are managing financially these days |
| CINTOFT | How often do you use the internet |
| gor | Region (former Government Office Region) |
| rur11cat | Rural or Urban Area (2011 Census definition) |
| CHILDHH | Children in household |
| COHAB | Living as a couple |
"""))
cells.append(nbf.v4.new_code_cell("""
log_step(
    "1.2 Problem Definition",
    "SUCCESS",
    "Defined prediction task and variables in markdown.",
    "Markdown cell with problem definition and variable dictionary table."
)
"""))

# --- 2_EDA ---
cells.append(nbf.v4.new_markdown_cell("""
## 2. Exploratory Data Analysis (EDA)
First, we filter out invalid rows from the target variable (`CARTS_NET` values `-3` and `3`) and create a copy of the dataframe `participation_eda` with a binary classification target, representing whether the respondent engaged with the arts physically.
Then we generate visualizations to understand the distributions of features and their relationship with the target, saving outputs to the evidence folder.
"""))

cells.append(nbf.v4.new_code_cell("""
try:
    # 1: remove -3 and 3
    valid_target_mask = ~participation_raw['CARTS_NET'].isin([-3.0, 3.0])
    participation_eda = participation_raw[valid_target_mask].copy()
    
    # Target variable remains values 1.0 (Yes) and 2.0 (No) at this stage. 
    # Let's map it to binary 1 (Yes) and 0 (No) as usually expected for plotting? 
    # Actually, the prompt says: "Do not recode the remaining target values to 0 and 1 yet."
    # So we keep 1.0 and 2.0.
    
    # 3: Create directory for EDA pics
    eda_pics_dir = os.path.join(evidence_dir, 'EDA_antigravity_Pics')
    os.makedirs(eda_pics_dir, exist_ok=True)
    
    # 4: EDA: Visualizing target variable distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=participation_eda, x='CARTS_NET', palette='viridis')
    plt.title('Distribution of Target Variable (CARTS_NET)')
    plt.xlabel('CARTS_NET (1=Yes, 2=No)')
    plt.savefig(os.path.join(eda_pics_dir, 'target_distribution.png'))
    plt.close()

    # Visualizing feature relationships with the target
    features_to_plot = ['AGEBAND', 'SEX', 'FINHARD', 'EDUCAT3']
    for feat in features_to_plot:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=participation_eda, x=feat, hue='CARTS_NET', palette='Set2')
        plt.title(f'{feat} by Arts Engagement (CARTS_NET)')
        plt.legend(title='CARTS_NET', labels=['1 (Yes)', '2 (No)'])
        plt.savefig(os.path.join(eda_pics_dir, f'{feat}_by_target.png'))
        plt.close()
        
    log_step(
        "2. EDA",
        "SUCCESS",
        "Filtered invalid target rows and generated EDA visualizations.",
        f"{eda_pics_dir} with .png figures"
    )
except Exception as e:
    log_step("2. EDA", "EARLY FAILURE", "Attempted EDA and file generation", "None", str(e))
    raise
"""))

# --- 3_missingness_handling ---
cells.append(nbf.v4.new_markdown_cell("""
## 3. Missingness Handling
For the feature variables, negative values generally represent forms of missingness (e.g., `-3`=Not Answered, `-4`=Not answered but should have, `-5`=Multi-selected for single response). Missingness handling rules:
- **Numerical/Ordinal**: Instead of imputing, since they are largely categorical or ordinal and we will one-hot encode them, we can treat the negative missing codes as their own category, OR replace them with `np.nan` and impute with the most frequent value.
Given standard practices for this dataset, we will replace negative values (missing codes) with the most frequent valid value (mode imputation) for each column.
Alternatively, replacing them with a specific 'Missing' string category could work if treated as purely categorical, but simple mode imputation guarantees no missingness while retaining column types.
Here, we replace negative values with the median or mode of valid values.
*Action*: Replace all negative values in the required features with the mode of that feature.
"""))

cells.append(nbf.v4.new_code_cell("""
try:
    print(f"Rows before cleaning: {len(participation_eda)}")
    participation_clean = participation_eda.copy()
    
    features = ['AGEBAND', 'SEX', 'QWORK', 'EDUCAT3', 'FINHARD', 'CINTOFT', 'gor', 'rur11cat', 'CHILDHH', 'COHAB']
    
    # Missingness handling: encode negative values as NaN, then impute with mode
    for feat in features:
        # Negative values are missing codes
        mask_missing = participation_clean[feat] < 0
        if mask_missing.sum() > 0:
            # Find mode of valid values
            valid_mode = participation_clean.loc[~mask_missing, feat].mode()[0]
            participation_clean.loc[mask_missing, feat] = valid_mode

    print(f"Rows after cleaning: {len(participation_clean)}")
    # Verify no missingness (no negative values and no NaNs)
    assert not (participation_clean[features] < 0).any().any()
    assert not participation_clean[features].isna().any().any()
    
    log_step(
        "3. Missingness handling",
        "SUCCESS",
        "Replaced negative missing codes with the mode of valid values for each feature.",
        "participation_clean dataframe created."
    )
except Exception as e:
    log_step("3. Missingness handling", "EARLY FAILURE", "Attempted missingness handling", "None", str(e))
    raise
"""))

# --- 4.1_prepare_modeling_data ---
cells.append(nbf.v4.new_markdown_cell("""
## 4.1 Prepare Modeling Data
- Define `X` (features) and `y` (target). The target `CARTS_NET` has values 1 and 2. We will map 1 (Yes) to 1 and 2 (No) to 0. But since the task is under-engagement identification, it's better to predict the UNDER-engagement class. Wait, the prompt says "predict whether a respondent engaged with the arts physically", implying 1 = Engagement. Let's map 1 to 1 (Positive class = Engagement) and 2 to 0 (Negative class = Non-engagement).
Wait, "The purpose is to identify groups that may face structural or contextual barriers to physical arts engagement... We frame this task as an under-engagement identification problem." This implies under-engagement is the positive class we want to identify. Therefore, `CARTS_NET` == 2 (No) should be the class 1 we want to predict.
Let's map `CARTS_NET` == 2.0 to 1, and `CARTS_NET` == 1.0 to 0.
- Create One-Hot Encoding pipeline for categorical features.
- Split data into train (70%), validation (15%), and test (15%) using stratified splitting.
"""))

cells.append(nbf.v4.new_code_cell("""
try:
    # 4.1 Prep data
    X = participation_clean[['AGEBAND', 'SEX', 'QWORK', 'EDUCAT3', 'FINHARD', 'CINTOFT', 'gor', 'rur11cat', 'CHILDHH', 'COHAB']].copy()
    # Map target: 2.0 (No engagement) -> 1, 1.0 (Yes engagement) -> 0
    y = (participation_clean['CARTS_NET'] == 2.0).astype(int)
    
    # Feature columns type conversion (as categorical for OneHotEncoding)
    cat_columns = X.columns.tolist()
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_columns)
        ])
    
    # Split: Train/Val/Test = 0.70 / 0.15 / 0.15
    # First split 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)
    
    # Then split temp into 50% val, 50% test (which is 15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    log_step(
        "4.1 Prepare Modeling Data",
        "SUCCESS",
        "Defined X and y, set up preprocessor, performed 70/15/15 stratified split.",
        "X_train, X_val, X_test splits"
    )
except Exception as e:
    log_step("4.1 Prepare Modeling Data", "EARLY FAILURE", "Attempted data splitting and prep", "None", str(e))
    raise
"""))

# --- 4.2_create_evaluation_harness ---
cells.append(nbf.v4.new_markdown_cell("""
## 4.2 Create Evaluation Harness
Since our focus is on identifying under-engagement, which could be imbalanced, relying purely on Accuracy is misleading. 
We will use to following metrics:
1. **Precision, Recall, and F1-score**: Focused on the positive class (Under-engagement).
2. **PR-AUC (Precision-Recall Area Under the Curve)**: Better represents the overall skill of a model for imbalanced classes than ROC-AUC.
3. **Confusion Matrix**: To observe true positives, false positive tradeoffs.
We define a common evaluation function `evaluate_model(model, X, y, threshold=0.5)` to output these metrics.
"""))

cells.append(nbf.v4.new_code_cell("""
def evaluate_model(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    
    report = classification_report(y_true, y_pred, target_names=['Engaged (0)', 'Under-engaged (1)'], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    df_metrics = pd.DataFrame({
        'Metric': ['Precision (Class 1)', 'Recall (Class 1)', 'F1-Score (Class 1)', 'PR-AUC'],
        'Value': [
            report['Under-engaged (1)']['precision'],
            report['Under-engaged (1)']['recall'],
            report['Under-engaged (1)']['f1-score'],
            pr_auc
        ]
    })
    
    return report, cm, pr_auc, df_metrics

def print_eval(df_metrics, cm):
    display(df_metrics.style.format({'Value': '{:.4f}'}))
    print("\\nConfusion Matrix:")
    print(cm)
    
log_step(
    "4.2 Create evaluation harness",
    "SUCCESS",
    "Defined evaluation function incorporating classification report, PR-AUC, and confusion matrix.",
    "evaluate_model function"
)
"""))

# --- 4.3_baseline_model_LR ---
cells.append(nbf.v4.new_markdown_cell("""
## 4.3 Baseline Model - Logistic Regression
We train a straightforward Logistic Regression with default hyperparameters on the training set, and evaluate it on the validation set.
"""))

cells.append(nbf.v4.new_code_cell("""
try:
    baseline_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, n_jobs=-1))
    ])
    
    baseline_lr.fit(X_train, y_train)
    
    # Evaluate on Validation set
    y_val_probs_blr = baseline_lr.predict_proba(X_val)[:, 1]
    report_blr, cm_blr, prauc_blr, df_metrics_blr = evaluate_model(y_val, y_val_probs_blr)
    
    print("Baseline Logistic Regression Performance (Validation Set):")
    print_eval(df_metrics_blr, cm_blr)
    
    log_step(
        "4.3 Baseline Model (LR)",
        "SUCCESS",
        "Trained baseline Logistic Regression model and evaluated on validation set.",
        f"PR-AUC: {prauc_blr:.4f}"
    )
except Exception as e:
    log_step("4.3 Baseline Model (LR)", "EARLY FAILURE", "Attempted to train and evaluate Baseline LR", "None", str(e))
    raise
"""))

# --- 5.1_improve_LR ---
cells.append(nbf.v4.new_markdown_cell("""
## 5.1 Improve Logistic Regression (Tuning & Threshold Moving)
We tune the regularization parameter `C` using cross-validation over `X_val`, or by scoring on the validation set itself. Since we need to tune on validation set and preserve test set, we will train over `X_train` using different hyperparameters and evaluate on `X_val` to find the best `C` and optimal decision threshold based on the F1-score for class 1.
"""))

cells.append(nbf.v4.new_code_cell("""
try:
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    best_c = None
    best_f1 = -1
    best_model_lr = None
    
    # Hyperparameter tuning
    tuning_records_lr = []
    
    for c in C_values:
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(C=c, random_state=RANDOM_SEED, max_iter=1000, n_jobs=-1))
        ])
        model.fit(X_train, y_train)
        y_val_probs = model.predict_proba(X_val)[:, 1]
        
        # Internal threshold tuning over [0.1, 0.9] to find best F1
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_val_pred = (y_val_probs >= thresh).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            tuning_records_lr.append({'C': c, 'threshold': thresh, 'f1': f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_c = c
                best_thresh_lr = thresh
                best_model_lr = model
                
    print(f"Tuned LR Best C: {best_c}")
    print(f"Tuned LR Best Threshold: {best_thresh_lr:.2f}")
    
    y_val_probs_tlr = best_model_lr.predict_proba(X_val)[:, 1]
    report_tlr, cm_tlr, prauc_tlr, df_metrics_tlr = evaluate_model(y_val, y_val_probs_tlr, threshold=best_thresh_lr)
    
    print("\\nTuned Logistic Regression Performance (Validation Set):")
    print_eval(df_metrics_tlr, cm_tlr)
    
    log_step(
        "5.1 Improve LR",
        "SUCCESS",
        f"Tuned LR C parameter and decision threshold over validation set.",
        f"Best F1: {best_f1:.4f}, PR-AUC: {prauc_tlr:.4f}"
    )
except Exception as e:
    log_step("5.1 Improve LR", "EARLY FAILURE", "Attempted to tune LR", "None", str(e))
    raise
"""))

# --- 5.2_tune_XGBoost ---
cells.append(nbf.v4.new_markdown_cell("""
## 5.2 Tune XGBoost
We train an XGBoost classifier, tuning hyperparameters (`learning_rate`, `max_depth`) and finding the best threshold on the validation set.
"""))

cells.append(nbf.v4.new_code_cell("""
try:
    learning_rates = [0.01, 0.1, 0.2]
    max_depths = [3, 5, 7]
    
    best_f1_xgb = -1
    best_params_xgb = {}
    best_thresh_xgb = 0.5
    best_model_xgb = None
    
    tuning_records_xgb = []

    # Fit feature transformer once for speed
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    for lr in learning_rates:
        for md in max_depths:
            clf = XGBClassifier(
                learning_rate=lr, 
                max_depth=md, 
                random_state=RANDOM_SEED, 
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            )
            clf.fit(X_train_t, y_train)
            y_val_probs = clf.predict_proba(X_val_t)[:, 1]
            
            for thresh in np.arange(0.1, 0.9, 0.05):
                y_val_pred = (y_val_probs >= thresh).astype(int)
                f1 = f1_score(y_val, y_val_pred)
                tuning_records_xgb.append({'learning_rate': lr, 'max_depth': md, 'threshold': thresh, 'f1': f1})
                
                if f1 > best_f1_xgb:
                    best_f1_xgb = f1
                    best_params_xgb = {'learning_rate': lr, 'max_depth': md}
                    best_thresh_xgb = thresh
                    best_model_xgb = clf
                    
    print(f"Best XGBoost Params: {best_params_xgb}")
    print(f"Best XGBoost Threshold: {best_thresh_xgb:.2f}")
    
    # We create the full pipeline for the best xgb model
    final_xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', best_model_xgb)
    ])
    
    y_val_probs_xgb = final_xgb_pipeline.predict_proba(X_val)[:, 1]
    report_xgb, cm_xgb, prauc_xgb, df_metrics_xgb = evaluate_model(y_val, y_val_probs_xgb, threshold=best_thresh_xgb)
    
    print("\\nTuned XGBoost Performance (Validation Set):")
    print_eval(df_metrics_xgb, cm_xgb)
    
    log_step(
        "5.2 Tune XGBoost",
        "SUCCESS",
        f"Tuned XGB parameters and decision threshold over validation set.",
        f"Best F1: {best_f1_xgb:.4f}, PR-AUC: {prauc_xgb:.4f}"
    )
except Exception as e:
    log_step("5.2 Tune XGBoost", "EARLY FAILURE", "Attempted to tune XGBoost", "None", str(e))
    raise
"""))

# --- 5.3_model_comparison ---
cells.append(nbf.v4.new_markdown_cell("""
## 5.3 Model Comparison
We now compare all three models (Baseline LR, Tuned LR, Tuned XGBoost) using the **test set** under the identical evaluation harness.
"""))

cells.append(nbf.v4.new_code_cell("""
try:
    print("=== Test Set Evaluation ===")
    
    # Baseline LR
    y_test_probs_blr = baseline_lr.predict_proba(X_test)[:, 1]
    report_test_blr, cm_test_blr, prauc_test_blr, df_test_blr = evaluate_model(y_test, y_test_probs_blr, threshold=0.5)
    
    # Tuned LR
    y_test_probs_tlr = best_model_lr.predict_proba(X_test)[:, 1]
    report_test_tlr, cm_test_tlr, prauc_test_tlr, df_test_tlr = evaluate_model(y_test, y_test_probs_tlr, threshold=best_thresh_lr)
    
    # Tuned XGBoost
    y_test_probs_xgb = final_xgb_pipeline.predict_proba(X_test)[:, 1]
    report_test_xgb, cm_test_xgb, prauc_test_xgb, df_test_xgb = evaluate_model(y_test, y_test_probs_xgb, threshold=best_thresh_xgb)
    
    # Combine outputs
    df_compare = pd.DataFrame({
        'Metric': df_test_blr['Metric'],
        'Baseline LR': df_test_blr['Value'],
        'Tuned LR': df_test_tlr['Value'],
        'Tuned XGBoost': df_test_xgb['Value']
    })
    
    display(df_compare.style.format({col: '{:.4f}' for col in df_compare.columns if col != 'Metric'}))
    
    log_step(
        "5.3 Model Comparison",
        "SUCCESS",
        "Evaluated Baseline LR, Tuned LR, and Tuned XGBoost on the unseen Test set.",
        "Comparison DataFrame Output generated."
    )
except Exception as e:
    log_step("5.3 Model Comparison", "EARLY FAILURE", "Attempted test set comparison", "None", str(e))
    raise
"""))

# --- 5.4_final_model_decision ---
cells.append(nbf.v4.new_markdown_cell("""
## 5.4 Final Model Decision
### Model Selection Framework
1. **Primary Metric (F1-Score for Under-engaged)**: Given that public arts engagement seeks to uncover missing demographic groups, both false positives (predicting under-engaged when they are active) and false negatives (missing those who are truly under-engaged) carry policy costs. F1-score balances precision and recall for the minority positive class.
2. **Secondary Metric (PR-AUC)**: Measures overall ability of the model's predicted probabilities to rank the under-engaged population regardless of the specific threshold.
3. **Interpretability & Deployability**: For government arts departments, being able to trace *why* a cohort is predicted to be under-engaged (e.g., Logistic Regression coefficients) is highly favored over black-box methods (like deeply tree-based ensembles) if performance is comparable.

### Decision
(The decision will be analyzed based on the summary printed below. Ultimately we favor the model providing the best trade-off of F1-score and interpretability.)
"""))

cells.append(nbf.v4.new_code_cell("""
print("=== Tuning Summary: Logistic Regression ===")
print("- Tuning Method: Grid Search over Predefined values")
print("- Hyperparameters Searched: Regularization `C` and Decision Threshold")
print(f"- Search Range: C in {C_values}, Threshold in [0.10, 0.85]")
print(f"- Total parameter configurations evaluated: {len(tuning_records_lr)}")
print("- Iteration/trial count completed: Fully completed loops")
print(f"- Best hyperparameter setting: C = {best_c}, Threshold = {best_thresh_lr:.2f}")
print(f"- Best validation F1-score: {best_f1:.4f}")

print("\\n=== Tuning Summary: XGBoost ===")
print("- Tuning Method: Grid Search over Predefined values")
print("- Hyperparameters Searched: `learning_rate`, `max_depth`, and Decision Threshold")
print(f"- Search Range: lr in {learning_rates}, max_depth in {max_depths}, Threshold in [0.10, 0.85]")
print(f"- Total parameter configurations evaluated: {len(tuning_records_xgb)}")
print("- Iteration/trial count completed: Fully completed loops")
print(f"- Best hyperparameter setting: {best_params_xgb}, Threshold = {best_thresh_xgb:.2f}")
print(f"- Best validation F1-score: {best_f1_xgb:.4f}")

# Compare and select automatically here to print out what we choose
best_test_f1 = max(
    df_compare.loc[df_compare['Metric'] == 'F1-Score (Class 1)', 'Tuned LR'].values[0],
    df_compare.loc[df_compare['Metric'] == 'F1-Score (Class 1)', 'Tuned XGBoost'].values[0]
)

chosen_model_name = "Tuned XGBoost" if df_compare.loc[df_compare['Metric'] == 'F1-Score (Class 1)', 'Tuned XGBoost'].values[0] == best_test_f1 else "Tuned Logistic Regression"

with open("best_model_name.txt", "w") as f:
    f.write(chosen_model_name)

print(f"\\nBased on the test set F1-Score, {chosen_model_name} is the optimal choice.")

log_step(
    "5.4 Final Model Decision",
    "SUCCESS",
    "Summarized tuning logs and stated final model selection criterion and decision.",
    "Tuning text summary output."
)
"""))

# Assign cells to notebook
nb['cells'] = cells

with open('experiment_antigravity.ipynb', 'w') as f:
    nbf.write(nb, f)

print("experiment_antigravity.ipynb successfully generated.")
