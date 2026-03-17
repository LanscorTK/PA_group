import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# Experiment: codex\n"
        "## 0. Setup\n"
        "This section initializes reproducibility controls, reviews the variable dictionary (without loading the data file), and creates the experiment tracking structure."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
DICT_PATH = DATA_DIR / "participation_2024-25_data_dictionary_cleaned.txt"
DATA_PATH = DATA_DIR / "participation_2024-25_experiment.tab"

EVIDENCE_DIR = BASE_DIR / "evidence_codex"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH = BASE_DIR / "run_log_codex.md"


def init_run_log() -> None:
    RUN_LOG_PATH.write_text("# Experiment Run Log: codex\\n\\n", encoding="utf-8")


def log_step(step_name: str, status: str, key_actions: str, key_outputs: str, warnings_or_errors: str = "None") -> None:
    with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"## Step: {step_name}\\n")
        f.write(f"- completion status: {status}\\n")
        f.write(f"- key actions: {key_actions}\\n")
        f.write(f"- key outputs: {key_outputs}\\n")
        f.write(f"- important warnings or errors: {warnings_or_errors}\\n\\n")


init_run_log()

# Read the variable dictionary in setup; do not load the main data file yet.
dictionary_text = DICT_PATH.read_text(encoding="utf-8", errors="ignore")
print(f"Dictionary loaded from {DICT_PATH} ({len(dictionary_text.splitlines())} lines).")
print(f"Evidence directory ready at: {EVIDENCE_DIR}")
print(f"Run log initialized at: {RUN_LOG_PATH}")

log_step(
    step_name="0. Setup",
    status="Completed",
    key_actions="Set global random seed to 42, initialized run log and evidence folder, read variable dictionary file only.",
    key_outputs=f"{RUN_LOG_PATH.name}, {EVIDENCE_DIR.name}/",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "# 1. Problem Framing and Data Foundations\n"
        "## 1.1 Dataset Ingestion and Schema Checks\n"
        "This subsection loads the tab-delimited dataset into `participation_raw`, reports dimensions, and checks that all required variables are present with sensible schema properties for this stage."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
required_variables = [
    "CARTS_NET",
    "AGEBAND",
    "SEX",
    "QWORK",
    "EDUCAT3",
    "FINHARD",
    "CINTOFT",
    "gor",
    "rur11cat",
    "CHILDHH",
    "COHAB",
]

participation_raw = pd.read_csv(DATA_PATH, sep="\\t")
print(f"Rows: {participation_raw.shape[0]}, Columns: {participation_raw.shape[1]}")

missing_required = [c for c in required_variables if c not in participation_raw.columns]
extra_columns = [c for c in participation_raw.columns if c not in required_variables]
duplicate_columns = participation_raw.columns[participation_raw.columns.duplicated()].tolist()

dtype_check = participation_raw[required_variables].dtypes.to_dict()
null_counts = participation_raw[required_variables].isna().sum().to_dict()

if missing_required:
    raise ValueError(f"Missing required variables: {missing_required}")

print("All required variables are present.")
print(f"Duplicate column names: {duplicate_columns}")
print(f"Extra columns beyond required list: {extra_columns}")
print("Null counts in required variables:")
print(pd.Series(null_counts))

schema_report = {
    "rows": int(participation_raw.shape[0]),
    "columns": int(participation_raw.shape[1]),
    "missing_required": missing_required,
    "duplicate_columns": duplicate_columns,
    "null_counts": null_counts,
    "dtypes": {k: str(v) for k, v in dtype_check.items()},
}

(pd.Series(schema_report["dtypes"]).rename("dtype").to_frame()).head()

log_step(
    step_name="1.1 Dataset ingestion and schema checks",
    status="Completed",
    key_actions="Loaded dataset into participation_raw, validated required variables, checked duplicates, dtypes, and null counts.",
    key_outputs=f"participation_raw shape={participation_raw.shape}; schema checks passed.",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 1.2 Prediction Task Definition\n"
        "The prediction task is a **binary classification** task: predict whether a respondent engaged with the arts physically in the last 12 months.\n\n"
        "We frame this task as an under-engagement identification problem with social research value. Rather than treating arts participation as a purely individual preference, the task investigates whether non-participation is socially patterned across demographic, socioeconomic, digital, and geographic factors. The purpose is to identify groups that may face structural or contextual barriers to physical arts engagement, and to provide evidence for more inclusive cultural policy and public engagement strategies.\n\n"
        "- target variable: `CARTS_NET`\n"
        "- feature variables: `AGEBAND, SEX, QWORK, EDUCAT3, FINHARD, CINTOFT, gor, rur11cat, CHILDHH, COHAB`\n\n"
        "For the target variable, rows with values `-3` and `3` will later be dropped as missing values so the task becomes binary. They are **not** dropped in this subsection.\n\n"
        "| Variable | Description (from dictionary) | Key coded values summary |\n"
        "|---|---|---|\n"
        "| `CARTS_NET` | Engaged (attended OR participated) with the arts physically in last 12 months | `1=Yes`, `2=No`, `-3/3` treated as missing in target handling |\n"
        "| `AGEBAND` | Respondent age band | `1=16-19` ... `15=85+`, `-3/997` non-informative |\n"
        "| `SEX` | Respondent gender | `1=Female`, `2=Male`, negative codes / `997` non-informative |\n"
        "| `QWORK` | Current working status | `1..10` substantive categories, negative codes / `997/999` non-informative |\n"
        "| `EDUCAT3` | Highest qualification | `1=Degree+`, `2=Other`, negative codes / `997/999` non-informative |\n"
        "| `FINHARD` | Financial management status | `1..5` substantive categories, negative codes / `997` non-informative |\n"
        "| `CINTOFT` | Internet use frequency | `1..5` substantive categories, negative codes non-informative |\n"
        "| `gor` | Region | `1..9` regions |\n"
        "| `rur11cat` | Rural/Urban category | `1=Rural`, `2=Urban` |\n"
        "| `CHILDHH` | Children in household | `0..4` substantive values, negative codes / `997` non-informative |\n"
        "| `COHAB` | Living as a couple | `1=Yes`, `2=No`, negative codes / `997` non-informative |"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
log_step(
    step_name="1.2 Problem definition",
    status="Completed",
    key_actions="Defined prediction objective, declared target/features, documented target missing-value rule and variable table.",
    key_outputs="Markdown definition completed in notebook.",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 2. Exploratory Data Analysis (EDA)\n"
        "This section prepares `participation_eda` for exploration by removing target values `-3` and `3`, creating a copied dataframe with a new target column, and producing EDA plots saved to the evidence folder."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
EDA_PICS_DIR = EVIDENCE_DIR / "EDA_codex_Pics"
EDA_PICS_DIR.mkdir(parents=True, exist_ok=True)

rows_before_target_filter = len(participation_raw)
participation_eda = participation_raw.loc[~participation_raw["CARTS_NET"].isin([-3, 3])].copy()

# Keep binary target as original 1/2 coding for now, per prompt.
participation_eda["target_binary"] = participation_eda["CARTS_NET"]
participation_eda = participation_eda.drop(columns=["CARTS_NET"])

rows_after_target_filter = len(participation_eda)
print(f"Rows before target cleaning: {rows_before_target_filter}")
print(f"Rows after removing CARTS_NET in {{-3, 3}}: {rows_after_target_filter}")
print("Target distribution (1=Yes engaged, 2=No engaged):")
print(participation_eda["target_binary"].value_counts(normalize=True).sort_index())

# Plot 1: target distribution
plt.figure(figsize=(6, 4))
(
    participation_eda["target_binary"]
    .map({1: "Engaged (1)", 2: "Not engaged (2)"})
    .value_counts()
    .reindex(["Engaged (1)", "Not engaged (2)"])
    .plot(kind="bar", color=["#4c78a8", "#f58518"])
)
plt.title("Target Distribution After Removing Missing Target Codes")
plt.ylabel("Count")
plt.xlabel("Target")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(EDA_PICS_DIR / "target_distribution.png", dpi=150)
plt.close()

# Plot 2: non-informative code rates by feature before cleaning
feature_cols = ["AGEBAND", "SEX", "QWORK", "EDUCAT3", "FINHARD", "CINTOFT", "gor", "rur11cat", "CHILDHH", "COHAB"]
non_informative_map = {
    "AGEBAND": {-3, 997},
    "SEX": {-5, -4, -3, 997},
    "QWORK": {-5, -4, -3, 997, 999},
    "EDUCAT3": {-5, -4, -3, 997, 999},
    "FINHARD": {-5, -4, -3, 997},
    "CINTOFT": {-5, -4, -3},
    "gor": set(),
    "rur11cat": set(),
    "CHILDHH": {-6, -5, -4, -3, 997},
    "COHAB": {-5, -4, -3, 997},
}

non_info_rates = {}
for col in feature_cols:
    codes = non_informative_map[col]
    if not codes:
        non_info_rates[col] = 0.0
    else:
        non_info_rates[col] = float(participation_eda[col].isin(codes).mean())

rate_df = pd.Series(non_info_rates).sort_values(ascending=False)
plt.figure(figsize=(9, 4.5))
rate_df.plot(kind="bar", color="#72b7b2")
plt.title("Rate of Non-informative Codes by Feature (Pre-cleaning)")
plt.ylabel("Proportion")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(EDA_PICS_DIR / "non_informative_rate_by_feature.png", dpi=150)
plt.close()

# Plot 3+: under-engagement rate by feature category
for col in feature_cols:
    grouped = (
        participation_eda.groupby(col)["target_binary"]
        .apply(lambda s: (s == 2).mean())
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(9, 4.5))
    grouped.plot(kind="bar", color="#e45756")
    plt.title(f"Under-engagement Rate by {col}")
    plt.ylabel("P(target=2 | category)")
    plt.xlabel(col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(EDA_PICS_DIR / f"{col}_under_engagement_rate.png", dpi=150)
    plt.close()

log_step(
    step_name="2. EDA",
    status="Completed",
    key_actions="Dropped target rows coded -3/3, created participation_eda with new target column, generated and saved EDA figures.",
    key_outputs=f"participation_eda shape={participation_eda.shape}; EDA figures saved to {EDA_PICS_DIR}",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 3. Missingness Handling\n"
        "Missingness is encoded as special numeric codes rather than `NaN`. Based on the variable dictionary, each feature uses variable-specific non-informative codes (for example, `-3` not answered, `997` prefer not to say, `999` don’t know).\n\n"
        "Handling rules used:\n"
        "- Replace variable-specific non-informative codes with `NaN`.\n"
        "- For ordinal-like features (`AGEBAND`, `CHILDHH`), impute with the median valid code.\n"
        "- For nominal features (`SEX`, `QWORK`, `EDUCAT3`, `FINHARD`, `CINTOFT`, `gor`, `rur11cat`, `COHAB`), impute with the most frequent valid category.\n"
        "- Keep all rows and ensure `participation_clean` has no missing values for feature columns."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
rows_before_cleaning = len(participation_eda)
participation_clean = participation_eda.copy()

missing_code_map = {
    "AGEBAND": {-3, 997},
    "SEX": {-5, -4, -3, 997},
    "QWORK": {-5, -4, -3, 997, 999},
    "EDUCAT3": {-5, -4, -3, 997, 999},
    "FINHARD": {-5, -4, -3, 997},
    "CINTOFT": {-5, -4, -3},
    "gor": set(),
    "rur11cat": set(),
    "CHILDHH": {-6, -5, -4, -3, 997},
    "COHAB": {-5, -4, -3, 997},
}

ordinal_features = ["AGEBAND", "CHILDHH"]
nominal_features = ["SEX", "QWORK", "EDUCAT3", "FINHARD", "CINTOFT", "gor", "rur11cat", "COHAB"]

summary_rows = []

for col, bad_codes in missing_code_map.items():
    n_bad = int(participation_clean[col].isin(bad_codes).sum()) if bad_codes else 0
    participation_clean[col] = participation_clean[col].replace(list(bad_codes), np.nan)

    if col in ordinal_features:
        fill_value = int(round(participation_clean[col].dropna().median()))
    else:
        fill_value = int(participation_clean[col].dropna().mode().iloc[0])

    participation_clean[col] = participation_clean[col].fillna(fill_value).astype(int)

    summary_rows.append(
        {
            "variable": col,
            "non_informative_replaced": n_bad,
            "imputation_strategy": "median" if col in ordinal_features else "mode",
            "imputation_value": fill_value,
        }
    )

rows_after_cleaning = len(participation_clean)
remaining_missing = int(participation_clean[[*missing_code_map.keys()]].isna().sum().sum())

print(f"Rows before cleaning: {rows_before_cleaning}")
print(f"Rows after cleaning: {rows_after_cleaning}")
print(f"Remaining feature missing values: {remaining_missing}")

missingness_summary = pd.DataFrame(summary_rows)
display(missingness_summary)

assert remaining_missing == 0, "participation_clean still contains missing values."

missingness_summary.to_csv(EVIDENCE_DIR / "missingness_summary_codex.csv", index=False)

log_step(
    step_name="3. Missingness handling",
    status="Completed",
    key_actions="Applied variable-specific non-informative code handling and imputation to feature variables.",
    key_outputs=f"participation_clean shape={participation_clean.shape}; no feature missing values; summary saved to {EVIDENCE_DIR / 'missingness_summary_codex.csv'}",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "# 4. Modeling Data and Evaluation Design\n"
        "## 4.1 Prepare Modeling Data\n"
        "Using `participation_clean`, this subsection defines `X` and `y`, sets model-specific preprocessing pipelines for Logistic Regression and XGBoost, and creates stratified train/validation/test splits in a 0.7/0.15/0.15 ratio."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
feature_variables = ["AGEBAND", "SEX", "QWORK", "EDUCAT3", "FINHARD", "CINTOFT", "gor", "rur11cat", "CHILDHH", "COHAB"]

X = participation_clean[feature_variables].copy()
# Positive class is under-engagement: CARTS_NET=2 (not engaged)
y = (participation_clean["target_binary"] == 2).astype(int)

categorical_features = feature_variables.copy()

lr_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

xgb_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_SEED,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_SEED,
)

print("Class prevalence (under-engaged=1):")
print(f"Full data: {y.mean():.4f}")
print(f"Train: {y_train.mean():.4f}, Validation: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")
print(f"Shapes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

log_step(
    step_name="4.1 Prepare modeling data",
    status="Completed",
    key_actions="Defined X/y, created separate preprocessing pipelines for LR and XGBoost, and produced stratified 70/15/15 splits.",
    key_outputs=f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 4.2 Create Evaluation Harness\n"
        "The task prioritizes identification of under-engagement (minority class), so evaluation must go beyond accuracy.\n\n"
        "Evaluation rules:\n"
        "- Use predicted probabilities and an explicit classification threshold.\n"
        "- Use the same metrics for Logistic Regression and XGBoost: `roc_auc`, `pr_auc`, `precision_under`, `recall_under`, `f1_under`, `f2_under`, `balanced_accuracy`.\n"
        "- Tune thresholds on the **validation set only** by maximizing `f2_under` (recall-emphasizing), reflecting policy preference to miss fewer under-engaged respondents.\n"
        "- Keep test set untouched until model comparison stage."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision_under": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_under": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_under": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_under": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics, cm


def find_best_threshold(y_true, y_prob, threshold_grid=None, objective="f2_under"):
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.10, 0.91, 0.05), 2)

    rows = []
    for t in threshold_grid:
        m, _ = evaluate_predictions(y_true, y_prob, threshold=t)
        rows.append(m)

    threshold_df = pd.DataFrame(rows).sort_values([objective, "pr_auc", "recall_under"], ascending=False)
    best_row = threshold_df.iloc[0]
    return float(best_row["threshold"]), threshold_df


def evaluate_model(model_name, y_true, y_prob, threshold):
    m, cm = evaluate_predictions(y_true, y_prob, threshold=threshold)
    out = {"model": model_name, **m}
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Engaged(0)", "Actual_Under(1)"],
        columns=["Pred_Engaged(0)", "Pred_Under(1)"],
    )
    return out, cm_df


log_step(
    step_name="4.2 Evaluation harness",
    status="Completed",
    key_actions="Implemented shared metric functions and validation-threshold tuning rule for both model families.",
    key_outputs="evaluate_predictions, find_best_threshold, evaluate_model functions defined.",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 4.3 Baseline Logistic Regression\n"
        "A baseline Logistic Regression model is trained with standard settings and evaluated on the validation set only using the predefined harness."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
baseline_lr = Pipeline(
    steps=[
        ("preprocess", lr_preprocessor),
        (
            "model",
            LogisticRegression(
                random_state=RANDOM_SEED,
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
            ),
        ),
    ]
)

baseline_lr.fit(X_train, y_train)
val_prob_baseline_lr = baseline_lr.predict_proba(X_val)[:, 1]
baseline_lr_val_result, baseline_lr_val_cm = evaluate_model(
    "Baseline Logistic Regression", y_val, val_prob_baseline_lr, threshold=0.50
)

baseline_lr_val_df = pd.DataFrame([baseline_lr_val_result]).set_index("model")
display(baseline_lr_val_df.round(4))
print("Validation confusion matrix (baseline LR):")
display(baseline_lr_val_cm)

baseline_lr_val_df.to_csv(EVIDENCE_DIR / "baseline_lr_validation_metrics_codex.csv")

log_step(
    step_name="4.3 Baseline LR",
    status="Completed",
    key_actions="Trained baseline Logistic Regression and evaluated on validation set at threshold 0.50.",
    key_outputs=f"Validation F2={baseline_lr_val_result['f2_under']:.4f}, PR-AUC={baseline_lr_val_result['pr_auc']:.4f}",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "# 5. Model Improvement, Comparison, and Decision\n"
        "## 5.1 Improve Logistic Regression\n"
        "Logistic Regression is tuned on the validation set only (same split), including decision-threshold tuning."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
lr_param_grid = {
    "model__C": [0.01, 0.1, 1.0, 3.0, 10.0],
    "model__penalty": ["l1", "l2"],
    "model__class_weight": [None, "balanced"],
}

lr_tuning_rows = []
trial = 0

for params in ParameterGrid(lr_param_grid):
    trial += 1
    candidate = Pipeline(
        steps=[
            ("preprocess", lr_preprocessor),
            ("model", LogisticRegression(random_state=RANDOM_SEED, max_iter=1500, solver="liblinear")),
        ]
    )
    candidate.set_params(**params)
    candidate.fit(X_train, y_train)

    val_prob = candidate.predict_proba(X_val)[:, 1]
    best_threshold, threshold_table = find_best_threshold(y_val, val_prob, objective="f2_under")
    val_result, _ = evaluate_model("candidate", y_val, val_prob, threshold=best_threshold)

    lr_tuning_rows.append(
        {
            "trial": trial,
            "C": params["model__C"],
            "penalty": params["model__penalty"],
            "class_weight": params["model__class_weight"],
            **val_result,
        }
    )

lr_tuning_df = pd.DataFrame(lr_tuning_rows).sort_values(["f2_under", "pr_auc", "recall_under"], ascending=False).reset_index(drop=True)
best_lr_row = lr_tuning_df.iloc[0]

best_lr_params = {
    "model__C": best_lr_row["C"],
    "model__penalty": best_lr_row["penalty"],
    "model__class_weight": best_lr_row["class_weight"],
}
best_lr_threshold = float(best_lr_row["threshold"])

# Refit tuned LR with best hyperparameters on train split
best_lr_model = Pipeline(
    steps=[
        ("preprocess", lr_preprocessor),
        ("model", LogisticRegression(random_state=RANDOM_SEED, max_iter=1500, solver="liblinear")),
    ]
)
best_lr_model.set_params(**best_lr_params)
best_lr_model.fit(X_train, y_train)

val_prob_tuned_lr = best_lr_model.predict_proba(X_val)[:, 1]
tuned_lr_val_result, tuned_lr_val_cm = evaluate_model("Tuned Logistic Regression", y_val, val_prob_tuned_lr, best_lr_threshold)

display(lr_tuning_df.head(10).round(4))
print("Validation confusion matrix (tuned LR):")
display(tuned_lr_val_cm)

lr_tuning_df.to_csv(EVIDENCE_DIR / "lr_tuning_results_codex.csv", index=False)

log_step(
    step_name="5.1 Improve LR",
    status="Completed",
    key_actions="Completed grid search over LR hyperparameters and validation threshold tuning.",
    key_outputs=f"Best LR params={best_lr_params}, best threshold={best_lr_threshold:.2f}, best validation F2={tuned_lr_val_result['f2_under']:.4f}",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 5.2 Tune XGBoost\n"
        "XGBoost is tuned on the same train/validation split with threshold tuning on validation only."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
xgb_param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.10],
    "model__subsample": [0.8, 1.0],
}

xgb_tuning_rows = []
trial = 0

for params in ParameterGrid(xgb_param_grid):
    trial += 1
    candidate = Pipeline(
        steps=[
            ("preprocess", xgb_preprocessor),
            (
                "model",
                XGBClassifier(
                    random_state=RANDOM_SEED,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    candidate.set_params(**params)
    candidate.fit(X_train, y_train)

    val_prob = candidate.predict_proba(X_val)[:, 1]
    best_threshold, threshold_table = find_best_threshold(y_val, val_prob, objective="f2_under")
    val_result, _ = evaluate_model("candidate", y_val, val_prob, threshold=best_threshold)

    xgb_tuning_rows.append(
        {
            "trial": trial,
            "n_estimators": params["model__n_estimators"],
            "max_depth": params["model__max_depth"],
            "learning_rate": params["model__learning_rate"],
            "subsample": params["model__subsample"],
            **val_result,
        }
    )

xgb_tuning_df = pd.DataFrame(xgb_tuning_rows).sort_values(["f2_under", "pr_auc", "recall_under"], ascending=False).reset_index(drop=True)
best_xgb_row = xgb_tuning_df.iloc[0]

best_xgb_params = {
    "model__n_estimators": int(best_xgb_row["n_estimators"]),
    "model__max_depth": int(best_xgb_row["max_depth"]),
    "model__learning_rate": float(best_xgb_row["learning_rate"]),
    "model__subsample": float(best_xgb_row["subsample"]),
}
best_xgb_threshold = float(best_xgb_row["threshold"])

tuned_xgb_model = Pipeline(
    steps=[
        ("preprocess", xgb_preprocessor),
        (
            "model",
            XGBClassifier(
                random_state=RANDOM_SEED,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            ),
        ),
    ]
)
tuned_xgb_model.set_params(**best_xgb_params)
tuned_xgb_model.fit(X_train, y_train)

val_prob_tuned_xgb = tuned_xgb_model.predict_proba(X_val)[:, 1]
tuned_xgb_val_result, tuned_xgb_val_cm = evaluate_model("Tuned XGBoost", y_val, val_prob_tuned_xgb, best_xgb_threshold)

display(xgb_tuning_df.head(10).round(4))
print("Validation confusion matrix (tuned XGBoost):")
display(tuned_xgb_val_cm)

xgb_tuning_df.to_csv(EVIDENCE_DIR / "xgb_tuning_results_codex.csv", index=False)

log_step(
    step_name="5.2 Tune XGBoost",
    status="Completed",
    key_actions="Completed grid search over XGBoost hyperparameters and validation threshold tuning.",
    key_outputs=f"Best XGB params={best_xgb_params}, best threshold={best_xgb_threshold:.2f}, best validation F2={tuned_xgb_val_result['f2_under']:.4f}",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 5.3 Model Comparison on Test Set\n"
        "At this stage only, the test set is used for final comparison of baseline Logistic Regression, tuned Logistic Regression, and tuned XGBoost under the same evaluation harness."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
# Test predictions
prob_test_baseline_lr = baseline_lr.predict_proba(X_test)[:, 1]
prob_test_tuned_lr = best_lr_model.predict_proba(X_test)[:, 1]
prob_test_tuned_xgb = tuned_xgb_model.predict_proba(X_test)[:, 1]

baseline_test_result, baseline_test_cm = evaluate_model("Baseline Logistic Regression", y_test, prob_test_baseline_lr, threshold=0.50)
tuned_lr_test_result, tuned_lr_test_cm = evaluate_model("Tuned Logistic Regression", y_test, prob_test_tuned_lr, threshold=best_lr_threshold)
tuned_xgb_test_result, tuned_xgb_test_cm = evaluate_model("Tuned XGBoost", y_test, prob_test_tuned_xgb, threshold=best_xgb_threshold)

comparison_df = pd.DataFrame([
    baseline_test_result,
    tuned_lr_test_result,
    tuned_xgb_test_result,
]).set_index("model")

display(comparison_df.round(4))

print("Confusion matrix: Baseline LR")
display(baseline_test_cm)
print("Confusion matrix: Tuned LR")
display(tuned_lr_test_cm)
print("Confusion matrix: Tuned XGBoost")
display(tuned_xgb_test_cm)

comparison_df.to_csv(EVIDENCE_DIR / "model_comparison_test_codex.csv")

log_step(
    step_name="5.3 Model comparison",
    status="Completed",
    key_actions="Evaluated baseline LR, tuned LR, and tuned XGBoost on the test set only.",
    key_outputs=f"Comparison table saved to {EVIDENCE_DIR / 'model_comparison_test_codex.csv'}",
)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 5.4 Final Model Decision\n"
        "### Quantitative Selection Framework\n"
        "A multi-dimensional framework is used to pick the final model from tuned LR and tuned XGBoost:\n"
        "1. **Priority metric: `f2_under` (40%)** to emphasize recall for under-engagement detection.\n"
        "2. **Ranking quality: `pr_auc` (25%)** for minority-class probability ranking.\n"
        "3. **Class-sensitive balance: `balanced_accuracy` (20%)** to avoid one-class dominance.\n"
        "4. **Operational usefulness: `precision_under` (15%)** to limit false alarms.\n\n"
        "Selection score = `0.40*f2_under + 0.25*pr_auc + 0.20*balanced_accuracy + 0.15*precision_under` on test-set outcomes (after all tuning is complete)."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """
print("=== Tuning Summary: Logistic Regression ===")
print("- tuning method used: grid search over fixed hyperparameter candidates + validation threshold sweep")
print("- hyperparameters searched: C, penalty, class_weight, threshold")
print(f"- search range/candidates: C={lr_param_grid['model__C']}, penalty={lr_param_grid['model__penalty']}, class_weight={lr_param_grid['model__class_weight']}, threshold=0.10..0.90 step 0.05")
print(f"- total parameter configurations evaluated: {len(list(ParameterGrid(lr_param_grid)))}")
print(f"- iteration or trial count completed: {len(lr_tuning_df)}")
print(f"- best hyperparameter setting found: C={best_lr_params['model__C']}, penalty={best_lr_params['model__penalty']}, class_weight={best_lr_params['model__class_weight']}, threshold={best_lr_threshold:.2f}")
print(f"- best validation performance under evaluation harness: f2_under={best_lr_row['f2_under']:.4f}, recall_under={best_lr_row['recall_under']:.4f}, pr_auc={best_lr_row['pr_auc']:.4f}")

print("\\n=== Tuning Summary: XGBoost ===")
print("- tuning method used: grid search over fixed hyperparameter candidates + validation threshold sweep")
print("- hyperparameters searched: n_estimators, max_depth, learning_rate, subsample, threshold")
print(f"- search range/candidates: n_estimators={xgb_param_grid['model__n_estimators']}, max_depth={xgb_param_grid['model__max_depth']}, learning_rate={xgb_param_grid['model__learning_rate']}, subsample={xgb_param_grid['model__subsample']}, threshold=0.10..0.90 step 0.05")
print(f"- total parameter configurations evaluated: {len(list(ParameterGrid(xgb_param_grid)))}")
print(f"- iteration or trial count completed: {len(xgb_tuning_df)}")
print(f"- best hyperparameter setting found: n_estimators={best_xgb_params['model__n_estimators']}, max_depth={best_xgb_params['model__max_depth']}, learning_rate={best_xgb_params['model__learning_rate']}, subsample={best_xgb_params['model__subsample']}, threshold={best_xgb_threshold:.2f}")
print(f"- best validation performance under evaluation harness: f2_under={best_xgb_row['f2_under']:.4f}, recall_under={best_xgb_row['recall_under']:.4f}, pr_auc={best_xgb_row['pr_auc']:.4f}")

final_candidates = comparison_df.loc[["Tuned Logistic Regression", "Tuned XGBoost"], ["f2_under", "pr_auc", "balanced_accuracy", "precision_under"]].copy()
final_candidates["selection_score"] = (
    0.40 * final_candidates["f2_under"]
    + 0.25 * final_candidates["pr_auc"]
    + 0.20 * final_candidates["balanced_accuracy"]
    + 0.15 * final_candidates["precision_under"]
)

display(final_candidates.round(4))

final_model_name = final_candidates["selection_score"].idxmax()
print(f"Selected final model: {final_model_name}")

final_candidates.to_csv(EVIDENCE_DIR / "final_model_selection_table_codex.csv")
comparison_df.to_csv(EVIDENCE_DIR / "model_comparison_test_codex.csv")

(BEST_MODEL_PATH := BASE_DIR / "best_model_name.txt").write_text(final_model_name, encoding="utf-8")

log_step(
    step_name="5.4 Final model decision",
    status="Completed",
    key_actions="Applied weighted quantitative framework to tuned LR and tuned XGBoost and selected final model.",
    key_outputs=f"Final model={final_model_name}; selection table saved to {EVIDENCE_DIR / 'final_model_selection_table_codex.csv'}",
)
"""
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3",
    },
}

with open("experiment_codex.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("experiment_codex.ipynb created.")
