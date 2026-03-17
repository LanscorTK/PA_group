# PA Group: Predictive Analytics Pipeline

## Overview
This repository contains the predictive analytics pipeline implemented for identifying physical arts under-engagement using the 2024-25 Participation dataset.

## Input Files Required
- `data/participation_2024-25_experiment.tab`: The raw dataset for modeling.
- `data/participation_2024-25_data_dictionary_cleaned.txt`: Data dictionary (reference).

## Instructions to Run the Notebook
The main experiment is structured in a Jupyter Notebook: `experiment_antigravity.ipynb`.
You can execute the entire pipeline sequentially from top to bottom without manual intervention:
```bash
jupyter nbconvert --execute --to notebook --inplace experiment_antigravity.ipynb
```
Or open the notebook in Jupyter/VSCode and run all cells.

## Outputs
- **`run_log_antigravity.md`**: Step-by-step progress and status tracking log of the execution.
- **`evidence_antigravity/`**: Contains artifacts generated during the execution (e.g., EDA visualizations).
- **In-notebook Metrics**: Output metrics such as Precision, Recall, F1-Score, PR-AUC, Confusion Matrix for Baseline LR, Tuned LR, and Tuned XGBoost.

## Reproducibility
- The global random seed is fixed (`np.random.seed(42)` and `random_state=42`) across the entire workflow (data splitting, modeling, initialization) to ensure the results remain consistent across different runs.
