# Experiment: claude_code

## Required Input Files

The following files must be available in `../../data/` (relative to this directory):

1. `participation_2024-25_experiment.tab` — UK Participation Survey dataset (tab-separated)
2. `participation_2024-25_data_dictionary_cleaned.txt` — Variable dictionary

## How to Run

```bash
cd agents/claude_code
pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace experiment_claude_code.ipynb
```

Or open `experiment_claude_code.ipynb` in Jupyter and run all cells sequentially.

## Outputs

| File | Description |
|------|-------------|
| `experiment_claude_code.ipynb` | Complete experiment notebook with all outputs |
| `run_log_claude_code.md` | Step-by-step execution log |
| `Report_claude_code.md` | Non-technical policy report (~400 words) |
| `requirements.txt` | Python dependencies |
| `evidence_claude_code/` | Evidence folder containing: |
| `  EDA_claude_code_Pics/` | EDA visualisation PNGs |
| `  baseline_lr_validation_metrics.csv` | Baseline LR validation results |
| `  lr_tuning_results.csv` | LR grid search results |
| `  xgb_tuning_results.csv` | XGBoost grid search results |
| `  test_model_comparison.csv` | Final test-set model comparison |
| `  model_selection_framework.csv` | Weighted model selection scores |
| `  missingness_handling_summary.csv` | Missingness handling details |

## Reproducibility

- **Random seed:** `RANDOM_STATE = 42` used throughout (numpy, random, sklearn, xgboost)
- **Stratified splits:** 70/15/15 train/validation/test with stratification on target
- **Relative paths only:** all file references use relative paths from this directory
- **Shared evaluation harness:** identical metrics applied to all models
- **Test set discipline:** test set used only once in Step 5.3 for final comparison
