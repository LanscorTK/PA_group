# Experiment: claude_code

## Required Input Files

- `../../data/participation_2024-25_experiment.tab` — UK Participation Survey 2024-25 dataset
- `../../data/participation_2024-25_data_dictionary_cleaned.txt` — Variable dictionary

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the notebook sequentially from top to bottom:
   ```
   jupyter notebook experiment_claude_code.ipynb
   ```
   Or execute non-interactively:
   ```
   jupyter nbconvert --to notebook --execute experiment_claude_code.ipynb
   ```

## Outputs Produced

| Output | Location |
|--------|----------|
| EDA visualisations (4 PNG files) | `evidence_claude_code/EDA_claude_code_Pics/` |
| Baseline LR validation metrics | `evidence_claude_code/baseline_lr_validation_metrics.csv` |
| LR tuning results | `evidence_claude_code/lr_tuning_results.csv` |
| XGBoost tuning results | `evidence_claude_code/xgb_tuning_results.csv` |
| Test set model comparison | `evidence_claude_code/test_model_comparison.csv` |
| Model selection framework | `evidence_claude_code/model_selection_framework.csv` |
| Missingness handling summary | `evidence_claude_code/missingness_handling_summary.csv` |
| Run log | `run_log_claude_code.md` |
| Report | `Report_claude_code.md` |

## Reproducibility

- **Fixed random seed**: `RANDOM_STATE = 42` used throughout for all randomised operations
- **Stratified splits**: Train/validation/test split (70/15/15) uses stratified sampling to preserve class proportions
- **Relative paths only**: All file references use relative paths
- **Shared evaluation harness**: Identical metric computation applied to all models
- **Test set discipline**: Test set used only once for final model comparison after all tuning is complete
