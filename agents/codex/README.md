# codex Experiment Package

## Required Input Files
Place these files under `./data` (already linked in this workspace):
- `participation_2024-25_data_dictionary_cleaned.txt`
- `participation_2024-25_experiment.tab`

## How To Run
From `agents/codex`, run:

```bash
python3 -m nbconvert --to notebook --execute --inplace experiment_codex.ipynb
```

This runs the notebook sequentially from top to bottom with no manual steps.

## Outputs Produced
- Main notebook: `experiment_codex.ipynb`
- Run log: `run_log_codex.md`
- Evidence folder: `evidence_codex/`
  - EDA figures: `evidence_codex/EDA_codex_Pics/*.png`
  - Missingness summary: `evidence_codex/missingness_summary_codex.csv`
  - Validation/test model metrics and tuning outputs:
    - `baseline_lr_validation_metrics_codex.csv`
    - `lr_tuning_results_codex.csv`
    - `xgb_tuning_results_codex.csv`
    - `model_comparison_test_codex.csv`
    - `final_model_selection_table_codex.csv`
- Final model text marker: `best_model_name.txt`
- Policy-facing report: `Report_codex.md`

## Reproducibility Controls
- Global random seed fixed to `42`.
- All randomized operations use `random_state=42`.
- Train/validation/test split is fixed and stratified (`0.7 / 0.15 / 0.15`).
- Tuning and evaluation use a shared, explicit evaluation harness and fixed threshold grid.
