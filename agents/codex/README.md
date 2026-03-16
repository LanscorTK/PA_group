# Codex Experiment Packaging

## Required Inputs
Place or keep these input files in `../../data/` relative to this folder:
- `participation_2024-25_experiment.tab`
- `participation_2024-25_data_dictionary_cleaned.txt`

## How To Run
From this directory (`agents/codex`):

```bash
pip install -r requirements.txt
jupyter notebook experiment_codex.ipynb
```

Run the notebook sequentially from top to bottom (Sections 0 to 5).

## Outputs Produced
Main outputs are saved under `evidence_codex/`:
- `EDA_codex_Pics/*.png` (EDA figures from Step 2)
- `missingness_handling_summary.csv` (Step 3 cleaning audit)
- `baseline_lr_validation_metrics.csv` (Step 4 validation metrics)
- `lr_tuning_results.csv` (Step 5.1 tuning trials)
- `xgb_tuning_results.csv` (Step 5.2 tuning trials)
- `test_model_comparison.csv` (Step 5.3 final test comparison)
- `model_selection_framework.csv` (Step 5.4 weighted final selection)

Additional deliverables in this workspace:
- `run_log_codex.md`
- `Report_codex.md`

## Reproducibility Measures
- Fixed global seed: `RANDOM_STATE = 42`.
- Stratified train/validation/test split with fixed random state.
- Relative paths only.
- Shared evaluation harness used consistently across all models.
- Test set reserved for final comparison only (after tuning).
