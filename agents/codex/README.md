# Codex Experiment Package

## Required input files
Place the following files in `./data` (already included in this workspace setup):
- `participation_2024-25_data_dictionary_cleaned.txt`
- `participation_2024-25_experiment.tab`

## How to run
From `agents/codex`:

```bash
python3 -m pip install -r requirements.txt
python3 -m nbconvert --to notebook --execute --inplace experiment_codex.ipynb
```

This executes the notebook top-to-bottom with no manual intervention.

## Outputs produced
- Main notebook: `experiment_codex.ipynb`
- Run log: `run_log_codex.md`
- Evidence folder: `evidence_codex/`
  - EDA figures: `evidence_codex/EDA_codex_Pics/*.png`
  - Schema, split, tuning, and evaluation outputs (`*.csv`, `*.json`, `*.png`)
- Policy-facing report: `Report_codex.md`

## Reproducibility controls
- Global random seed fixed to `42`.
- All split and model randomness set with `random_state=42`.
- Same train/validation/test split reused across all modeling steps.
- Test set held out until final comparison stage.
