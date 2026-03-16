# Experiment: claude_code

## Required Input Files

The following files must be available at `../../data/` relative to this directory:

- `participation_2024-25_experiment.tab` — Tab-separated data file (34,378 rows, 11 columns)
- `participation_2024-25_data_dictionary_cleaned.txt` — Variable dictionary describing coded values

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the notebook from this directory:
   ```
   jupyter nbconvert --to notebook --execute experiment_claude_code.ipynb
   ```
   Or open in Jupyter and run all cells sequentially from top to bottom.

## Outputs

| Output | Location |
|--------|----------|
| Experiment notebook | `experiment_claude_code.ipynb` |
| Run log | `run_log_claude_code.md` |
| EDA figures (.png) | `evidence_claude_code/EDA_claude_code_Pics/` |
| Test set comparison | `evidence_claude_code/test_set_comparison.csv` |
| Model selection scores | `evidence_claude_code/model_selection_scores.csv` |
| Non-technical report | `Report_claude_code.md` |

## Reproducibility

- **Random seed**: `RANDOM_STATE = 42` is used throughout for all random operations (`random.seed`, `np.random.seed`, `train_test_split`, model training).
- **Relative paths**: All file paths are relative to the notebook directory.
- **Sequential execution**: The notebook is designed to run from top to bottom without manual intervention.
- **Data split**: 70% train / 15% validation / 15% test, stratified by target class.
- **No test set leakage**: The test set is used only once, for final model comparison in Step 5.3.
