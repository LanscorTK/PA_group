# Antigravity AI Agent: Predictive Analytics Experiment

This package contains the end-to-end execution of a machine learning workflow for the PA Group Coursework, focusing on predicting arts under-engagement using the UK Participation Survey (2024-25).

## Required Input Files
To run the notebook `experiment_antigravity.ipynb`, ensure the following two files are placed in the same directory as the notebook (symlinks are acceptable):
1. `participation_2024-25_experiment.tab` - The main data subset.
2. `participation_2024-25_data_dictionary_cleaned.txt` - Variable dictionary used for schema and missingness logic definition.

## How to Run the Notebook
1. Install dependencies from the `requirements.txt` file (e.g., `pip install -r requirements.txt`).
2. Run the notebook sequentially from top to bottom. No manual intervention is required.
    - via Jupyter Lab/Notebook interface: Click "Restart & Run All".
    - via command line: `jupyter nbconvert --to notebook --execute experiment_antigravity.ipynb --inplace`.

## Outputs Produced and Save Locations
All outputs are localized to this directly:
- **`EDA_antigravity_Pics/`**: A folder containing `.png` versions of all EDA visualizations (e.g., target distribution and feature relationships).
- **`run_log_antigravity.md`**: An audit trail documenting step-by-step progress, completion statuses, and any key actions or warnings.
- **`Report_antigravity.md`**: A concise, non-technical policy report deriving insights from the model evaluations.

## Methods Supporting Reproducibility
- **Global Random Seed Control**: Established `random_state=42` entirely throughout data splitting, model training, and tuning to ensure identical pipeline results on independent runs.
- **Relative Path Consistency**: All inputs are loaded and outputs strictly written utilizing relative paths from the current directory.
- **Sequential Execution**: The notebook is built dynamically keeping cells linear. Tests passed guaranteeing zero internal or upward data dependencies allowing straight-line execution.
