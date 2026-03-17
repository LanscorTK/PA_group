# antigravity Experiment Packaging

## Required Input Files
The experiment requires the following files to be present in the `./data/` folder relative to the notebook:
- `participation_2024-25_experiment.tab` - The raw data file.
- `participation_2024-25_data_dictionary_cleaned.txt` - The data dictionary.

## How to Run the Notebook
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure the `./data/` folder contains the required inputs.
3. Open `experiment_antigravity.ipynb` in Jupyter Notebook or JupyterLab.
4. Run all cells sequentially from top to bottom (`Cell > Run All`). The notebook is designed to execute without any manual intervention.

## Outputs Produced
- The notebook itself contains textual outputs, printed summaries, and metrics.
- Visual outputs (Exploratory Data Analysis charts and Confusion Matrices) are automatically saved as `.png` files into the `./evidence_antigravity/EDA_antigravity_Pics/` directory.
- A run log documenting the experiment's step-by-step progress is maintained in `run_log_antigravity.md` in the root directory.

## Reproducibility Steps
- A global random seed (`SEED = 42`) is explicitly defined at the beginning of the notebook.
- This seed is passed into all stochastic algorithms (`train_test_split`, `LogisticRegression`, `XGBClassifier`) to ensure reproducible model training and evaluation.
- Relative paths are used strictly for data ingestion and output saving to ensure cross-system compatibility.
- Data schema checking and transparent handling of missing values (dropping non-informative codes `< 0` and `>= 997`) guarantee that the models ingest clean and consistent data.
