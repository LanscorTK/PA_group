# 1.1 Dataset ingestion and schema checks
file_path = "participation_2024-25_experiment.tab"
df_raw = pd.read_csv(file_path, sep='\t')
participation_raw = df_raw.copy()

print(f"Dataset Shape: {participation_raw.shape[0]} rows and {participation_raw.shape[1]} columns.")

required_columns = [
    'CARTS_NET', 'AGEBAND', 'SEX', 'QWORK', 'EDUCAT3', 
    'FINHARD', 'CINTOFT', 'gor', 'rur11cat', 'CHILDHH', 'COHAB'
]

# Check 1: Required variables presence
missing_cols = [col for col in required_columns if col not in participation_raw.columns]
if not missing_cols:
    print("Schema Check 1: All required variables are present.")
else:
    print(f"Schema Check 1 FAILED: Missing variables {missing_cols}")

# Check 2: Duplicate columns
duplicate_cols = participation_raw.columns[participation_raw.columns.duplicated()]
if len(duplicate_cols) == 0:
    print("Schema Check 2: No duplicate columns found.")
else:
    print(f"Schema Check 2 FAILED: Duplicate columns {duplicate_cols}")

# Check 3: Data types
non_numeric_cols = participation_raw[required_columns].select_dtypes(exclude=['number']).columns
if len(non_numeric_cols) == 0:
    print("Schema Check 3: All required columns are numeric (as expected from coded categorical variables).")
else:
    print(f"Schema Check 3 FAILED: Non-numeric columns found {non_numeric_cols}")

from IPython.display import display
print("\nStatistical summary of required columns:")
display(participation_raw[required_columns].describe())
