# 3.2 Cleaning Implementation
participation_clean = participation_eda.copy()
original_row_count = participation_clean.shape[0]

# Rule 1 & 3: Drop invalid rows for Scale/Ordinal & Geographic variables
drop_cols = ['AGEBAND', 'CHILDHH', 'gor', 'rur11cat']
for col in drop_cols:
    # Keep only rows where the value is >= 0 and < 997
    participation_clean = participation_clean[(participation_clean[col] >= 0) & (participation_clean[col] < 997)]

# Rule 2: Recode missing values for Nominal variables to 999.0
recode_cols = ['SEX', 'QWORK', 'EDUCAT3', 'FINHARD', 'CINTOFT', 'COHAB']
for col in recode_cols:
    invalid_mask = (participation_clean[col] < 0) | (participation_clean[col] >= 997)
    participation_clean.loc[invalid_mask, col] = 999.0

final_row_count = participation_clean.shape[0]

print(f"Rows before missingness handling: {original_row_count}")
print(f"Rows after missingness handling: {final_row_count}")
print(f"Total rows dropped: {original_row_count - final_row_count}")

# Verify no invalid values remain in the drop columns
for col in drop_cols:
    assert participation_clean[col].min() >= 0, f"Missing values remain in {col}"
    assert participation_clean[col].max() < 997, f"Missing values remain in {col}"

print("\nMissingness handling complete. Data is ready for encoding and training.")
