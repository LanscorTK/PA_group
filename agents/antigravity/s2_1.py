import os
import matplotlib.pyplot as plt
import seaborn as sns

# 2.1 Target Conversion and Data Extraction
valid_targets = [1.0, 2.0]
df_filtered = participation_raw[participation_raw['CARTS_NET'].isin(valid_targets)].copy()

# Add a new binary target variable "arts_engaged" without recoding to 0/1 yet
df_filtered['arts_engaged'] = df_filtered['CARTS_NET']

# Drop the original CARTS_NET column
participation_eda = df_filtered.drop(columns=['CARTS_NET']).copy()

print(f"Original rows: {participation_raw.shape[0]}")
print(f"Excluded rows with target values -3 or 3: {participation_raw.shape[0] - participation_eda.shape[0]}")
print(f"Rows remaining for EDA & modeling: {participation_eda.shape[0]}")

# Create folder for saving figures
pic_dir = "EDA_antigravity_Pics"
os.makedirs(pic_dir, exist_ok=True)
print(f"Created sub-directory: '{pic_dir}' for EDA visuals.")

# 2.2 Exploratory Data Analysis & Visualization
sns.set_theme(style="whitegrid")

# 1. Target Variable Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=participation_eda, x='arts_engaged', palette='viridis')
plt.title("Distribution of Arts Engagement (Target Variable)\n1 = Yes, 2 = No")
plt.xlabel("Arts Engaged")
plt.ylabel("Count")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "dist_target_variable.png"), dpi=150)
plt.close()

# 2. Feature Distribution by Target: AGEBAND
plt.figure(figsize=(10, 6))
sns.countplot(data=participation_eda, x='AGEBAND', hue='arts_engaged', palette='viridis')
plt.title("Arts Engagement by Age Band (AGEBAND)")
plt.xlabel("Age Band Code")
plt.ylabel("Count")
plt.legend(title='Arts Engaged (1=Yes, 2=No)')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "dist_AGEBAND_by_target.png"), dpi=150)
plt.close()

# 3. Feature Distribution by Target: RUR11CAT (Rural/Urban)
plt.figure(figsize=(8, 5))
sns.countplot(data=participation_eda, x='rur11cat', hue='arts_engaged', palette='mako')
plt.title("Arts Engagement by Rural/Urban Class (rur11cat)")
plt.xlabel("Rural/Urban Code")
plt.ylabel("Count")
plt.legend(title='Arts Engaged')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "dist_RUR11CAT_by_target.png"), dpi=150)
plt.close()

# 4. Feature Distribution by Target: FINHARD (Financial Hardship)
plt.figure(figsize=(8, 5))
sns.countplot(data=participation_eda, x='FINHARD', hue='arts_engaged', palette='flare')
plt.title("Arts Engagement by Financial Hardship (FINHARD)")
plt.xlabel("Financial Hardship Code")
plt.ylabel("Count")
plt.legend(title='Arts Engaged')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, "dist_FINHARD_by_target.png"), dpi=150)
plt.close()

print("All visualisations have been saved successfully to EDA_antigravity_Pics/.")
