import pandas as pd

# 1. Load dataset
file_path = "/Users/frangkysupit/Downloads/survey/mental-heath-in-tech-2016_20161114.csv"  # make sure file is in the same folder as your script
df = pd.read_csv(file_path)

# 2. Drop features with >=50% missing values
threshold = len(df) * 0.5
df_cleaned = df.dropna(thresh=threshold, axis=1)

print("Original shape:", df.shape)
print("After dropping columns:", df_cleaned.shape)

# 3. Fill remaining NaN with the mode of each column
for col in df_cleaned.columns:
    mode_val = df_cleaned[col].mode()[0]  # get most frequent value
    df_cleaned[col].fillna(mode_val, inplace=True)

# 4. Verify
print("Remaining NaN values:", df_cleaned.isna().sum().sum())

# 5. (Optional) Save the cleaned dataset
df_cleaned.to_csv("mental_health_cleaned.csv", index=False)
print("Cleaned dataset saved as mental_health_cleaned.csv")