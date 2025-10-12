import pandas as pd

# 1. Load cleaned dataset
file_path = "/Users/frangkysupit/Downloads/survey/mental_health_cleaned.csv"  # <-- update with your path
df = pd.read_csv(file_path)

# 2. Apply One-Hot Encoding to categorical columns
df_ohe = pd.get_dummies(df, drop_first=True, dtype=int)  # drop_first=True avoids dummy variable trap

# 3. Save encoded dataset
output_path = "/Users/frangkysupit/Downloads/survey/mental_health_encoded.csv"
df_ohe.to_csv(output_path, index=False)

# 4. Print info
print("Original shape:", df.shape)
print("After One-Hot Encoding:", df_ohe.shape)
print("Number of features (columns) after OHE:", df_ohe.shape[1])
print("One-hot encoded dataset saved as:", output_path)
