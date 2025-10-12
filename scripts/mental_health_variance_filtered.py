import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# 1. Load dataset
file_path = "/Users/frangkysupit/Downloads/survey/mental_health_encoded.csv"
df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# 2. Separate features and target (if any)
# If your dataset has a target column, exclude it here
# Example: target = df['target']; X = df.drop('target', axis=1)
X = df.copy()

# 3. Apply Variance Threshold
# Threshold=0 removes only features with zero variance (same value for all rows)
# You can increase it (e.g., 0.01) to drop features with very little variance
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X)

# 4. Get selected feature names
selected_features = X.columns[selector.get_support()]

# 5. Create new DataFrame with selected features
df_reduced = pd.DataFrame(X_var, columns=selected_features)

# 6. Save result
output_path = "/Users/frangkysupit/Downloads/survey/mental_health_variance_filtered.csv"
df_reduced.to_csv(output_path, index=False)

print("After removing zero-variance features:", df_reduced.shape)
print("Number of features removed:", len(X.columns) - len(selected_features))
print("Filtered dataset saved as:", output_path)
