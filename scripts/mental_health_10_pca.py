import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the filtered dataset
df = pd.read_csv("/Users/frangkysupit/Downloads/survey/mental_health_variance_filtered.csv")

# Standardize the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply PCA with 10 components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio per component
explained_variance_ratio = pca.explained_variance_ratio_ * 100
cumulative_variance = explained_variance_ratio.cumsum()

# Print total variance explained by 10 components
total_var_explained = cumulative_variance[-1]
print(f"Total variance explained by top 10 PCs: {total_var_explained:.2f}%")
