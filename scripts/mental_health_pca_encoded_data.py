# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load encoded dataset
df = pd.read_csv("/Users/frangkysupit/Downloads/survey/mental_health_encoded.csv")

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance_ratio = pca.explained_variance_ratio_ * 100
cumulative_variance = explained_variance_ratio.cumsum()

# Print cumulative variance retained
print(f"Cumulative variance retained by top 2 PCs: {cumulative_variance[-1]:.2f}%")
