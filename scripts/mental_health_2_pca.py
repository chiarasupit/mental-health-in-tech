# ============================================
# Principal Component Analysis (PCA) with Clustering Visualization
# ============================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 2. Load dataset
file_path = "/Users/frangkysupit/Downloads/survey/mental_health_variance_filtered.csv"
df = pd.read_csv(file_path)

# 3. Select numeric features
X = df.select_dtypes(include=['number'])

# 4. Standardize data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. Run K-Means clustering with k = 2 (based on previous silhouette analysis)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 7. Visualize the PCA projection with clusters
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', s=25, alpha=0.7)
plt.title("PCA Projection (Top 2 Components) with K-Means Clusters (k=2)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# 8. Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance by Principal Components:")
print(f"PC1: {explained_variance[0]*100:.2f}%")
print(f"PC2: {explained_variance[1]*100:.2f}%")

# 9. Optional – visualize explained variance
plt.figure(figsize=(6,4))
plt.bar(['PC1', 'PC2'], explained_variance * 100, color='skyblue')
plt.title("Explained Variance Ratio of Top 2 Principal Components")
plt.ylabel("Percentage of Total Variance Explained")
plt.show()
