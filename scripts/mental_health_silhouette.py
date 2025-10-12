# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score

# 1. Load dataset
file_path = "/Users/frangkysupit/Downloads/survey/mental_health_variance_filtered.csv"
df = pd.read_csv(file_path)

# 2. Ensure only numeric data is used
X = df.select_dtypes(include=[np.number])

# 3. Prepare range of clusters
k_values = range(2, 7)
silhouette_scores = []

# 4. Run KMeans and Silhouette analysis for each k
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    
    # Calculate silhouette score
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"For k = {k}, Average Silhouette Score = {score:.4f}")
    
    # Plot silhouette visualization
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(X)
    visualizer.show()

# 5. Plot silhouette scores summary
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
plt.title("Silhouette Score vs Number of Clusters (k)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.grid(True)
plt.show()
