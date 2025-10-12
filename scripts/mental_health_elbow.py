# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# 1. Load your dataset
file_path = "/Users/frangkysupit/Downloads/survey/mental_health_variance_filtered.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# 2. Ensure data is numeric
X = df.select_dtypes(include=[np.number])

# 3. Create a K-Means model and Elbow Visualizer
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(1, 10), metric='distortion', timings=False)

# 4. Fit visualizer on your dataset
visualizer.fit(X)

# 5. Display the elbow plot
visualizer.show()

# Optional: print optimal k value
print("Optimal number of clusters (k):", visualizer.elbow_value_)
