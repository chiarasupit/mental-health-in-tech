import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tabulate import tabulate  # for clean table display
import numpy as np

# 1. Load dataset
df = pd.read_csv("/Users/frangkysupit/Downloads/survey/mental_health_variance_filtered.csv")

# 2. Apply K-Means clustering (k=2)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(df)

# 3. Interpret cluster labels based on mean values
df_temp = df.copy()
df_temp['Cluster'] = clusters
cluster_means = df_temp.groupby('Cluster').mean().mean(axis=1)

if cluster_means[0] > cluster_means[1]:
    cluster_labels = {0: 'Has Mental Health Disorder', 1: 'No Mental Health Disorder'}
else:
    cluster_labels = {0: 'No Mental Health Disorder', 1: 'Has Mental Health Disorder'}

df['Cluster'] = [cluster_labels[c] for c in clusters]

# 4. Prepare for Chi-Square test
X = df.drop(columns=['Cluster'])
y = df['Cluster'].apply(lambda x: 1 if x == 'Has Mental Health Disorder' else 0)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply Chi-Square test
chi_scores, p_values = chi2(X_scaled, y)

# 6. Compile results into a DataFrame
chi2_results = pd.DataFrame({
    'Feature': X.columns,
    'Chi2_Score': chi_scores,
    'P_Value': p_values
})

# 7. Add significance stars
def significance(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''
chi2_results['Significance'] = chi2_results['P_Value'].apply(significance)

# 8. Sort results by importance
chi2_results = chi2_results.sort_values(by='Chi2_Score', ascending=False)

# 9. Display top 5 in formatted table
top5 = chi2_results.head(5)
print("\n=== Top 5 Features Contributing to Mental Health Cluster Separation ===\n")
print(tabulate(top5, headers='keys', tablefmt='grid', showindex=False, floatfmt=(".4f", ".4f", ".4e", "s")))

# === New Section: Save table as an image ===
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
table_data = [["Feature", "Chi2_Score", "P_Value", "Significance"]] + top5.values.tolist()
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("Top 5 Chi-Square Features for Mental Health Cluster Separation", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig("/Users/frangkysupit/Downloads/survey/top5_chi_square_table.png", bbox_inches='tight', dpi=300)
plt.close()
print("\n✅ Table image saved as 'top5_chi_square_table.png'\n")

# 10. Visualization — Top 5 features
plt.figure(figsize=(12, 7))
bars = plt.barh(top5['Feature'][::-1], top5['Chi2_Score'][::-1], color='skyblue')
plt.xlabel("Chi-Square Score", fontsize=12)
plt.title("Top 5 Features Contributing to Mental Health Cluster Separation", fontsize=14, pad=20)
plt.tight_layout(pad=3.0)

# Annotate significance stars
for i, (score, sig) in enumerate(zip(top5['Chi2_Score'][::-1], top5['Significance'][::-1])):
    plt.text(score + 0.5, i, sig, va='center', fontsize=12, color='black')

plt.show()

# 11. Save updated dataset
df.to_csv("/Users/frangkysupit/Downloads/survey/mental_health_with_clusters.csv", index=False)
print("\nClustered dataset saved as 'mental_health_with_clusters.csv'")
print(f"\nCluster label meanings: {cluster_labels}")