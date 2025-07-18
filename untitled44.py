# -*- coding: utf-8 -*-
"""Untitled44.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YqvtoFif7sChkUEQVxVLDxfMw5arBqNv
"""

# 📦 Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 📁 Step 1: Load Dataset
data = load_iris()
X = data.data
y_true = data.target  # actual labels (for reference only)

# 📉 Optional PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 📍 Step 2: Fit K-Means & Assign Cluster Labels
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# 📊 Step 3: Elbow Method to Find Optimal K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(X)
    inertias.append(kmeans_model.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, marker='o', color='teal')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# 🎨 Step 4: Visualize Clusters (2D)
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title('K-Means Clustering Visualization (PCA Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# 🧠 Step 5: Evaluate Using Silhouette Score
score = silhouette_score(X, cluster_labels)
print(f"Silhouette Score for K={kmeans.n_clusters}: {score:.3f}")