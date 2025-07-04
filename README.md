
# 🌸 K-Means Clustering on Iris Dataset

This project demonstrates unsupervised clustering using the **K-Means** algorithm on the **Iris dataset** along with PCA-based visualization and evaluation using the silhouette score.

---

## 📁 Dataset
We use the **Iris dataset**, a classic dataset with 150 samples and 4 features representing measurements of iris flowers of 3 species.

---

## 🛠️ Tasks Performed

### 1. 📦 Import Libraries
We import necessary Python libraries for clustering, visualization, and evaluation.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

---

### 2. 📁 Load Dataset

```python
data = load_iris()
X = data.data
y_true = data.target  # actual labels (for reference only)
```

---

### 3. 📉 PCA for 2D Visualization

To visualize in 2D, we reduce dimensions using PCA.

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

---

### 4. 📍 Fit K-Means Model

```python
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)
```

---

### 5. 📊 Elbow Method to Find Optimal K

We compute inertia for K = 1 to 10 and plot to find the elbow point.

```python
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(X)
    inertias.append(kmeans_model.inertia_)
```

📌 Plotting the Elbow Curve:

```python
plt.plot(K_range, inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
```

---

### 6. 🎨 Visualize Clusters (PCA-reduced)

```python
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title('K-Means Clustering Visualization (PCA Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
```

---

### 7. 🧠 Evaluate using Silhouette Score

```python
score = silhouette_score(X, cluster_labels)
print(f"Silhouette Score for K={kmeans.n_clusters}: {score:.3f}")
```

---

## ✅ Final Output

- 📌 **Elbow curve** to choose best K.
- 📌 **PCA scatter plot** to visualize clusters.
- 📌 **Silhouette Score** for evaluating clustering performance.

---

## 👩‍💻 Tools Used

- Python
- Scikit-learn
- Matplotlib
- Numpy

---

## 📎 Notes

- The elbow method helps in selecting the right number of clusters.
- PCA is used only for 2D visualization and not for clustering.
- Silhouette score provides a measure of how well the clustering is performed.

---

## 📌 Sample Output
```
Silhouette Score for K=3: 0.552
```

