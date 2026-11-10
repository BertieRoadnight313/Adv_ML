"""
===============================================================================
Topic: Advanced Clustering Methods (K-Means, Fuzzy C-Means, DBSCAN, Hierarchical)
Libraries: scikit-learn, fcmeans, pandas, numpy, matplotlib, seaborn
===============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from fcmeans import FCM  # Install via: pip install fuzzy-c-means (Import as 'fcmeans')

# Set working directory to the script's directory (ensures relative paths work)
# EXAM TIP: If running in Jupyter Notebook, __file__ is not defined; use os.getcwd() instead.
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

# =============================================================================
# STEP 1: DATA LOADING & INITIAL EXPLORATION
# =============================================================================
df = pd.read_csv("Mall_Customers.csv")

# 1. Dataset Dimensions: (rows, columns)
print("--- Data Shape ---")
print(df.shape)

# 2. View First 5 Rows: Quick sanity check of column contents
print("\n--- First 5 Rows ---")
print(df.head())

# 3. Data Types & Memory Usage: Inspect non-null counts and column types
print("\n--- Data Info ---")
print(df.info())

# 4. Check Missing Values: Count NaNs per column
# EXAM TIP: Distance-based algorithms (KMeans, DBSCAN, etc.) CANNOT handle missing values!
print("\n--- Missing Value Count ---")
print(df.isna().sum())

# 5. Summary Statistics: Mean, std, min, max, percentiles for numerical/categorical data
print("\n--- Summary Statistics ---")
print(df.describe(include="all"))


# =============================================================================
# STEP 2: FEATURE SELECTION & FEATURE SCALING
# =============================================================================
# Select numerical continuous features relevant for customer segmentation
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()

# Feature Scaling using StandardScaler (Z-score normalization: mean=0, std=1)
# WHY SCALING IS MANDATORY FOR CLUSTERING:
# Clustering models rely on Euclidean Distance: d(p, q) = sqrt(sum((p_i - q_i)^2))
# Unscaled features with larger ranges (e.g., Annual Income in thousands vs. Age)
# will unfairly dominate the distance calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Scaled Feature Inspection ---")
print("Scaled Data Shape:", X_scaled.shape)
print("Approx. means (should be ~0.0):", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations (should be ~1.0):", X_scaled.std(axis=0).round(3))


# =============================================================================
# STEP 3: HYPERPARAMETER TUNING - CHOOSING OPTIMAL K (CLUSTERS)
# =============================================================================
# Evaluate cluster quality across candidate K values from 2 to 10
ks = range(2, 11)
inertias, silhouettes = [], []

for k in ks:
    # n_init=10 runs KMeans 10 times with different random seeds and picks the best initialization.
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    
    # Inertia: Sum of squared distances of samples to their closest cluster center (Within-Cluster SSE).
    # Lower is better, but decreases monotonically with K. Look for the "Elbow point".
    inertias.append(model.inertia_)
    
    # Silhouette Score: Measures how well-separated clusters are (range: -1 to +1).
    # +1 = well-clustered, 0 = overlapping clusters, -1 = misclassified. Higher is better.
    silhouettes.append(silhouette_score(X_scaled, labels))

# Plotting Elbow Curve and Silhouette Scores side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Elbow Method
axes[0].plot(list(ks), inertias, marker="o", color="b")
axes[0].set_title("Elbow Plot (Inertia vs. K)")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-Cluster SSE)")

# Plot 2: Silhouette Score Method
axes[1].plot(list(ks), silhouettes, marker="o", color="g")
axes[1].set_title("Silhouette Score by K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

# Display tabular evaluation summary
k_eval_df = pd.DataFrame({"K": list(ks), "Inertia": inertias, "Silhouette": silhouettes}).round(3)
print("\n--- Evaluation Metrics for K ---")
print(k_eval_df)


# =============================================================================
# STEP 4: HARD CLUSTERING BASELINE - K-MEANS
# =============================================================================
# Fit KMeans model with chosen K (e.g., K=5 based on highest Silhouette / Elbow justification)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# Cluster Profiling: Mean value of features in original units per cluster
# EXAM TIP: Always interpret clusters using ORIGINAL unscaled units so recommendations make practical sense!
print("\n--- KMeans Cluster Profiling (Original Units Mean) ---")
print(df.groupby("kmeans_cluster")[features].mean().round(1))

# Count number of data points assigned to each cluster
print("\n--- KMeans Cluster Size Distribution ---")
print(df["kmeans_cluster"].value_counts().sort_index())

# Visualize KMeans Segmentation (Annual Income vs Spending Score)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2],
                hue="kmeans_cluster", palette="tab10", style="kmeans_cluster", s=70)
plt.title("KMeans Clustering Segments (K=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.show()


# =============================================================================
# STEP 5: SOFT / FUZZY CLUSTERING - FUZZY C-MEANS (FCM)
# =============================================================================
# Unlike KMeans (hard assignment: point belongs 100% to 1 cluster),
# Fuzzy C-Means outputs a probability/membership matrix U where each point
# has partial membership across ALL clusters (sum of memberships = 1.0).

fcm = FCM(n_clusters=k, random_state=42)
fcm.fit(X_scaled)

# U: Membership matrix of shape (n_samples, n_clusters)
U = fcm.u
centres_scaled = fcm.centers

print("\n--- FCM Membership Matrix Analysis ---")
print("Membership matrix shape (N x K):", U.shape)
print("First 5 membership vectors (probabilities across 5 clusters):\n", np.round(U[:5], 3))
print("Row sums verification (must all equal 1.0):\n", U.sum(axis=1)[:5].round(4))

# Derived Fuzzy Metrics:
# 1. fcm_cluster: Hard label derived from cluster with highest membership degree
df["fcm_cluster"] = U.argmax(axis=1)

# 2. fcm_max_membership: Confidence/certainty level of assignment (closer to 1 = crisp, ~1/K = ambiguous)
df["fcm_max_membership"] = U.max(axis=1)

# 3. fcm_membership_gap: Difference between top 1st and 2nd highest memberships.
#    Large gap = confident assignment; Small gap (~0) = boundary/ambiguous data point.
sorted_U = np.sort(U, axis=1)
df["fcm_membership_gap"] = sorted_U[:, -1] - sorted_U[:, -2]


# =============================================================================
# STEP 6: EVALUATING FUZZY HYPERPARAMETERS (CANDIDATE K IN FCM)
# =============================================================================
# Compare membership sharpness across candidate cluster counts (K = 3, 4, 5, 6)
comparison = []
for candidate_k in [3, 4, 5, 6]:
    model = FCM(n_clusters=candidate_k, random_state=42)
    model.fit(X_scaled)
    U_k = model.u
    sorted_memberships = np.sort(U_k, axis=1)
    max_membership = U_k.max(axis=1)
    gap = sorted_memberships[:, -1] - sorted_memberships[:, -2]
    
    comparison.append({
        "K": candidate_k,
        "mean_max_membership": max_membership.mean(),    # Higher mean = sharper cluster boundaries
        "median_max_membership": np.median(max_membership),
        "mean_top_two_gap": gap.mean()                  # Higher gap = less overlap between clusters
    })

print("\n--- FCM Candidate K Comparison ---")
print(pd.DataFrame(comparison).round(3))


# =============================================================================
# STEP 7: VISUALIZATION & PROFILING FUZZY C-MEANS
# =============================================================================
# 1. FCM Provisional Scatterplot (Max Membership Labels)
plot_df = df.copy()
plt.figure(figsize=(7, 5))
sns.scatterplot(data=plot_df, x=features[1], y=features[2],
                hue="fcm_cluster", palette="tab10", style="fcm_cluster", s=70)
plt.title("FCM: Provisional Labels by Maximum Membership")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.show()

# 2. Distribution of Maximum Membership (Assignment Sharpness / Certainty)
plt.figure(figsize=(7, 4))
sns.histplot(df["fcm_max_membership"], bins=20, kde=True, color="purple")
plt.xlabel("Maximum FCM Membership")
plt.title("Cluster Assignment Sharpness (How confident are assignments?)")
plt.tight_layout()
plt.show()

# 3. Profile FCM Clusters in Original Units
print("\n--- FCM Cluster Profiling (Original Units Mean) ---")
print(df.groupby("fcm_cluster")[features].mean().round(1))


# =============================================================================
# STEP 8: ALTERNATIVE CLUSTERING ALGORITHMS
# =============================================================================

# -----------------------------------------------------------------------------
# OPTION 1: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# -----------------------------------------------------------------------------
# Key Parameters:
# - eps (epsilon): Max distance between two samples to be considered neighbors.
# - min_samples: Min number of samples in a neighborhood to form a core point.
# ADVANTAGES:
# - Does NOT require pre-specifying number of clusters (K).
# - Discovers arbitrary-shaped clusters and automatically flags noise (-1 label).
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.7, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

print("\n--- DBSCAN Cluster Distribution ---")
# EXAM TIP: Label '-1' indicates noise / outliers detected by DBSCAN!
print(pd.Series(db_labels).value_counts().sort_index())


# -----------------------------------------------------------------------------
# OPTION 2: HIERARCHICAL CLUSTERING (Agglomerative Clustering)
# -----------------------------------------------------------------------------
# Key Parameters:
# - n_clusters: Target number of clusters (K).
# - linkage="ward": Minimizes total within-cluster variance (similar to KMeans objective).
# ADVANTAGES:
# - Creates a hierarchical tree (Dendrogram) showing sub-cluster relationships.
# - Deterministic results (no random centroid initialization).
from sklearn.cluster import AgglomerativeClustering

hier = AgglomerativeClustering(n_clusters=k, linkage="ward")
hier_labels = hier.fit_predict(X_scaled)

print("\n--- Agglomerative Hierarchical Cluster Distribution ---")
print(pd.Series(hier_labels).value_counts().sort_index())
