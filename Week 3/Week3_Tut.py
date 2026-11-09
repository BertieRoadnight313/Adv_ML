# Advanced Clustering Methods
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from fcmeans import FCM # install as 'pip install fuzzy-c-means' - not fcmeans

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Inspect the Data
df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))

# Select features and scale the data

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled Data Shape:", X_scaled.shape)
print("Approx. means:", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations:", X_scaled.std(axis=0).round(3))

# Choose K using evidence

ks = range(2, 11)
inertias, silhouettes = [], []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(ks), inertias, marker="o")
axes[0].set_title("Elbow Plot")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set_title("Silhouette by K")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

pd.DataFrame({"K": list(ks), "Inertia": inertias,
              "silhouette": silhouettes}).round(3)


# Fit the hard-clustering baseline

k = 5 # replace with the value you can justify
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# profile clusters in original feature units
print(df.groupby("kmeans_cluster")[features].mean().round(1))
print(df["kmeans_cluster"].value_counts().sort_index())

# Visualize the clusters

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2],
                hue="kmeans_cluster", palette="tab10")
plt.title("KMeans Clustering Segments")
plt.tight_layout()
plt.show() #Part D - Fit the hard clustering baseline


# Part E - Fit the fuzzy C-Means

fcm = FCM(n_clusters=k, random_state=42)
fcm.fit(X_scaled)

U = fcm.u
centres_scaled = fcm.centers

print("Membership matrix shape:", U.shape)
print("First five membership vectors:\n", np.round(U[:5], 3))
print("Row sums (should be about 1):", U.sum(axis=1)[:5].round(4))

df["fcm_cluster"] = U.argmax(axis=1)
df["fcm_max_membership"] = U.max(axis=1)
sorted_U = np.sort(U, axis=1)
df["fcm_membership_gap"] = sorted_U[:,-1] - sorted_U[:,-2]




