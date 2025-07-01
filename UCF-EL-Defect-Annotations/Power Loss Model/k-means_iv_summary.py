import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import re

# I expanded/changed some of David's code from the original k-means.py
# His used m55_iv_el.txt, this one is intended to run on iv_summary.txt

# File paths
iv_el_file_path = r'C:\Users\Isaac\Downloads\iv_summary.txt' #this is my path to iv_summary, please don't forget to change
defect_percentages_file_path = r'C:\Users\Isaac\Downloads\M55Dataframe.csv' #this is my path to M55DataFrame, please don't forget to change
metadata_file_path = r"C:\Users\Isaac\Downloads\m55_module_metadata.txt" #this is my path to m55_module_metadata.txt, please don't forget to change

# Read in data
iv_el_df = pd.read_csv(iv_el_file_path, sep="\t")

# Load tab-separated CSV properly
defects_df = pd.read_csv(defect_percentages_file_path)

# Clean up column names
defects_df.columns = defects_df.columns.str.strip()
metadata_df = pd.read_csv(metadata_file_path, sep="\t")

# Extract consistent 10-digit serial number as 'id'
iv_el_df['id'] = iv_el_df['Serial_Number'].astype(str).str.extract(r'(\d{10})')
defects_df['id'] = defects_df['Filename'].str.extract(r'(\d{10})')

# Extract ID
defects_df['id'] = defects_df['Filename'].str.extract(r'(\d{10})')

#Locating each file based on Serial number
metadata_df["serial_number"] = metadata_df["serial_number"].astype(str)
metadata_df["id"] = metadata_df["serial_number"].str.extract(r'(\d{10})')

# Merge defect percentages with metadata
defects_with_meta = defects_df.merge(metadata_df, on="id", how="left")

# Merge that with IV/Electrical data
merged_df = pd.merge(defects_with_meta, iv_el_df, on="id", how="inner")


# Feature columns (ensure they exist in merged_df)
features_cols = ['Crack', 'Contact', 'Interconnect', 'Corrosion',
                 'Isc_(A)', 'Voc_(V)', 'Pmp_(W)']

if features.empty:
    raise ValueError("No valid rows left after dropping NaN and 0 values.")

# Drop rows with any missing values in selected features
features = merged_df[features_cols].dropna()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# KMeans clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)
merged_df.loc[features.index, 'cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA with clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f'Clusters: {num_clusters}, Random seed: 42')
plt.grid(True)
plt.tight_layout()
plt.show()

#Save results
merged_df.to_csv(r'C:\Users\Isaac\Downloads\clustered_iv_defects.csv', index=False)
#Change to location of your csv -> It was just easier to save to my downloads at the time



