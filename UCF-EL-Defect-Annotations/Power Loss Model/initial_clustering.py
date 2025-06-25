import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
import seaborn as sns

iv_el_file_path = r'C:\Users\light\Downloads\M55\M55\Tabular_Data\m55_iv_el.txt'
defect_percentages_file_path = r'C:\Users\light\Downloads\M55\M55Dataframe.csv'

# Read the files in into csv
iv_el_df = pd.read_csv(iv_el_file_path, delimiter='\t')
defects_df = pd.read_csv(defect_percentages_file_path)

# Create a new column based on the unique 10 digit sequence
iv_el_df['id'] = iv_el_df['txt_filename'].str.extract(r'(\d{10})')
defects_df['id'] = defects_df['Filename'].str.extract(r'(\d{10})')

# Merge the defect dataframe and the IV EL dataframes together
merged_df = pd.merge(iv_el_df, defects_df, on='id', how='inner')

merged_df_export_file_path = ""

# Optional if want to download the merged csv
# merged_df.to_csv(merged_df_export_file_path, index=False)

# print(f"Merged dataframe shape: {merged_df.shape}")

# Select specific features
features_cols = ['Crack', 'Contact', 'Interconnect', 'Corrosion',
                 'Isc_(A)', 'Voc_(V)','Pmp_(W)']

features = merged_df[features_cols].dropna()

if features.empty:
    raise ValueError("No valid rows left after dropping NaN and 0 values.")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Run K means clustering
num_clusters = 4
random_seed = 42
kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster column
merged_df.loc[features.index, 'cluster'] = clusters

# Run PCA to visualize
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the cluster
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title(f'Clusters: {num_clusters}, Random seed: {random_seed}')
plt.colorbar(scatter, label='Cluster')
plt.show()
