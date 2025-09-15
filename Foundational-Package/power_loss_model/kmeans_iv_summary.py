import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Updated script from the original k-means.py
# Previously using m55_iv_el.txt in k-means.py, this one is intended to run on iv_summary.txt

def run_kmeans_iv_summary(
    iv_el_file_path: str,
    defect_percentages_file_path: str,
    metadata_file_path: str,
    output_dir: str,
):
    """
    Script to perform KMeans clustering on IV summary data combined with defect percentages and metadata:
      - Reads three inputs (IV summary, defects CSV, metadata)
      - Performs KMeans + PCA
      - Displays the PCA plot and also saves it
      - Saves merged results to output/clustered_iv_defects.csv

    Args:
        iv_el_file_path (str): path to iv_summary.txt (tab-separated)
        defect_percentages_file_path (str): path to M55Dataframe.csv
        metadata_file_path (str): path to m55_module_metadata.txt (tab-separated)
        output_dir (str): folder to write outputs (csv and png)
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    
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

    # Drop rows with any missing values in selected features
    features = merged_df[features_cols].dropna()

    if features.empty:
        raise ValueError("No valid rows left after dropping NaN and 0 values.")

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

    # Save PCA plot to output
    pca_path = output_dir / "kmeans_pca.png"
    plt.savefig(pca_path, bbox_inches="tight")
    plt.show()

    #Save results
    csv_path = output_dir / "clustered_iv_defects.csv"
    merged_df.to_csv(csv_path, index=False)

    # Confirmation print
    print(f"Saved clustered dataframe to: {csv_path}")
    print(f"Saved PCA plot to: {pca_path}")

    # Return paths for callers
    return {"csv_path": str(csv_path), "pca_plot": str(pca_path)}
