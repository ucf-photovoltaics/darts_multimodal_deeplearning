import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def run_heatmap(file_path: str, output_dir: str):
    """
    Generate correlation heatmap between defects and performance metrics.
    Saves both the heatmap figure and the correlation matrix to output_dir.
    Also displays the heatmap inline.

    Args:
        file_path (str): Path to clustered_iv_defects.csv
        output_dir (str): Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(file_path)

    # Calculate correlations
    correlation_matrix = df[['Crack', 'Contact', 'Interconnect', 'Corrosion',
                             'Vmp_(V)', 'Voc_(V)', 'Imp_(A)', 'Isc_(A)',
                             'Pmp_(W)', 'Efficiency_(%)', 'FF']].corr()

    # Save correlation matrix to text file
    results_file = output_dir / "heatmap_results.txt"
    correlation_matrix.to_string(open(results_file, "w"))
    print(f"Correlation matrix saved to: {results_file}")

    # Plotting the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Defects and Performance Metrics')

    # Save the figure as PNG
    fig_path = output_dir / "heatmap.png"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Heatmap image saved to: {fig_path}")

    # Display inline in Marimo
    plt.show()
