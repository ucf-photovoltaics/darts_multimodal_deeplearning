import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data (File we created via k-means_iv_summary.py)
df = pd.read_csv('clustered_iv_defects.csv')

# Calculate correlations
correlation_matrix = df[['Crack', 'Contact', 'Interconnect', 'Corrosion', 'Vmp_(V)', 'Voc_(V)', 'Pmp_(W)', 'Efficiency_(%)']].corr()

#Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Between Defects and Performance Metrics')
plt.show()
