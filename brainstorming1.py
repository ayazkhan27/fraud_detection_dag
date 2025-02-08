import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load your training data; adjust the filename if needed.
df = pd.read_csv("resampled_train_data_no_ohe.csv")

# -------------------------------------------------------------------
# 1. Plot the distribution of absolute Pearson correlations among base discretized features
# -------------------------------------------------------------------

# Define the base features as those ending with '_disc' (but not including 'Amount_disc')
base_features = [col for col in df.columns if col.endswith('_disc') and col != 'Amount_disc']

# Compute the absolute correlation matrix for the base features
corr_matrix = df[base_features].corr().abs()

# Extract the upper triangle values (excluding the diagonal)
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
correlations = corr_matrix.where(mask).stack().values

# -------------------------------------------------------------------
# 2. Plot the distribution of p-values from chi-squared tests for candidate features
# -------------------------------------------------------------------

# Define candidate features as those ending with '_disc' or '_interaction' (excluding 'Amount_disc')
candidate_features = [col for col in df.columns 
                      if (col.endswith('_disc') or col.endswith('_interaction')) 
                      and col != 'Amount_disc']

p_values = []
for col in candidate_features:
    contingency_table = pd.crosstab(df[col], df['Class'])
    # Calculate chi-squared test statistics; p is the p-value.
    _, p, _, _ = chi2_contingency(contingency_table)
    p_values.append(p)

# -------------------------------------------------------------------
# Plot both histograms side by side
# -------------------------------------------------------------------
plt.figure(figsize=(14, 6))

# Left subplot: Distribution of absolute correlations
plt.subplot(1, 2, 1)
plt.hist(correlations, bins=30, edgecolor='black')
plt.xlabel("Absolute Pearson Correlation")
plt.ylabel("Frequency")
plt.title("Distribution of Absolute Correlations\namong Base Discretized Features")

# Right subplot: Distribution of chi-squared p-values
plt.subplot(1, 2, 2)
plt.hist(p_values, bins=30, edgecolor='black')
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.title("Distribution of Chi-Square p-values\nfor Candidate Features")

plt.tight_layout()
plt.show()
