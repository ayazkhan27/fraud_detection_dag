# 03_resampling.py

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTENC

def resample_data(file_path="discretized_data.csv", preprocessed_path="preprocessed_data.csv"):
    """Resamples the data using SMOTE-NC, reintroducing 'Amount' as continuous."""

    df_disc = pd.read_csv(file_path)  # Discretized data
    df_preprocessed = pd.read_csv(preprocessed_path) #Original, Preprocessed data.

    # Re-integrate the continuous 'Amount' column
    df_disc['Amount'] = df_preprocessed['Amount']

    # Separate features (X) and target (y)
    X = df_disc.drop('Class', axis=1)
    y = df_disc['Class']

    # Identify categorical feature indices (all _disc columns + 'Amount')
    categorical_features = [i for i, col in enumerate(X.columns) if col.endswith('_disc')]

    # SMOTE-NC
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42, k_neighbors=min(5, y.value_counts().min() - 1))
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    print("Original class distribution:", np.bincount(y))
    print("Resampled class distribution:", np.bincount(y_resampled))

    return X_resampled, y_resampled

if __name__ == '__main__':
    X_resampled, y_resampled = resample_data()

    # Combine resampled features and target
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_resampled.columns), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)


    # Save resampled data
    df_resampled.to_csv("resampled_data.csv", index=False)
    print("\nResampled data saved to resampled_data.csv")

    # Print dtypes
    print("\nData types of resampled data:")
    for col in df_resampled.columns:
        print(f" {col}: {df_resampled[col].dtype}")