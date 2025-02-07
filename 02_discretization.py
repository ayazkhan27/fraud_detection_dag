# 02_discretization.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.utils import resample

def improved_cluster_discretize(feature_values, k_range=range(2, 7)):  # Limit k_range
    """Discretizes a feature using clustering, with outlier handling and sampling."""
    X = feature_values.values.reshape(-1, 1)

    # --- Robust Outlier Handling ---
    Q1 = np.quantile(X, 0.25)
    Q3 = np.quantile(X, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outlier_indices_lower = np.where(X < lower_bound)[0]
    outlier_indices_upper = np.where(X > upper_bound)[0]

    max_outlier_percentage = 0.10
    if len(outlier_indices_lower) / len(X) > max_outlier_percentage:
        lower_bound = np.quantile(X, max_outlier_percentage)
        outlier_indices_lower = np.where(X < lower_bound)[0]

    if len(outlier_indices_upper) / len(X) > max_outlier_percentage:
        upper_bound = np.quantile(X, 1 - max_outlier_percentage)
        outlier_indices_upper = np.where(X > upper_bound)[0]

    X_no_outliers = X[(X >= lower_bound) & (X <= upper_bound)]
    X_no_outliers = X_no_outliers.reshape(-1, 1)

    print(f"    Shape of X_no_outliers: {X_no_outliers.shape}")

    best_score = -1
    best_k = None
    best_labels = None

    k_range = [k for k in k_range if k <= len(X_no_outliers) // 2]
    if not k_range:
        k_range = [1]

    for k in k_range:
        print(f"    Trying k = {k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        if len(X_no_outliers) >= k:
            labels = kmeans.fit_predict(X_no_outliers)

            if len(np.unique(labels)) > 1:
                # --- Sampling for Silhouette Score ---
                sample_size = min(10000, len(X_no_outliers))  # Sample at most 10,000 points
                X_sample = resample(X_no_outliers, n_samples=sample_size, random_state=42)
                labels_sample = kmeans.predict(X_sample) #Predict based on fit.
                score = silhouette_score(X_sample, labels_sample)
            else:
                score = -1
            print(f"      Silhouette score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        else:
            print(f"    Skipping k={k} (not enough data points)")
            continue

    # Assign outlier labels
    final_labels = np.zeros(len(X), dtype=int)

    if best_labels is not None:
        final_labels[(X[:, 0] >= lower_bound) & (X[:, 0] <= upper_bound)] = best_labels + 1 #Fix is here.

    final_labels[outlier_indices_lower] = 0
    final_labels[outlier_indices_upper] = best_k + 1 if best_k is not None else 1 if len(k_range) > 0 else 0

    outlier_bins = {"lower": 0, "upper": best_k + 1 if best_k is not None else 1 if len(k_range) > 0 else 0}

    return final_labels, best_k, best_score, outlier_bins

if __name__ == '__main__':
    df = pd.read_csv("preprocessed_data.csv")
    features_to_discretize = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    discretization_info = {}

    for feature in features_to_discretize:
        print(f"Discretizing feature: {feature}")
        labels, optimal_k, score, outlier_bins = improved_cluster_discretize(df[feature])
        df[f"{feature}_disc"] = labels
        discretization_info[feature] = {
            "optimal_k": optimal_k,
            "silhouette_score": score,
            "outlier_bins": outlier_bins
        }
        print(f"Feature {feature}: Optimal clusters = {optimal_k}, Silhouette Score = {score:.3f}, Outlier Bins = {outlier_bins}")

    print("\nDiscretization Information:")
    print(discretization_info)

    df.to_csv("discretized_data.csv", index=False)
    print("\nDiscretized data saved to discretized_data.csv")

    print("\nValue Counts for Discretized Features:")
    for feature in features_to_discretize:
        print(f"\n--- {feature}_disc ---")
        print(df[f"{feature}_disc"].value_counts())

    print("\n--- Cross-Tabulation of Original Class vs. Discretized Amount (Outlier Check) ---")
    print(pd.crosstab(df['Class'], df['Amount_disc']))

    for feature in features_to_discretize:
        if feature != 'Amount':
            print(f"\n--- Cross-Tabulation of Original Class vs. Discretized {feature} (Outlier Check) ---")
            print(pd.crosstab(df['Class'], df[f'{feature}_disc']))