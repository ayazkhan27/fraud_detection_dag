# 03_feature_engineering_resampling_train.py

import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTENC
import pickle
from scipy.stats import chi2_contingency

def fast_discretize(feature_values, num_bins=5):
    """
    Discretizes a feature using quantile-based binning (pd.qcut) for fast computation.
    This is fully data-driven and domain-agnostic.
    
    Args:
        feature_values (pd.Series): The continuous feature to discretize.
        num_bins (int): The target number of bins.
    
    Returns:
        pd.Series: Discretized values as integers.
        dict: A dictionary containing the bin edges (for test time).
    """
    try:
        # Use pd.qcut to create 'num_bins' quantile bins.
        # duplicates='drop' handles cases where there aren’t enough unique values.
        discretized, bin_edges = pd.qcut(feature_values, q=num_bins, duplicates='drop', retbins=True, labels=False)
    except Exception as e:
        # If qcut fails (should rarely happen), fall back to pd.cut with equally spaced bins.
        discretized, bin_edges = pd.cut(feature_values, bins=num_bins, retbins=True, labels=False)
    
    # Save the bin edges for later use in test-time transformation.
    info = {"num_bins": num_bins, "bin_edges": bin_edges.tolist()}
    return discretized, info

def create_interaction_features(df, parents_of_class):
    """
    Creates interaction features by string concatenation.
    Since the number of parents is small, this nested loop is not computationally heavy.
    """
    for parent1 in parents_of_class:
        for parent2 in parents_of_class:
            if parent1 != parent2:
                # Create a new column by concatenating the string representation of the two features.
                df[f'{parent1}_{parent2}_interaction'] = df[parent1].astype(str) + "_" + df[parent2].astype(str)
    return df

def feature_engineering_and_resampling(X_train_path="X_train.csv", y_train_path="y_train.csv", num_bins=5):
    """
    Performs feature engineering and resampling on the training data.
    This version replaces the heavy KMeans-based discretization with fast, quantile-based discretization.
    
    Steps:
      1. Read training data.
      2. For each feature to discretize, apply pd.qcut.
      3. Create interaction features for a predefined set of parent features.
      4. Run chi-squared tests between each discretized/interaction feature and the target to select promising features.
      5. Resample the training data using SMOTENC.
      6. Save the promising features list and discretization information.
    
    Returns:
      None (saves processed files to disk).
    """
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    df_train = pd.concat([X_train, y_train], axis=1)
    
    # List of features to discretize
    features_to_discretize = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    discretization_info = {}

    # Replace heavy clustering-based discretization with fast quantile binning.
    for feature in features_to_discretize:
        discretized, info = fast_discretize(df_train[feature], num_bins=num_bins)
        df_train[f"{feature}_disc"] = discretized
        # Save any extra info you might need later (e.g., for transforming test data)
        discretization_info[feature] = info
        print(f"Feature {feature}: Discretized into {info['num_bins']} bins with edges: {info['bin_edges']}")
    
    # --- Interaction Features (Using a small, predefined set of parents) ---
    placeholder_parents = ['V1_disc', 'V7_disc', 'V12_disc', 'V14_disc', 'Time_disc']
    df_train = create_interaction_features(df_train, placeholder_parents)
    
    # --- Chi-squared Tests (After discretization and interactions, Before SMOTE) ---
    p_value_threshold = 0.05
    promising_features = []
    # Evaluate each discretized and interaction feature (except Amount_disc is always kept)
    for col in df_train.columns:
        if (col.endswith('_disc') or col.endswith('_interaction')) and col != 'Amount_disc':
            contingency_table = pd.crosstab(df_train[col], df_train['Class'])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            print(f"Chi-squared test for {col}: p-value = {p:.4f}")
            if p < p_value_threshold:
                promising_features.append(col)
                print(f"Adding {col} to promising_features (p-value: {p:.4f})")
    promising_features.append("Amount_disc")  # Always include Amount_disc
    print(f"Promising features (based on chi-squared with Class): {promising_features}")

    # --- Save Promising Features and Discretization Info ---
    with open("promising_features.pkl", "wb") as f:
        pickle.dump(promising_features, f)
    print("Promising features list saved to promising_features.pkl")
    
    with open("discretization_info.pkl", "wb") as f:
        pickle.dump(discretization_info, f)
    print("Discretization information saved to discretization_info.pkl")
    
    # --- SMOTE-NC (After discretization and interactions, BEFORE any encoding) ---
    X_train_processed = df_train.drop('Class', axis=1)
    y_train_processed = df_train['Class']
    categorical_features = [i for i, col in enumerate(X_train_processed.columns) if col.endswith(('_disc', '_interaction'))]
    
    # SMOTENC to address class imbalance; note that fewer features means faster SMOTE.
    from imblearn.over_sampling import SMOTENC  # Ensure import is here
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42,
                       k_neighbors=min(5, y_train_processed.value_counts().iloc[-1] - 1))
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train_processed, y_train_processed)
    
    print("Original class distribution:", np.bincount(y_train_processed))
    print("Resampled class distribution:", np.bincount(y_train_resampled))
    
    # Convert back to DataFrame (SMOTE outputs numpy arrays)
    df_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train_processed.columns)
    df_train_resampled['Class'] = y_train_resampled
    
    # --- Save the Feature-Engineered and Resampled Training Data ---
    df_train_resampled.to_csv("resampled_train_data_no_ohe.csv", index=False)
    print("\nFeature-engineered and resampled train data saved to resampled_train_data_no_ohe.csv")

if __name__ == '__main__':
    feature_engineering_and_resampling()
