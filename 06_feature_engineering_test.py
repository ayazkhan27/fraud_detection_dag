# 06_feature_engineering_test.py

import pandas as pd
import pickle

def feature_engineer_test(X_test_path="X_test.csv",
                          discretization_info_path="discretization_info.pkl"):
    """Applies feature engineering to the test set using the learned discretization info.
    
    This version uses the saved bin edges from the fast discretization routine and does not perform one-hot encoding,
    keeping the representation consistent with training.
    """
    X_test = pd.read_csv(X_test_path)

    # --- Discretize ---
    features_to_discretize = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    with open(discretization_info_path, "rb") as f:
        discretization_info = pickle.load(f)
    
    for feature in features_to_discretize:
        # Retrieve the saved bin edges from training.
        bin_edges = discretization_info[feature]['bin_edges']
        labels = list(range(len(bin_edges) - 1))
        # Use pd.cut with the saved bin edges.
        # Fill NaN values with 0 to avoid conversion issues.
        X_test[f"{feature}_disc"] = pd.cut(X_test[feature],
                                           bins=bin_edges,
                                           labels=labels,
                                           include_lowest=True).fillna(0).astype(int)

    # --- Interaction Features ---
    # Load the learned Bayesian network model to get the parents of the 'Class' node.
    with open("bayesian_network_model.pkl", "rb") as f:
        model = pickle.load(f)
    parents_of_class = model.get_parents('Class')
    
    # Filter to only use base discretized features (i.e. skip those already labeled as interactions).
    base_parents = [p for p in parents_of_class if '_interaction' not in p]
    
    # Create interaction features by concatenating the string representations of each pair of base parents.
    for parent1 in base_parents:
        for parent2 in base_parents:
            if parent1 != parent2:
                interaction_col = f'{parent1}_{parent2}_interaction'
                # Only create the interaction if it doesn't already exist
                if interaction_col not in X_test.columns:
                    X_test[interaction_col] = X_test[parent1].astype(str) + "_" + X_test[parent2].astype(str)

    # --- Save the Feature-Engineered Test Data ---
    X_test.to_csv("feature_engineered_test_data_no_ohe.csv", index=False)
    print("\nFeature-engineered test data saved to feature_engineered_test_data.csv")
    
    return X_test

if __name__ == '__main__':
    X_test_engineered = feature_engineer_test()
    print(f"\nShape of engineered test data: {X_test_engineered.shape}")
    print(X_test_engineered.info())
