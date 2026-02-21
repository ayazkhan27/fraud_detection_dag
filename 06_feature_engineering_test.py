# 06_feature_engineering_test.py

import pandas as pd
import pickle
from utils import load_config, setup_logger

logger = setup_logger(__name__)

def feature_engineer_test(config):
    """Applies feature engineering to the test set using learned discretization info."""
    
    in_path = config["paths"]["X_test"]
    out_path = config["paths"]["feature_engineered_test"]
    disc_info_path = config["paths"]["discretization_info"]
    model_path = config["paths"]["bayesian_network_model"]
    
    logger.info(f"Loading test features from {in_path}")
    X_test = pd.read_csv(in_path)

    # --- Discretize ---
    features_to_discretize = config["preprocessing"]["features_to_scale"]
    logger.info(f"Loading discretization bin edges from {disc_info_path}")
    
    with open(disc_info_path, "rb") as f:
        discretization_info = pickle.load(f)
    
    for feature in features_to_discretize:
        bin_edges = discretization_info[feature]['bin_edges']
        labels = list(range(len(bin_edges) - 1))
        
        X_test[f"{feature}_disc"] = pd.cut(
            X_test[feature],
            bins=bin_edges,
            labels=labels,
            include_lowest=True
        ).fillna(0).astype(int)

    # --- Interaction Features ---
    logger.info(f"Loading Bayesian Model to determine parents of Class from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    parents_of_class = model.get_parents('Class')
    base_parents = [p for p in parents_of_class if '_interaction' not in p]
    
    logger.info(f"Generating interaction features among base parents: {base_parents}")
    for parent1 in base_parents:
        for parent2 in base_parents:
            if parent1 != parent2:
                interaction_col = f'{parent1}_{parent2}_interaction'
                if interaction_col not in X_test.columns:
                    X_test[interaction_col] = X_test[parent1].astype(str) + "_" + X_test[parent2].astype(str)

    # --- Save Output ---
    X_test.to_csv(out_path, index=False)
    logger.info(f"Feature-engineered test data saved to {out_path} with shape {X_test.shape}")
    
    return X_test

if __name__ == '__main__':
    config = load_config()
    feature_engineer_test(config)
