# 03_feature_engineering_resampling_train.py

import pandas as pd
import numpy as np
import pickle
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from imblearn.over_sampling import SMOTENC
from utils import load_config, setup_logger
from tqdm import tqdm

logger = setup_logger(__name__)

def fast_discretize(feature_values, num_bins=5):
    """Discretizes a feature using quantile-based binning."""
    try:
        discretized, bin_edges = pd.qcut(feature_values, q=num_bins, duplicates='drop', retbins=True, labels=False)
    except Exception as e:
        discretized, bin_edges = pd.cut(feature_values, bins=num_bins, retbins=True, labels=False)
    return discretized, {"num_bins": num_bins, "bin_edges": bin_edges.tolist()}

def create_interaction_features(df, parents_of_class):
    """Creates interaction features by string concatenation."""
    for parent1 in tqdm(parents_of_class, desc="Creating Interactions"):
        for parent2 in parents_of_class:
            if parent1 != parent2:
                col_name = f'{parent1}_{parent2}_interaction'
                if col_name not in df:
                    df[col_name] = df[parent1].astype(str) + "_" + df[parent2].astype(str)
    return df

def feature_engineering_and_resampling(config):
    X_train_path = config["paths"]["X_train"]
    y_train_path = config["paths"]["y_train"]
    
    logger.info("Loading training splits...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    df_train = pd.concat([X_train, y_train], axis=1)
    
    # 1. Discretization
    features_to_discretize = config["preprocessing"]["features_to_scale"]
    num_bins = config["feature_engineering"]["num_bins"]
    discretization_info = {}

    logger.info(f"Discretizing {len(features_to_discretize)} features...")
    for feature in tqdm(features_to_discretize, desc="Discretizing features"):
        discretized, info = fast_discretize(df_train[feature], num_bins=num_bins)
        df_train[f"{feature}_disc"] = discretized
        discretization_info[feature] = info
    
    # 2. Interaction Features
    interaction_parents = config["feature_engineering"]["interaction_parents_candidates"]
    logger.info("Generating interaction terms...")
    df_train = create_interaction_features(df_train, interaction_parents)
    
    # 3. Feature Selection: Chi-squared AND Mutual Information
    p_value_threshold = config["feature_engineering"]["p_value_threshold"]
    mi_threshold = config["feature_engineering"]["mutual_information_threshold"]
    promising_features = set()
    
    candidate_cols = [col for col in df_train.columns if (col.endswith('_disc') or col.endswith('_interaction')) and col != 'Amount_disc']
    logger.info("Evaluating feature importance metrics (Chi2 and MI)...")
    
    for col in tqdm(candidate_cols, desc="Evaluating Features"):
        contingency_table = pd.crosstab(df_train[col], df_train['Class'])
        _, p, _, _ = chi2_contingency(contingency_table)
        mi_score = mutual_info_score(df_train[col], df_train['Class'])
        
        if p < p_value_threshold and mi_score > mi_threshold:
            promising_features.add(col)

    promising_features.add("Amount_disc")
    promising_features_list = list(promising_features)
    logger.info(f"Total promising features selected: {len(promising_features_list)}")

    prom_out = config["paths"]["promising_features"]
    disc_out = config["paths"]["discretization_info"]
    with open(prom_out, "wb") as f: pickle.dump(promising_features_list, f)
    with open(disc_out, "wb") as f: pickle.dump(discretization_info, f)
    logger.info("Artifacts saved.")
    
    # 4. SMOTE-NC Resampling
    X_train_processed = df_train.drop('Class', axis=1)
    y_train_processed = df_train['Class']
    categorical_features = [i for i, col in enumerate(X_train_processed.columns) if col.endswith(('_disc', '_interaction'))]
    
    k_neighbors = min(config["feature_engineering"]["smote_k_neighbors"], y_train_processed.value_counts().iloc[-1] - 1)
    logger.info(f"Initializing SMOTENC (k={k_neighbors})... This might take a few minutes. Please wait.")
    
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=config["pipeline"]["random_state"], k_neighbors=k_neighbors)
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train_processed, y_train_processed)
    
    logger.info(f"Original class distribution: {np.bincount(y_train_processed)}")
    logger.info(f"Resampled class distribution: {np.bincount(y_train_resampled)}")
    
    df_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train_processed.columns)
    df_train_resampled['Class'] = y_train_resampled
    
    out_resampled = config["paths"]["resampled_train"]
    logger.info("Saving resampled training data... (in chunks)")
    
    chunk_size = 50000
    num_chunks = (len(df_train_resampled) // chunk_size) + 1
    
    header = True
    for i, chunk in enumerate(tqdm(np.array_split(df_train_resampled, num_chunks), total=num_chunks, desc="Saving Resampled CSV")):
        chunk.to_csv(out_resampled, mode='a' if not header else 'w', index=False, header=header)
        header = False
        
    logger.info(f"Saved resampled training data to {out_resampled}")

if __name__ == '__main__':
    config = load_config()
    feature_engineering_and_resampling(config)
