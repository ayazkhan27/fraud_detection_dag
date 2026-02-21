# 04_dag_structure_learning_train.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
import pickle
from tqdm import tqdm
from utils import load_config, setup_logger

logger = setup_logger(__name__)

def learn_dag_structure(config):
    """
    Learns the DAG structure using a robust whitelist and Tabu Search to escape local optima.
    """
    train_file_path = config["paths"]["resampled_train"]
    logger.info(f"Loading resampled data from {train_file_path}")
    
    df = pd.read_csv(train_file_path)
    if 'Amount' in df.columns:
        df = df.drop('Amount', axis=1)

    max_unique = config["feature_engineering"]["max_unique_cardinality"]
    cols_to_keep = [col for col in df.columns if col == 'Class' or df[col].nunique() <= max_unique]
    df = df[cols_to_keep]
    all_nodes = list(df.columns)
    logger.info(f"Filtered nodes with cardinality <= {max_unique}: {len(all_nodes)}")

    prom_path = config["paths"]["promising_features"]
    with open(prom_path, "rb") as f:
        promising_features = pickle.load(f)
    promising_features = [feat for feat in promising_features if feat in all_nodes]

    correlation_threshold = config["feature_engineering"]["correlation_threshold"]
    non_interaction_features = [feat for feat in promising_features if not feat.endswith('_interaction')]
    
    if non_interaction_features:
        corr_matrix = df[non_interaction_features].corr()
        upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
        pruned_interactions = []
        for i, j in tqdm(zip(*upper_tri_indices), total=len(upper_tri_indices[0]), desc="Evaluating feature correlations"):
            feat1, feat2 = non_interaction_features[i], non_interaction_features[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= correlation_threshold:
                interaction_name = f"{feat1}_{feat2}_interaction"
                if interaction_name in all_nodes:
                    pruned_interactions.append(interaction_name)
                    
        for inter in pruned_interactions:
            if inter not in promising_features:
                promising_features.append(inter)

    whitelist = []
    for feature in promising_features:
        if feature in all_nodes:
            whitelist.append((feature, 'Class'))
            
    if 'Time_disc' in all_nodes and 'Amount_disc' in all_nodes:
        whitelist.append(('Time_disc', 'Amount_disc'))
        
    for i, feat1 in enumerate(non_interaction_features):
        for feat2 in non_interaction_features[i+1:]:
            interaction_name = f"{feat1}_{feat2}_interaction"
            if interaction_name in all_nodes:
                whitelist.append((interaction_name, 'Class'))
                whitelist.append((feat1, interaction_name))
                whitelist.append((feat2, interaction_name))

    initial_dag = BayesianNetwork()
    initial_dag.add_nodes_from(all_nodes)
    
    tabu_length = config["dag_learning"]["tabu_length"]
    max_iter = config["dag_learning"]["max_iter"]
    
    logger.info(f"Starting HillClimbSearch (tabu_length={tabu_length}, max_iter={max_iter})")
    hc = HillClimbSearch(df)
    
    # HillClimbSearch natively takes some 'show_progress' argument kwargs usually, checking it inline
    logger.info("Executing Search Algorithm. This is an intensive process and might run for several minutes...")
    
    best_model = hc.estimate(
        scoring_method=BicScore(df),
        white_list=whitelist,
        start_dag=initial_dag,
        tabu_length=tabu_length,
        max_iter=max_iter,
        show_progress=True # Enable internal progress bar from pgmpy if supported
    )
    
    learned_dag = BayesianNetwork(best_model.edges())

    for u, v in list(learned_dag.edges()):
        if "_interaction" in u or "_interaction" in v:
            if not learned_dag.has_node(u): learned_dag.add_node(u)
            if not learned_dag.has_node(v): learned_dag.add_node(v)
            if not learned_dag.has_edge(u, v): learned_dag.add_edge(u, v)

    logger.info(f"DAG Learning Complete. Found {len(learned_dag.edges())} edges.")

    edges_path = config["paths"]["learned_dag_edges"]
    model_path = config["paths"]["bayesian_network_model"]
    
    with open(edges_path, "w") as f:
        for edge in learned_dag.edges():
            f.write(f"{edge[0]} -> {edge[1]}\n")
            
    with open(model_path, "wb") as f:
        pickle.dump(learned_dag, f)
        
    logger.info("Successfully exported DAG artifacts.")
    return learned_dag

if __name__ == '__main__':
    config = load_config()
    learn_dag_structure(config)
