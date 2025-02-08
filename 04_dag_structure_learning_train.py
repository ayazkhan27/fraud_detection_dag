# 04_dag_structure_learning_train.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
import pickle

def learn_dag_structure(train_file_path="resampled_train_data_no_ohe.csv"):
    """
    Learns the DAG structure using a whitelist based on statistically selected promising features.
    It uses vectorized correlation-based pruning for interaction features and filters out
    high-cardinality columns to avoid huge contingency tables.
    Instead of using the PC algorithm for initialization, we skip it and initialize HillClimbSearch
    from an empty network (with all nodes), relying on the robust whitelist.
    """

    # 1. Load the discretized, resampled data (without one-hot encoding)
    df = pd.read_csv(train_file_path)
    df = df.drop('Amount', axis=1)  # Drop the continuous 'Amount' column

    # 2. Filter out high-cardinality columns (except 'Class') to avoid huge contingency tables.
    max_unique_threshold = 50  # Adjust as needed
    cols_to_keep = [col for col in df.columns if col == 'Class' or df[col].nunique() <= max_unique_threshold]
    df = df[cols_to_keep]
    all_nodes = list(df.columns)

    # 3. Load promising features (from Script 03) and retain only those present in the filtered data.
    with open("promising_features.pkl", "rb") as f:
        promising_features = pickle.load(f)
    promising_features = [feat for feat in promising_features if feat in all_nodes]
    print(f"Promising features (loaded and filtered): {promising_features}")

    # 4. --- Vectorized Correlation-Based Interaction Pruning ---
    # For non-interaction promising features (discretized numeric columns), compute the correlation matrix.
    correlation_threshold = 0.4
    non_interaction_features = [feat for feat in promising_features if not feat.endswith('_interaction')]
    if non_interaction_features:
        corr_matrix = df[non_interaction_features].corr()
        # Get the indices for the upper triangle (excluding the diagonal)
        upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
        pruned_interactions = []
        for i, j in zip(*upper_tri_indices):
            feat1 = non_interaction_features[i]
            feat2 = non_interaction_features[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= correlation_threshold:
                interaction_name = f"{feat1}_{feat2}_interaction"
                if interaction_name in all_nodes:
                    pruned_interactions.append(interaction_name)
                    print(f"Adding interaction {interaction_name} (corr={corr_value:.2f})")
        # Extend promising features with these pruned interactions if not already present
        for inter in pruned_interactions:
            if inter not in promising_features:
                promising_features.append(inter)
    print(f"Updated promising features: {promising_features}")

    # 5. --- Whitelist Construction ---
    whitelist = []
    # (a) For each promising feature, add an edge from the feature to 'Class'
    for feature in promising_features:
        if feature in all_nodes:
            whitelist.append((feature, 'Class'))
    # (b) Allow 'Time_disc' to influence 'Amount_disc' if both exist
    if 'Time_disc' in all_nodes and 'Amount_disc' in all_nodes:
        whitelist.append(('Time_disc', 'Amount_disc'))
    # (c) For each pair of non-interaction promising features, if an interaction exists, add edges:
    for i, feat1 in enumerate(non_interaction_features):
        for feat2 in non_interaction_features[i+1:]:
            interaction_name = f"{feat1}_{feat2}_interaction"
            if interaction_name in all_nodes:
                whitelist.append((interaction_name, 'Class'))
                whitelist.append((feat1, interaction_name))
                whitelist.append((feat2, interaction_name))
    print(f"Whitelist: {whitelist}")  # DEBUG

    # 6. --- DAG Learning via HillClimbSearch Only ---
    # Initialize an empty Bayesian network and add all nodes.
    initial_dag = BayesianNetwork()
    initial_dag.add_nodes_from(all_nodes)
    
    # Run HillClimbSearch using the robust whitelist starting from the empty network.
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=BicScore(df),
                             white_list=whitelist,
                             start_dag=initial_dag,
                             max_iter=500)
    learned_dag = BayesianNetwork(best_model.edges())

    # Post-processing: ensure that interaction feature nodes and edges are present.
    for u, v in list(learned_dag.edges()):
        if "_interaction" in u or "_interaction" in v:
            if not learned_dag.has_node(u):
                learned_dag.add_node(u)
            if not learned_dag.has_node(v):
                learned_dag.add_node(v)
            if not learned_dag.has_edge(u, v):
                learned_dag.add_edge(u, v)

    print("Learned DAG edges:")
    print(learned_dag.edges())
    print("Nodes in the learned DAG:")
    print(learned_dag.nodes())

    return learned_dag

if __name__ == '__main__':
    learned_dag = learn_dag_structure()

    # Save the learned DAG structure to a text file.
    with open("learned_dag_edges.txt", "w") as f:
        for edge in learned_dag.edges():
            f.write(f"{edge[0]} -> {edge[1]}\n")
    print("\nLearned DAG edges saved to learned_dag_edges.txt")

    # Save the Bayesian Network model using pickle.
    with open("bayesian_network_model.pkl", "wb") as f:
        pickle.dump(learned_dag, f)
    print("Bayesian Network Model Saved")
