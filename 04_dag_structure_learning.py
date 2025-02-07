# 04_dag_structure_learning.py

import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork

def learn_dag_structure(file_path="resampled_data.csv"):
    """Learns the DAG structure, restricting parent nodes using a whitelist/blacklist."""
    df = pd.read_csv(file_path)
    df = df.drop('Amount', axis=1)  # Drop continuous Amount

    # --- Whitelist ---
    all_nodes = list(df.columns)
    all_nodes_except_class = [node for node in all_nodes if node != 'Class']
    whitelist = []

    # Allow the most promising features to be parents of 'Class'
    promising_features = ['V1_disc', 'V3_disc', 'V7_disc', 'V10_disc', 'V12_disc',
                         'V14_disc', 'V16_disc', 'V17_disc','V18_disc', 'V20_disc', 'V21_disc',
                         'Amount_disc', 'Time_disc']
    for feature in promising_features:
        whitelist.append((feature, 'Class'))

    # Allow some inter-feature relationships (based on domain knowledge and to
    # allow for indirect effects). We'll allow Time to influence Amount.
    whitelist.append(('Time_disc', 'Amount_disc'))

    # We also allow the V features to have some relationships *among themselves*
    # This is a bit of a concession to the fact that we don't have the original
    # feature meanings, and they *are* the result of a PCA, which implies some
    # underlying relationships.  We keep it limited, though. We allow V1-V14 to potentially affect V15-V28
    for i in range(1, 15):  # V1 to V14
      for j in range(15, 29): #V15 to V28
        whitelist.append((f'V{i}_disc', f'V{j}_disc'))



    # Initialize HillClimbSearch
    hc = HillClimbSearch(df)

    # Run structure learning with BicScore and whitelist
    best_model = hc.estimate(scoring_method=BicScore(df), white_list=whitelist)

    # Create a BayesianNetwork object
    learned_dag = BayesianNetwork(best_model.edges())
    print("Learned DAG edges:")
    print(learned_dag.edges())

    return learned_dag

if __name__ == '__main__':
    learned_dag = learn_dag_structure()

    # Save the learned DAG structure
    with open("learned_dag_edges.txt", "w") as f:
        for edge in learned_dag.edges():
            f.write(f"{edge[0]} -> {edge[1]}\n")
    print("\nLearned DAG edges saved to learned_dag_edges.txt")