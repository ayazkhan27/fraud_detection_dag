# 05_parameter_learning.py

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

def learn_parameters(dag_edges_file="learned_dag_edges.txt", data_file="resampled_data.csv"):
    """Learns the parameters (CPDs) of the Bayesian network.

    Args:
        dag_edges_file (str): Path to the file containing the DAG edges.
        data_file (str): Path to the resampled data CSV file.

    Returns:
        pgmpy.models.BayesianNetwork: The Bayesian network with learned parameters.
    """
    # Reconstruct the DAG structure from the edges file
    edges = []
    with open(dag_edges_file, "r") as f:
        for line in f:
            u, v = line.strip().replace("'", "").split(" -> ")
            edges.append((u, v))
    model = BayesianNetwork(edges)

    # Load the resampled data
    df = pd.read_csv(data_file)
    df = df.drop('Amount', axis=1) #Drop nondiscretized Amount.


    # Learn the parameters using Maximum Likelihood Estimation
    model.fit(df, estimator=MaximumLikelihoodEstimator)

    # Print the CPDs (for inspection)
    for cpd in model.get_cpds():
        print(f"CPD for {cpd.variable}:")
        print(cpd)

    return model

if __name__ == '__main__':
    model_with_parameters = learn_parameters()

    # Save the model with parameters.  We'll use pickle for this.
    import pickle
    with open("bayesian_network_model.pkl", "wb") as f:
        pickle.dump(model_with_parameters, f)
    print("\nBayesian network model with parameters saved to bayesian_network_model.pkl")