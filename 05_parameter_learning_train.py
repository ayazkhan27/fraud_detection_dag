# 05_parameter_learning_train.py

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator # Switched from MaximumLikelihoodEstimator
import pickle
from utils import load_config, setup_logger

logger = setup_logger(__name__)

def learn_parameters(config):
    """Learns parameters (CPDs) using a Bayesian Estimator with a Dirichlet prior."""
    
    dag_edges_file = config["paths"]["learned_dag_edges"]
    data_file = config["paths"]["resampled_train"]
    model_out = config["paths"]["bayesian_network_model"]
    
    logger.info(f"Reconstructing DAG from {dag_edges_file}")
    edges = []
    with open(dag_edges_file, "r") as f:
        for line in f:
            u, v = line.strip().replace("'", "").split(" -> ")
            edges.append((u, v))
    model = BayesianNetwork(edges)

    logger.info(f"Loading data {data_file}")
    df = pd.read_csv(data_file)
    if 'Amount' in df.columns:
        df = df.drop('Amount', axis=1)

    # Learn the parameters using Bayesian Estimator (BDeu prior)
    # This prevents the 0-probability assignments for unseen edge cases mathematically
    equiv_sample_size = config["dag_learning"]["bdeu_equivalent_sample_size"]
    logger.info(f"Fitting parameters using BayesianEstimator (BDeu, equiv_sample_size={equiv_sample_size})")
    
    model.fit(
        df, 
        estimator=BayesianEstimator, 
        prior_type="BDeu", 
        equivalent_sample_size=equiv_sample_size
    )

    # Save the model
    with open(model_out, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Bayesian network model with calibrated CPDs saved to {model_out}")

    return model

if __name__ == '__main__':
    config = load_config()
    learn_parameters(config)
