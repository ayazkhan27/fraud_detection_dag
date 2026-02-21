# 09_causal_inference.py

import pickle
from pgmpy.inference import CausalInference
import pandas as pd
from utils import load_config, setup_logger

logger = setup_logger(__name__)

def perform_causal_inference(config):
    """Performs causal inference using the do-calculus."""

    model_path = config["paths"]["bayesian_network_model"]
    logger.info(f"Loading trained DAG from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Create a CausalInference object
    logger.info("Initializing Causal Inference engine (do-calculus)")
    causal_infer = CausalInference(model)

    # --- Define Interventions and Estimate Causal Effects ---
    parents_of_class = model.get_parents('Class')
    logger.info(f"Identified parents of Class: {parents_of_class}")

    # 1. Baseline Probability of Fraud
    baseline_query = causal_infer.query(variables=['Class'])
    baseline_prob_fraud = baseline_query.values[1]
    logger.info(f"Baseline P(Class=1): {baseline_prob_fraud:.4f}")

    # 2. Interventions
    results = []
    interventions = {}
    for parent in parents_of_class:
        if parent != "Class":
            for value in model.get_cpds(parent).state_names[parent]:
                interventions[(parent, value)] = value

    logger.info(f"Evaluating {len(interventions)} possible causal interventions...")
    for intervention, value in interventions.items():
        try:
            causal_effect = causal_infer.query(variables=['Class'], do={intervention[0]: intervention[1]})
            prob_fraud = causal_effect.values[1]
            change_in_prob = prob_fraud - baseline_prob_fraud
            results.append({
                'intervention': f'do({intervention[0]}={value})',
                'prob_fraud': prob_fraud,
                'change_from_baseline': change_in_prob
            })
        except Exception as e:
            logger.debug(f"Error during causal int. for do({intervention}={value}): {e}")
            results.append({
                'intervention': f'do({intervention[0]}={value})',
                'prob_fraud': None,
                'change_from_baseline': None
            })

    results_df = pd.DataFrame(results)
    
    out_path = config["paths"]["causal_inference_results"]
    results_df.to_csv(out_path, index=False)
    logger.info(f"Causal inference complete. Results saved to {out_path}")
    
    return results_df

if __name__ == '__main__':
    config = load_config()
    results_df = perform_causal_inference(config)
    print("\n--- Causal Inference Summary ---")
    print(results_df.dropna().head(10)) # Print top valid results