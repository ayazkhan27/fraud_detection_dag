# 09_causal_inference.py

import pickle
from pgmpy.inference import CausalInference
import pandas as pd

def perform_causal_inference(model_path="bayesian_network_model.pkl"):
    """Performs causal inference using the do-calculus.

    Args:
        model_path (str): Path to the saved Bayesian network model.

    Returns:
        pandas.DataFrame: Results of causal inference, suitable for analysis.
    """

    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Create a CausalInference object
    causal_infer = CausalInference(model)

    # --- Define Interventions and Estimate Causal Effects ---

    # Get the parents of 'Class' (for later comparison)
    parents_of_class = model.get_parents('Class')
    print(f"Parents of Class: {parents_of_class}")

    # 1. Baseline Probability of Fraud (without intervention)
    baseline_query = causal_infer.query(variables=['Class'])
    baseline_prob_fraud = baseline_query.values[1]
    print(f"\nBaseline P(Class=1): {baseline_prob_fraud:.4f}")

    # 2. Interventions on Key Features (Discretized)
    results = []  # Store results in a list of dictionaries
    interventions = {}
    for parent in parents_of_class:
        if parent != "Class": #Do not intervene on the class itself.
            for value in model.get_cpds(parent).state_names[parent]:
                interventions[(parent, value)] = value #Store for the loop.

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
        except ValueError as e:
            print(f"Error during causal inference for do({intervention}={value}): {e}")
            results.append({ #Append a value even if there is an error.
                'intervention': f'do({intervention[0]}={value})',
                'prob_fraud': None,
                'change_from_baseline': None
            })
            continue #Move on

    # Create a pandas DataFrame for easy analysis and display
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == '__main__':
    results_df = perform_causal_inference()
    print("\n--- Causal Inference Results ---")
    print(results_df)
    results_df.to_csv("causal_inference_results.csv", index=False)
    print("\nCausal inference results saved to causal_inference_results.csv")