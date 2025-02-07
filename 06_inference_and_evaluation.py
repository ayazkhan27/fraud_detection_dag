# 06_inference_and_evaluation.py

import pandas as pd
import numpy as np
import pickle
from pgmpy.inference import VariableElimination
from sklearn.metrics import precision_recall_curve, auc, classification_report

def evaluate_model(model_path="bayesian_network_model.pkl", data_path="discretized_data.csv"):
    """Performs inference and evaluates the model's performance.

    Args:
        model_path (str): Path to the saved Bayesian network model (pickle file).
        data_path (str): Path to the discretized data CSV file.

    Returns:
        None (prints evaluation metrics)
    """

    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load the *original* discretized data (NOT the resampled data)
    df_original = pd.read_csv(data_path)
    df_original = df_original.drop('Amount', axis = 1) # Drop the Amount

    # Split into features (X) and target (y) - using the *original* data
    X_original = df_original.drop('Class', axis=1)
    y_original = df_original['Class']

    # Create inference engine
    infer = VariableElimination(model)

    # Make predictions on the original data
    predicted_probs = []
    for _, row in X_original.iterrows():
        evidence = row.to_dict()
        # Ensure that only discretized columns from the model are in evidence
        filtered_evidence = {k: v for k, v in evidence.items() if k in model.nodes}

        try:
            prob_fraud = infer.query(variables=['Class'], evidence=filtered_evidence).values[1]
            predicted_probs.append(prob_fraud)
        except ValueError as e:
            print(f"Error during inference: {e}")
            print(f"Problematic evidence: {filtered_evidence}")  # Print the problematic evidence
            # Instead of returning, append a default value (e.g., 0) and continue
            predicted_probs.append(0)
            continue # Skip to the next iteration

    # Convert to NumPy array for sklearn
    predicted_probs = np.array(predicted_probs)

    # Evaluate performance (AUPRC, classification report)
    precision, recall, thresholds = precision_recall_curve(y_original, predicted_probs)
    auprc = auc(recall, precision)
    print(f"AUPRC: {auprc:.4f}")

    # Use a threshold to generate class predictions (for classification report)
    threshold = 0.5
    predicted_classes = (predicted_probs >= threshold).astype(int)
    print("\nClassification Report (Threshold = 0.5):")
    print(classification_report(y_original, predicted_classes))
    return precision, recall, thresholds, auprc

if __name__ == '__main__':
    precision, recall, thresholds, auprc = evaluate_model()

        # Plot of the auprc
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve, AUPRC = {auprc:.4f}")
    plt.savefig("PR_Curve.png")
    print("\n Plotted PR Curve, and outputted as PR_Curve.png in root.")