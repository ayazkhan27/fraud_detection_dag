# 07_inference_and_evaluation.py

import pandas as pd
import numpy as np
import pickle
from pgmpy.inference import VariableElimination
from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt

def load_model(model_path="bayesian_network_model.pkl"):
    """Loads the Bayesian network model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def cast_evidence_value(model, variable, value):
    """
    Casts an evidence value to the appropriate type based on the CPD state names.
    It first checks the candidate value as-is, then its string version, then tries integer conversion.
    If none match, it defaults to the first valid state.
    """
    try:
        cpd = model.get_cpds(variable)
        if cpd is not None:
            states = cpd.state_names[variable]
            # Try value as-is.
            if value in states:
                return value
            # Try string version.
            if str(value) in states:
                return str(value)
            # Try integer conversion.
            try:
                int_val = int(value)
                if int_val in states:
                    return int_val
            except Exception:
                pass
            print(f"Warning: Value '{value}' for variable {variable} not found in valid states {states}. Defaulting to {states[0]}.")
            return states[0]
    except Exception as e:
        print(f"Error casting evidence for variable {variable} with value {value}: {e}")
    return value

def predict_probabilities(model, data):
    """Predicts fraud probabilities for given data using variable elimination."""
    infer = VariableElimination(model)
    predicted_probs = []

    # We want to use only the columns in data that are also nodes in the model.
    # If a model node is missing from the test data, we simply do not provide evidence for it.
    available_columns = list(set(model.nodes) & set(data.columns))
    # Exclude 'Class' because it is the target.
    if "Class" in available_columns:
        available_columns.remove("Class")
    
    # Subset the test data to only those available columns.
    evidence_data = data[available_columns]

    for _, row in evidence_data.iterrows():
        evidence = row.to_dict()
        # For each variable, cast its value robustly.
        filtered_evidence = {var: cast_evidence_value(model, var, val) for var, val in evidence.items()}
        try:
            # Query for the probability of Class = 1.
            query_result = infer.query(variables=['Class'], evidence=filtered_evidence)
            prob_fraud = query_result.values[1]
            predicted_probs.append(prob_fraud)
        except ValueError as e:
            print(f"Error during inference: {e}")
            print(f"Problematic evidence: {filtered_evidence}")
            predicted_probs.append(0)  # Append a default value if error occurs
            continue
    return np.array(predicted_probs)

def find_optimal_threshold(y_true, predicted_probs):
    """Finds the optimal threshold based on the F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, predicted_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, precision[optimal_idx], recall[optimal_idx], f1_scores[optimal_idx]

def evaluate_at_threshold(y_true, predicted_probs, threshold):
    """Evaluates performance at a specific threshold."""
    predicted_classes = (predicted_probs >= threshold).astype(int)
    print(f"\nClassification Report (Threshold = {threshold:.4f}):")
    print(classification_report(y_true, predicted_classes))

if __name__ == '__main__':
    model = load_model()
    
    # Load the feature-engineered test data and the true labels.
    X_test = pd.read_csv("feature_engineered_test_data_no_ohe.csv")
    y_test = pd.read_csv("y_test.csv")
    if "Class" in y_test.columns:
        y_test = y_test["Class"]

    predicted_probs = predict_probabilities(model, X_test)
    
    optimal_threshold, best_precision, best_recall, best_f1 = find_optimal_threshold(y_test, predicted_probs)
    print(f"\nOptimal Threshold (based on F1-score): {optimal_threshold:.4f}")
    print(f"Precision at Optimal Threshold: {best_precision:.4f}")
    print(f"Recall at Optimal Threshold: {best_recall:.4f}")
    print(f"F1-score at Optimal Threshold: {best_f1:.4f}")

    evaluate_at_threshold(y_test, predicted_probs, optimal_threshold)

    # Visualization is handled in a separate script.
