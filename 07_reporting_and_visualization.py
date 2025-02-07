# 07_reporting_and_visualization.py

import pandas as pd
import numpy as np
import pickle
from pgmpy.inference import VariableElimination
from sklearn.metrics import precision_recall_curve, auc, classification_report
import matplotlib.pyplot as plt
import networkx as nx

def load_model(model_path="bayesian_network_model.pkl"):
    """Loads the Bayesian network model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_probabilities(model, data):
    """Predicts fraud probabilities for given data."""
    infer = VariableElimination(model)
    predicted_probs = []
    for _, row in data.iterrows():
        evidence = row.to_dict()
        filtered_evidence = {k: v for k, v in evidence.items() if k in model.nodes}
        try:
            prob_fraud = infer.query(variables=['Class'], evidence=filtered_evidence).values[1]
            predicted_probs.append(prob_fraud)
        except ValueError:
            predicted_probs.append(0)  # Handle potential errors
    return np.array(predicted_probs)

def find_optimal_threshold(y_true, predicted_probs):
    """Finds the optimal threshold based on the F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, predicted_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.nanargmax(f1_scores)  # Use nanargmax to handle potential NaN values
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, precision[optimal_idx], recall[optimal_idx], f1_scores[optimal_idx]

def evaluate_at_threshold(y_true, predicted_probs, threshold):
    """Evaluates performance at a specific threshold."""
    predicted_classes = (predicted_probs >= threshold).astype(int)
    print(f"\nClassification Report (Threshold = {threshold:.4f}):")
    print(classification_report(y_true, predicted_classes))

def visualize_dag(model, filename="bayesian_network.png"):
    """Visualizes the DAG."""
    nx_graph = nx.DiGraph()
    for u, v in model.edges():
        nx_graph.add_edge(u, v)

    pos = nx.spring_layout(nx_graph, seed=42)  # For consistent layout
    plt.figure(figsize=(16, 12))  # Larger figure size
    nx.draw(nx_graph, pos, with_labels=True, node_size=2000, node_color="skyblue",
            font_size=10, font_weight="bold", arrowsize=20, width=2)
    plt.title("Learned Bayesian Network Structure", fontsize=16)
    plt.savefig(filename)
    print(f"\nDAG visualization saved to {filename}")


if __name__ == '__main__':
    model = load_model()
    df_original = pd.read_csv("discretized_data.csv")
    df_original = df_original.drop('Amount', axis=1)
    X_original = df_original.drop('Class', axis=1)
    y_original = df_original['Class']

    predicted_probs = predict_probabilities(model, X_original)

    # Find optimal threshold
    optimal_threshold, best_precision, best_recall, best_f1 = find_optimal_threshold(y_original, predicted_probs)
    print(f"\nOptimal Threshold (based on F1-score): {optimal_threshold:.4f}")
    print(f"Precision at Optimal Threshold: {best_precision:.4f}")
    print(f"Recall at Optimal Threshold: {best_recall:.4f}")
    print(f"F1-score at Optimal Threshold: {best_f1:.4f}")

    # Evaluate at optimal threshold
    evaluate_at_threshold(y_original, predicted_probs, optimal_threshold)

    # Visualize DAG
    visualize_dag(model)