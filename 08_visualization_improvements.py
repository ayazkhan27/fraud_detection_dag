# 08_visualization_improvements.py

import os
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFWriter
from networkx.drawing.nx_pydot import graphviz_layout

# Ensure Graphviz is in PATH (fixes common import issue)
os.environ["PATH"] += os.pathsep + "/usr/bin/"

# Try importing pydot, handle potential import failure
try:
    import pydot
except ImportError:
    print("Warning: pydot is not installed properly. Ensure 'pydot' and 'graphviz' are installed.")


def load_model(model_path="bayesian_network_model.pkl"):
    """Loads the Bayesian network model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def visualize_dag(model, save_path="improved_bayesian_network.png"):
    """Generates an improved DAG visualization of the Bayesian Network."""
    nx_graph = nx.DiGraph(model.edges())

    try:
        pos = graphviz_layout(nx_graph, prog="neato")  # "neato" optimizes node placement
    except:
        print("Warning: Falling back to spring layout for DAG visualization.")
        pos = nx.spring_layout(nx_graph)

    plt.figure(figsize=(12, 8))
    nx.draw(nx_graph, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="gray", font_size=10, font_weight="bold")
    plt.title("Bayesian Network Structure (Improved)")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Improved DAG visualization saved to {save_path}")


def visualize_markov_blanket(model, target_variable="Class", save_path="markov_blanket.png"):
    """Visualizes the Markov Blanket of the target variable."""
    blanket = model.get_markov_blanket(target_variable)
    subgraph = nx.DiGraph()

    for node in blanket + [target_variable]:
        subgraph.add_node(node)

    for edge in model.edges():
        if edge[0] in blanket or edge[1] in blanket or edge[0] == target_variable or edge[1] == target_variable:
            subgraph.add_edge(*edge)

    try:
        pos = graphviz_layout(subgraph, prog="neato")
    except:
        pos = nx.spring_layout(subgraph)

    plt.figure(figsize=(10, 6))
    nx.draw(subgraph, pos, with_labels=True, node_size=2500, node_color="lightcoral", edge_color="black", font_size=10, font_weight="bold")
    plt.title(f"Markov Blanket of '{target_variable}'")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Markov Blanket visualization saved to {save_path}")


def compute_auprc(y_true, predicted_probs, save_path="auprc_curve.png"):
    """Computes and visualizes the Precision-Recall curve with AUPRC."""
    precision, recall, _ = precision_recall_curve(y_true, predicted_probs)
    auprc_score = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"AUPRC = {auprc_score:.4f}")
    plt.fill_between(recall, precision, alpha=0.2, color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (AUPRC)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ AUPRC plot saved to {save_path}")


if __name__ == '__main__':
    model = load_model()

    # --- Improved DAG Visualization ---
    visualize_dag(model)

    # --- Markov Blanket Visualization ---
    visualize_markov_blanket(model)
