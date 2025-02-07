# 08_visualization_improvements.py

import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from networkx.drawing.nx_pydot import graphviz_layout

def visualize_dag(model, filename="improved_bayesian_network.png"):
    """Visualizes the DAG with improved clarity."""

    nx_graph = nx.DiGraph(model.edges())

    # --- 1. Improved Layout ---
    # Use graphviz for layout (requires pydot and graphviz to be installed)
    pos = graphviz_layout(nx_graph, prog="neato")  # "neato" is good for these kinds of graphs

    # --- 2. Node Styling ---
    node_colors = []
    for node in nx_graph.nodes():
        if node == 'Class':
            node_colors.append('red')  # Highlight the 'Class' node
        elif "_disc" in node:
            node_colors.append('skyblue') #Keep skyblue for other nodes
        else:
             node_colors.append('lightgray') # Should not happen, Amount was dropped.

    # --- 3. Edge Styling ---
    edge_colors = []
    edge_widths = []
    for u, v in nx_graph.edges():
        if v == 'Class':
            edge_colors.append('red')  # Highlight edges going into 'Class'
            edge_widths.append(2.0)
        else:
            edge_colors.append('black')
            edge_widths.append(0.5)

    # --- 4. Draw the Graph ---
    plt.figure(figsize=(18, 12))  # Even larger figure size
    nx.draw(nx_graph, pos, with_labels=True, node_size=2500, node_color=node_colors,
            font_size=10, font_weight="bold", arrowsize=20, width=edge_widths,
            edge_color=edge_colors, edge_cmap=plt.cm.Reds)

    plt.title("Improved Bayesian Network Structure", fontsize=18)
    plt.savefig(filename)
    print(f"\nImproved DAG visualization saved to {filename}")

    #--- 5. Markov Blanket Visualization ---
    mb_nodes = list(model.get_markov_blanket('Class')) + ['Class'] # Include class
    mb_graph = nx_graph.subgraph(mb_nodes)
    pos_mb = graphviz_layout(mb_graph, prog="neato")

    node_colors_mb = ['red' if node == 'Class' else 'skyblue' for node in mb_graph.nodes()]
    edge_colors_mb = ['red' if v == 'Class' else 'black' for u, v in mb_graph.edges()]
    edge_widths_mb = [2.0 if v == 'Class' else 0.5 for u, v in mb_graph.edges()]

    plt.figure(figsize=(12, 8))
    nx.draw(mb_graph, pos_mb, with_labels=True, node_size=2000, node_color=node_colors_mb,
           font_size=10, font_weight='bold', arrowsize=20, width=edge_widths_mb,
           edge_color=edge_colors_mb)
    plt.title("Markov Blanket of Class Node", fontsize=16)
    plt.savefig("markov_blanket.png")
    print(f"Markov Blanket visualization saved to markov_blanket.png")


if __name__ == '__main__':
    # Load the trained model (from the pickle file created in step 5)
    with open("bayesian_network_model.pkl", "rb") as f:
        model = pickle.load(f)
    visualize_dag(model)