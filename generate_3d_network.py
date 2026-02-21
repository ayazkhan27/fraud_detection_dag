import networkx as nx
import plotly.graph_objects as go
import pickle
from pgmpy.models import BayesianNetwork
import numpy as np

def generate_3d_network():
    print("Loading learned Bayesian Network...")
    with open("bayesian_network_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    G = nx.DiGraph(model.edges())
    
    # Identify Markov Blanket for 'Class'
    target = 'Class'
    if target in G.nodes():
        parents = list(G.predecessors(target))
        children = list(G.successors(target))
        spouses = []
        for child in children:
            spouses.extend(list(G.predecessors(child)))
        markov_blanket = set(parents + children + spouses)
        if target in markov_blanket:
            markov_blanket.remove(target)
    else:
        markov_blanket = set()
        print("Warning: 'Class' node not found in DAG.")

    # Calculate 3D layout
    pos = nx.spring_layout(G, dim=3, seed=42, k=0.5)

    # Extract node coordinates
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    z_nodes = [pos[node][2] for node in G.nodes()]
    
    # Colors and Sizes
    node_colors = []
    node_sizes = []
    hover_texts = []
    
    for node in G.nodes():
        if node == target:
            node_colors.append('#ff0055') # Neon pink/red for Target
            node_sizes.append(25)
            hover_text = f"<b>{node}</b><br>TARGET VARIABLE"
        elif node in markov_blanket:
            node_colors.append('#00e5ff') # Cyan for Markov Blanket
            node_sizes.append(15)
            hover_text = f"<b>{node}</b><br>Markov Blanket (Strong Driver)"
        else:
            node_colors.append('#4a4a8a') # Dim purple for others
            node_sizes.append(8)
            hover_text = f"<b>{node}</b>"
            
        hover_texts.append(hover_text)

    # Create Edges
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    # Plotly Traces
    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.2)', width=2),
        hoverinfo='none'
    )

    trace_nodes = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.9,
            line=dict(width=2, color='#ffffff')
        ),
        text=[n if n == target or n in markov_blanket else "" for n in G.nodes()],
        textposition="top center",
        textfont=dict(color="white", size=10),
        hoverinfo='text',
        hovertext=hover_texts
    )

    # Layout
    layout = go.Layout(
        title=dict(text="<b>Causal Bayesian Network: Fraud Drivers (3D Engine)</b>", font=dict(size=24, color='#ffffff')),
        paper_bgcolor='#0a0a1a', # Deep space background
        plot_bgcolor='#0a0a1a',
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            bgcolor='#0a0a1a'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig = go.Figure(data=[trace_edges, trace_nodes], layout=layout)
    
    # Save as interactive HTML
    out_file = "Causal_Network_3D.html"
    fig.write_html(out_file)
    print(f"Stunning 3D network saved to {out_file}")

if __name__ == "__main__":
    generate_3d_network()
