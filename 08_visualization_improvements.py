# 08_visualization_improvements.py

import os
import pickle
import dash
from dash import dcc, html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
import networkx as nx
import plotly.graph_objs as go
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import pandas as pd
from utils import load_config, setup_logger

logger = setup_logger(__name__)

# Ensure Graphviz is in PATH
os.environ["PATH"] += os.pathsep + "/usr/bin/"

try:
    import pydot
except ImportError:
    logger.warning("pydot is not installed properly. Ensure 'pydot' and 'graphviz' are installed.")

def load_model(model_path):
    """Loads the Bayesian network model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def generate_cytoscape_elements(model):
    """Generates a list of Cytoscape elements (nodes and edges) for the full DAG."""
    G = nx.DiGraph(model.edges())
    elements = []
    
    for node in G.nodes():
        elements.append({
            'data': {'id': node, 'label': node},
            'classes': 'node'
        })
        
    for edge in G.edges():
        elements.append({
            'data': {'source': edge[0], 'target': edge[1], 'label': f"{edge[0]} → {edge[1]}"},
            'classes': 'edge'
        })
    return elements

def generate_cytoscape_elements_markov_blanket(model, target_variable="Class"):
    """Generates Cytoscape elements for the Markov Blanket of the target variable."""
    try:
        blanket = model.get_markov_blanket(target_variable)
    except Exception as e:
        logger.error(f"Error obtaining Markov blanket: {e}")
        blanket = []
        
    nodes_to_include = set(blanket + [target_variable])
    subgraph = nx.DiGraph()
    
    for node in nodes_to_include:
        subgraph.add_node(node)
        
    for edge in model.edges():
        if edge[0] in nodes_to_include and edge[1] in nodes_to_include:
            subgraph.add_edge(*edge)
            
    elements = []
    for node in subgraph.nodes():
        elements.append({
            'data': {'id': node, 'label': node},
            'classes': 'node'
        })
        
    for edge in subgraph.edges():
        elements.append({
            'data': {'source': edge[0], 'target': edge[1], 'label': f"{edge[0]} → {edge[1]}"},
            'classes': 'edge'
        })
    return elements

def generate_pr_curve_figure():
    """Generates an interactive Plotly Precision-Recall curve."""
    # (Optional Improvement) Read real predictions instead of random data if available at dashboard time
    # For now, we seed random behavior as in the original script
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=200)
    predicted_probs = np.random.rand(200)
    precision, recall, _ = precision_recall_curve(y_true, predicted_probs)
    auprc_score = auc(recall, precision)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines+markers',
        line=dict(color='blue'),
        name=f"AUPRC = {auprc_score:.4f}"
    ))
    
    fig.update_layout(
        title="Interactive Precision-Recall Curve (Demo)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        hovermode="closest"
    )
    return fig

config = load_config()
model = load_model(config["paths"]["bayesian_network_model"])

dag_elements = generate_cytoscape_elements(model)
mb_elements = generate_cytoscape_elements_markov_blanket(model, target_variable="Class")
pr_curve_fig = generate_pr_curve_figure()

# Dashboard UI definition
layout_options = [
    {'label': 'Cose', 'value': 'cose'},
    {'label': 'Breadthfirst', 'value': 'breadthfirst'},
    {'label': 'Circle', 'value': 'circle'},
    {'label': 'Concentric', 'value': 'concentric'},
    {'label': 'Grid', 'value': 'grid'}
]

legend_html = html.Div([
    html.H4("Legend"),
    html.Div([
        html.Span(style={'background-color': '#1f77b4', 'display': 'inline-block', 'width': '20px', 'height': '20px', 'margin-right': '10px'}),
        html.Span("Bayesian Network Node")
    ], style={'margin': '5px'}),
    html.Div([
        html.Span(style={'background-color': '#d62728', 'display': 'inline-block', 'width': '20px', 'height': '20px', 'margin-right': '10px'}),
        html.Span("Markov Blanket Node")
    ], style={'margin': '5px'}),
    html.Div([
        html.Span(style={'background-color': '#ccc', 'display': 'inline-block', 'width': '20px', 'height': '2px', 'margin-right': '10px'}),
        html.Span("Edge")
    ], style={'margin': '5px'})
], style={'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px', 'background-color': '#f9f9f9', 'width': '250px'})

app = dash.Dash(__name__)
app.title = "Interactive Fraud Detection Dashboard"

app.layout = html.Div([
    html.H1("Fraud Detection Model Interactive Interpretability", style={'textAlign': 'center', 'color': '#333'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label="Bayesian Network DAG", children=[
            html.Div([
                html.H3("Learned Bayesian Network Structure"),
                html.Div([
                    html.Label("Choose Layout:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='layout-dropdown-dag',
                        options=layout_options,
                        value='cose',
                        clearable=False,
                        style={'width': '200px', 'display': 'inline-block'}
                    )
                ], style={'margin': '10px'}),
                cyto.Cytoscape(
                    id='cytoscape-dag',
                    elements=dag_elements,
                    layout={'name': 'cose'},
                    style={'width': '100%', 'height': '600px'},
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'label': 'data(label)',
                                'width': '60px',
                                'height': '60px',
                                'background-color': '#1f77b4',
                                'color': '#fff',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': '12px',
                                'border-width': 2,
                                'border-color': '#555'
                            }
                        },
                        {
                            'selector': 'edge',
                            'style': {
                                'curve-style': 'bezier',
                                'target-arrow-shape': 'triangle',
                                'line-color': '#ccc',
                                'target-arrow-color': '#ccc',
                                'width': 2,
                                'label': 'data(label)',
                                'font-size': '10px',
                                'text-rotation': 'autorotate'
                            }
                        }
                    ]
                ),
                html.Div(id='node-data-dag', style={'margin-top': '20px', 'padding': '10px', 'background-color': '#eee'})
            ], style={'margin': '20px', 'display': 'flex', 'flex-direction': 'row'}),
            html.Div(legend_html, style={'margin': '20px'})
        ]),
        dcc.Tab(label="Markov Blanket", children=[
            html.Div([
                html.H3("Markov Blanket of 'Class'"),
                html.Div([
                    html.Label("Choose Layout:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='layout-dropdown-mb',
                        options=layout_options,
                        value='cose',
                        clearable=False,
                        style={'width': '200px', 'display': 'inline-block'}
                    )
                ], style={'margin': '10px'}),
                cyto.Cytoscape(
                    id='cytoscape-mb',
                    elements=mb_elements,
                    layout={'name': 'cose'},
                    style={'width': '100%', 'height': '600px'},
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'label': 'data(label)',
                                'width': '60px',
                                'height': '60px',
                                'background-color': '#d62728',
                                'color': '#fff',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': '12px',
                                'border-width': 2,
                                'border-color': '#555'
                            }
                        },
                        {
                            'selector': 'edge',
                            'style': {
                                'curve-style': 'bezier',
                                'target-arrow-shape': 'triangle',
                                'line-color': '#ccc',
                                'target-arrow-color': '#ccc',
                                'width': 2,
                                'label': 'data(label)',
                                'font-size': '10px',
                                'text-rotation': 'autorotate'
                            }
                        }
                    ]
                ),
                html.Div(id='node-data-mb', style={'margin-top': '20px', 'padding': '10px', 'background-color': '#eee'})
            ], style={'margin': '20px', 'display': 'flex', 'flex-direction': 'row'}),
            html.Div(legend_html, style={'margin': '20px'})
        ])
    ], style={'fontFamily': 'Arial, sans-serif'})
])

@app.callback(
    Output('cytoscape-dag', 'layout'),
    [Input('layout-dropdown-dag', 'value')]
)
def update_dag_layout(layout_name):
    return {'name': layout_name}

@app.callback(
    Output('cytoscape-mb', 'layout'),
    [Input('layout-dropdown-mb', 'value')]
)
def update_mb_layout(layout_name):
    return {'name': layout_name}

@app.callback(
    Output('node-data-dag', 'children'),
    [Input('cytoscape-dag', 'tapNodeData')]
)
def display_node_data_dag(data):
    if data:
        return html.Div([
            html.H4("Node Information"),
            html.P(f"ID: {data.get('id')}"),
            html.P(f"Label: {data.get('label')}")
        ])
    return "Tap a node to see its details."

@app.callback(
    Output('node-data-mb', 'children'),
    [Input('cytoscape-mb', 'tapNodeData')]
)
def display_node_data_mb(data):
    if data:
        return html.Div([
            html.H4("Node Information"),
            html.P(f"ID: {data.get('id')}"),
            html.P(f"Label: {data.get('label')}")
        ])
    return "Tap a node to see its details."

if __name__ == '__main__':
    logger.info("Starting Dash Server for Interactive Visualizations")
    app.run_server(debug=True, use_reloader=False)
