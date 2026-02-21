# 07_inference_and_evaluation.py

import pandas as pd
import numpy as np
import pickle
from pgmpy.inference import VariableElimination
from sklearn.metrics import precision_recall_curve, classification_report
from utils import load_config, setup_logger
from tqdm import tqdm

logger = setup_logger(__name__)

def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def cast_evidence_value(model, variable, value):
    try:
        cpd = model.get_cpds(variable)
        if cpd is not None:
            states = cpd.state_names[variable]
            if value in states: return value
            if str(value) in states: return str(value)
            try:
                int_val = int(value)
                if int_val in states: return int_val
            except Exception: pass
            return states[0]
    except Exception as e:
        pass
    return value

def predict_probabilities(model, data):
    logger.info("Starting inference on test data using Variable Elimination...")
    infer = VariableElimination(model)
    predicted_probs = []

    available_columns = list(set(model.nodes) & set(data.columns))
    if "Class" in available_columns:
        available_columns.remove("Class")
    
    evidence_data = data[available_columns]
    
    for idx, row in tqdm(evidence_data.iterrows(), total=len(evidence_data), desc="Variable Elimination Inference"):
        evidence = row.to_dict()
        filtered_evidence = {var: cast_evidence_value(model, var, val) for var, val in evidence.items()}
        try:
            query_result = infer.query(variables=['Class'], evidence=filtered_evidence)
            prob_fraud = query_result.values[1]
            predicted_probs.append(prob_fraud)
        except Exception as e:
            predicted_probs.append(0)
            
    return np.array(predicted_probs)

def evaluate(y_true, predicted_probs):
    logger.info("Evaluating Inference Model Performance")
    precision, recall, thresholds = precision_recall_curve(y_true, predicted_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    optimal_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    logger.info(f"Optimal Threshold: {optimal_threshold:.4f} | F1: {f1_scores[optimal_idx]:.4f}")
    predicted_classes = (predicted_probs >= optimal_threshold).astype(int)
    
    print(f"\nClassification Report (Threshold = {optimal_threshold:.4f}):")
    print(classification_report(y_true, predicted_classes))

def run_inference(config):
    model_path = config["paths"]["bayesian_network_model"]
    test_path = config["paths"]["feature_engineered_test"]
    y_test_path = config["paths"]["y_test"]
    
    model = load_model(model_path)
    
    logger.info(f"Loading test datasets ({test_path})")
    X_test = pd.read_csv(test_path)
    y_test = pd.read_csv(y_test_path)["Class"]
    
    predicted_probs = predict_probabilities(model, X_test)
    evaluate(y_test, predicted_probs)
    
if __name__ == '__main__':
    config = load_config()
    run_inference(config)
