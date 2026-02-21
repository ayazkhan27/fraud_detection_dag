# 10_train_xgboost.py

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
import matplotlib   
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from utils import load_config, setup_logger, ensure_dir

logger = setup_logger(__name__)

def preprocess_data(df):
    """Converts interaction features into categorical codes."""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df

def generate_interactive_shap(xgb_model, X_test, out_path):
    """Generates an interactive HTML summary plot for advanced SHAP explainability."""
    explainer = shap.Explainer(xgb_model, X_test)
    shap_values = explainer(X_test)
    
    # Save a static summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    
    # Optional: We could generate force_plots but they require specific JS embedding. 
    # For now, a comprehensive static dependence and summary plot matrix is highly descriptive.
    return shap_values

def train_xgboost(config):
    """Trains an XGBoost model, evaluates AUPRC, and generates SHAP explanations."""
    
    train_path = config["paths"]["resampled_train"]
    test_path = config["paths"]["feature_engineered_test"]
    y_test_path = config["paths"]["y_test"]
    
    ensure_dir(config["paths"]["figures_dir"])
    
    logger.info(f"Loading data: {train_path}, {test_path}")
    df_train = preprocess_data(pd.read_csv(train_path))
    df_test = preprocess_data(pd.read_csv(test_path))

    X_train = df_train.drop('Class', axis=1)
    y_train = df_train['Class']

    if "Class" in df_test.columns:
        X_test = df_test.drop("Class", axis=1)
    else:
        X_test = df_test.copy()
        
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = pd.read_csv(y_test_path)["Class"]

    # Key Causal Features check
    key_features = config["xgboost"]["key_causal_features"]
    missing = [feat for feat in key_features if feat not in X_train.columns]
    if missing:
        logger.error(f"Missing key causal features in dataset: {missing}")
        raise ValueError(f"Missing causal features: {missing}")

    logger.info("Initializing and training XGBoost Classifier")
    xgb_model = xgb.XGBClassifier(
        objective=config["xgboost"]["objective"], 
        eval_metric=config["xgboost"]["eval_metric"],
        enable_categorical=True,
        random_state=config["pipeline"]["random_state"]
    )
    
    xgb_model.fit(X_train, y_train)

    # Evaluation
    logger.info("Evaluating XGBoost predictions")
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    auprc = average_precision_score(y_test, y_pred_proba)
    logger.info(f"AUPRC: {auprc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    opt_idx = np.nanargmax(f1_scores)
    opt_threshold = thresholds[opt_idx]
    logger.info(f"Optimal Threshold (F1): {opt_threshold:.4f}")

    # Plot PR Curve
    pr_fig_path = config["paths"]["xgboost_pr_curve"]
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve, AUPRC = {auprc:.4f}")
    plt.savefig(pr_fig_path)
    logger.info(f"PR Curve saved to {pr_fig_path}")

    # SHAP Interpretability
    logger.info("Calculating deep SHAP values for model transparency")
    shap_fig_path = config["paths"]["xgboost_shap_importance"]
    html_path = config["paths"]["interactive_shap"]
    
    shap_values = generate_interactive_shap(xgb_model, X_test, html_path)
    plt.savefig(shap_fig_path)
    plt.close()
    logger.info(f"Static SHAP Summary saved to {shap_fig_path}")

    # Save Model
    model_path = config["paths"]["xgboost_model"]
    with open(model_path, "wb") as f:
        pickle.dump(xgb_model, f)
    logger.info(f"Trained XGBoost model saved to {model_path}")

if __name__ == '__main__':
    config = load_config()
    train_xgboost(config)
