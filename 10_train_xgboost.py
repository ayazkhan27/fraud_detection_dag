import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
import matplotlib   
matplotlib.use("Agg")  # Add this at the top before plotting
import matplotlib.pyplot as plt
import pickle

# **Feature selection based on causal inference**
KEY_FEATURES = ["V4_disc", "V16_disc", "V14_disc_V12_disc_interaction"]

def preprocess_data(df):
    """
    Converts interaction features (object dtype) into categorical codes.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes  # Convert category to integer codes
    return df

def train_xgboost(train_data_path="resampled_train_data_no_ohe.csv", test_data_path="feature_engineered_test_data_no_ohe.csv"):
    """Trains an XGBoost model on the feature-engineered data with improved fraud recall."""

    # Load training and test data
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    # Preprocess both datasets to convert object columns into categorical codes
    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)

    # Split training data into features (X) and target (y)
    X_train = df_train.drop('Class', axis=1)
    y_train = df_train['Class']

    # Ensure the test data has the same columns as X_train
    if "Class" in df_test.columns:
        X_test = df_test.drop("Class", axis=1)
    else:
        X_test = df_test.copy()
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)  # Ensure column alignment

    # Load the true labels for the test set
    y_test = pd.read_csv("y_test.csv")["Class"]

    # **Ensure key causal features are included**
    missing_features = [feat for feat in KEY_FEATURES if feat not in X_train.columns]
    if missing_features:
        raise ValueError(f"Missing key causal features in dataset: {missing_features}")

    # Initialize the XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", 
                                  eval_metric="aucpr",
                                  enable_categorical=True,  # Enable categorical variable handling
                                  use_label_encoder=False, 
                                  random_state=42)
    
    # Train the model
    xgb_model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    # Calculate AUPRC
    auprc = average_precision_score(y_test, y_pred_proba)
    print(f"AUPRC: {auprc:.4f}")

    # Determine the optimal threshold using the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold (F1-score): {optimal_threshold:.4f}")

    # **Test different thresholds for recall improvement**
    for threshold in [0.95, 0.97, optimal_threshold]:
        print(f"\n--- Evaluating at Threshold {threshold:.4f} ---")
        y_pred = (y_pred_proba >= threshold).astype(int)
        print(classification_report(y_test, y_pred))

    # Plot and save the PR curve
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve, AUPRC = {auprc:.4f}")
    plt.savefig("xgboost_pr_curve.png")
    print("\nXGBoost PR Curve saved to xgboost_pr_curve.png")

    # --- Feature Importance Analysis using SHAP ---
    explainer = shap.Explainer(xgb_model, X_train)
    shap_values = explainer(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("xgboost_shap_importance.png")
    print("\nSHAP Feature Importance saved to xgboost_shap_importance.png")

    # Save the trained model
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    print("XGBoost model saved to xgboost_model.pkl")

    return xgb_model

if __name__ == '__main__':
    train_xgboost()
