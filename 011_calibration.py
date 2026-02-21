# 011_calibration.py
import numpy as np
import pandas as pd
import pickle
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_config, setup_logger, ensure_dir

logger = setup_logger(__name__)

def preprocess_for_xgb(df):
    """Convert object columns to category codes for XGBoost."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

def perform_calibration(config):
    test_path = config["paths"]["feature_engineered_test"]
    y_test_path = config["paths"]["y_test"]
    model_path = config["paths"]["xgboost_model"]
    
    ensure_dir(config["paths"]["figures_dir"])
    
    logger.info(f"Loading files: {test_path}")
    X_test = pd.read_csv(test_path)
    y_test = pd.read_csv(y_test_path)
    if "Class" in y_test.columns:
        y_test = y_test["Class"]

    X_test = preprocess_for_xgb(X_test)

    logger.info(f"Loading XGBoost model from {model_path}")
    with open(model_path, "rb") as f:
        xgb_model = pickle.load(f)

    expected_features = xgb_model.get_booster().feature_names
    X_test = X_test.reindex(columns=expected_features, fill_value=0)

    # Raw predicted probabilities
    logger.info("Scoring uncalibrated baseline")
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    brier_uncalibrated = brier_score_loss(y_test, y_pred_proba)
    logger.info(f"Brier Score (Uncalibrated): {brier_uncalibrated:.4f}")

    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)

    # Plot Uncalibrated
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Uncalibrated")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot (Uncalibrated)")
    plt.legend()
    uncal_path = config["paths"]["calibration_uncalibrated"]
    plt.savefig(uncal_path)
    plt.close()
    logger.info(f"Uncalibrated plot saved to {uncal_path}")

    # --- Platt Scaling ---
    logger.info("Applying Platt Scaling (sigmoid calibration)")
    calibrated_clf = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_test, y_test)

    y_pred_proba_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]
    brier_calibrated = brier_score_loss(y_test, y_pred_proba_calibrated)
    logger.info(f"Brier Score (Platt Scaling): {brier_calibrated:.4f}")

    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_test, y_pred_proba_calibrated, n_bins=10)

    # Plot Platt Scaling
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Platt Scaling")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot (Platt Scaling)")
    plt.legend()
    platt_path = config["paths"]["calibration_platt"]
    plt.savefig(platt_path)
    plt.close()
    logger.info(f"Platt Scaling plot saved to {platt_path}")

    # --- Isotonic Regression (Optional but highly robust for XGBoost) ---
    logger.info("Applying Isotonic Regression")
    calibrated_clf_iso = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
    calibrated_clf_iso.fit(X_test, y_test)

    y_pred_proba_iso = calibrated_clf_iso.predict_proba(X_test)[:, 1]
    brier_calibrated_iso = brier_score_loss(y_test, y_pred_proba_iso)
    logger.info(f"Brier Score (Isotonic): {brier_calibrated_iso:.4f}")

    fraction_of_positives_iso, mean_predicted_value_iso = calibration_curve(y_test, y_pred_proba_iso, n_bins=10)
    
    # Plot Isotonic Regression
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value_iso, fraction_of_positives_iso, "s-", label="Isotonic Regression")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot (Isotonic Regression)")
    plt.legend()
    iso_path = config["paths"]["calibration_isotonic"]
    plt.savefig(iso_path)
    plt.close()
    logger.info(f"Isotonic plot saved to {iso_path}")

if __name__ == '__main__':
    config = load_config()
    perform_calibration(config)
