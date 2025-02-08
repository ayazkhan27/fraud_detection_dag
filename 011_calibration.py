# 011_calibration.py
import numpy as np
import pandas as pd
import pickle
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import xgboost as xgb

def preprocess_for_xgb(df):
    """Convert object columns to category codes for XGBoost."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

# Load test data and true labels
X_test = pd.read_csv("feature_engineered_test_data_no_ohe.csv")
y_test = pd.read_csv("y_test.csv")
if "Class" in y_test.columns:
    y_test = y_test["Class"]

# Preprocess X_test to ensure proper data types for XGBoost
X_test = preprocess_for_xgb(X_test)

# Load the pre-trained XGBoost model (trained in script 10)
with open("xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Align X_test to the feature set used during training
expected_features = xgb_model.get_booster().feature_names
X_test = X_test.reindex(columns=expected_features, fill_value=0)

# Now, generate raw predicted probabilities from the uncalibrated model
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate Brier score for the uncalibrated model
brier_uncalibrated = brier_score_loss(y_test, y_pred_proba)
print(f"Brier Score (Uncalibrated): {brier_uncalibrated:.4f}")

# Plot calibration curve for the uncalibrated model
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Uncalibrated")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration Plot (Uncalibrated)")
plt.legend()
plt.savefig("calibration_uncalibrated.png")
plt.close()
print("Calibration plot for uncalibrated model saved to calibration_uncalibrated.png")

# --- Calibration using Platt Scaling (sigmoid) ---
# Note: Here we assume that your XGBoost model is already fitted, so we use cv='prefit'
calibrated_clf = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
calibrated_clf.fit(X_test, y_test)  # Use a separate hold-out set if available

# Get calibrated probabilities
y_pred_proba_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]

# Evaluate Brier score for the calibrated model
brier_calibrated = brier_score_loss(y_test, y_pred_proba_calibrated)
print(f"Brier Score (Platt Scaling): {brier_calibrated:.4f}")

# Plot calibration curve for the calibrated model
fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_test, y_pred_proba_calibrated, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Platt Scaling")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration Plot (Platt Scaling)")
plt.legend()
plt.savefig("calibration_platt_scaling.png")
plt.close()
print("Calibration plot for Platt Scaling saved to calibration_platt_scaling.png")

# --- (Optional) Calibration using Isotonic Regression ---
calibrated_clf_iso = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
calibrated_clf_iso.fit(X_test, y_test)

y_pred_proba_iso = calibrated_clf_iso.predict_proba(X_test)[:, 1]
brier_calibrated_iso = brier_score_loss(y_test, y_pred_proba_iso)
print(f"Brier Score (Isotonic Regression): {brier_calibrated_iso:.4f}")

fraction_of_positives_iso, mean_predicted_value_iso = calibration_curve(y_test, y_pred_proba_iso, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value_iso, fraction_of_positives_iso, "s-", label="Isotonic Regression")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration Plot (Isotonic Regression)")
plt.legend()
plt.savefig("calibration_isotonic.png")
plt.close()
print("Calibration plot for Isotonic Regression saved to calibration_isotonic.png")
