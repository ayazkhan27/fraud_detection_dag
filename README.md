# Credit Card Fraud Detection with Interpretable Bayesian Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project develops a credit card fraud detection system using a **Bayesian Network**, prioritizing *interpretability* and *adaptability*. Unlike "black-box" machine learning models, a Bayesian Network provides a **probabilistic and graphical representation** of the relationships between transaction features and the likelihood of fraud. This allows for not only prediction but also an *explanation* of why a transaction is flagged as potentially fraudulent.

The dataset used is the publicly available **Kaggle Credit Card Fraud Dataset**, which contains anonymized credit card transactions pre-processed with **Principal Component Analysis (PCA)**. To further enhance fraud detection capabilities, the project also integrates an **XGBoost classifier** for performance comparison.

This approach is designed to be **domain-agnostic**, meaning the methodology can be adapted to other anomaly detection tasks beyond fraud detection.

## Project Goals

- Develop a robust **fraud detection system** combining Bayesian Networks and XGBoost.
- Prioritize **interpretability** to understand the factors driving fraud predictions.
- Create a **domain-agnostic** pipeline adaptable to different datasets.
- Handle **severe class imbalance** (0.172% fraud cases) using **SMOTE-NC**.
- Demonstrate a **complete data science workflow**, from preprocessing to model evaluation.
- Explore **potential causal relationships** using do-calculus and Bayesian inference.

## Methodology

The pipeline follows a **structured, iterative** approach:

### **1. Data Ingestion and Preprocessing (`01_data_ingestion_and_preprocessing.py`)**
- Loads the `creditcard.csv` dataset.
- Standardizes numerical features (`V1` to `V28`, `Time`, `Amount`) using `StandardScaler`.
- Saves the processed dataset as `preprocessed_data.csv`.

### **2. Train/Test Split (`02_train_test_split.py`)**
- Splits the dataset into **80% training and 20% testing**.
- Ensures **stratified sampling** due to class imbalance.
- Saves `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.

### **3. Feature Engineering & Resampling (Train) (`03_feature_engineering_resampling_train.py`)**
- **Discretizes** continuous features using **quantile-based binning (`pd.qcut`)**.
- Creates **interaction features** between key variables.
- Uses **Chi-Squared tests** to select **statistically significant** features.
- **Applies SMOTE-NC** to balance class distribution.
- Saves processed data as `resampled_train_data_no_ohe.csv`.

### **4. Bayesian Network DAG Structure Learning (`04_dag_structure_learning_train.py`)**
- Uses **HillClimbSearch with BicScore** to learn a **Directed Acyclic Graph (DAG)**.
- Constructs a **whitelist of high-impact features** to improve DAG accuracy.
- Saves the learned **Bayesian Network structure** (`learned_dag_edges.txt`).

### **5. Parameter Learning (`05_parameter_learning_train.py`)**
- Estimates **Conditional Probability Distributions (CPDs)** for the learned Bayesian Network.
- Updates and saves the model as `bayesian_network_model.pkl`.

### **6. Feature Engineering (Test) (`06_feature_engineering_test.py`)**
- Loads `X_test.csv` and applies the **same bin edges** as the training set.
- Creates **interaction features** based on the Bayesian DAG structure.
- Saves the processed test data as `feature_engineered_test_data_no_ohe.csv`.

### **7. Inference & Evaluation (`07_inference_and_evaluation.py`)**
- Performs **Bayesian inference using Variable Elimination**.
- Finds the **optimal fraud detection threshold** using **Precision-Recall curves**.
- Evaluates the **AUPRC, Precision, Recall, and F1-score**.

### **8. Visualization (`08_visualization_improvements.py`)**
- **Visualizes the Bayesian DAG** using Graphviz.
- **Plots the Markov Blanket** of the `Class` node.
- **Generates a Precision-Recall curve (AUPRC)**.

### **9. Causal Inference (`09_causal_inference.py`)**
- Performs **causal inference** using do-calculus.
- Tests **intervention effects on fraud probability**.
- Saves results as `causal_inference_results.csv`.

### **10. XGBoost Training & Evaluation (`10_train_xgboost.py`)**
- Trains an **XGBoost model** using causally significant features.
- Computes **SHAP values** for interpretability.
- Saves the trained model as `xgboost_model.pkl`.

## Results Summary

### **Bayesian Network Performance (at optimal threshold)**
- **Precision:** 0.03
- **Recall:** 0.78
- **F1-score:** 0.06

### **XGBoost Performance (at optimal threshold)**
- **AUPRC:** 0.8103
- **Precision:** 0.87
- **Recall:** 0.78
- **F1-score:** 0.82

### **Causal Inference Insights**
- Causal inference results are stored in `causal_inference_results.csv`.
- Bayesian Networks provided insights into **feature relationships affecting fraud likelihood**.

## Future Work

- **Dynamic Bayesian Networks (DBNs):** Extend the model to capture **temporal fraud patterns**.
- **Hybrid Model:** Combine Bayesian Networks with **Neural Networks** for fraud detection.
- **Alternative Discretization:** Explore **decision-tree-based binning** instead of quantile binning.
- **Ensemble Methods:** Combine Bayesian inference with XGBoost predictions.
- **Integration with Real-Time Systems:** Deploy a **streaming fraud detection model** using **Kafka & FastAPI**.

## Dataset Information

The dataset is available on Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- `V1-V28`: PCA components.
- `Time`: Seconds since first transaction.
- `Amount`: Transaction amount.
- `Class`: **1 = Fraud, 0 = Legitimate.**

## Installation & Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `imblearn`, `pgmpy`, `networkx`, `matplotlib`, `xgboost`, `shap`, `scipy`, `pydot`

**Additional Graphviz Installation (Required for DAG visualization)**
```bash
sudo apt-get install graphviz graphviz-dev  # Debian-based systems
```

## Usage

Run the full pipeline:
```bash
python main.py
```

## License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

