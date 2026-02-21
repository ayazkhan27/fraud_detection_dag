# Master-Grade Credit Card Fraud Detection Pipeline

An industry-standard, high-performance Directed Acyclic Graph (DAG) pipeline designed for deep causal inference, interpretable machine learning, and extreme class imbalance handling in financial fraud detection.

## 🌟 Overview
This project transforms a traditional black-box fraud detection model into a fully transparent, statistically rigorous Bayesian pipeline. Using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), this system uncovers the causal mechanisms behind fraudulent transactions, applies robust handling techniques for imbalanced data, and provides deep interpretability engines.

## 🚀 Key Features & Mathematical Innovations

* **Centralized Configuration Hub (`config.yaml`)**: Manage all tunable hyperparameters, input/output paths, and preprocessing boundaries in one place.
* **Mutual Information Feature Selection**: Moves beyond linear/univariate Chi-squared limits. Features mathematically proven to reduce information entropy about the target are prioritized.
* **Robust Bayesian DAG Structure Learning**: Implements **Tabu Search** with a Bayesian Information Criterion (BIC) scoring proxy to intelligently escape local graphical structural optima.
* **Bayesian Estimation with BDeu Priors**: Completely overrides standard Maximum Likelihood Estimator (MLE) fatal flaws handling $0$ probabilities, embedding mathematically sound Laplace smoothing for unseen edge states.
* **Causal Do-Calculus Engine**: Intervenes on predicted features to simulate macro-level structural shifts on baseline fraud probability.
* **Probability Calibration**: Corrects tree-based black-box probabilities to perfectly match real-world observed frequencies via Platt Scaling and Isotonic Regression.
* **Interactive TQDM Visualizations**: Rich real-time progress logging across processing milestones.

## 🛠️ Architecture and File Structure
The environment is deeply decoupled into logical processing steps driven by the master script.

```text
├── config.yaml                                  # Master Pipeline Configuration 
├── utils.py                                     # Standardized Logging & Operations
├── main.py                                      # Master Pipeline Orchestrator
├── 01_data_ingestion_and_preprocessing.py       # Data Loading & Standard Scaler
├── 02_train_test_split.py                       # Stratified Splitting
├── 03_feature_engineering_resampling_train.py   # Quantile Discretization, Mutual Info & SMOTENC
├── 04_dag_structure_learning_train.py           # Correlation Pruning & Tabu Search Structure Learning
├── 05_parameter_learning_train.py               # Bayesian Estimation with BDeu Priors
├── 06_feature_engineering_test.py               # Feature Consistency Application
├── 07_inference_and_evaluation.py               # Variable Elimination Inference Engine
├── 08_visualization_improvements.py             # Dash/Cytoscape Interactive Dashboard
├── 09_causal_inference.py                       # Do-Calculus Inference Evaluation
├── 011_calibration.py                           # Sigmoid & Isotonic Distribution Rescaling
└── 10_train_xgboost.py                          # Deep interpretability (SHAP) Model Execution
```

## ⚙️ How to Run
Ensure `requirements.txt` dependencies are fulfilled. The entire pipeline respects `config.yaml` definitions.

```bash
# To run the entire pipeline seamlessly end-to-end:
python3 main.py

# To run the pipeline AND asynchronously launch the Dashboard at the conclusion:
python3 main.py --dashboard

# To execute a specific step from the pipeline (e.g. debugging/testing):
python3 main.py --step 1_data_ingestion
```

### Dashboard Access
If running the dashboard visualization engine, head over to `http://127.0.0.1:8050` in your web browser. Here you can manipulate graph layouts dynamically (Cose, Breadthfirst, Grid, Circle) to map out structural causation and target variable Markov Blankets.

## 📊 Analytics and Interpretations
All analytical artifacts (precision-recall curves, calibration plots, SHAP metrics) are deposited in the `/Figures/` directory. Numerical data frames reflecting causal intervention estimations map directly to `causal_inference_results.csv`.
