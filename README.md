# Credit Card Fraud Detection with Interpretable Bayesian Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project develops a credit card fraud detection system using a Bayesian network, prioritizing interpretability and adaptability.  Unlike "black box" machine learning models, a Bayesian network provides a visual and probabilistic representation of the relationships between transaction features and the likelihood of fraud.  This allows for not only prediction but also *explanation* of why a transaction is flagged as potentially fraudulent. The project utilizes a publicly available dataset of anonymized credit card transactions, pre-processed with Principal Component Analysis (PCA).  The approach is designed to be domain-agnostic, making it potentially adaptable to other anomaly detection tasks.

## Project Goals

*   Develop a robust fraud detection system with high accuracy (measured by AUPRC).
*   Prioritize *interpretability* to understand the factors driving fraud predictions.
*   Create a *domain-agnostic* pipeline adaptable to other datasets.
*   Handle the *class imbalance* inherent in fraud data (very few fraudulent transactions).
*   Demonstrate a complete data science workflow, from data loading to model evaluation and visualization.
*   Explore *potential causal relationships* between transaction features and fraud.

## Methodology

The project follows a multi-stage pipeline:

1.  **Data Ingestion and Preprocessing:** Loads the dataset, checks for missing values, and standardizes the continuous features.
2.  **Domain-Agnostic Discretization:** Converts continuous features into discrete bins using a robust clustering approach (k-means with silhouette analysis and outlier handling). This is done *without* relying on domain-specific rules.
3.  **Resampling:** Addresses the class imbalance using SMOTE-NC (Synthetic Minority Oversampling Technique for Nominal and Continuous features), creating a balanced dataset for training.
4.  **DAG Structure Learning:** Learns the structure of the Bayesian network (a Directed Acyclic Graph, or DAG) using a hybrid approach (HillClimbSearch with BIC scoring) and a whitelist to constrain the search space and prevent memory issues.
5.  **Parameter Learning:** Estimates the conditional probability distributions (CPDs) for each node in the DAG using Maximum Likelihood Estimation (MLE).
6.  **Inference and Evaluation:** Uses the trained Bayesian network to predict the probability of fraud for new transactions and evaluates performance using AUPRC (Area Under the Precision-Recall Curve) and a classification report (with an optimized threshold).
7.  **Visualization and Reporting:** Visualizes the learned DAG, highlighting key relationships and the Markov blanket of the 'Class' (fraud) node. Calculates the optimal prediction threshold.

## Files

The project is organized into the following Python scripts:

*   **`01_data_ingestion_and_preprocessing.py`:**
    *   Loads the `creditcard.csv` dataset.
    *   Checks for missing values.
    *   Standardizes the continuous features (V1-V28, Time, Amount) using `StandardScaler`.
    *   Saves the preprocessed data to `preprocessed_data.csv`.
*   **`02_discretization.py`:**
    *   Loads the `preprocessed_data.csv` file.
    *   Performs domain-agnostic discretization of continuous features using k-means clustering with silhouette analysis and IQR-based outlier handling.
    *   Creates new columns for the discretized features (e.g., `V1_disc`, `Amount_disc`).
    *   Saves the discretized data to `discretized_data.csv`.
*   **`03_resampling.py`:**
    *   Loads the `discretized_data.csv` file and `preprocessed_data.csv`.
    *   Reintroduces the continuous `Amount` column for use in the SMOTE-NC step.
    *   Addresses class imbalance using SMOTE-NC, which handles mixed continuous and categorical data.
    *   Saves the resampled data to `resampled_data.csv`.
*   **`04_dag_structure_learning.py`:**
    *   Loads the `resampled_data.csv` file.
    *   Learns the DAG structure using `HillClimbSearch` with the `BicScore` and a whitelist for edges.
    *   Saves the learned DAG edges to `learned_dag_edges.txt`.
*   **`05_parameter_learning.py`:**
    *   Loads the DAG structure from `learned_dag_edges.txt` and the data from `resampled_data.csv`.
    *   Estimates the conditional probability distributions (CPDs) using `MaximumLikelihoodEstimator`.
    *   Saves the complete Bayesian network model (structure + parameters) to `bayesian_network_model.pkl`.
*   **`06_inference_and_evaluation.py`:**
    *   Loads the trained model (`bayesian_network_model.pkl`) and the *original* discretized data (`discretized_data.csv`).
    *   Performs inference to predict fraud probabilities.
    *   Evaluates performance using AUPRC and a classification report (with a default threshold of 0.5).
    *  Outputs a precision-recall curve.
*   **`07_reporting_and_visualization.py`:**
    *   Loads the trained model (`bayesian_network_model.pkl`) and original data.
    *   Finds the optimal threshold for classification based on the F1-score.
    *   Evaluates performance using the optimal threshold.
    *   Generates improved visualizations of the Bayesian network.
*   **`08_visualization_improvements.py`:**
    *   Loads the Bayesian Network.
    *   Visualizes the DAG using `graphviz` with improved layout, node coloring, and edge highlighting.
    *   Visualizes the Markov blanket of the 'Class' node.
    *   Saves the improved visualization as `improved_bayesian_network.png` and `markov_blanket.png`.
*   **`main.py`:**
    *   Executes the entire pipeline by running the individual scripts in the correct order.
*  **`requirements.txt`:**
    *    Specifies the required libraries to run this project.
* **`README.md`**
    * This file.

## Dataset

The project uses the "Credit Card Fraud Detection" dataset, which can be found on Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  Download the `creditcard.csv` file and place it in the project's root directory.

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It is highly imbalanced, with only 0.172% of transactions being fraudulent.  The features V1-V28 are the result of a PCA transformation, and their original meanings are not provided due to confidentiality issues.  The 'Time' feature represents the seconds elapsed between each transaction and the first transaction in the dataset.  The 'Amount' feature is the transaction amount.  The 'Class' feature is the target variable (1 for fraud, 0 for legitimate).

## Requirements

The following Python libraries are required:

*   pandas
*   numpy
*   scikit-learn
*   imbalanced-learn (imblearn)
*   pgmpy
*   networkx
*   matplotlib
*   pydot
*   graphviz (install separately - see below)

You can install the Python dependencies using pip:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with contents:

```
pandas
numpy
scikit-learn
imblearn
pgmpy
networkx
matplotlib
pydot
```

You also need to install Graphviz separately.  Instructions can be found on the Graphviz website: [https://graphviz.org/download/](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://www.google.com/url?sa=E%26source=gmail%26q=https://graphviz.org/download/). For Debian-based Linux distributions (like Ubuntu), you can use:

```bash
sudo apt-get install graphviz graphviz-dev
```

## Usage

To run the entire pipeline, execute the `main.py` script from the terminal:

```bash
python main.py
```

This will run all the steps in sequence, generating the output files and printing progress information to the console.

## Results

The final model achieves an AUPRC of 0.7842.  The optimal threshold for classification (based on maximizing the F1-score) is 0.9993, resulting in a precision of 0.8762 and a recall of 0.7622. The learned Bayesian network structure and the Markov blanket visualization provide insights into the key factors influencing fraud predictions. The `improved_bayesian_network.png` and `markov_blanket.png` files provide visualizations of the network.

## Future Work

Potential future improvements include:

*   **Causal Inference:**  Performing formal causal inference using the do-calculus to quantify the impact of interventions.
*   **Dynamic Bayesian Networks:** Extending the model to a Dynamic Bayesian Network to capture temporal dependencies.
*   **Ensemble Methods:** Combining the Bayesian network with other machine learning models for improved prediction accuracy.
*   **Model Monitoring and Retraining:** Implementing a system for automatically monitoring model performance and retraining when necessary.
*   **Feature Interaction Exploration:** Investigating interactions between features to identify more complex fraud patterns.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/url?sa=E&source=gmail&q=LICENSE) file for details.
