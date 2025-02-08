# main.py

import subprocess

def run_script(script_name):
    """Runs a Python script and handles potential errors."""
    try:
        subprocess.run(['python', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        exit(1)  # Exit the main script if any sub-script fails

if __name__ == '__main__':
    print("Starting Credit Card Fraud Detection Pipeline")

    print("\n--- Step 1: Data Ingestion and Preprocessing ---")
    run_script('01_data_ingestion_and_preprocessing.py')

    print("\n--- Step 2: Train/Test Split ---")
    run_script('02_train_test_split.py')

    print("\n--- Step 3: Feature Engineering and Resampling (Train Set) ---")
    run_script('03_feature_engineering_resampling_train.py')

    print("\n--- Step 4: DAG Structure Learning (Train Set) ---")
    run_script('04_dag_structure_learning_train.py')

    print("\n--- Step 5: Parameter Learning (Train Set) ---")
    run_script('05_parameter_learning_train.py') #Train

    print("\n--- Step 6: Feature Engineering (Test Set) ---")
    run_script('06_feature_engineering_test.py')

    print("\n--- Step 7: Inference and Evaluation ---")
    run_script('07_inference_and_evaluation.py')

    print("\n--- Step 8: Improved Visualization ---")
    run_script('08_visualization_improvements.py')

    print("\n--- Step 9: Causal Inference ---")
    run_script('09_causal_inference.py')

    print("\n--- Step 10: Train XGBoost Model ---")
    run_script('10_train_xgboost.py')


    print("\n--- Pipeline Complete ---")