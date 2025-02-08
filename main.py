# main.py

import subprocess
import sys

def run_script(script_name, wait=True):
    """Runs a Python script.
    
    Args:
        script_name (str): The script to run.
        wait (bool): If True, wait for the script to finish; if False, run in the background.
    """
    try:
        if wait:
            subprocess.run(['python', script_name], check=True)
        else:
            subprocess.Popen(['python', script_name])
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)  # Exit the main script if any sub-script fails

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
    run_script('05_parameter_learning_train.py')

    print("\n--- Step 6: Feature Engineering (Test Set) ---")
    run_script('06_feature_engineering_test.py')

    print("\n--- Step 7: Inference and Evaluation ---")
    run_script('07_inference_and_evaluation.py')

    print("\n--- Step 9: Causal Inference ---")
    run_script('09_causal_inference.py')

    print("\n--- Step 10: Train XGBoost Model ---")
    run_script('10_train_xgboost.py')

    print("\n--- Step 11: Test Calibration ---")
    run_script('011_calibration.py')

    # Launch Step 8 (Improved Visualization) last, non-blocking.
    print("\n--- Launching Improved Visualization Dashboard (Non-Blocking) ---")
    run_script('08_visualization_improvements.py', wait=False)

    print("\n--- Pipeline Complete ---")
