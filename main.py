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

    print("\n--- Step 2: Discretization ---")
    run_script('02_discretization.py')

    print("\n--- Step 3: Resampling ---")
    run_script('03_resampling.py')

    print("\n--- Step 4: DAG Structure Learning ---")
    run_script('04_dag_structure_learning.py')

    print("\n--- Step 5: Parameter Learning ---")
    run_script('05_parameter_learning.py')

    print("\n--- Step 6: Inference and Evaluation ---")
    run_script('06_inference_and_evaluation.py')

    print("\n--- Step 7: Reporting and Visualization ---")
    run_script('07_reporting_and_visualization.py')

    print("\n--- Step 8: Improved Visualization ---")
    run_script('08_visualization_improvements.py')

    print("\n--- Pipeline Complete ---")