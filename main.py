# main.py

import subprocess
import sys
import argparse
from tqdm import tqdm
from utils import setup_logger, load_config

logger = setup_logger(__name__)

PIPELINE_STEPS = [
    ('1_data_ingestion', '01_data_ingestion_and_preprocessing.py'),
    ('2_split_data', '02_train_test_split.py'),
    ('3_feature_engineering_train', '03_feature_engineering_resampling_train.py'),
    ('4_dag_structure', '04_dag_structure_learning_train.py'),
    ('5_parameter_learning', '05_parameter_learning_train.py'),
    ('6_feature_engineering_test', '06_feature_engineering_test.py'),
    ('7_inference', '07_inference_and_evaluation.py'),
    ('9_causal_inference', '09_causal_inference.py'),
    ('10_xgboost', '10_train_xgboost.py'),
    ('11_calibration', '011_calibration.py')
]

def run_script(script_name, wait=True):
    try:
        if wait:
            # Check=True will surface any errors and halt the master script execution 
            subprocess.run([sys.executable, script_name], check=True)
            print("") # spacer
        else:
            subprocess.Popen([sys.executable, script_name])
    except subprocess.CalledProcessError as e:
        logger.error(f"FATAL: Execution of '{script_name}' failed -> {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for Fraud Detection DAG")
    parser.add_argument('--step', type=str, help='Run a specific step (e.g., "1_data_ingestion"). If not specified, runs all steps.')
    parser.add_argument('--dashboard', action='store_true', help='Launch visualization dashboard asynchronously at end.')
    args = parser.parse_args()

    config = load_config()
    logger.info("Configuration Hub Loaded Successfuly.")
    
    if args.step:
        step_map = dict(PIPELINE_STEPS)
        if args.step not in step_map:
            logger.error(f"Invalid step name: {args.step}. Available: {list(step_map.keys())}")
            sys.exit(1)
        run_script(step_map[args.step])
    else:
        # Run all sequential steps with a master progress bar
        print("\n" + "="*50)
        print("  MASTER FRAUD DETECTION DAG PIPELINE STARTING")
        print("="*50 + "\n")
        
        for step_name, script_name in tqdm(PIPELINE_STEPS, desc="Overall Pipeline Progress", position=0, leave=True):
            print(f"\n--- Running Step: {step_name} ({script_name}) ---")
            run_script(script_name)

    if args.dashboard:
        logger.info("Starting Dashboard in the background...")
        run_script('08_visualization_improvements.py', wait=False)

if __name__ == '__main__':
    main()
