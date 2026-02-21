# utils.py
import yaml
import logging
import sys
import os

def load_config(config_path="config.yaml"):
    """Loads the central YAML configuration file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML configuration: {exc}")
        sys.exit(1)

def setup_logger(name):
    """Sets up a standardized logger for the pipeline modules."""
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't already have handlers to prevent duplicate logs
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        # Create console handler with formatting
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def ensure_dir(dir_path):
    """Ensures a directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
