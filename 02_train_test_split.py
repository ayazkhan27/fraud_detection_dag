# 02_train_test_split.py
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_config, setup_logger

logger = setup_logger(__name__)

def split_data(config):
    """Splits the data into training and testing sets."""
    
    file_path = config["paths"]["preprocessed_data"]
    logger.info(f"Reading preprocessed data from {file_path}")
    df = pd.read_csv(file_path)

    # Split into features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    random_state = config["pipeline"]["random_state"]
    test_size = config["pipeline"]["test_size"]
    
    logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")
    # Split into training and testing sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    out_X_train = config["paths"]["X_train"]
    out_X_test = config["paths"]["X_test"]
    out_y_train = config["paths"]["y_train"]
    out_y_test = config["paths"]["y_test"]

    # Save training and testing sets
    X_train.to_csv(out_X_train, index=False)
    X_test.to_csv(out_X_test, index=False)
    y_train.to_csv(out_y_train, index=False, header=True)
    y_test.to_csv(out_y_test, index=False, header=True)

    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Testing set size: {X_test.shape}")
    logger.info(f"Data split into {out_X_train}, {out_X_test}, {out_y_train}, {out_y_test}")

if __name__ == '__main__':
    config = load_config()
    split_data(config)
