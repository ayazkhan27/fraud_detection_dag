# 01_data_ingestion_and_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import load_config, setup_logger
from tqdm import tqdm
import os

logger = setup_logger(__name__)

def load_and_preprocess(config):
    """Loads, cleans, and standardizes dataset with progress tracking."""
    file_path = config["paths"]["raw_data"]
    
    logger.info(f"Loading data from {file_path}")
    
    # 284807 rows is standard for creditcard dataset
    chunk_size = 50000
    estimated_rows = 284807 
    total_chunks = (estimated_rows // chunk_size) + 1
    
    try:
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        df_list = []
        for chunk in tqdm(chunks, total=total_chunks, desc="Reading CSV Chunks"):
            df_list.append(chunk)
        df = pd.concat(df_list, ignore_index=True)
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        raise
        
    missing_sum = df.isnull().sum().sum()
    logger.info(f"Total missing values before handling: {missing_sum}")

    features_to_scale = config["preprocessing"]["features_to_scale"]
    logger.info("Applying StandardScaler to continuous features...")
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    logger.info("Data ingestion and standardization complete.")
    return df

if __name__ == '__main__':
    config = load_config()
    df_processed = load_and_preprocess(config)
    
    out_path = config["paths"]["preprocessed_data"]
    logger.info("Saving preprocessed DataFrame in chunks...")
    
    chunk_size = 50000
    num_chunks = (len(df_processed) // chunk_size) + 1
    
    header = True
    for i, chunk in enumerate(tqdm(np.array_split(df_processed, num_chunks), total=num_chunks, desc="Saving CSV")):
        chunk.to_csv(out_path, mode='a' if not header else 'w', index=False, header=header)
        header = False
        
    logger.info(f"Preprocessed DataFrame saved to: {out_path} with shape {df_processed.shape}")
