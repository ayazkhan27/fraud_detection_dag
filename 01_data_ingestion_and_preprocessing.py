# 01_data_ingestion_and_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path="creditcard.csv"):
    """Loads, cleans, and standardizes the credit card dataset.

    Args:
        file_path (str): Path to the CSV dataset.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """
    # Data Loading & Exploration
    df = pd.read_csv(file_path)

    # Check for Missing Values (using isnull() and sum())
    print("Missing Values Before Handling:\n", df.isnull().sum())

    # Standardization
    features_to_scale = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    print("\nFirst 5 Rows of Preprocessed Data:")
    print(df.head())

    return df

if __name__ == '__main__':
    df_processed = load_and_preprocess()
    # Save processed data for the next step
    df_processed.to_csv("preprocessed_data.csv", index=False)

    #Output Summary for verification
    print("\nSummary of Preprocessing:")
    print(f"  Shape of the processed DataFrame: {df_processed.shape}")
    print(f"  Data types of the columns:\n{df_processed.dtypes}")