# 02_train_test_split.py
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path="preprocessed_data.csv"):
    """Splits the data into training and testing sets."""

    df = pd.read_csv(file_path)

    # Split into features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split into training and testing sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save training and testing sets
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False, header=True)
    y_test.to_csv("y_test.csv", index=False, header=True)

    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print("Data split into X_train.csv, X_test.csv, y_train.csv, y_test.csv")


if __name__ == '__main__':
    split_data()