import argparse
import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression

def train(file: str) -> None:
    df = pd.read_csv(file)
    df.dropna()
    y = np.array(df['Hogwarts House'])
    X = np.array(df[["Herbology", "Divination", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]])
    # nan_rows = np.isnan(X).any(axis=1)
    X[np.isnan(X)] = -1

    # min_val = X.min(axis=0)  # Minimum value for each feature
    # max_val = X.max(axis=0)  # Maximum value for each feature

    # X = (X - min_val) / (max_val - min_val)
    mean = X.mean(axis=0)  # Mean value for each feature
    std_dev = X.std(axis=0)  # Standard deviation for each feature

    X = (X - mean) / std_dev
    LogReg = LogisticRegression()
    LogReg.fit(X, y)
    LogReg.predict()
    

def main() -> None:
    parser = argparse.ArgumentParser(description='Logistic Regression model train with Gradient Descent')

    parser.add_argument(
        '--csv_file',
        default='../datasets/dataset_train.csv',
        help='Path to a data CSV file (default = dataset_train.csv)',
    )

    args = parser.parse_args()
    train(args.csv_file)
    


if __name__ == '__main__':
    main()