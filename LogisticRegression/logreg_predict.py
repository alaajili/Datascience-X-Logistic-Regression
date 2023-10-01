import argparse
import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression


def predict(file: str, weights_file: str) -> None:
    df = pd.read_csv(file)
    X = np.array(df[["Herbology", "Divination", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]])
    # X[np.isnan(X)] = -1
    for i in range(X.shape[1]):
        feature_mean = np.nanmean(X[:, i])
        X[np.isnan(X[:, i]), i] = feature_mean

    min_val = X.min(axis=0)  # Minimum value for each feature
    max_val = X.max(axis=0)  # Maximum value for each feature
    X = (X - min_val) / (max_val - min_val)

    # mean = X.mean(axis=0)  # Mean value for each feature
    # std_dev = X.std(axis=0)  # Standard deviation for each feature

    # X = (X - mean) / std_dev
    X = np.c_[np.ones((X.shape[0], 1)), X]

    LogReg = LogisticRegression()
    predictions = LogReg.predict(X)
    df = pd.DataFrame({'Hogwarts House': predictions})
    df.to_csv('houses.csv', index_label='Index')

    true = pd.read_csv('../datasets/dataset_truth.csv')
    y_true = true['Hogwarts House']
    y_pred = df['Hogwarts House']
    # print(y_true)
    # accuracy_score = np.round(np.sum(y_true == predictions))
    print('accuracy score ==> ', np.sum(y_true == y_pred) / len(y_true))


def main() -> None:
    parser = argparse.ArgumentParser(description='predict values with weights')

    parser.add_argument(
        '--csv_file',
        default='../datasets/dataset_test.csv',
        help='Path to a data CSV file (default = dataset_train.csv)',
    )

    parser.add_argument(
        '--weights_file',
        default='weights.npy',
        help='Path to a your weights file (default = weights.npy)',
    )

    args = parser.parse_args()
    predict(args.csv_file, args.weights_file)

if __name__ == '__main__':
    main()