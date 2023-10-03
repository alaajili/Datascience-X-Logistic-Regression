import argparse
import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression
from preprocessor import preprocessor


def predict(file: str, weights_file: str) -> None:
    df = pd.read_csv(file)
    X = np.array(df[["Herbology", "Divination", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]])

    X = preprocessor(X)

    thetas = np.load(weights_file)

    LogReg = LogisticRegression()
    predictions = LogReg.predict(X, thetas)
    df = pd.DataFrame({'Hogwarts House': predictions})
    df.to_csv('houses.csv', index_label='Index')

    true = pd.read_csv('../datasets/dataset_truth.csv')
    y_true = true['Hogwarts House']
    y_pred = df['Hogwarts House']
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