import argparse
import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression
from preprocessor import preprocessor
import time

def train(file: str) -> None:
    df = pd.read_csv(file)
    y = np.array(df['Hogwarts House'])
    X = np.array(df[["Herbology", "Divination", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]])

    X = preprocessor(X)

    LogReg = LogisticRegression()
    LogReg.fit(X, y)
    y_pred = LogReg.predict(X)
    print('accuracy score ==> ', np.sum(y == y_pred) / len(y))


def main() -> None:
    parser = argparse.ArgumentParser(description='Logistic Regression model train with Gradient Descent')

    parser.add_argument(
        '--csv_file',
        default='../datasets/dataset_train.csv',
        help='Path to a data CSV file (default = dataset_train.csv)',
    )

    args = parser.parse_args()
    a=time.time()
    train(args.csv_file)
    b=time.time()
    print('it took:', b-a)

    


if __name__ == '__main__':
    main()