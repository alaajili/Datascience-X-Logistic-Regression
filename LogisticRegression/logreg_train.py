import argparse
import pandas as pd
import numpy as np

def train(file: str) -> None:
    df = pd.read_csv(file)

    y = np.array(df['Hogwarts House'])
    X = np.array(df[["Herbology", "Divination", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]])
    print(y)
    

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