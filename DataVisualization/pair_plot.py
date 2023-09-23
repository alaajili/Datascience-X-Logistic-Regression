import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot(file):
    df = pd.read_csv(file)
    sns.set_theme(style="ticks")

    numeric_data = df.select_dtypes(include='number').reset_index(drop=True)

    sns.pairplot(data=numeric_data, hue='Hogwarts House', diag_kind='hist')
    plt.show()





def main():
    parser = argparse.ArgumentParser(description='A scatter plot to compare two features')

    parser.add_argument(
        '--csv_file',
        default='../datasets/dataset_train.csv',
        help='Path to a data CSV file (default = dataset_train.csv)',
    )

    args = parser.parse_args()
    pair_plot(args.csv_file)

if __name__ == '__main__':
    main()
