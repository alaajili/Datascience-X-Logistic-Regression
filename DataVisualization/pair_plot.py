import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def pair_plot(file):
    df = pd.read_csv(file)

    numeric_data = df.select_dtypes(include='number').drop(columns=['Index'])
    numeric_data['Hogwarts House'] = df['Hogwarts House']

    g = sns.pairplot(numeric_data, hue='Hogwarts House', diag_kind='hist', height=1.1)

    for ax in g.axes.flat:
        ax.set_ylabel(ax.get_ylabel().replace(' ', '\n'))
        ax.set_xlabel(ax.get_xlabel().replace(' ', '\n'))


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
