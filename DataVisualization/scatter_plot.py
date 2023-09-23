import argparse
import matplotlib.pyplot as plt
import pandas as pd

def scatter_plot(file, feature1, feature2):
    print(file, feature1, feature2)

    df = pd.read_csv(file)
    unique_houses = df['Hogwarts House'].unique()

    color_map = {house: f'C{i}' for i, house in enumerate(unique_houses)}
    plt.figure(figsize=(8,6))
    
    for house in unique_houses:
        data = df[df['Hogwarts House'] == house].dropna()
        plt.scatter(data[feature1], data[feature2], alpha=0.7, label=f'{house}', color=color_map.get(house))
    
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='A scatter plot to compare two features')

    parser.add_argument(
        '--csv_file',
        default='../datasets/dataset_train.csv',
        help='Path to a data CSV file (default = dataset_train.csv)',
    )
    parser.add_argument(
        '-f1', '--feature1',
        default='Flying',
        help='name of the first feature (default = Flying)'
    )
    parser.add_argument(
        '-f2', '--feature2',
        default='Potions',
        help='name of the second feature (default = Potions)'
    )

    args = parser.parse_args()
    scatter_plot(args.csv_file, args.feature1, args.feature2)

if __name__ == '__main__':
    main()