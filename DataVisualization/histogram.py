import argparse
import matplotlib.pyplot as plt
import pandas as pd


def histogram(file, course):
    # print(file, course)

    df = pd.read_csv(file)

    unique_houses = df['Hogwarts House'].unique()
    
    color_map = {house: f'C{i}' for i, house in enumerate(unique_houses)}
    plt.figure(figsize=(8,6))
    plt.ylabel('Frequency')

    for house in unique_houses:
        data = df[df['Hogwarts House'] == house].dropna()
        plt.hist(data[course], bins=10, alpha=0.7, label=f'{house}', color=color_map.get(house))

    plt.legend()
    # plt.xlim(-4, 4)
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description='A histogram display the score of the four houses for a given course')

    parser.add_argument(
        '--csv_file',
        default='../datasets/dataset_train.csv',
        help='Path to a data CSV file (default = dataset_train.csv)',
    )
    parser.add_argument(
        '--course',
        default='Flying',
        help='Name of the course (default = Flying)'
    )

    args = parser.parse_args()
    histogram(args.csv_file, args.course)


if __name__ == '__main__':
    main()