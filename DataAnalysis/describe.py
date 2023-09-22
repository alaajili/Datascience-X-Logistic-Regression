import pandas as pd
import argparse
from description import Description
    

def describe(file):
    df = pd.read_csv(file)
    description = Description(df)

    summary = {}

    for column in df.columns:
        if df[column].dtype in [int, float]:
            values = description.get_values(column)
            summary[column] = [
                values['count'],
                values['mean'],
                values['std'],
                values['min'],
                values['25%'],
                values['50%'],
                values['75%'],
                values['max'],
            ]

    summary_df = pd.DataFrame(summary)
    summary_df.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    return summary_df

def main():
    parser = argparse.ArgumentParser(description='describe statistics for a CSV file.')

    parser.add_argument(
        'csv_file',
        help='Path to a data CSV file'
    )

    args = parser.parse_args()

    summary = describe(args.csv_file)

    print(summary)


if __name__ == '__main__':
    main()
