import pandas as pd
import argparse
from helper import Helper
    

def describe(file):
    df = pd.read_csv(file)
    helper = Helper(df)

    summary = {}

    for column in df.columns:
        if df[column].dtype in [int, float]:
            summary[column] = [
                helper.count(column),
                helper.mean(column),
                df[column].std(),
                helper.min(column),
                helper.quantile(25, column),
                df[column].median(),
                df[column].quantile(0.75),
                helper.max(column),
            ]

    summary_df = pd.DataFrame(summary);
    summary_df.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    return summary_df

def main():
    parser = argparse.ArgumentParser(description='describe statistics for a CSV file.')

    parser.add_argument(
        'csv_file',
        help='Path to a data CSV file'
    )

    args = parser.parse_args();

    summary = describe(args.csv_file)

    print(summary)


if __name__ == '__main__':
    main()
