import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write a file to parquet format')
    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('output', type=str, help='output file')
    args = parser.parse_args()
    if '.txt' in args.input:
        df = pd.read_table(args.input, na_values=['MAX'])  # For off targets
    else:
        raise ValueError('Only accepts txt files at the moment')
    if '.parquet' in args.output:
        df.to_parquet(args.output)
    else:
        raise ValueError('Please include .parquet ending in output')
