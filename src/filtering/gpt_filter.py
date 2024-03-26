import argparse
import csv

from src.evaluate.main import ASSERTION_TYPES
from io import StringIO


def filter_row_if_no_valid_junit_assertion(input_file, output_file):
    with open(input_file, 'rb') as infile, open(output_file, 'w', newline='') as outfile:
        text = infile.read().replace(b"\0",b"").decode(encoding="utf-8")
        file = StringIO(text)
        reader = csv.reader(file)
        writer = csv.writer(outfile)
        for row in reader:
            expected_assertion, *assertions = row
            filtered_assertion_list = list(filter(lambda token: token in ASSERTION_TYPES, [tokens.split()[0] for tokens in assertions]))
            if len(filtered_assertion_list) == len(assertions):
                writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Filter CSV data based on a specified column and value.')
    parser.add_argument("-i","--input_file", help="Input CSV file", required=True)
    parser.add_argument("-o","--output_file", help="Input CSV file", required=True)

    args = parser.parse_args()

    filter_row_if_no_valid_junit_assertion(args.input_file, args.output_file)

if __name__ == '__main__':
    main()