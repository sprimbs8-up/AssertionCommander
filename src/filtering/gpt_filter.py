import argparse
import csv
from io import StringIO
from pathlib import Path

from src.evaluate.main import ASSERTION_TYPES


def filter_row_if_no_valid_junit_assertion(input_file: Path, output_file: Path) -> None:
    """
    Filter rows in a CSV file based on the presence of valid JUnit assertions.

    Args:
        input_file (Path): Path to the input CSV file.
        output_file (Path): Path to the output CSV file.
    """
    with Path.open(input_file, "rb") as infile, Path.open(
        output_file, "w", newline=""
    ) as outfile:
        # Replace \0 bytes by empty byte.
        text = infile.read().replace(b"\0", b"").decode(encoding="utf-8")
        file = StringIO(text)
        reader = csv.reader(file)
        writer = csv.writer(outfile)
        for row in reader:
            expected_assertion, *assertions = row
            filtered_assertion_list = list(
                filter(
                    lambda token: token in ASSERTION_TYPES,
                    [tokens.split()[0] for tokens in assertions],
                )
            )
            if len(filtered_assertion_list) == len(assertions):
                writer.writerow(row)


def main() -> None:
    """Parses the arguments and filters all rows that have no valid junit assertion."""

    parser = argparse.ArgumentParser(
        description="Filter CSV data based on a specified column and value."
    )
    parser.add_argument("-i", "--input_file", help="Input CSV file", required=True)
    parser.add_argument("-o", "--output_file", help="Input CSV file", required=True)

    args = parser.parse_args()

    filter_row_if_no_valid_junit_assertion(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
