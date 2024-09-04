import argparse
import logging
from pathlib import Path


def handle_files(
    allowed_lines_path: Path, input_file_path: Path, output_file_path: Path
) -> None:
    """
    Handle files based on allowed lines and exports only them.

    Args:
        allowed_lines_path (Path): Path to the file containing allowed line numbers.
        input_file_path (Path): Path to the input file.
        output_file_path (Path): Path to the output file.
    """
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    # filter allowed lines
    with Path.open(allowed_lines_path) as allowed_lines_file:
        allowed_lines = [int(line.strip()) for line in allowed_lines_file.readlines()]

    # exports them
    with Path.open(input_file_path) as input_file, Path.open(
        output_file_path, "w"
    ) as output_file:
        counter = 0
        for prediction in input_file:
            if counter in allowed_lines:
                output_file.write(prediction)
            counter += 1


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    )
    parser = argparse.ArgumentParser(description="Exports the data lines.")
    parser.add_argument(
        "-l", "--allowed-lines", help="The allowed lines.", required=True
    )
    parser.add_argument(
        "-i", "--input-file", help="The input prediction file.", required=True
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="The directory where the results should be saved.",
        required=True,
    )

    args = parser.parse_args()
    allowed_lines = Path(args.allowed_lines)
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    handle_files(allowed_lines, input_file, output_file)


if __name__ == "__main__":
    main()
