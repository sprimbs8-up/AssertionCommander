import argparse
import csv
import dataclasses
import json
import logging
from pathlib import Path
from tqdm import tqdm


@dataclasses.dataclass
class VariantComparator:
    def __init__(
        self,
        total: int,
        correct_in_both: int,
        correct_in_raw: int,
        correct_in_abstract: int,
    ):
        self.total = total
        self.correct_in_both = correct_in_both
        self.correct_in_raw = correct_in_raw
        self.correct_in_abstract = correct_in_abstract

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "correct_in_both": self.correct_in_both,
            "correct_in_raw": self.correct_in_raw,
            "correct_in_abstract": self.correct_in_abstract,
        }


def compare_files(
    raw_file_path: Path, abstract_file_path: Path, output_file_path: Path
):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_file_path) as raw_file, open(
        abstract_file_path
    ) as abstract_file, open(output_file_path, "w") as output_file:
        raw_file_reader = csv.reader(raw_file)
        abstract_file_reader = csv.reader(abstract_file)
        both_correct = 0
        correct_in_raw_variant = 0
        correct_in_abstract_variant = 0
        total = 0
        for raw_data, abstract_data in tqdm(
            zip(raw_file_reader, abstract_file_reader, strict=False), leave=False
        ):
            expected_raw, *raw_predictions = raw_data
            expected_abstract, *abstract_predictions = abstract_data
            if _clean(expected_raw) != _clean(expected_abstract):
                logging.warning("The expected assertions do no match.")
                continue
            expected = _clean(expected_raw)
            raw_predictions_list = list(map(_clean, raw_predictions))
            abstract_predictions_list = list(map(_clean, abstract_predictions))
            if (
                expected in raw_predictions_list
                and expected in abstract_predictions_list
            ):
                both_correct += 1
            elif expected in raw_predictions_list:
                correct_in_raw_variant += 1
            elif expected in abstract_predictions_list:
                correct_in_abstract_variant += 1
            total += 1
        variant_comparator_container = VariantComparator(
            total, both_correct, correct_in_raw_variant, correct_in_abstract_variant
        )
        json.dump(variant_comparator_container.to_dict(), output_file)
        logging.info(
            f"Total {total} - Correct in both variants: {both_correct} - Correct in raw variant: {correct_in_raw_variant} - Correct in abstract variant: {correct_in_abstract_variant}"
        )
        logging.info(
            f"Total {total} - Correct in both variants: {both_correct / total * 100}% - Correct in raw variant: {correct_in_raw_variant / total * 100}% - Correct in abstract variant: {correct_in_abstract_variant/total * 100}%"
        )


def _clean(assertion: str) -> str:
    return (
        assertion.replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("\xa0", "")
        .replace("　　", "")
    )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Perform a model prediction comparison for the raw and abstract models."
    )
    parser.add_argument(
        "-r", "--raw-file", help="Input Raw Prediction File", required=True
    )
    parser.add_argument(
        "-a", "--abstract-file", help="Input Abstract Prediction File", required=True
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="The directory where the results should be saved.",
        required=True,
    )

    args = parser.parse_args()
    raw_file_path = Path(args.raw_file)
    absolute_file_path = Path(args.abstract_file)
    output_file = Path(args.output_file)
    compare_files(raw_file_path, absolute_file_path, output_file)


if __name__ == "__main__":
    main()
