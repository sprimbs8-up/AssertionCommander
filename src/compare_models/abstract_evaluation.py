import argparse
import csv
import dataclasses
import json
import logging
from pathlib import Path

from tqdm import tqdm

ABSTRACT_TOKEN_IDENTIFIERS = ["IDENT","METHOD","CHAR","STRING","INT","BOOL","BYTE","SHORT","DOUBLE","FLOAT"]

def abstract_evaluation(
    data_file_path: Path, abstract_file_path: Path, output_file_path: Path
) -> None:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(data_file_path) as data_file, Path.open(
        abstract_file_path
    ) as abstract_file, Path.open(output_file_path, "w") as output_file:
        abstract_file_reader = csv.reader(abstract_file)
        dict_counter = []
        trans_dict_len = 0
        for data, abstract_data in tqdm(
            zip(data_file, abstract_file_reader, strict=False), leave=False
        ):
            loaded_dict = json.loads(data)
            translation_dict = loaded_dict["dict"]
            trans_dict_len +=len(translation_dict)
            loaded_assertion = loaded_dict["labels"]
            expected_abstract, *abstract_predictions = abstract_data
            if expected_abstract != loaded_assertion:
                logging.info("Error!")
                continue
            counter = 0
            abstract_preds_spaces = abstract_predictions[0].replace(","," , ").replace("."," . ").replace("!="," !=").replace("?"," ? ").replace("'"," ' ")
            abstract_predictions_split = abstract_preds_spaces.split()
            unknown = set()
            for idx, predicted_assertion_token in enumerate(abstract_predictions_split):
                if predicted_assertion_token not in translation_dict and is_abstract_token(predicted_assertion_token):
                    counter+=1
                    unknown.add(predicted_assertion_token)
            if counter >=4:
                print("\n===" +str(counter))
                print(abstract_preds_spaces)
                print(expected_abstract)
                print(unknown)
                print("\n===")
            dict_counter.append(counter)
        available_values = set(dict_counter)
        min_value = min(available_values)
        max_value = max(available_values)
        failing_assert_types = {val:0 for val in range(min_value, max_value+1)}
        for el in dict_counter:
            failing_assert_types[el] += 1
        print(failing_assert_types)
        print(trans_dict_len / len(dict_counter))
        json.dump(failing_assert_types, output_file)


def is_abstract_token(token: str):
    if token == "TRY_CATCH":
        return False
    identifier, *rest = token.split("_")
    if len(rest) != 1:
        return False
    return identifier in ABSTRACT_TOKEN_IDENTIFIERS


def _clean(assertion: str) -> str:
    return (
        assertion.replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("\xa0", "")
        .replace("　　", "")
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Perform a model prediction comparison for the raw and abstract models."
    )
    parser.add_argument(
        "-a", "--abstract-file", help="Input Abstract Prediction File", required=True
    )
    parser.add_argument(
        "-d", "--data-file", help="Input Data File", required=True
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="The directory where the results should be saved.",
        required=True,
    )

    args = parser.parse_args()
    data_file = Path(args.data_file)
    absolute_file_path = Path(args.abstract_file)
    output_file = Path(args.output_file)
    abstract_evaluation(data_file, absolute_file_path, output_file)


if __name__ == "__main__":
    main()
