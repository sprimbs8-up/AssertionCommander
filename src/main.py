import argparse
import logging
import sys

from tqdm.contrib.logging import logging_redirect_tqdm

from src.commander import AssertionCommander
from src.dataset_type import parse_type


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument(
        "--root-dir",
        default="evaluation-data",
        type=str,
        dest="root_dir",
        help="Specifies the root directory of the data.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        dest="batch_size",
        help="Specifies the batch size for processing data. Default is 32.",
    )

    parser.add_argument(
        "--top-k",
        default=1,
        type=int,
        dest="top_k",
        help="Determines the number of top results to consider. Default is 1.",
    )

    parser.add_argument(
        "--model-url",
        default="http://localhost:8080/",
        type=str,
        dest="model_url",
        help="Specifies the URL where the model is hosted. Default is http://localhost:8080/.",
    )

    parser.add_argument(
        "--number-assertions",
        default=1,
        type=int,
        dest="num_assert",
        help="Sets the number of assertions to be considered. Default is 1.",
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        dest="model",
        help="Specifies the type or identifier of the model to be used. This argument is required.",
    )
    parser.add_argument(
        "--export",
        default="file:console",
        type=str,
        dest="exporter",
        help="Specifies the type or identifier of the model to be used. This argument is required.",
    )
    parser.add_argument(
        "--export-dir",
        default="results",
        type=str,
        dest="export_dir",
        help="Specifies the root directory of the metrics to be exported.",
    )
    parser.add_argument(
        "--type",
        default="test",
        type=str,
        dest="type",
        help="Specifies the type or identifier of the model to be used. This argument is required.",
    )
    parser.add_argument(
        "--pred-file",
        default=None,
        type=str,
        dest="pred_file",
        help="Specifies the file for already created predictions.",
    )
    parser.add_argument(
        "--epoch",
        default=None,
        type=str,
        dest="epoch",
        help="Specifies the epoch under evaluation.",
    )
    parser.add_argument("--data-type", dest="data_type", default=None, help="The type of the data. Possible inputs: { None, raw, abstract, test_method }")

    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    _log_args(args)

    root_dir = args.root_dir
    batch_size = args.batch_size
    top_k = args.top_k
    model_url = args.model_url
    num_assert = args.num_assert
    model = args.model
    exporters = args.exporter
    dataset_type = args.type
    export_dir = args.export_dir
    pred_file = args.pred_file
    epoch = args.epoch
    data_type = args.data_type
    commander: AssertionCommander = AssertionCommander(
        model_url=model_url,
        model_name=model,
        batch_size=batch_size,
        top_k=top_k,
        assertion_number=num_assert,
        exporters=exporters,
        dataset_type=parse_type(dataset_type),
        root_dir=root_dir,
        export_dir=export_dir,
        cached_predictions_file=pred_file,
        data_type=data_type,
        epoch=epoch,
    )
    commander.evaluate()
    return 0


def _log_args(args: argparse.Namespace) -> None:
    logging.info("Configuration:")  #
    max_length = max([len(arg) for arg in vars(args)])
    for arg in vars(args):
        logging_str = f"- {arg}:{' '*(max_length - len(arg)+2)}{getattr(args, arg)}"
        logging.info(logging_str)


def run():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    )
    parsed_args = parse_arguments()
    with logging_redirect_tqdm():
        sys.exit(main(parsed_args))
if __name__ == "__main__":
    run()
