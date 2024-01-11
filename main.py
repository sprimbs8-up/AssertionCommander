import argparse
import sys
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
from src.commander import AssertionCommander


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")

    parser.add_argument(
        "--batch_size",
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

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> int:
    batch_size = args.batch_size
    top_k = args.top_k
    model_url = args.model_url
    num_assert = args.num_assert
    model = args.model
    exporters = args.exporter
    commander: AssertionCommander = AssertionCommander(
        model_url=model_url,
        model_name=model,
        batch_size=batch_size,
        top_k=top_k,
        assertion_number=num_assert,
        exporters=exporters,
    )
    _log_args(batch_size, model, model_url, num_assert, top_k, exporters)
    commander.evaluate()
    return 0


def _log_args(batch_size, model, model_url, num_assert, top_k, exporters) -> None:
    logging.info("Configuration:")
    logging.info(f"- Batch Size:           {batch_size}")
    logging.info(f"- Top-k:                {top_k}")
    logging.info(f"- Model URL:            {model_url}")
    logging.info(f"- Number of Assertions: {num_assert}")
    logging.info(f"- Selected Model:       {model}")
    logging.info(f"- Exporters:            {exporters}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    )
    parsed_args = parse_arguments()
    with logging_redirect_tqdm():
        sys.exit(main(parsed_args))
