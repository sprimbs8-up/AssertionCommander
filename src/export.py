import abc
import json
import logging
from typing import List, Dict

from src.models import Models


class Exporter(abc.ABC):
    def __init__(self, model: Models, assertion_number: int):
        self.model: Models = model
        self.assertion_number: int = assertion_number

    @abc.abstractmethod
    def export_predictions(
        self, references: List[str], top_k_predictions: List[List[str]]
    ):
        pass

    @abc.abstractmethod
    def export_metrics(self, metrics: Dict[str, float]):
        pass


class LoggingExporter(Exporter):
    def export_predictions(
        self, references: List[str], top_k_predictions: List[List[str]]
    ):
        for ex, top_k in zip(references, top_k_predictions):
            output: str = "EXPECTED:" + json.dumps(ex) + "  TOP_K:" + json.dumps(top_k)
            logging.info(output)

    def export_metrics(self, metrics: Dict[str, float]):
        logging.info("=" * 100)
        logging.info("Metrics:")
        logging.info(metrics)
        logging.info("=" * 100)


def get_exporters_from_str(
    exporters: str, model: Models, assertion_number: int
) -> set[Exporter]:
    return set(
        [
            parse_exporter(exporter, model, assertion_number)
            for exporter in exporters.split(":")
        ]
    )


def parse_exporter(exporter: str, model: Models, assertion_number: int) -> Exporter:
    match exporter:
        case "logging":
            return LoggingExporter(model, assertion_number)
    raise NotImplementedError(f'The exporter "{exporter}" is not available.')
