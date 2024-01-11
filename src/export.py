import abc
import json
import logging
import csv
from pathlib import Path
from typing import List, Dict

from src.models import Models


class Exporter(abc.ABC):
    def __init__(self, model: Models, assertion_number: int, top_k: int):
        self.model: Models = model
        self.assertion_number: int = assertion_number
        self.top_k = top_k

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    @abc.abstractmethod
    def export_predictions(
        self, references: List[str], top_k_predictions: List[List[str]]
    ):
        pass

    @abc.abstractmethod
    def export_metrics(self, metrics: Dict[str, float]):
        pass

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class ConsoleExporter(Exporter):
    def export_predictions(
        self, references: List[str], top_k_predictions: List[List[str]]
    ):
        pass

    def export_metrics(self, metrics: Dict[str, float]):
        logging.info("=" * 100)
        logging.info("Metrics:")
        logging.info(metrics)
        logging.info("=" * 100)

    def initialize(self):
        pass

    def close(self):
        pass


class FileWriterExporter(Exporter):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        top_k: int,
        default_dir: str = "results",
    ):
        super().__init__(model, assertion_number, top_k)
        self.prediction_file = None
        self.metric_file = None
        self.default_dir = default_dir
        self.base_path: Path = (
            Path(self.default_dir) / str(self.assertion_number) / self.model.name
        )
        self.prediction_dir: Path = self.base_path / "predictions"
        self.prediction_file_path: Path = self.prediction_dir / f"top-{top_k}.csv"
        self.metric_dir: Path = self.base_path / "metrics"
        self.metric_file_path: Path = self.metric_dir / f"top-{top_k}.json"

    def export_predictions(
        self, references: List[str], top_k_predictions: List[List[str]]
    ):
        for ref, top_k in zip(references, top_k_predictions):
            csv_writer = csv.writer(self.prediction_file)
            csv_writer.writerow([ref.replace("\n", "")] + top_k)

    def export_metrics(self, metrics: Dict[str, float]):
        json.dump(metrics, self.metric_file)

    def initialize(self):
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_file = open(self.prediction_file_path, mode="w")
        self.metric_file = open(self.metric_file_path, mode="w")

    def close(self):
        self.prediction_file.close()
        self.metric_file.close()


class CombinedExporter(Exporter):
    def __init__(self, model: Models, assertion_number: int, top_k: int):
        super().__init__(model, assertion_number, top_k)
        self.exporters = {
            ConsoleExporter(model, assertion_number, top_k),
            FileWriterExporter(model, assertion_number, top_k),
        }

    def export_predictions(
        self, references: List[str], top_k_predictions: List[List[str]]
    ):
        for exporter in self.exporters:
            exporter.export_predictions(references, top_k_predictions)

    def export_metrics(self, metrics: Dict[str, float]):
        for exporter in self.exporters:
            exporter.export_metrics(metrics)

    def initialize(self):
        for exporter in self.exporters:
            exporter.initialize()

    def close(self):
        for exporter in self.exporters:
            exporter.close()


def get_exporters_from_str(
    exporters: str, model: Models, assertion_number: int, top_k: int
) -> CombinedExporter:
    return CombinedExporter(model, assertion_number, top_k)


def parse_exporter(
    exporter: str, model: Models, assertion_number: int, top_k: int
) -> Exporter:
    match exporter:
        case "console":
            return ConsoleExporter(model, assertion_number, top_k)
        case "file":
            return FileWriterExporter(model, assertion_number, top_k)
    raise NotImplementedError(f'The exporter "{exporter}" is not available.')
