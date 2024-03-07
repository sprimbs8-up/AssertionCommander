import abc
import csv
import json
import logging
from pathlib import Path

from src.dataset_type import DatasetType
from src.models import Models


class Exporter(abc.ABC):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        top_k: int,
        default_dir: str,
        dataset_type: DatasetType,
        no_pred_export: bool,
        raw_file_path: str,
        epoch: str,
    ) -> None:
        self.model: Models = model
        self.assertion_number: int = assertion_number
        self.top_k = top_k
        self.default_dir = default_dir
        self.dataset_type = dataset_type
        self.no_pred_export = no_pred_export
        self.epoch = epoch

    def __enter__(self) -> object:
        self.initialize()
        return self

    def __exit__(self, *exc_details: object) -> None:
        self.close()

    @abc.abstractmethod
    def export_predictions(
        self, references: list[str], top_k_predictions: list[list[str]]
    ) -> None:
        pass

    @abc.abstractmethod
    def export_metrics(self, metrics: dict[str, float]) -> None:
        pass

    @abc.abstractmethod
    def initialize(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class ConsoleExporter(Exporter):
    def export_predictions(
        self, references: list[str], top_k_predictions: list[list[str]]
    ) -> None:
        pass

    def export_metrics(self, metrics: dict[str, any]) -> None:
        logging.info("=" * 100)
        epoch_text = f"(Epoch {self.epoch})" if self.epoch is not None else ""
        logging.info("Metrics: %s", epoch_text)
        maximal_key = max([len(key) for key in metrics])
        for metric in metrics:
            if type(metrics[metric]) is dict:
                maximal_key_metric = max([len(key) for key in metrics[metric]])
                metric_dict = metrics[metric]
                logging.info(
                    " - %s:",
                    metric,
                )
                for m in metric_dict:
                    logging.info(
                        "   + %s:%s%s",
                        m,
                        " " * (maximal_key_metric - len(m) + 1),
                        metric_dict[m],
                    )
            else:
                logging.info(
                    " - %s:%s%s",
                    metric,
                    " " * (maximal_key - len(metric) + 1),
                    metrics[metric],
                )

        logging.info("=" * 100)

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass


class FileWriterExporter(Exporter):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        top_k: int,
        default_dir: str,
        dataset_type: DatasetType,
        no_pred_export: bool,
        raw_file_path: str,
        epoch: str,
    ) -> None:
        super().__init__(
            model,
            assertion_number,
            top_k,
            default_dir,
            dataset_type,
            no_pred_export,
            raw_file_path,
            epoch,
        )
        self.prediction_file = None
        self.metric_file = None
        self.default_dir = default_dir
        self.base_path: Path = (
            Path(self.default_dir) / str(self.assertion_number) / self.model.name
        )
        if raw_file_path is not None:
            self.base_path = self.base_path / raw_file_path
        appendix: str = (
            f"epoch-{'%02d' % int(self.epoch)}." if self.epoch is not None else ""
        )
        self.prediction_dir: Path = self.base_path / "predictions"
        self.prediction_file_path: Path = (
            self.prediction_dir
            / f"{appendix}{self.dataset_type.type_name}_top-{top_k}.csv"
        )
        self.metric_dir: Path = self.base_path / "metrics"
        self.metric_file_path: Path = (
            self.metric_dir
            / f"{appendix}{self.dataset_type.type_name}_top-{top_k}.json"
        )

    def export_predictions(
        self, references: list[str], top_k_predictions: list[list[str]]
    ) -> None:
        if self.prediction_file is not None:
            for ref, top_k in zip(references, top_k_predictions, strict=False):
                csv_writer = csv.writer(self.prediction_file)
                csv_writer.writerow([ref.replace("\n", ""), *top_k])

    def export_metrics(self, metrics: dict[str, float]) -> None:
        json.dump(metrics, self.metric_file)

    def initialize(self) -> None:
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        if not self.no_pred_export:
            self.prediction_file = Path.open(self.prediction_file_path, mode="w")
        self.metric_file = Path.open(self.metric_file_path, mode="w")

    def close(self) -> None:
        if not self.no_pred_export:
            self.prediction_file.close()
        self.metric_file.close()


class CombinedExporter(Exporter):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        top_k: int,
        default_dir: str,
        dataset_type: DatasetType,
        exporters_str: str,
        no_pred_export: bool,
        raw_file: str,
        epoch: str,
    ) -> None:
        super().__init__(
            model,
            assertion_number,
            top_k,
            default_dir,
            dataset_type,
            no_pred_export,
            raw_file,
            epoch,
        )
        self.exporters = _get_exporters_from_str(
            exporters_str,
            model,
            assertion_number,
            top_k,
            default_dir,
            dataset_type,
            no_pred_export,
            raw_file,
            epoch,
        )

    def export_predictions(
        self, references: list[str], top_k_predictions: list[list[str]]
    ) -> None:
        for exporter in self.exporters:
            exporter.export_predictions(references, top_k_predictions)

    def export_metrics(self, metrics: dict[str, float]) -> None:
        for exporter in self.exporters:
            exporter.export_metrics(metrics)

    def initialize(self) -> None:
        for exporter in self.exporters:
            exporter.initialize()

    def close(self) -> None:
        for exporter in self.exporters:
            exporter.close()


def _get_exporters_from_str(
    exporters: str,
    model: Models,
    assertion_number: int,
    top_k: int,
    default_dir: str,
    dataset_type: DatasetType,
    no_pred_export: bool,
    raw_file: str,
    epoch: str,
) -> set[Exporter]:
    return {
        _parse_exporter(
            exporter,
            model,
            assertion_number,
            top_k,
            default_dir,
            dataset_type,
            no_pred_export,
            raw_file,
            epoch,
        )
        for exporter in exporters.split(":")
    }


def _parse_exporter(
    exporter: str,
    model: Models,
    assertion_number: int,
    top_k: int,
    default_dir: str,
    dataset_type: DatasetType,
    no_pred_export: bool,
    raw_file: str,
    epoch: str,
) -> Exporter:
    match exporter:
        case "console":
            return ConsoleExporter(
                model,
                assertion_number,
                top_k,
                default_dir,
                dataset_type,
                no_pred_export,
                raw_file,
                epoch,
            )
        case "file":
            return FileWriterExporter(
                model,
                assertion_number,
                top_k,
                default_dir,
                dataset_type,
                no_pred_export,
                raw_file,
                epoch,
            )
    error_msg = f'The exporter "{exporter}" is not available.'
    raise NotImplementedError(error_msg)
