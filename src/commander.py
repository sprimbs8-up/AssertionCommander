import json
import logging
import sys
from enum import Enum
from typing import List, Dict
import requests

from src.data_loader import DataLoader, AtlasDataLoader, build_data_loader
from src.export import Exporter, parse_exporter
from src.metrics import MetricComputer, CombinedMetricComputer
from src.models import parse_model, Models
from src.export import get_exporters_from_str


def _build_prediction(input_strings: List[str], top_k: int):
    return {"preprocessed_codes": input_strings, "prediction_count": top_k}


class AssertionCommander:
    def __init__(
        self,
        model_url: str,
        model_name: str,
        batch_size: int,
        top_k: int,
        assertion_number: int,
        metric_evaluators: MetricComputer = None,
        exporters: str = None,
    ):
        self.model_url: str = model_url
        self.model_name: Models = parse_model(model_name)
        self.batch_size: int = batch_size
        self.top_k: int = top_k
        self.metric_evaluators = metric_evaluators
        self.assertion_number = assertion_number
        self.data_loader = build_data_loader(
            self.model_name, assertion_number, self.batch_size
        )
        if self.metric_evaluators is None:
            self.metric_evaluators = CombinedMetricComputer(self.top_k)

        self.exporters = get_exporters_from_str(
            exporters, self.model_name, self.assertion_number, self.top_k
        )

    def _predict(self, input_strings: List[str], top_k: int) -> List[List[str]]:
        prediction_response = requests.post(
            url=self.model_url, json=_build_prediction(input_strings, top_k)
        )
        if prediction_response.status_code != 200:
            logging.error("Error occurred in server! Maybe reduce batch size!")
            sys.exit(1)
        prediction_response_json = prediction_response.json()
        predictions = []
        for prediction_response_obj in prediction_response_json:
            top_k_predictions = prediction_response_obj["predicted_sub_tokens"]
            combined_assertions = [
                " ".join(assertions) for assertions in top_k_predictions
            ]
            predictions.append(combined_assertions)
        return predictions

    def _export_predictions(
        self, expected: List[str], top_k_predictions: List[List[str]]
    ) -> None:
        self.exporters.export_predictions(
            references=expected, top_k_predictions=top_k_predictions
        )

    def _export_metrics(self, metrics: Dict[str, float]):
        self.exporters.export_metrics(metrics)

    def evaluate(self) -> None:
        current_metrics = {}
        with self.data_loader, self.exporters:
            progress_bar = self.data_loader.load_data_stepwise()
            for ref, inputs in progress_bar:
                progress_bar.set_description(self._get_metrics_for_bar(current_metrics), refresh=True)
                predictions = self._predict(inputs, self.top_k)
                self._export_predictions(ref, predictions)
                self.metric_evaluators.add_to_batch(
                    references=ref, top_k_predictions_batch=predictions
                )
                current_metrics = self.metric_evaluators.compute_metrics()

        #metrics: Dict[str, float] = self.metric_evaluators.compute_metrics()
            self._export_metrics(current_metrics)

    def _get_metrics_for_bar(self, metrics: Dict[str, float]):
        accuracy = metrics["accuracy"] if "accuracy" in metrics else 0
        syntactic_correct = metrics["syntactic_correct"] if "syntactic_correct" in metrics else 0
        bleu = metrics["bleu"] if "bleu" in metrics else 0
        return f"Eval [acc: {round(accuracy, 2)}, cor: {round(syntactic_correct, 2)}, bleu: {round(bleu, 2)}]"
