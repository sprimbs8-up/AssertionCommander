import logging
import sys

import requests

from src.data_loader import build_data_loader
from src.dataset_type import DatasetType
from src.export import CombinedExporter
from src.metrics import CombinedMetricComputer, MetricComputer
from src.models import Models, parse_model


def _build_prediction(input_strings: list[str], top_k: int) -> dict[str, float]:
    return {"preprocessed_codes": input_strings, "prediction_count": top_k}


def _get_metrics_for_bar(metrics: dict[str, float]) -> str:
    accuracy = metrics["accuracy"] if "accuracy" in metrics else 0
    syntactic_correct = (
        metrics["syntactic_correct"] if "syntactic_correct" in metrics else 0
    )
    bleu = metrics["bleu"] if "bleu" in metrics else 0
    return f"Eval [acc: {round(accuracy, 2)}, cor: {round(syntactic_correct, 2)}, bleu: {round(bleu, 2)}]"


class AssertionCommander:
    def __init__(
            self,
            model_url: str,
            model_name: str,
            batch_size: int,
            top_k: int,
            assertion_number: int,
            dataset_type: DatasetType,
            root_dir: str,
            export_dir: str,
            cached_predictions_file: str,
            raw_data: bool,
            epoch: str,
            metric_evaluators: MetricComputer = None,
            exporters: str = None,
    ) -> None:
        self.model_url: str = model_url
        self.model: Models = parse_model(model_name)
        self.batch_size: int = batch_size
        self.top_k: int = top_k
        self.assertion_number = assertion_number
        self.dataset_type = dataset_type
        self.root_dir = root_dir
        self.export_dir = export_dir
        self.metric_evaluators = metric_evaluators
        self.data_loader = build_data_loader(
            model=self.model,
            assertion_number=self.assertion_number,
            batch_size=self.batch_size,
            dataset=self.dataset_type,
            default_data_dir=self.root_dir,
            cache_pred_dir=cached_predictions_file,
            top_k=top_k,
            raw_data=raw_data,
        )
        self.epoch = epoch
        if self.metric_evaluators is None:
            self.metric_evaluators = CombinedMetricComputer(self.top_k)
        self.pred_export = cached_predictions_file is None
        self.exporters = CombinedExporter(
            model=self.model,
            assertion_number=self.assertion_number,
            top_k=self.top_k,
            default_dir=self.export_dir,
            dataset_type=self.dataset_type,
            exporters_str=exporters,
            no_pred_export=not self.pred_export,
            raw_file="raw" if raw_data else "abstract",
            epoch=self.epoch
        )

    def _predict(self, input_strings: list[str], top_k: int) -> list[list[str]]:
        prediction_response = requests.post(
            url=self.model_url, json=_build_prediction(input_strings, top_k)
        )
        if prediction_response.status_code != 200:
            should_retry = self._handle_request_errors()
            if should_retry:
                try:
                    return self._predict(input_strings, top_k)
                except Exception:
                    text = input("Continue? [y]/n\n")
                    if text.startswith("y"):
                        return self._predict(input_strings, top_k)
        prediction_response_json = prediction_response.json()
        predictions = []
        for prediction_response_obj in prediction_response_json:
            top_k_predictions = prediction_response_obj["predicted_sub_tokens"]
            combined_assertions = [
                " ".join(assertions) for assertions in top_k_predictions
            ]
            predictions.append(combined_assertions)
        return predictions

    @staticmethod
    def _handle_request_errors() -> bool:
        error_msg = "An Error occurred in server part. Please see server logs!"
        logging.exception(error_msg)
        text = input("Continue? [y]/n\n")
        return text.startswith("y")

    def _export_predictions(
            self, expected: list[str], top_k_predictions: list[list[str]]
    ) -> None:
        self.exporters.export_predictions(
            references=expected, top_k_predictions=top_k_predictions
        )

    def _export_metrics(self, metrics: dict[str, float]) -> None:
        self.exporters.export_metrics(metrics)

    def _check_correct_epoch(self) -> bool:
        if self.epoch is not None:
            prediction_response = requests.get(url=self.model_url + "/epoch")
            server_epoch = prediction_response.json()
            if server_epoch != int(self.epoch):
                logging.warning("The epochs from the server and the given epoch are different. Stopping Evaluation.")
                logging.warning("Server Epoch: %s", server_epoch)
                logging.warning("Config Epoch: %s", self.epoch)
                return False
        return True

    def evaluate(self) -> None:
        correct_epoch = self._check_correct_epoch()
        if not correct_epoch:
            return
        current_metrics = {}
        with self.data_loader, self.exporters:
            progress_bar = self.data_loader.load_data_stepwise()
            for ref, inputs, *optional in progress_bar:
                progress_bar.set_description(
                    _get_metrics_for_bar(current_metrics), refresh=True
                )
                if self.pred_export:
                    predictions = self._predict(inputs, self.top_k)
                    if optional is not None and len(optional) > 0:
                        predictions, ref = self._convert_to_raw_tokens(
                            optional[0], predictions, ref
                        )
                    self._export_predictions(ref, predictions)
                else:
                    predictions = inputs

                self.metric_evaluators.add_to_batch(
                    references=ref, top_k_predictions_batch=predictions
                )
                current_metrics = self.metric_evaluators.compute_metrics()
            self._export_metrics(current_metrics)

    def _convert_to_raw_tokens(self, optional, predictions, ref):
        raw_refs = []
        raw_preds = []
        for r, pred, optional_dict in zip(ref, predictions, optional):
            ref_tokens = r.split()
            pred_tokens = [
                self._normalize_token_list(
                    [self._normalize_token(pred) for pred in k_pred.split()]
                )
                for k_pred in pred
            ]
            refs = [optional_dict[t] if t in optional_dict else t for t in ref_tokens]
            preds = [
                [optional_dict[t] if t in optional_dict else t for t in top_k_pred]
                for top_k_pred in pred_tokens
            ]
            raw_refs.append(" ".join(refs))
            raw_preds.append([" ".join(p) for p in preds])
        ref = raw_refs
        predictions = raw_preds
        return predictions, ref

    def _normalize_token(self, token):
        normalized_token = token
        for character in [".", ","]:
            normalized_token = f" {character} ".join(normalized_token.split(character))
        return normalized_token

    def _normalize_token_list(self, token_list):
        return (" ".join(token_list)).split()
