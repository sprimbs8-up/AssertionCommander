import sys
from enum import Enum
from typing import List, Dict
import requests

from src.data_loader import DataLoader, AtlasDataLoader
from src.metrics import MetricComputer, CombinedMetricComputer
from src.models import parse_model, Models


class AssertionCommander:
    def __init__(
            self,
            model_url: str,
            model_name: str,
            batch_size: int,
            top_k: int,
            assertion_number: int,
            metric_evaluators: MetricComputer = None,
    ):
        self.model_url: str = model_url
        self.model_name: Models = parse_model(model_name)
        self.batch_size: int = batch_size
        self.top_k: int = top_k
        self.metric_evaluators = metric_evaluators
        self.assertion_number = assertion_number
        self.data_loader = AtlasDataLoader(self.model_name, assertion_number, self.batch_size)
        if self.metric_evaluators is None:
            self.metric_evaluators = CombinedMetricComputer(self.top_k)

    def predict(self, input_strings: List[str], top_k: int) -> List[List[str]]:
        el = {"preprocessed_codes": input_strings, "prediction_count": top_k}
        prediction_response = requests.post(url=self.model_url, json=el)
        if prediction_response.status_code != 200:
            print("Error occurred in server! Maybe reduce batch size!")
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

    def evaluate(self) -> Dict[str, float]:
        with self.data_loader:
            for ref, inputs in self.data_loader.load_data_stepwise():
                predictions = self.predict(inputs, self.top_k)
                self.metric_evaluators.add_to_batch(
                    references=ref, top_k_predictions_batch=predictions
                )

        return self.metric_evaluators.compute_metrics()
