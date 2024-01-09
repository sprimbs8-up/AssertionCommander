import sys
from itertools import islice
from typing import List, Dict
import requests

from src.metrics import MetricComputer, CombinedMetricComputer

references = "evaluation-data/10/atlas/assertLines.txt"
input_methods = "evaluation-data/10/atlas/testMethods.txt"


class AssertionCommander:
    def __init__(self, model_url: str, model_name: str, batch_size: int, top_k: int,
                 metric_evaluators: List[MetricComputer] = None):
        self.model_url: str = model_url
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.top_k: int = top_k
        self.metric_evaluators = metric_evaluators
        if self.metric_evaluators is None:
            self.metric_evaluators = [CombinedMetricComputer(self.top_k)]

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
            combined_assertions = [" ".join(assertions) for assertions in top_k_predictions]
            predictions.append(combined_assertions)
        return predictions

    def evaluate(self) -> Dict[str, float]:
        with open(references, "r") as reference_file, open(input_methods, "r") as input_file:

            for ref, inputs in zip(
                    iter(lambda: tuple(islice(reference_file, self.batch_size)), ()),
                    iter(lambda: tuple(islice(input_file, self.batch_size)), ()),
            ):
                predictions = self.predict(inputs, self.top_k)
                for metric_computer in self.metric_evaluators:
                    metric_computer.add_to_batch(
                        references=ref, top_k_predictions_batch=predictions
                    )
        metrics = {}
        for metric_computer in self.metric_evaluators:
            metrics.update(metric_computer.compute_metrics())
        return metrics
