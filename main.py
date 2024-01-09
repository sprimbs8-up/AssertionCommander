from typing import List

import evaluate
import requests

from src.metrics import (
    AssertionTypeMetricComputer,
    SyntacticCorrectnessMetricComputer,
    ClassicalMetricComputer,
    MetricComputer,
    CombinedMetricComputer,
)

from itertools import islice

references = "evaluation-data/10/atlas/assertLines.txt"
input_methods = "evaluation-data/10/atlas/testMethods.txt"
batch_size = 1  # Or whatever chunk size you want

url="http://localhost:8080/"

def predict(input_strings: List[str], top_k=1) -> List[List[str]]:
    el = {
        "preprocessed_codes": input_strings,
        "prediction_count": top_k
    }
    prediction_response = requests.post("http://localhost:8080", json=el).json()
    predictions = []
    for prediction_response_obj in prediction_response:
        top_k_predictions = prediction_response_obj["predicted_sub_tokens"]
        combined_assertions = [" ".join(assertions) for assertions in top_k_predictions]
        predictions.append(combined_assertions)
    print(predictions)
    return predictions


def get_metrics() -> CombinedMetricComputer:
    return CombinedMetricComputer(
        [
           # ClassicalMetricComputer(),
            SyntacticCorrectnessMetricComputer(),
           # AssertionTypeMetricComputer(),
        ]
    )


with open(references, "r") as reference_file, open(input_methods, "r") as input_file:
    metric_computer = get_metrics()
    for ref, inputs in zip(
        iter(lambda: tuple(islice(reference_file, batch_size)), ()),
        iter(lambda: tuple(islice(input_file, batch_size)), ()),
    ):
        predictions = predict(inputs)
        metric_computer.add_to_batch(references=ref, top_k_predictions_batch=predictions)
    print(metric_computer.compute_metrics())
