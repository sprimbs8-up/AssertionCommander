import sys
from typing import List

import requests

from src.metrics import (
    MetricComputer,
    CombinedMetricComputer,
)

from itertools import islice

references = "evaluation-data/10/atlas/assertLines.txt"
input_methods = "evaluation-data/10/atlas/testMethods.txt"
batch_size = 50  # Or whatever chunk size you want
top_k = 20
url = "http://localhost:8080/"


def predict(input_strings: List[str], top_k: int) -> List[List[str]]:
    el = {"preprocessed_codes": input_strings, "prediction_count": top_k}
    prediction_response = requests.post("http://localhost:8080", json=el)
    if prediction_response.status_code != 200:
        print("Error occurred in server! Maybe reduce batch size!")
        sys.exit(1)
    prediction_response_json = prediction_response.json()
    predictions = []
    for prediction_response_obj in prediction_response_json:
        top_k_predictions = prediction_response_obj["predicted_sub_tokens"]
        combined_assertions = [" ".join(assertions) for assertions in top_k_predictions]
        predictions.append(combined_assertions)
    print(predictions)
    return predictions


def get_metrics(top_k_predictions: int) -> MetricComputer:
    return CombinedMetricComputer(top_k_predictions)


with open(references, "r") as reference_file, open(input_methods, "r") as input_file:
    metric_computer = get_metrics(top_k)
    for ref, inputs in zip(
        iter(lambda: tuple(islice(reference_file, batch_size)), ()),
        iter(lambda: tuple(islice(input_file, batch_size)), ()),
    ):
        predictions = predict(inputs, top_k)
        metric_computer.add_to_batch(
            references=ref, top_k_predictions_batch=predictions
        )
    print(metric_computer.compute_metrics())
