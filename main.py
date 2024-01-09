from typing import List

import evaluate
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
batch_size = 10  # Or whatever chunk size you want


def predict(input_strings: List[str]) -> List[str]:
    return input_strings


def get_metrics() -> CombinedMetricComputer:
    return CombinedMetricComputer(
        [
            ClassicalMetricComputer(),
            SyntacticCorrectnessMetricComputer(),
            AssertionTypeMetricComputer(),
        ]
    )


with open(references, "r") as reference_file, open(input_methods, "r") as input_file:
    metric_computer = get_metrics()
    for ref, inputs in zip(
        iter(lambda: tuple(islice(reference_file, batch_size)), ()),
        iter(lambda: tuple(islice(input_file, batch_size)), ()),
    ):
        predictions = predict(inputs)
        metric_computer.add_to_batch(references=ref, predictions=predictions)
    print(metric_computer.compute_metrics())
