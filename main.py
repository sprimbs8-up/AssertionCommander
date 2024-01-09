import evaluate
from src.metrics import (
    AssertionTypeMetricComputer,
    SyntacticCorrectnessMetricComputer,
    ClassicalMetricComputer,
)


from itertools import islice


def process(a, b):
    print(list(a))
    print(list(b))


filename = "evaluation-data/10/atlas/assertLines.txt"
filename2 = "evaluation-data/10/atlas/testMethods.txt"
batch_size = 16  # Or whatever chunk size you want
with open(filename, "r") as f, open(filename2, "r") as g:
    for a, b in zip(
        iter(lambda: tuple(islice(f, batch_size)), ()),
        iter(lambda: tuple(islice(g, batch_size)), ()),
    ):
        process(a, b)
