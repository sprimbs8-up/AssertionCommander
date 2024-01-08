import evaluate
from src.metrics import AssertionTypeMetricComputer, SyntacticCorrectnessMetricComputer

bleu = evaluate.load("accuracy")
predictions = [
    "assertEquals ( result , 2 )",
    "assertEquals ( result , 2 )",
    'assertEquals ( "result" , 2 )',
    'assertEquals ( "result" , 2 , 2)',
]
references = [
    "assertEquals ( result . isEmpty ( ) )",
    "assertTrue ( result , 2 )",
    "assertEquals ( result , 2 )",
    "assertEquals ( result , 2 )",
]

metrics = [AssertionTypeMetricComputer(), SyntacticCorrectnessMetricComputer()]

for metric in metrics:
    for pred, ref in zip(predictions, references):
        metric.add_to_batch(predictions=[pred], references=[ref])
    print(metric.compute_metrics())
