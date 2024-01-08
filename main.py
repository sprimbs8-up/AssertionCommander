import evaluate
from src.metrics import AssertionTypeMetricComputer, SyntacticCorrectnessMetricComputer, ClassicalMetricComputer

bleu = evaluate.load("accuracy")
predictions = [
    "assertEquals ( result , 2 )",
    "assertEquals ( result , 2 )",
    'assertEquals ( "result" , 2 )',
    'assertTrue ( "result" , 2 , 2)',
]
references = [
    "assertEquals ( result . isEmpty ( ) )",
    "assertEquals ( result , 2 )",
    "assertEquals ( result , 2 )",
    "assertEquals ( result , 2 )",
]


metrics = [ClassicalMetricComputer(), AssertionTypeMetricComputer(), SyntacticCorrectnessMetricComputer()]
metrics_dict = {}
for metric in metrics:
    for pred, ref in zip(predictions, references):
        metric.add_to_batch(predictions=[pred], references=[ref])
    metrics_dict.update(metric.compute_metrics())
print(metrics_dict)
