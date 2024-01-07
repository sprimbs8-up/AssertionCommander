import abc
from typing import List, Dict

import evaluate


class MetricComputer(abc.ABC):
    @abc.abstractmethod
    def add_to_batch(self, references: List[str], predictions: List[str]) -> None:
        pass

    @abc.abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        pass


def _extract_assertion_types(assertion: List[str]) -> List[str]:
    return [assert_statement.split()[0] for assert_statement in assertion]


class AssertionTypeMetricComputer(MetricComputer):
    def __init__(self):
        self.assertionTypeDict = {
            "assertEquals": 0,
            "assertNotEquals": 1,
            "assertTrue": 2,
            "assertFalse": 3,
            "assertNull": 4,
            "assertNotNull": 5,
            "assertThrows": 6,
            "TRY_CATCH": 7,
        }
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.f1_score = evaluate.load("f1")

    def add_to_batch(self, references: List[str], predictions: List[str]) -> None:
        ref_assertions = _extract_assertion_types(references)
        pred_assertions = _extract_assertion_types(predictions)
        ref_assertions_number = self.convert_assertion_list_to_number_list(
            ref_assertions
        )
        pred_assertions_number = self.convert_assertion_list_to_number_list(
            pred_assertions
        )
        self.precision.add_batch(
            predictions=pred_assertions_number, references=ref_assertions_number
        )
        self.recall.add_batch(
            predictions=pred_assertions_number, references=ref_assertions_number
        )
        self.f1_score.add_batch(
            predictions=pred_assertions_number, references=ref_assertions_number
        )

    def convert_assertion_list_to_number_list(self, assertion: List[str]) -> List[int]:
        return [
            self.convert_assertion_to_number(assert_type) for assert_type in assertion
        ]

    def convert_assertion_to_number(self, assertion: str) -> int:
        if assertion in self.assertionTypeDict:
            return self.assertionTypeDict[assertion]
        else:
            return -1

    def compute_metrics(self) -> Dict[str, float]:
        final_precision_score = self.precision.compute(average="macro", zero_division=0)
        final_recall_score = self.recall.compute(average="macro", zero_division=0)
        final_f1_score = self.f1_score.compute(average="macro")

        final_metrics = {}
        final_metrics.update(final_precision_score)
        final_metrics.update(final_recall_score)
        final_metrics.update(final_f1_score)
        return final_metrics


class SyntacticCorrectnessMetricComputer(MetricComputer):
    def __init__(self):
        self.syntactic_correct: int = 0
        self.total: int = 0

    def compute_metrics(self) -> Dict[str, float]:
        return {"syntactic_correct": float(self.syntactic_correct) / float(self.total)}

    def add_to_batch(self, references: List[str], predictions: List[str]) -> None:
        self.syntactic_correct += len(predictions) // 2
        self.total += len(predictions)
