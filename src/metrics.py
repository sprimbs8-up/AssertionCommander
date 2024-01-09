import abc
from typing import List, Dict

import evaluate
import subprocess
import json


class MetricComputer(abc.ABC):
    @abc.abstractmethod
    def add_to_batch(self, references: List[str], predictions: List[str]) -> None:
        pass

    @abc.abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        pass


def _extract_assertion_types(assertion: List[str]) -> List[str]:
    return [assert_statement.split()[0] for assert_statement in assertion]


class CombinedMetricComputer(MetricComputer):
    def __init__(self, metric_computers: List[MetricComputer]):
        self.metric_computers = metric_computers

    def add_to_batch(self, references: List[str], predictions: List[str]) -> None:
        for computer in self.metric_computers:
            computer.add_to_batch(references=references, predictions=predictions)

    def compute_metrics(self) -> Dict[str, float]:
        metric_dict = {}
        for computer in self.metric_computers:
            metric_dict.update(computer.compute_metrics())
        return metric_dict


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
        self.failure_batches: int = 0
        self.total: int = 0

    def compute_metrics(self) -> Dict[str, float]:
        return {
            "syntactic_correct": float(self.syntactic_correct) / float(self.total),
            "failure_batches": self.failure_batches,
        }

    def add_to_batch(self, references: List[str], predictions: List[str]) -> None:
        self._compute_syntactic_correct_predictions(predictions)
        self.total += len(predictions)

    def _compute_syntactic_correct_predictions(self, predictions) -> None:
        cmd = [
            "java",
            "-jar",
            "libs/assertions.jar",
            "check",
            "--codes",
            json.dumps(predictions),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr is not None and result.stderr != "":
            print(result.stderr)
        list_res = json.loads(result.stdout.strip())
        if len(predictions) == len(list_res):
            self.syntactic_correct += sum([1 for res in list_res if res])
        else:
            self.failure_batches += 1


class ClassicalMetricComputer(MetricComputer):
    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.correct_predictions: int = 0
        self.total: int = 0

    def compute_metrics(self) -> Dict[str, float]:
        metric_dict = {}
        bleu_metrics = self.bleu.compute()
        metric_dict["bleu"] = bleu_metrics["bleu"]
        metric_dict["accuracy"] = self.correct_predictions / self.total
        return metric_dict

    def add_to_batch(self, references: List[str], predictions: List[str]) -> None:
        self.bleu.add_batch(references=references, predictions=predictions)
        self._accuracy(references, predictions)

    def _accuracy(self, references: List[str], predictions: List[str]):
        equality = [
            self._clean(r) == self._clean(p) for r, p in zip(references, predictions)
        ]
        self.correct_predictions += len([_ for _ in equality if _])
        self.total += len(equality)

    @staticmethod
    def _clean(assertion: str):
        return assertion.replace(" ", "")
