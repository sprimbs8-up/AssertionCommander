import abc
from functools import reduce
from typing import List, Dict

import evaluate
import subprocess
import json


class MetricComputer(abc.ABC):
    @abc.abstractmethod
    def add_to_batch(self, references: List[str], top_k_predictions_batch: List[List[str]]) -> None:
        pass

    @abc.abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        pass


def _extract_assertion_types(assertion: List[str]) -> List[str]:
    return [assert_statement.split()[0] for assert_statement in assertion]


class CombinedMetricComputer(MetricComputer):
    def __init__(self, metric_computers: List[MetricComputer]):
        self.metric_computers = metric_computers

    def add_to_batch(self, references: List[str], top_k_predictions_batch: List[List[str]]) -> None:
        for computer in self.metric_computers:
            computer.add_to_batch(references=references, top_k_predictions_batch=top_k_predictions_batch)

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

    def add_to_batch(self, references: List[str], top_k_predictions_batch: List[List[str]]) -> None:
        ref_assertions = _extract_assertion_types(references)
        pred_assertions = _extract_assertion_types(top_k_predictions_batch)
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

    def add_to_batch(self, references: List[str], top_k_predictions_batch: List[List[str]]) -> None:
        self._compute_syntactic_correct_predictions(top_k_predictions_batch)
        self.total += len(top_k_predictions_batch)

    def _compute_syntactic_correct_predictions(self, top_k_predictions_batch: List[List[str]]) -> None:
        flatten_prediction_batch = []
        for top_k in top_k_predictions_batch:
            flatten_prediction_batch.extend(top_k)
        cmd = [
            "java",
            "-jar",
            "libs/assertions.jar",
            "check",
            "--codes",
            json.dumps(flatten_prediction_batch),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr is not None and result.stderr != "":
            print(result.stderr)
        list_res = json.loads(result.stdout.strip())
        n = 1
        groups = [list_res[i:i + n] for i in range(0, len(list_res), n)]
        if len(flatten_prediction_batch) == len(groups):
            self.syntactic_correct += sum([ 1 if any(res) else 0  for res in groups])
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

    def add_to_batch(self, references: List[str], top_k_predictions_batch: List[str]) -> None:
        self.bleu.add_batch(references=references, predictions=top_k_predictions_batch)
        self._accuracy(references, top_k_predictions_batch)

    def _accuracy(self, references: List[str], top_k_predictions_batch: List[str]):
        equality = [
            self._clean(r) == self._clean(p) for r, p in zip(references, top_k_predictions_batch)
        ]
        self.correct_predictions += len([_ for _ in equality if _])
        self.total += len(equality)

    @staticmethod
    def _clean(assertion: str):
        return assertion.replace(" ", "")
