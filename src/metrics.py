import abc
import json
import os
import subprocess

import evaluate
from nltk.translate.bleu_score import sentence_bleu


class MetricComputer(abc.ABC):
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    @abc.abstractmethod
    def add_to_batch(
        self, references: list[str], top_k_predictions_batch: list[list[str]]
    ) -> None:
        pass

    @abc.abstractmethod
    def compute_metrics(self) -> dict[str, float]:
        pass


def _extract_assertion_types(assertion: list[str]) -> list[str]:
    return [assert_statement.split()[0] for assert_statement in assertion]


class CombinedMetricComputer(MetricComputer):
    def __init__(
        self, top_k: int, metric_computers: list[MetricComputer] = None
    ) -> None:
        super().__init__(top_k)
        self.metric_computers = metric_computers
        if metric_computers is None:
            self.metric_computers = [
                AccuracyMetricComputer(top_k),
                SyntacticCorrectnessMetricComputer(top_k),
                AssertionTypeMetricComputer(top_k),
                BleuMetricComputer(top_k),
            ]

    def add_to_batch(
        self, references: list[str], top_k_predictions_batch: list[list[str]]
    ) -> None:
        for computer in self.metric_computers:
            computer.add_to_batch(
                references=references, top_k_predictions_batch=top_k_predictions_batch
            )

    def compute_metrics(self) -> dict[str, float]:
        metric_dict = {}
        for computer in self.metric_computers:
            metric_dict.update(computer.compute_metrics())
        return metric_dict


class AssertionTypeMetricComputer(MetricComputer):
    def __init__(self, top_k: int) -> None:
        super().__init__(top_k)
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

    def add_to_batch(
        self, references: list[str], top_k_predictions_batch: list[list[str]]
    ) -> None:
        ref_assertions = _extract_assertion_types(references)
        pred_assertions = [
            _extract_assertion_types(top_k) for top_k in top_k_predictions_batch
        ]
        ref_assertions_number = self.convert_assertion_list_to_number_list(
            ref_assertions
        )
        pred_assertions_numbers = [
            self.convert_assertion_list_to_number_list(pred) for pred in pred_assertions
        ]
        pred_assertions_number = [
            self.select_usable_number(types, expected)
            for types, expected in zip(
                pred_assertions_numbers, ref_assertions_number, strict=False
            )
        ]
        self.precision.add_batch(
            predictions=pred_assertions_number, references=ref_assertions_number
        )
        self.recall.add_batch(
            predictions=pred_assertions_number, references=ref_assertions_number
        )
        self.f1_score.add_batch(
            predictions=pred_assertions_number, references=ref_assertions_number
        )

    @staticmethod
    def select_usable_number(possible_types: list[int], expected: int) -> int:
        if expected in possible_types:
            return expected

        return possible_types[0]

    def convert_assertion_list_to_number_list(self, assertion: list[str]) -> list[int]:
        return [
            self.convert_assertion_to_number(assert_type) for assert_type in assertion
        ]

    def convert_assertion_to_number(self, assertion: str) -> int:
        if assertion in self.assertionTypeDict:
            return self.assertionTypeDict[assertion]

        return -1

    def compute_metrics(self) -> dict[str, float]:
        final_precision_score = self.precision.compute(average="macro", zero_division=0)
        final_recall_score = self.recall.compute(average="macro", zero_division=0)
        final_f1_score = self.f1_score.compute(average="macro")

        final_metrics = {}
        final_metrics.update(final_precision_score)
        final_metrics.update(final_recall_score)
        final_metrics.update(final_f1_score)
        return final_metrics


class SyntacticCorrectnessMetricComputer(MetricComputer):
    def __init__(self, top_k: int) -> None:
        super().__init__(top_k)
        self.syntactic_correct: int = 0
        self.failure_batches: int = 0
        self.total: int = 0

    def compute_metrics(self) -> dict[str, float]:
        syntactic_correct = (
            float(self.syntactic_correct) / float(self.total) if self.total > 0 else 0
        )
        return {
            "syntactic_correct": syntactic_correct,
            "failure_batches": self.failure_batches,
        }

    def add_to_batch(
        self, references: list[str], top_k_predictions_batch: list[list[str]]
    ) -> None:
        super().add_to_batch(references, top_k_predictions_batch)
        self._compute_syntactic_correct_predictions(top_k_predictions_batch)
        self.total += len(top_k_predictions_batch)

    def _compute_syntactic_correct_predictions(
        self, top_k_predictions_batch: list[list[str]]
    ) -> None:
        flatten_prediction_batch = []
        for top_k_pred in top_k_predictions_batch:
            flatten_prediction_batch.extend(top_k_pred)
        java_cmd = os.getenv("JAVA_HOME")
        java_cmd = java_cmd if java_cmd is not None else "java"
        cmd = [
            java_cmd,
            "-jar",
            "libs/assertions.jar",
            "check",
            "--codes",
            json.dumps(flatten_prediction_batch),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stderr is not None and result.stderr != "":
            print(result.stderr)
        list_res = json.loads(result.stdout.strip())

        groups = [
            list_res[i : i + self.top_k] for i in range(0, len(list_res), self.top_k)
        ]
        if len(top_k_predictions_batch) == len(groups):
            self.syntactic_correct += sum([1 if any(res) else 0 for res in groups])
        else:
            self.failure_batches += 1


def _clean(assertion: str) -> str:
    return assertion.replace(" ", "").replace("\n", "")


def _clean_list(assertions: list[str]) -> list[str]:
    return [_clean(assertion) for assertion in assertions]


class AccuracyMetricComputer(MetricComputer):
    def __init__(self, top_k: int) -> None:
        super().__init__(top_k)
        self.correct_predictions: int = 0
        self.total: int = 0

    def compute_metrics(self) -> dict[str, float]:
        accuracy = self.correct_predictions / self.total if self.total > 0 else 0
        return {"accuracy": accuracy}

    def add_to_batch(
        self, references: list[str], top_k_predictions_batch: list[list[str]]
    ) -> None:
        self._accuracy(references, top_k_predictions_batch)

    def _accuracy(
        self, references: list[str], top_k_predictions_batch: list[list[str]]
    ) -> None:
        equality = [
            _clean(r) in _clean_list(p)
            for r, p in zip(references, top_k_predictions_batch, strict=False)
        ]
        self.correct_predictions += len([_ for _ in equality if _])
        self.total += len(equality)


class BleuMetricComputer(MetricComputer):
    def __init__(self, top_k: int) -> None:
        super().__init__(top_k)
        self.bleu_score_sum: float = 0.0
        self.bleu_scores: list[float] = []

    def add_to_batch(
        self, references: list[str], top_k_predictions_batch: list[list[str]]
    ) -> None:
        for ref, top_k_preds in zip(references, top_k_predictions_batch, strict=False):
            top_k_bleu_score = max(
                [
                    sentence_bleu(references=[ref], hypothesis=top_k_pred, weights=[1])
                    for top_k_pred in top_k_preds
                ]
            )
            self.bleu_score_sum += top_k_bleu_score
            self.bleu_scores.append(top_k_bleu_score)

    def compute_metrics(self) -> dict[str, float]:
        return {"bleu": self.bleu_score_sum / len(self.bleu_scores)}
