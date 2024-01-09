import abc
from functools import reduce
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu


import evaluate
import subprocess
import json


class MetricComputer(abc.ABC):
    def __init__(self, top_k: int):
        self.top_k = top_k

    @abc.abstractmethod
    def add_to_batch(
        self, references: List[str], top_k_predictions_batch: List[List[str]]
    ) -> None:
        pass

    @abc.abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        pass


def _extract_assertion_types(assertion: List[str]) -> List[str]:
    return [assert_statement.split()[0] for assert_statement in assertion]


class CombinedMetricComputer(MetricComputer):
    def __init__(self, top_k: int):
        super().__init__(top_k)
        self.metric_computers = [
            ClassicalMetricComputer(top_k),
            SyntacticCorrectnessMetricComputer(top_k),
            AssertionTypeMetricComputer(top_k),
            BleuMetricComputer(top_k)
        ]

    def add_to_batch(
        self, references: List[str], top_k_predictions_batch: List[List[str]]
    ) -> None:
        for computer in self.metric_computers:
            computer.add_to_batch(
                references=references, top_k_predictions_batch=top_k_predictions_batch
            )

    def compute_metrics(self) -> Dict[str, float]:
        metric_dict = {}
        for computer in self.metric_computers:
            metric_dict.update(computer.compute_metrics())
        return metric_dict


class AssertionTypeMetricComputer(MetricComputer):
    def __init__(self, top_k: int):
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
        self, references: List[str], top_k_predictions_batch: List[List[str]]
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
            for types, expected in zip(pred_assertions_numbers, ref_assertions_number)
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
    def select_usable_number(possible_types: List[int], expected: int) -> int:
        if expected in possible_types:
            return expected
        else:
            return possible_types[0]

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
    def __init__(self, top_k: int):
        super().__init__(top_k)
        self.syntactic_correct: int = 0
        self.failure_batches: int = 0
        self.total: int = 0

    def compute_metrics(self) -> Dict[str, float]:
        return {
            "syntactic_correct": float(self.syntactic_correct) / float(self.total),
            "failure_batches": self.failure_batches,
        }

    def add_to_batch(
        self, references: List[str], top_k_predictions_batch: List[List[str]]
    ) -> None:
        self._compute_syntactic_correct_predictions(top_k_predictions_batch)
        self.total += len(top_k_predictions_batch)

    def _compute_syntactic_correct_predictions(
        self, top_k_predictions_batch: List[List[str]]
    ) -> None:
        flatten_prediction_batch = []
        for top_k_pred in top_k_predictions_batch:
            flatten_prediction_batch.extend(top_k_pred)
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

        groups = [
            list_res[i : i + self.top_k] for i in range(0, len(list_res), self.top_k)
        ]
        if len(top_k_predictions_batch) == len(groups):
            self.syntactic_correct += sum([1 if any(res) else 0 for res in groups])
        else:
            self.failure_batches += 1


def _clean(assertion: str):
    return assertion.replace(" ", "")


def _clean_list(assertions: List[str]) -> List[str]:
    return [_clean(assertion) for assertion in assertions]


class ClassicalMetricComputer(MetricComputer):
    def __init__(self, top_k: int):
        super().__init__(top_k)
        self.correct_predictions: int = 0
        self.total: int = 0

    def compute_metrics(self) -> Dict[str, float]:
        metric_dict = {}
        metric_dict["accuracy"] = self.correct_predictions / self.total
        return metric_dict

    def add_to_batch(
        self, references: List[str], top_k_predictions_batch: List[List[str]]
    ) -> None:
        self._accuracy(references, top_k_predictions_batch)

    def _accuracy(
        self, references: List[str], top_k_predictions_batch: List[List[str]]
    ):
        equality = [
            _clean(r) in _clean_list(p)
            for r, p in zip(references, top_k_predictions_batch)
        ]
        self.correct_predictions += len([_ for _ in equality if _])
        self.total += len(equality)


class BleuMetricComputer(MetricComputer):
    def __init__(self, top_k: int):
        super().__init__(top_k)
        self.bleu_score_sum:float = 0.0
        self.bleu_scores: List[float] = []

    def add_to_batch(
        self, references: List[str], top_k_predictions_batch: List[List[str]]
    ) -> None:
        for ref, top_k_preds in zip(references, top_k_predictions_batch):
            top_k_bleu_score = max([sentence_bleu(references=[ref], hypothesis=top_k_pred, weights=[1]) for top_k_pred in top_k_preds])
            self.bleu_score_sum += top_k_bleu_score
            self.bleu_scores.append(top_k_bleu_score)

    def compute_metrics(self) -> Dict[str, float]:
        return {"bleu": self.bleu_score_sum / len(self.bleu_scores)}

    # def _extract_from_data(self, data: list[Datapoint]):
    #
    #
    #     target_labels: list = []
    #     top_prediction_labels: list = []
    #     in_top_k = 0
    #
    #     for d in data:
    #         target_labels.append(d.target_label)
    #         top_prediction_labels.append(d.prediction_results[0].label)
    #         bleu = sentence_bleu(
    #             references=[d.target_subtokenised_label],
    #             hypothesis=d.prediction_results[0].subtokenised_label,
    #             weights=[1],
    #         )
    #         bleu_score_sum += bleu
    #         bleu_scores.append(bleu)
    #
    #         if any(pred.label == d.target_label for pred in d.prediction_results):
    #             in_top_k += 1
    #
    #     metrics = Metrics(
    #         f1_score(
    #             y_true=target_labels,
    #             y_pred=top_prediction_labels,
    #             average="weighted",
    #         ),
    #         bleu_score_sum / len(data),
    #         in_top_k / len(data),
    #     )
    #
    #     return (
    #         target_labels,
    #         top_prediction_labels,
    #         bleu_score_sum,
    #         bleu_scores,
    #         in_top_k,
    #         metrics,
    #     )
