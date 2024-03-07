import abc
import csv
import json

import pandas
from itertools import islice
from pathlib import Path
from typing import Iterable, Any

from tqdm import tqdm

from src.dataset_type import DatasetType
from src.models import Models


class DataLoader(abc.ABC):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        batch_size: int,
        default_data_dir: str,
        dataset: DatasetType,
        raw_set: bool,
    ) -> None:
        self.model = model
        self.assertion_number = assertion_number
        self.batch_size = batch_size
        self.default_data_dir = default_data_dir
        self.dataset = dataset
        self.raw_set = raw_set

    def __enter__(self) -> object:
        self.load_files()
        return self

    def __exit__(self, *exc_details: object) -> None:
        self.close_files()

    @abc.abstractmethod
    def load_files(self) -> None:
        pass

    @abc.abstractmethod
    def load_data_stepwise(self) -> tqdm:
        pass

    @abc.abstractmethod
    def close_files(self) -> None:
        pass

    @abc.abstractmethod
    def get_number_data_points(self) -> int:
        pass

    def _get_dataset_type_dir(self) -> str:
        match self.dataset:
            case DatasetType.TEST:
                return "testing"
            case DatasetType.TRAINING:
                return "training"
            case DatasetType.VALIDATION:
                return "validation"


class AtlasDataLoader(DataLoader):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        batch_size: int,
        default_data_dir: str,
        dataset: DatasetType,
        raw_set: bool,
    ) -> None:
        super().__init__(
            model, assertion_number, batch_size, default_data_dir, dataset, raw_set
        )
        type_dir = "raw" if raw_set else "abstract"
        self.references_file_path: Path = (
            Path(self.default_data_dir)
            / str(self.assertion_number)
            / self.model.name
            / type_dir
            / self._get_dataset_type_dir()
            / "assertLines.txt"
        )
        self.input_file_path: Path = (
            Path(self.default_data_dir)
            / str(self.assertion_number)
            / self.model.name
            / type_dir
            / self._get_dataset_type_dir()
            / "testMethods.txt"
        )
        self.abstract_dict_file: Path = (
            (
                Path(self.default_data_dir)
                / str(self.assertion_number)
                / self.model.name
                / type_dir
                / self._get_dataset_type_dir()
                / "dict.jsonl"
            )
            if not raw_set
            else None
        )
        self.input_file = None
        self.dict_file = None
        self.ref_file = None
        self.num_data_elements: int = self.get_number_data_points()

    def get_number_data_points(self) -> int:
        with Path.open(self.input_file_path) as inputs:
            return len(inputs.readlines())

    def load_files(self) -> None:
        self.ref_file = Path.open(self.references_file_path)
        self.input_file = Path.open(self.input_file_path)
        if not self.raw_set:
            self.dict_file = Path.open(self.abstract_dict_file)

    def load_data_stepwise(self) -> tqdm:
        total = self.num_data_elements // self.batch_size
        if self.num_data_elements % self.batch_size != 0:
            total += 1
        if self.raw_set:
            return tqdm(
                zip(
                    iter(lambda: tuple(islice(self.ref_file, self.batch_size)), ()),
                    iter(lambda: tuple(islice(self.input_file, self.batch_size)), ()),
                    strict=False,
                ),
                total=total,
                desc=f"Evaluating {self.model.name}-{self.assertion_number}",
            )
        else:
            return tqdm(
                zip(
                    iter(lambda: tuple(islice(self.ref_file, self.batch_size)), ()),
                    iter(lambda: tuple(islice(self.input_file, self.batch_size)), ()),
                    map(
                        self.line_to_dict,
                        iter(
                            lambda: tuple(islice(self.dict_file, self.batch_size)), ()
                        ),
                    ),
                    strict=False,
                ),
                total=total,
                desc=f"Evaluating {self.model.name}-{self.assertion_number}",
            )

    def line_to_dict(self, line):
        return [json.loads(l) for l in line]

    def close_files(self) -> None:
        self.ref_file.close()
        self.input_file.close()
        if self.dict_file is not None:
            self.dict_file.close()


class TogaDataLoader(DataLoader):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        batch_size: int,
        default_data_dir: str,
        dataset: DatasetType,
        raw_set: bool,
    ):
        super().__init__(
            model, assertion_number, batch_size, default_data_dir, dataset, raw_set
        )
        self.toga_data_path: Path = (
            Path(self.default_data_dir)
            / str(self.assertion_number)
            / self.model.name
            / "combined"
            / (self._get_dataset_type_dir() + ".csv")
        )
        self.num_data_elements: int = self.get_number_data_points()

    def load_files(self) -> None:
        self.toga_file = Path.open(self.toga_data_path)

    def load_data_stepwise(self) -> tqdm:
        total = self.num_data_elements // self.batch_size
        if self.num_data_elements % self.batch_size != 0:
            total += 1
        prediction_reader = csv.DictReader(
            self.toga_file,
            fieldnames=["idx", "fm", "test", "assertion", "docstring"],
        )
        next(prediction_reader)
        return tqdm(
            map(
                self._extract_references_and_inputs,
                iter(lambda: tuple(islice(prediction_reader, self.batch_size)), ()),
            ),
            total=total,
            desc=f"Evaluating {self.model.name}-{self.assertion_number}",
        )

    def close_files(self) -> None:
        self.toga_file.close()

    def get_number_data_points(self) -> int:
        with Path.open(self.toga_data_path) as inputs:
            return len(inputs.readlines()) - 1

    def _extract_references_and_inputs(self, tuple_predictions):
        pred_list = list(tuple_predictions)
        references = [ref["assertion"] for ref in pred_list]
        predictions = [ref["fm"] + "[SEP]" + ref["test"] + "[SEP]" for ref in pred_list]
        return references, predictions


class AsserT5DataLoader(DataLoader):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        batch_size: int,
        default_data_dir: str,
        dataset: DatasetType,
        raw_set: bool,
    ):
        super().__init__(
            model, assertion_number, batch_size, default_data_dir, dataset, raw_set
        )
        self.assert5_datapath: Path = (
            Path(self.default_data_dir)
            / str(self.assertion_number)
            / self.model.name
            / ("raw" if raw_set else "abstract")
            / (self._get_dataset_type_dir() + ".jsonl")
        )
        self.num_data_elements: int = self.get_number_data_points()

    def load_files(self) -> None:
        self.asserT5_file = Path.open(self.assert5_datapath)

    def load_data_stepwise(self) -> tqdm:
        total = self.num_data_elements // self.batch_size
        if self.num_data_elements % self.batch_size != 0:
            total += 1

        return tqdm(
            map(
                self._convert_to_token_dict,
                iter(lambda: tuple(islice(self.asserT5_file, self.batch_size)), ()),
            ),
            total=total,
            desc=f"Evaluating {self.model.name}-{self.assertion_number}",
        )

    def close_files(self) -> None:
        self.asserT5_file.close()

    def get_number_data_points(self) -> int:
        with Path.open(self.assert5_datapath) as inputs:
            return len(inputs.readlines())

    def _convert_to_token_dict(self, input_tuple):
        input_list = list(input_tuple)
        json_dict_list = [json.loads(row) for row in input_list]
        labels = [el["labels"] for el in json_dict_list]
        references = [el["inputIDs"] for el in json_dict_list]
        dicts = [el["dict"] if "dict" in el else {} for el in json_dict_list]
        return labels, references, dicts


class CachedPredictionsDataLoader(DataLoader):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        batch_size: int,
        default_data_dir: str,
        dataset: DatasetType,
        cached_predictions_file: str,
        top_k: int,
        raw_set: bool,
    ) -> None:
        super().__init__(
            model, assertion_number, batch_size, default_data_dir, dataset, raw_set
        )
        self.cached_predictions_file = cached_predictions_file
        self.predictions_file = None
        self.num_data_elements: int = self.get_number_data_points()
        self.top_k = top_k

    def load_files(self) -> None:
        self.predictions_file = Path.open(Path(self.cached_predictions_file), "r")

    def load_data_stepwise(self) -> Iterable[Any]:
        total = self.num_data_elements // self.batch_size
        if self.num_data_elements % self.batch_size != 0:
            total += 1
        prediction_reader = csv.reader(
            x.replace("\0", "") for x in self.predictions_file
        )
        return tqdm(
            map(
                self._split,
                iter(lambda: tuple(islice(prediction_reader, self.batch_size)), ()),
            ),
            total=total,
            desc=f"Evaluating {self.model.name}-{self.assertion_number}",
        )

    def _split(self, tuple_predictions):
        ref_pred_list = list(tuple_predictions)
        references = [ref[0] for ref in ref_pred_list]
        predictions = [pred[1 : self.top_k + 1] for pred in ref_pred_list]
        return references, predictions

    def close_files(self) -> None:
        self.predictions_file.close()

    def get_number_data_points(self) -> int:
        with Path.open(Path(self.cached_predictions_file)) as inputs:
            return len(inputs.readlines())


def build_data_loader(
    model: Models,
    assertion_number: int,
    batch_size: int,
    dataset: DatasetType,
    default_data_dir: str,
    cache_pred_dir: str,
    top_k: int,
    raw_data: bool,
) -> DataLoader:
    if cache_pred_dir is not None:
        return CachedPredictionsDataLoader(
            model=model,
            assertion_number=assertion_number,
            batch_size=batch_size,
            dataset=dataset,
            default_data_dir=default_data_dir,
            cached_predictions_file=cache_pred_dir,
            top_k=top_k,
            raw_set=raw_data,
        )
    match model:
        case Models.ATLAS | Models.DOUBLE_TRANSFORMERS:
            return AtlasDataLoader(
                model=model,
                assertion_number=assertion_number,
                batch_size=batch_size,
                dataset=dataset,
                default_data_dir=default_data_dir,
                raw_set=raw_data,
            )
        case Models.TOGA:
            return TogaDataLoader(
                model=model,
                assertion_number=assertion_number,
                batch_size=batch_size,
                dataset=dataset,
                default_data_dir=default_data_dir,
                raw_set=raw_data,
            )
        case Models.CODE_2_SEQ:
            raise NotImplementedError
        case Models.ASSERT5:
            return AsserT5DataLoader(
                model=model,
                assertion_number=assertion_number,
                batch_size=batch_size,
                dataset=dataset,
                default_data_dir=default_data_dir,
                raw_set=raw_data,
            )
    raise ValueError
