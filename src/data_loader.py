import abc
import csv
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
    ) -> None:
        self.model = model
        self.assertion_number = assertion_number
        self.batch_size = batch_size
        self.default_data_dir = default_data_dir
        self.dataset = dataset

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


class AtlasDataLoader(DataLoader):
    def __init__(
        self,
        model: Models,
        assertion_number: int,
        batch_size: int,
        default_data_dir: str,
        dataset: DatasetType,
    ) -> None:
        super().__init__(model, assertion_number, batch_size, default_data_dir, dataset)

        self.references_file_path: Path = (
            Path(self.default_data_dir)
            / str(self.assertion_number)
            / self.model.name
            / self._get_dataset_type_dir()
            / "assertLines.txt"
        )
        self.input_file_path: Path = (
            Path(self.default_data_dir)
            / str(self.assertion_number)
            / self.model.name
            / self._get_dataset_type_dir()
            / "testMethods.txt"
        )
        self.input_file = None
        self.ref_file = None
        self.num_data_elements: int = self.get_number_data_points()

    def _get_dataset_type_dir(self) -> str:
        match self.dataset:
            case DatasetType.TEST:
                return "testing"
            case DatasetType.TRAINING:
                return "training"
            case DatasetType.VALIDATION:
                return "validation"

    def get_number_data_points(self) -> int:
        with Path.open(self.input_file_path) as inputs:
            return len(inputs.readlines())

    def load_files(self) -> None:
        self.ref_file = Path.open(self.references_file_path)
        self.input_file = Path.open(self.input_file_path)

    def load_data_stepwise(self) -> tqdm:
        total = self.num_data_elements // self.batch_size
        if self.num_data_elements % self.batch_size != 0:
            total += 1
        return tqdm(
            zip(
                iter(lambda: tuple(islice(self.ref_file, self.batch_size)), ()),
                iter(lambda: tuple(islice(self.input_file, self.batch_size)), ()),
                strict=False,
            ),
            total=total,
            leave=False,
            desc=f"Evaluating {self.model.name}-{self.assertion_number}",
        )

    def close_files(self) -> None:
        self.ref_file.close()
        self.input_file.close()


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
    ) -> None:
        super().__init__(model, assertion_number, batch_size, default_data_dir, dataset)
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
        prediction_reader = csv.reader(self.predictions_file)
        return tqdm(
            map(
                self._split,
                iter(lambda: tuple(islice(prediction_reader, self.batch_size)), ()),
            ),
            total=total,
            leave=False,
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
        )
    match model:
        case Models.ATLAS | Models.DOUBLE_TRANSFORMERS:
            return AtlasDataLoader(
                model=model,
                assertion_number=assertion_number,
                batch_size=batch_size,
                dataset=dataset,
                default_data_dir=default_data_dir,
            )
        case Models.TOGA, Models.CODE_2_SEQ:
            raise NotImplementedError
    raise ValueError
