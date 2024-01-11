import abc
from itertools import islice
from pathlib import Path

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


def build_data_loader(
    model: Models,
    assertion_number: int,
    batch_size: int,
    dataset: DatasetType,
    default_data_dir: str,
) -> DataLoader:
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
