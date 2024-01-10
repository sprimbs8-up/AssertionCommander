import abc
from itertools import islice
from pathlib import Path
from contextlib import contextmanager
from tqdm import tqdm


class DataLoader(abc.ABC):
    def __init__(
        self,
        model_name: str,
        assertion_number: int,
        batch_size: int,
        default_data_dir: str = "evaluation-data",
    ):
        self.model_name = model_name
        self.assertion_number = assertion_number
        self.batch_size = batch_size
        self.default_data_dir = default_data_dir

    @abc.abstractmethod
    def load_files(self) -> None:
        pass

    @abc.abstractmethod
    def load_data_stepwise(self):
        pass

    @abc.abstractmethod
    def close_files(self) -> None:
        pass

    @abc.abstractmethod
    def get_number_data_points(self) -> int:
        pass


class AtlasDataLoader(DataLoader):
    def __init__(self, model_name: str, assertion_number: int, batch_size: int):
        super().__init__(model_name, assertion_number, batch_size)

        self.references_file_path: Path = (
            Path(self.default_data_dir)
            / str(self.assertion_number)
            / str(self.model_name)
            / "assertLines.txt"
        )
        self.input_file_path: Path = (
            Path(self.default_data_dir)
            / str(self.assertion_number)
            / str(self.model_name)
            / "testMethods.txt"
        )
        self.input_file = None
        self.ref_file = None
        self.num_data_elements: int = self.get_number_data_points()

    def get_number_data_points(self) -> int:
        with open(self.input_file_path, "r") as inputs:
            return len(inputs.readlines())

    def load_files(self):
        self.ref_file = open(self.references_file_path, "r")
        self.input_file = open(self.input_file_path, "r")

    def load_data_stepwise(self):
        total = self.num_data_elements // self.batch_size
        if self.num_data_elements % self.batch_size != 0:
            total += 1
        return tqdm(
            zip(
                iter(lambda: tuple(islice(self.ref_file, self.batch_size)), ()),
                iter(lambda: tuple(islice(self.input_file, self.batch_size)), ()),
            ),
            total=total,
            leave=False,
            desc=f"Evaluating {self.model_name}-{self.assertion_number}",
        )

    def close_files(self):
        self.ref_file.close()
        self.input_file.close()
