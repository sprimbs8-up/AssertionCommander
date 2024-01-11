from enum import Enum


class DatasetType(Enum):
    TRAINING = "train"
    TEST = "test"
    VALIDATION = "val"

    def __new__(cls, *args: object) -> object:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, type_name: str) -> None:
        self._type_name: str = type_name

    @property
    def type_name(self) -> str:
        return self._type_name


def parse_type(type_string: str) -> DatasetType | None:
    types = [
        data_type
        for data_type in list(DatasetType)
        if data_type.type_name == type_string
    ]
    return None if len(types) == 0 else types[0]
