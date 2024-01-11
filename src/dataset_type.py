from enum import Enum
from typing import Optional


class DatasetType(Enum):
    TRAINING = "train"
    TEST = "test"
    VALIDATION = "val"

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, type: str):
        self._type: str = type

    @property
    def type(self):
        return self._type


def parse_type(type_string) -> Optional[DatasetType]:
    models = [
        type_str for type_str in list(DatasetType) if type_str.name == type_string
    ]
    return None if len(models) == 0 else models[0]
