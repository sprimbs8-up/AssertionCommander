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
    types = [type_str for type_str in list(DatasetType) if type_str.type == type_string]
    return None if len(types) == 0 else types[0]
