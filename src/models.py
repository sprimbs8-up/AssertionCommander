from enum import Enum
from typing import Optional


class Models(Enum):
    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, name: str):
        self._name: str = name

    @property
    def name(self):
        return self._name

    ATLAS = "atlas"
    DOUBLE_TRANSFORMERS = "double-transformers"
    TOGA = "toga"
    CODE_2_SEQ = "code2seq"


def parse_model(model_string) -> Optional[Models]:
    models = [model for model in list(Models) if model.name == model_string]
    return None if len(models) == 0 else models[0]
