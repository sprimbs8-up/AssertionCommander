from enum import Enum


class Models(Enum):
    def __new__(cls, *args: object) -> object:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, name: str) -> None:
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    ATLAS = "atlas"
    DOUBLE_TRANSFORMERS = "double-transformers"
    TOGA = "toga"
    ASSERT5 = "asserT5"
    GPT = "gpt"


def parse_model(model_string: str) -> Models | None:
    models = [model for model in list(Models) if model.name == model_string]
    return None if len(models) == 0 else models[0]
