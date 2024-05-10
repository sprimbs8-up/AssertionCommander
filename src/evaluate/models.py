from enum import Enum


class Models(Enum):
    """Enum class representing different models."""

    ATLAS = "atlas"
    DOUBLE_TRANSFORMERS = "double-transformers"
    TOGA = "toga"
    ASSERT5 = "asserT5"
    GPT = "gpt"

    def __new__(cls, *args: object) -> object:
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, name: str) -> None:
        self._name: str = name

    @property
    def name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self._name


def parse_model(model_string: str) -> Models | None:
    """
    Parse a model string and return the corresponding Models enum value.

    Args:
        model_string (str): The model string.

    Returns:
        Models | None: The corresponding Models enum value, or None if not found.
    """
    models = [model for model in list(Models) if model.name == model_string]
    return None if len(models) == 0 else models[0]
