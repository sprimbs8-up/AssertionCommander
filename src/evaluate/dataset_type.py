from enum import Enum


class DatasetType(Enum):
    """
    Enumeration representing different types of datasets.

    Attributes:
        TRAINING (DatasetType): Training dataset.
        TEST (DatasetType): Test dataset.
        TEST_FILTERED (DatasetType): Filtered test dataset without any data leakage problem.
        VALIDATION (DatasetType): Validation dataset.
    """

    TRAINING = "train"
    TEST = "test"
    TEST_FILTERED = "test_filtered"
    VALIDATION = "val"

    def __new__(cls, *args: object) -> object:
        """
        Create a new instance of the DatasetType enum.

        Args:
            *args (object): Variable number of arguments.

        Returns:
            object: A new instance of the DatasetType enum.
        """
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, type_name: str) -> None:
        self._type_name: str = type_name

    @property
    def type_name(self) -> str:
        """
        Get the name of the dataset type.

        Returns:
            str: The name of the dataset type.
        """
        return self._type_name


def parse_type(type_string: str) -> DatasetType | None:
    """
    Parse a string to obtain the corresponding DatasetType enum value.

    Args:
        type_string (str): The string representing the dataset type.

    Returns:
        DatasetType | None: The corresponding DatasetType enum value if found, otherwise None.
    """
    types = [
        data_type
        for data_type in list(DatasetType)
        if data_type.type_name == type_string
    ]
    return None if len(types) == 0 else types[0]
