"""Functions for loading and validating project JSON files."""

import json
from pathlib import Path
from pydantic import TypeAdapter

from ..models import FunctionDefinition, TestCase


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the JSON file containing function definitions.

    Returns:
        A list of validated function definitions.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    adapter = TypeAdapter(list[FunctionDefinition])
    return adapter.validate_python(data)


def load_test_cases(path: Path) -> list[TestCase]:
    """Load and validate test cases from a JSON file.

    Args:
        path: Path to the JSON file containing test cases.

    Returns:
        A list of validated test cases.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    adapter = TypeAdapter(list[TestCase])
    return adapter.validate_python(data)
