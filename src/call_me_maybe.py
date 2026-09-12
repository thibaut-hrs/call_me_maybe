"""Program entry point."""

from pathlib import Path
from llm_sdk import Small_LLM_Model
from models import FunctionDefinition
from loaders import load_function_definitions, load_test_cases
from constrained_decoder import (
    generate_constrained,
    get_function_names,
    get_function_definition
)
from json_generator import generate_function_call


def main() -> None:
    """Load and display project data."""
    functions = load_function_definitions(
        Path("functions_definition.json")
    )
    test_cases = load_test_cases(
        Path("function_calling_tests.json")
    )

    llm = Small_LLM_Model(device='cpu')
    generated_text: str = generate_function_call(
        llm,
        functions,
        "What is the sum of 2 and 3?"
    )
    print(f"Generated text: {generated_text}")
    


if __name__ == "__main__":
    main()
# "parameters": {"a": 2.0, "b": 3.0}}'