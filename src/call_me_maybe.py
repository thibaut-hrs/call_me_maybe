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
import time


def main() -> None:
    """Load and display project data."""
    start = time.perf_counter()
    functions = load_function_definitions(
        Path("functions_definition.json")
    )
    test_cases = load_test_cases(
        Path("function_calling_tests.json")
    )

    llm = Small_LLM_Model(device='cpu')

    vocab_size: int = len(llm.get_logits_from_input_ids([0]))
    vocabulary = [
        llm.decode([token_id])
        for token_id in range(vocab_size)
    ]

    generated_text: str = generate_function_call(
        llm,
        vocabulary,
        functions,
        "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    )
    print(f"Generated text: {generated_text}")
    end = time.perf_counter()
    print(f"Generation: {end - start:.2f}s")


if __name__ == "__main__":
    main()
# "parameters": {"a": 2.0, "b": 3.0}}'