from pathlib import Path
from llm_sdk import Small_LLM_Model
from .loaders import load_function_definitions, load_test_cases
from .json_generator import generate_output_file
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
    prompts: list[str] = [case.prompt for case in test_cases]

    llm = Small_LLM_Model(device='cpu')

    vocab_size: int = len(llm.get_logits_from_input_ids([0]))
    vocabulary = [
        llm.decode([token_id])
        for token_id in range(vocab_size)
    ]

    generate_output_file(
        llm,
        vocabulary,
        functions,
        prompts,
        "output.json"
    )
    
    end = time.perf_counter()
    print(f"Generation: {end - start:.2f}s")


if __name__ == "__main__":
    main()
