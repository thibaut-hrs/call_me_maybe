from pathlib import Path
from llm_sdk import Small_LLM_Model
from .loaders import load_function_definitions, load_test_cases
from .json_generator import generate_output_file
import time
import argparse
from json import JSONDecodeError
from pydantic import ValidationError


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate function calls from natural language prompts."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="File with all the function definitions the LLM will use"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="File with the testing prompts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="File containing the generated function calls"
    )
    return parser.parse_args()


def main() -> None:
    """Load and display project data."""

    start = time.perf_counter()
    args = parse_arguments()
    last_try_open: str = ""

    try:
        last_try_open = args.functions_definition
        functions = load_function_definitions(
            Path(args.functions_definition)
        )
        last_try_open = args.input
        test_cases = load_test_cases(
            Path(args.input)
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
            args.output
        )

        end = time.perf_counter()
        print(f"Generation: {end - start:.2f}s")

    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        print(f"Error when opening file: {e}")

    except JSONDecodeError as e:
        print(f"JSON format error with file {last_try_open}: {e}")

    except ValidationError as e:
        print(f"Wrong field detected in file {last_try_open}:\n{e}")

    except (KeyboardInterrupt, EOFError) as e:
        print(f"Programm manually interrupted by user")


if __name__ == "__main__":
    main()
