"""Program entry point."""

from pathlib import Path

from src.loaders import load_function_definitions, load_test_cases


def main() -> None:
    """Load and display project data."""
    functions = load_function_definitions(
        Path("functions_definition.json")
    )
    test_cases = load_test_cases(
        Path("function_calling_tests.json")
    )

    print("Functions:")
    for function in functions:
        print(function)

    print("\nTest cases:")
    for test_case in test_cases:
        print(test_case)


if __name__ == "__main__":
    main()
