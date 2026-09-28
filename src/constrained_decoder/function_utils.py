from ..models import FunctionDefinition


def get_function_names(functions: list[FunctionDefinition]) -> list[str]:
    """Extract the names of a list of function definitions.

    Args:
        functions: The function definitions to extract names from.

    Returns:
        A list containing the name of each function.
    """
    return [function.name for function in functions]


def get_function_descriptions(
    functions: list[FunctionDefinition]
) -> list[str]:
    """Extract the descriptions of a list of function definitions.

    Args:
        functions: The function definitions to extract descriptions from.

    Returns:
        A list containing the description of each function.
    """
    return [function.description for function in functions]


def get_function_definition(
        functions: list[FunctionDefinition],
        generated_text: str
) -> FunctionDefinition:
    """Find the function definition matching the generated function name.

    The function name is extracted from the last space-separated part of
    generated_text and compared with the name of each provided function.

    Args:
        functions: The available function definitions.
        generated_text: The generated text containing the function name.

    Returns:
        The function definition whose name matches the generated function
        name.

    Raises:
        ValueError: If the generated function name does not match any of the
            provided function definitions.
    """
    function_name: str = generated_text.split(' ')[-1]
    for function in functions:
        if function.name == function_name:
            return function

    raise ValueError(
        f"'{function_name}' does not match any function "
        "in the functions_definition.json file."
    )
