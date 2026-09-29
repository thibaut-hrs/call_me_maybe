from ..models import FunctionDefinition
from .function_utils import (
    get_function_names,
    get_function_descriptions,
)


def generate_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Generate a prompt for selecting a function from a user request.

    The generated prompt provides the language model with the available
    function names and their descriptions, and instructs it to return only
    the name of the function that best matches the user request.

    Args:
        prompt: The user's original request.
        functions: The available function definitions used for function
            selection.

    Returns:
        A formatted prompt instructing the language model to select one
        function name.
    """
    function_names = get_function_names(functions)
    function_descriptions = get_function_descriptions(functions)

    available_functions: list[str] = []
    for i in range(len(function_names)):
        available_functions.append(
            f'{function_names[i]}: {function_descriptions[i]}'
        )

    return f"""
    You are a function router. Read the request and choose the best function
    by reading the description. Return ONLY the function name.

    Available functions:
    {available_functions}

    User request: {prompt}

    Return exactly one function name, and nothing else:
    """


def generate_arg_prompt(
    prompt: str,
    function: FunctionDefinition,
    parameter: str
) -> str:
    """Generate a prompt for inferring a function argument value.

    The generated prompt provides the language model with the user's request,
    the selected function definition, its parameters, and the name of the
    argument whose value must be inferred.

    Args:
        prompt: The user's original request.
        function: The function definition for which an argument value is
            being generated.
        parameter: The name of the function parameter whose value must be
            inferred.

    Returns:
        A formatted prompt instructing the language model to return the
        requested argument as a JSON string value.
    """

    parameters: str = ""
    for key, value in function.parameters.items():
        parameters += f"- {key}: {value}\n"

    return f"""
    Complete one argument of a function call.

    User request:
    {prompt}

    Function name:
    {function.name}

    Function description:
    {function.description}

    Function parameters:
    {parameters}
    Requested argument:
    {parameter}

    Infer the value of the requested argument from the user request and the
    function definition. Return exactly one JSON string value
    and nothing after it.

    {parameter} value: \""""
