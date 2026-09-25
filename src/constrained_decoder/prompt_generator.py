from ..models import FunctionDefinition
from .function_utils import (
    get_function_names,
    get_function_descriptions,
)


def generate_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
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
