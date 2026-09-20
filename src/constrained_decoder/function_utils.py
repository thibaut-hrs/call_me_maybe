from models import FunctionDefinition


def get_function_names(functions: list[FunctionDefinition]) -> list[str]:
    return [function.name for function in functions]


def get_function_descriptions(
    functions: list[FunctionDefinition]
) -> list[str]:
    return [function.description for function in functions]


def get_function_definition(
        functions: list[FunctionDefinition],
        generated_text: str
) -> FunctionDefinition:
    function_name: str = generated_text.split(' ')[-1]
    for function in functions:
        if function.name == function_name:
            return function

    raise ValueError(
        f"'{function_name}' does not match any function "
        "in the functions_definition.json file."
    )
