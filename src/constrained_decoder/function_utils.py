from models import FunctionDefinition


def get_function_names(functions: list[FunctionDefinition]) -> list[str]:
    return [function.name for function in functions]


def get_function_descriptions(functions: list[FunctionDefinition]) -> list[str]:
    return [function.description for function in functions]


def get_function_definition(
        functions: list[FunctionDefinition],
        generated_text: str,
        prompt: str
) -> FunctionDefinition:
    description: str = generated_text[len(prompt):]
    for function in functions:
        if function.description == description:
            return function

    raise ValueError(
        f"'{description}' does not match any function "
        "description in the functions_definition.json file."
    )
