from models import FunctionDefinition


def get_function_names(functions: list[FunctionDefinition]) -> list[str]:
    return [function.name for function in functions]
