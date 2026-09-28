from llm_sdk import Small_LLM_Model
from ..models import FunctionDefinition
from ..constrained_decoder import (
    generate_constrained,
    generate_number,
    generate_string,
    get_function_definition,
    get_function_names,
    generate_prompt
)
import json
from pathlib import Path


def generate_function_call(
    llm: Small_LLM_Model,
    vocabulary: list[str],
    functions: list[FunctionDefinition],
    prompt: str
) -> str:
    """Generate a JSON function call from a user prompt.

    The function first selects the function that best matches the user
    prompt, then generates the corresponding JSON structure and infers
    the value of each function parameter using constrained decoding.

    Args:
        llm: The language model used to generate the function call.
        vocabulary: The model vocabulary mapping token IDs to token texts.
        functions: The available function definitions.
        prompt: The user request from which the function call is generated.

    Returns:
        A JSON-formatted string representing the generated function call.
    """

    print(f'Genrating a function call from a prompt: "{prompt}"')

    context: str = generate_prompt(prompt, functions)

    token_batch = llm.encode(context)
    token_ids = token_batch.tolist()[0]
    logits = llm.get_logits_from_input_ids(token_ids)

    target_texts = get_function_names(functions)
    context, token_ids = generate_constrained(
        llm,
        vocabulary,
        context,
        token_ids,
        target_texts
    )
    function = get_function_definition(functions, context)

    text_generated: str = ""

    target_texts = ['{"prompt": ' + json.dumps(prompt) + ', ']
    generated, token_ids = generate_constrained(
        llm,
        vocabulary,
        context,
        token_ids,
        target_texts
    )
    context, text_generated = context + generated, text_generated + generated

    target_texts = ['"name": "' + function.name + '", "parameters": {']
    generated, token_ids = generate_constrained(
        llm,
        vocabulary,
        context,
        token_ids,
        target_texts
    )
    context, text_generated = context + generated, text_generated + generated

    param_count: int = len(function.parameters)

    i: int = 0
    for key, value in function.parameters.items():
        i += 1
        target_texts = ['"' + key + '": ']
        generated, token_ids = generate_constrained(
            llm,
            vocabulary,
            context,
            token_ids,
            target_texts
        )
        context = context + generated
        text_generated = text_generated + generated

        if value.type == "number":
            token_full_list = list(range(len(logits)))
            end_token_ids = [
                token_id
                for token_id in token_full_list
                if (
                    llm.decode([token_id]) == " "
                    or llm.decode([token_id]) == ","
                )
            ]
            generated, token_ids = generate_number(
                llm,
                vocabulary,
                token_ids,
                end_token_ids
            )
            context = context + generated
            text_generated = text_generated + generated

        elif value.type == "string":
            generated, token_ids = generate_string(
                llm,
                vocabulary,
                token_ids,
                prompt,
                function,
                key
            )
            context = context + generated
            text_generated = text_generated + generated

        if i < param_count:
            target_texts = [", "]
            generated, token_ids = generate_constrained(
                llm,
                vocabulary,
                context,
                token_ids,
                target_texts
            )
            context = context + generated
            text_generated = text_generated + generated

    target_texts = ["}}"]
    generated, token_ids = generate_constrained(
        llm,
        vocabulary,
        context,
        token_ids,
        target_texts
    )
    context, text_generated = context + generated, text_generated + generated

    return text_generated


def generate_all_function_calls(
    llm: Small_LLM_Model,
    vocabulary: list[str],
    functions: list[FunctionDefinition],
    prompts: list[str],
) -> list[dict]:
    """Generate function calls for a list of user prompts.

    Each prompt is processed independently by generate_function_call(),
    and the resulting JSON strings are parsed into Python dictionaries.

    Args:
        llm: The language model used to generate the function calls.
        vocabulary: The model vocabulary mapping token IDs to token texts.
        functions: The available function definitions.
        prompts: The user prompts for which function calls should be
            generated.

    Returns:
        A list of dictionaries containing the generated function calls.
    """

    result: list[dict] = []

    for prompt in prompts:
        generated_text = generate_function_call(
            llm,
            vocabulary,
            functions,
            prompt
        )
        result.append(json.loads(generated_text))

    return result


def write_function_calls(
    function_calls: list[dict],
    output_path: Path,
) -> None:
    """Write generated function calls to a JSON file.

    The output directory is created if it does not already exist.

    Args:
        function_calls: The function calls to write to the output file.
        output_path: The path of the JSON file to create.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(function_calls, file, indent=4)


def generate_output_file(
    llm: Small_LLM_Model,
    vocabulary: list[str],
    functions: list[FunctionDefinition],
    prompts: list[str],
    output_path: Path,
) -> None:
    """Generate function calls and write them to an output JSON file.

    This function orchestrates the complete generation pipeline: it generates
    a function call for each prompt and writes the resulting calls to the
    specified output file.

    Args:
        llm: The language model used to generate the function calls.
        vocabulary: The model vocabulary mapping token IDs to token texts.
        functions: The available function definitions.
        prompts: The user prompts to process.
        output_path: The path of the JSON file to create.
    """

    function_calls = generate_all_function_calls(
        llm,
        vocabulary,
        functions,
        prompts
    )
    write_function_calls(function_calls, output_path)
    print(
        "Function calls succesfully written in JSON format "
        f"in the file '{output_path}' !"
    )
