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


def generate_function_call(
    llm: Small_LLM_Model,
    vocabulary: list[str],
    functions: FunctionDefinition,
    prompt: str
) -> str:

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


# def generate_json() -> list[dict]:
#     results = []

#     for test_case in test_cases:
#         result = generate_function_call(...)
#         results.append(result)

#     return results
