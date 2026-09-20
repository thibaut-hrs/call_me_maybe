from llm_sdk import Small_LLM_Model
import re
from .prompt_generator import generate_arg_prompt
from models import FunctionDefinition
import json


NUMBER_PATTERN = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?)")


def is_valid_number_candidate(
    current_text: str,
    candidate_text: str,
) -> bool:
    number: str = current_text + candidate_text
    if number == "-":
        return True
    return bool(NUMBER_PATTERN.fullmatch(number))


def is_valid_number(current_text: str) -> bool:
    return bool(NUMBER_PATTERN.fullmatch(current_text))


def is_valid_candidate(
    current_text: str,
    candidate_text: str,
    target_texts: list[str]
) -> bool:
    for target_text in target_texts:
        if (
            target_text.startswith(current_text + candidate_text)
            and candidate_text != ""
        ):
            return True
    return False


def get_valid_token_ids(
    vocabulary: list[str],
    candidate_ids: list[int],
    current_text: str,
    target_texts: list[str],
) -> list[int]:
    return [
        candidate_id for candidate_id in candidate_ids
        if is_valid_candidate(
            current_text,
            vocabulary[candidate_id],
            target_texts
        )
    ]


def get_valid_number_token_ids(
    vocabulary: list[str],
    candidate_ids: list[int],
    current_number: str
) -> list[int]:
     return [
        candidate_id
        for candidate_id in candidate_ids
        if is_valid_number_candidate(
            current_number,
            vocabulary[candidate_id]
        )
    ]
     


def get_best_token_id(
    valid_token_ids: list[int],
    logits: list[float],
) -> int:
    return max(valid_token_ids, key=lambda id: logits[id])


def is_generation_complete(
    current_text: str,
    target_texts: list[str],
) -> bool:
    return current_text in target_texts


def generate_constrained(
    llm: Small_LLM_Model,
    vocabulary: list[str],
    current_text: str,
    token_ids: list[int],
    target_texts: list[str],
) -> tuple[str, list[int]]:

    target_texts = [current_text + text for text in target_texts]

    token_ids = token_ids.copy()

    generated: str = ""

    while True:
        logits: list[float] = llm.get_logits_from_input_ids(token_ids)
        candidate_ids: list[int] = list(range(len(logits)))
        valid_ids: list[int] = get_valid_token_ids(
            vocabulary, candidate_ids, current_text, target_texts
        )
        best_token_id = get_best_token_id(valid_ids, logits)

        token_ids.append(best_token_id)
        decoded_token = vocabulary[best_token_id]
        generated += decoded_token
        current_text += decoded_token

        if is_generation_complete(current_text, target_texts):
            break

    return (generated, token_ids)


def generate_number(
    llm: Small_LLM_Model,
    vocabulary: list[str],
    token_ids: list[int],
    end_token_ids: list[int]
) -> tuple[str, list[int]]:

    token_ids = token_ids.copy()
    number: str = ""

    while True:
        logits: list[float] = llm.get_logits_from_input_ids(token_ids)
        candidate_ids: list[int] = list(range(len(logits)))
        valid_ids: list[int] = get_valid_number_token_ids(
            vocabulary,
            candidate_ids,
            number
        )

        if is_valid_number(number):
            valid_ids.extend(end_token_ids)

        best_token_id = get_best_token_id(valid_ids, logits)

        if best_token_id in end_token_ids:
            break

        token_ids.append(best_token_id)
        number += vocabulary[best_token_id]

    return (number, token_ids)


def generate_string(
        llm: Small_LLM_Model,
        vocabulary: list[str],
        token_ids: list[int],
        prompt: str,
        function: FunctionDefinition,
        parameter: str
) -> tuple[str, list[int]]:
    def validate_string_token(
        token: str,
        escaped: bool
    ) -> tuple[bool, bool, bool]:

        if token == "":
            return False, escaped, False

        closes_string = False

        for character in token:
            if escaped:
                if character not in ('"', "\\", "/", "b", "f", "n", "r", "t"):
                    return False, escaped, False

                escaped = False
                continue

            if character == "\\":
                escaped = True
                continue

            if character == '"':
                closes_string = True
                continue

            if ord(character) < 32:
                return False, escaped, False

        return True, escaped, closes_string

    token_ids = token_ids.copy()
    prompt: str = generate_arg_prompt(prompt, function, parameter)
    prompt_token_batch = llm.encode(prompt)
    prompt_token_ids = prompt_token_batch.tolist()[0]

    string: str = '"'
    token_ids += llm.encode('"')

    escaped: bool = False

    while True:
        logits = llm.get_logits_from_input_ids(prompt_token_ids)
        candidate_ids = list(range(len(logits)))

        valid_ids = []

        for token_id in candidate_ids:
            token = vocabulary[token_id]

            valid, _, _ = validate_string_token(token, escaped)

            if valid:
                valid_ids.append(token_id)

        best_token_id = get_best_token_id(valid_ids, logits)
        decoded_token = vocabulary[best_token_id]

        valid, escaped, closes_string = validate_string_token(
            decoded_token,
            escaped
        )

        token_ids.append(best_token_id)
        prompt_token_ids.append(best_token_id)
        string += decoded_token

        if closes_string and not escaped:
            break

    json.dumps(string)
    return (string, token_ids)



