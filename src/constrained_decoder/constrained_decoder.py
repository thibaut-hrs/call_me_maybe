from llm_sdk import Small_LLM_Model
import re
from .prompt_generator import generate_arg_prompt
from ..models import FunctionDefinition
import json


NUMBER_PATTERN = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?)")


def is_valid_number_candidate(
    current_text: str,
    candidate_text: str,
) -> bool:
    """Check whether a candidate token can extend a valid number.

    The candidate is appended to the current text and the resulting string
    is checked against the number pattern. A standalone minus sign is also
    accepted as an intermediate state during number generation.

    Args:
        current_text: The number text generated so far.
        candidate_text: The token text to append to the current number.

    Returns:
        True if the candidate produces a valid or potentially valid number,
        otherwise False.
    """

    number: str = current_text + candidate_text
    if number == "-":
        return True
    return bool(NUMBER_PATTERN.fullmatch(number))


def is_valid_number(current_text: str) -> bool:
    """Check whether a string represents a valid number.

    Args:
        current_text: The text to validate as a number.

    Returns:
        True if the text matches the supported number format, otherwise False.
    """
    return bool(NUMBER_PATTERN.fullmatch(current_text))


def is_valid_candidate(
    current_text: str,
    candidate_text: str,
    target_texts: list[str]
) -> bool:
    """Check whether a token can extend the current text toward a target.

    A candidate is valid when appending it to the current text produces a
    prefix of at least one target text.

    Args:
        current_text: The text generated so far.
        candidate_text: The token text being considered.
        target_texts: The possible complete target texts.

    Returns:
        True if the candidate can lead to one of the target texts,
        otherwise False.
    """
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
    """Filter token IDs to those that can extend the current text.

    Args:
        vocabulary: The model vocabulary mapping token IDs to token texts.
        candidate_ids: The token IDs to evaluate.
        current_text: The text generated so far.
        target_texts: The possible complete target texts.

    Returns:
        A list containing only the candidate token IDs that can extend
        current_text toward at least one target text.
    """
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
    """Filter token IDs to those that can extend the current number.

    Args:
        vocabulary: The model vocabulary mapping token IDs to token texts.
        candidate_ids: The token IDs to evaluate.
        current_number: The number text generated so far.

    Returns:
        A list containing only the candidate token IDs that preserve a
        valid number format.
    """
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
    """Select the highest-scoring token among valid candidates.

    Args:
        valid_token_ids: Token IDs that satisfy the current decoding
            constraints.
        logits: The model logits indexed by token ID.

    Returns:
        The token ID with the highest logit among the valid candidates.
    """
    return max(valid_token_ids, key=lambda id: logits[id])


def is_generation_complete(
    current_text: str,
    target_texts: list[str],
) -> bool:
    """Check whether the generated text matches a target text.

    Args:
        current_text: The text generated so far.
        target_texts: The possible complete target texts.

    Returns:
        True if current_text exactly matches one of the target texts,
        otherwise False.
    """
    return current_text in target_texts


def generate_constrained(
    llm: Small_LLM_Model,
    vocabulary: list[str],
    current_text: str,
    token_ids: list[int],
    target_texts: list[str],
) -> tuple[str, list[int]]:
    """Generate text while restricting tokens to valid target prefixes.

    At each generation step, only tokens that keep the generated text as a
    prefix of at least one target text are considered. Among those tokens,
    the one with the highest model logit is selected.

    Args:
        llm: The language model used to generate the next-token logits.
        vocabulary: The model vocabulary mapping token IDs to token texts.
        current_text: The text already present before constrained generation.
        token_ids: The token IDs corresponding to the current model input.
        target_texts: The possible texts that the model is allowed to
            generate after current_text.

    Returns:
        A tuple containing the generated text and the updated list of token
        IDs.
    """

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
    """Generate a valid number using constrained token decoding.

    Tokens that would produce an invalid number are excluded from the
    candidates. Once the generated text forms a valid number, end tokens
    are also allowed so that generation can terminate.

    Args:
        llm: The language model used to generate the next-token logits.
        vocabulary: The model vocabulary mapping token IDs to token texts.
        token_ids: The token IDs corresponding to the current model input.
        end_token_ids: Token IDs that can terminate number generation.

    Returns:
        A tuple containing the generated number as a string and the updated
        list of token IDs.
    """

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
    """Generate a JSON-compatible string using constrained decoding.

    The function builds the argument-generation prompt and constrains token
    generation so that the resulting value follows JSON string rules.
    Invalid escape sequences and control characters are rejected, while an
    unescaped closing quote terminates the generated string.

    Args:
        llm: The language model used to generate the string value.
        vocabulary: The model vocabulary mapping token IDs to token texts.
        token_ids: The token IDs accumulated for the function call.
        prompt: The original user prompt used to build the argument prompt.
        function: The function definition containing the parameter being
            generated.
        parameter: The name of the function parameter whose string value is
            being generated.

    Returns:
        A tuple containing the generated JSON string and the updated list of
        token IDs.
    """

    def validate_string_token(
        token: str,
        escaped: bool
    ) -> tuple[bool, bool, bool]:
        """Validate a token according to JSON string escaping rules.

        Args:
            token: The token text to validate.
            escaped: Whether the previous token ended with an escape
                character.

        Returns:
            A tuple containing:
                - Whether the token is valid.
                - Whether the token leaves an unfinished escape sequence.
                - Whether the token contains a closing quote.
        """
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
    arg_prompt: str = generate_arg_prompt(prompt, function, parameter)
    prompt_token_batch = llm.encode(arg_prompt)
    prompt_token_ids = prompt_token_batch.tolist()[0]

    string: str = '"'
    token_ids += llm.encode('"').tolist()[0]

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

    return (string, token_ids)
