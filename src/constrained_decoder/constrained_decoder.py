from llm_sdk import Small_LLM_Model


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
    llm: Small_LLM_Model,
    candidate_ids: list[int],
    current_text: str,
    target_texts: list[str],
) -> list[int]:
    return [
        candidate_id for candidate_id in candidate_ids
        if is_valid_candidate(
            current_text,
            llm.decode([candidate_id]),
            target_texts
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
    current_text: str,
    token_ids: list[int],
    target_texts: list[str],
) -> tuple[str, list[int]]:

    print(f"Initial text: {current_text}")
    print(f"Initial tokens: {token_ids}")
    print(target_texts)

    while True:
        logits: list[float] = llm.get_logits_from_input_ids(token_ids)
        candidate_ids: list[int] = list(range(len(logits)))
        valid_ids: list[int] = get_valid_token_ids(
            llm, candidate_ids, current_text, target_texts
        )
        print(f"Number of valid IDs: {len(valid_ids)}")
        best_token_id = get_best_token_id(valid_ids, logits)

        token_ids.append(best_token_id)
        current_text += llm.decode([best_token_id])

        if is_generation_complete(current_text, target_texts):
            break

    return (current_text, token_ids)
