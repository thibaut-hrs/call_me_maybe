from llm_sdk import Small_LLM_Model
import json


def main() -> None:
    llm = Small_LLM_Model(device='cpu')

    prompt: str = "What is the sum of 40 and 2?"

    token_batch = llm.encode(prompt)
    initial_token_ids = token_batch.tolist()[0]

    token_ids = initial_token_ids.copy()
    for i in range(40):
        logits = llm.get_logits_from_input_ids(token_ids)
        token_ids.append(logits.index(max(logits)))
    print("Prompt: ", prompt)
    initial_len: int = len(initial_token_ids)
    print("LLM: ", llm.decode(token_ids[initial_len:]))

def is_valid_candidate(
    current_text: str,
    candidate_text: str,
    target_texts: list[str]
) -> bool:
    for target_text in target_texts:
        if (target_text.startswith(current_text + candidate_text)):
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

if __name__ == "__main__":
    llm = Small_LLM_Model(device='cpu')
    current = "Je mange "
    candidate = "des pates"
    target = "Je mange des pates"
    tokens_batch = llm.encode(candidate)
    tokens_ids = tokens_batch.tolist()[0]
    print(get_valid_token_ids(llm, tokens_ids, current, [target]))

    print(llm.decode(4340))
    print(type(llm.decode(4340)))

    print(llm.decode([4340]))
    print(type(llm.decode([4340])))

    decoded = llm.decode([4340])

    if isinstance(decoded, list):
        print(type(decoded[0]))