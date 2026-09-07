from llm_sdk import Small_LLM_Model
import json


def main() -> None:
    llm = Small_LLM_Model(device='cpu')

    prompt: str = "Sometimes I put pinapples on pizza"

    token_batch = llm.encode(prompt)
    initial_token_ids = token_batch.tolist()[0]

    token_ids = initial_token_ids.copy()
    for i in range(40):
        logits = llm.get_logits_from_input_ids(token_ids)
        token_ids.append(logits.index(max(logits)))
    print(llm.decode(token_ids))


if __name__ == "__main__":
    main()
