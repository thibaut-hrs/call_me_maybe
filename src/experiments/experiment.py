from llm_sdk import Small_LLM_Model
import json


def main() -> None:
    llm = Small_LLM_Model(device='cpu')

    prompt: str = "How are you ?"
    print(f"Original text: {prompt}")

    token_batch = llm.encode(prompt)
    token_ids = token_batch.tolist()[0]
    print(f"Token IDs: {token_ids}")
    print(f"token_ids type: {type(token_ids)}")
    print(f"First element type {type(token_ids[0])}")

    logits = llm.get_logits_from_input_ids(token_ids)
    print(f"logits type: {type(logits)}")
    print(f"logits len: {len(logits)}")
    print(f"Ten first logits: {logits[:10]}\n")

    tokenizer_path: str = llm.get_path_to_tokenizer_file()
    print(f"tokenizer path: {tokenizer_path}")
    with open(tokenizer_path, "r") as file:
        data = json.load(file)
        print(f"data type: {type(data)}")
        for key in data:
            print(key)
        


if __name__ == "__main__":
    main()
