from llm_sdk import Small_LLM_Model


def main() -> None:
    llm = Small_LLM_Model(device='cpu')

    prompt: str = "How are you ?"
    print(f"Original text: {prompt}")

    token_batch = llm.encode(prompt)
    print(f"\nencoded: {token_batch}")

    print(f"Type: {type(token_batch)}")

    print(f"Shape: {token_batch.shape}")

    print(
        f"token_batch.tolist() = {token_batch.tolist()}, "
        f"type = {type(token_batch.tolist())}"
    )

    decoded: str = ""
    for token_sequence in token_batch:
        decoded += llm.decode(token_sequence)
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()
