*This project has been created as part of the 42 curriculum by thours.*

# Call Me Maybe

## Description

**Call Me Maybe** is a Python project about function calling with Large Language Models (LLMs).

The program takes natural-language prompts and generates structured JSON function calls. It uses the provided `Qwen/Qwen3-0.6B` model through the `llm_sdk` package.

The main challenge is to obtain reliable structured output from a small language model. Instead of relying only on prompting, this project uses **constrained decoding** to restrict the tokens that can be generated at each step.

The generated output contains:
- `prompt`: the original user request;
- `name`: the selected function name;
- `parameters`: the generated arguments with the expected types.

## Instructions

### Requirements

- Python 3.10+
- `uv`

Install the project dependencies with:

```bash
uv sync
```

### Run

By default:

```bash
uv run python -m src
```

Custom input/output files can be provided:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

The default input files are located in `data/input/`. The output directory is created automatically when needed.

## Algorithm explenation

The generation pipeline works in several stages:

1. The user prompt is passed to the LLM with the available function names and descriptions.
2. The LLM selects a function name using constrained decoding.
3. The selected function definition determines the expected parameters and their types.
4. The JSON structure is generated piece by piece using constrained decoding.
5. Number parameters are generated with a numeric constraint allowing only valid integer or floating-point representations.
6. String parameters are generated with JSON string validation, including escape sequences and control-character checks.
7. The completed JSON object is written to the output file.

For each constrained generation step, the model produces logits for all possible tokens. The implementation filters the vocabulary to tokens that keep the partial output valid, then selects the highest-scoring valid token. This prevents the decoder from generating tokens that would break the expected structure.

### Token index optimization
The model vocabulary contains a large number of tokens. Checking every token at every generation step is unnecessarily expensive, especially when only a small subset of tokens can possibly continue the current text.

To reduce this cost, the project builds a token index once when the program starts.

### KV Cache optimization
Autoregressive language models generate text one token at a time. Without a KV cache, the complete sequence would have to be processed again at every generation step:

```text
prompt
prompt + token 1
prompt + token 1 + token 2  
prompt + token 1 + token 2 + token 3
...
```

As the generated sequence becomes longer, this causes a large amount of redundant computation.

The implementation uses the **Key-Value (KV) cache** provided by the Transformer model to avoid this repeated work.

The initial prompt is processed normally:
```text
prompt
  ↓
LLM
  ↓
logits + KV cache
```

For the next generation step, only the newly generated token is passed to the model together with the previously computed cache:
```text
new token + previous KV cache
              ↓
             LLM
              ↓
    new logits + new KV cache
```

The process is repeated for each generated token.

The complete list of token IDs is still maintained by the program for decoding and generation logic, but it no longer needs to be passed through the Transformer at every step.

The KV cache is used by constrained generation, string generation, and number generation. It is particularly useful for longer generations, where repeatedly processing the complete sequence would otherwise become increasingly expensive.

## Design Decisions

- **Qwen/Qwen3-0.6B:** the required default model is used through the provided `Small_LLM_Model` interface.
- **Constrained decoding:** function names, JSON structure, numbers, and strings are constrained during generation rather than relying solely on the model's ability to follow instructions.
- **Pydantic validation:** input function definitions and test cases are validated with Pydantic models.
- **Separated responsibilities:** prompt construction, constrained decoding, JSON generation, file loading, and data models are implemented in separate modules.
- **Greedy selection:** among valid candidate tokens, the token with the highest logit is selected.
- **Token indexing:** the vocabulary is indexed once to reduce the number of tokens that need to be checked during constrained decoding.
- **KV caching:** the Transformer KV cache is reused between generation steps to avoid recomputing the previously processed sequence.

## Performance analysis

The project targets the requirements specified by the subject:

- valid, parseable JSON for every generated output;
- correct function selection and argument extraction;
- processing of the complete test set within a reasonable time;
- graceful handling of missing files, invalid JSON, invalid data, and user interruption.

Constrained decoding improves structural reliability by preventing invalid tokens from being selected. The implementation is designed around token-by-token generation, so the LLM is repeatedly queried as the output is built.

## Challenges faced

The main technical challenge was implementing constrained decoding with the model's token vocabulary. A token does not necessarily correspond to a complete word or value, so validation has to work with partial generated text.

Numbers require special handling because generation may temporarily contain an incomplete value such as `-`. Strings require additional validation for JSON escaping and control characters.

The other major performance issue came from the autoregressive nature of LLM generation. Initially, each new token required the complete sequence generated so far to be processed again by the model. This became particularly expensive for longer strings and constrained generations.

This was optimized by implementing the model's KV cache. The first forward pass processes the complete input and creates the cache. Subsequent forward passes receive only the newly generated token and the existing cache. This avoids recomputing the attention keys and values for all previously processed tokens.

This was optimized by implementing a token indexayion and the model's KV cache. These optimizations were implemented progressively so that each change could be benchmarked independently and the correctness of the constrained decoder could be preserved.

## Testing strategy

The implementation was tested using the provided JSON input format and different prompts, including:

- arithmetic operations with different numbers;
- string arguments;
- multiple parameters;
- generated outputs written to a new output file;
- invalid or missing input files;
- JSON and Pydantic validation errors.
- With and without KV Cache

The generated output can be checked with a JSON parser and against the function definitions to verify its structure and parameter types.

## Example usage

Input:

```text
What is the sum of 2 and 3?
```

A corresponding function definition could be:

```json
{
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
        "a": {"type": "number"},
        "b": {"type": "number"}
    },
    "returns": {"type": "number"}
}
```

The generated function call follows the required structure:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2,
        "b": 3
    }
}
```

## Resources

- 42 `Call Me Maybe` subject — project specification and constrained-decoding requirements.
- Qwen3 documentation and model information.
- Python `json` documentation.
- Python `argparse` documentation.
- Pydantic documentation.
- The provided `llm_sdk` package and its `Small_LLM_Model` interface.

### AI usage

AI tools were used as a development aid for understanding technical concepts, reviewing implementation choices, identifying potential issues, and improving documentation such as docstrings and this README.
