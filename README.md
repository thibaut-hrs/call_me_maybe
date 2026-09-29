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

## Algorithm

The generation pipeline works in several stages:

1. The user prompt is passed to the LLM with the available function names and descriptions.
2. The LLM selects a function name using constrained decoding.
3. The selected function definition determines the expected parameters and their types.
4. The JSON structure is generated piece by piece using constrained decoding.
5. Number parameters are generated with a numeric constraint allowing only valid integer or floating-point representations.
6. String parameters are generated with JSON string validation, including escape sequences and control-character checks.
7. The completed JSON object is written to the output file.

For each constrained generation step, the model produces logits for all possible tokens. The implementation filters the vocabulary to tokens that keep the partial output valid, then selects the highest-scoring valid token. This prevents the decoder from generating tokens that would break the expected structure.

## Design Decisions

- **Qwen/Qwen3-0.6B:** the required default model is used through the provided `Small_LLM_Model` interface.
- **Constrained decoding:** function names, JSON structure, numbers, and strings are constrained during generation rather than relying solely on the model's ability to follow instructions.
- **Pydantic validation:** input function definitions and test cases are validated with Pydantic models.
- **Separated responsibilities:** prompt construction, constrained decoding, JSON generation, file loading, and data models are implemented in separate modules.
- **Greedy selection:** among valid candidate tokens, the token with the highest logit is selected.

## Performance and Reliability

The project targets the requirements specified by the subject:

- valid, parseable JSON for every generated output;
- correct function selection and argument extraction;
- processing of the complete test set within a reasonable time;
- graceful handling of missing files, invalid JSON, invalid data, and user interruption.

Constrained decoding improves structural reliability by preventing invalid tokens from being selected. The implementation is designed around token-by-token generation, so the LLM is repeatedly queried as the output is built.

## Challenges

The main technical challenge was implementing constrained decoding with the model's token vocabulary. A token does not necessarily correspond to a complete word or value, so validation has to work with partial generated text.

Numbers require special handling because generation may temporarily contain an incomplete value such as `-`. Strings require additional validation for JSON escaping and control characters.

## Testing

The implementation was tested using the provided JSON input format and different prompts, including:

- arithmetic operations with different numbers;
- string arguments;
- multiple parameters;
- generated outputs written to a new output file;
- invalid or missing input files;
- JSON and Pydantic validation errors.

The generated output can be checked with a JSON parser and against the function definitions to verify its structure and parameter types.

## Example

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
