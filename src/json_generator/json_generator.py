from llm_sdk import Small_LLM_Model
from models import FunctionDefinition
from constrained_decoder import (
    generate_constrained,
    get_function_names,
    get_function_definition
)


def generate_function_call(
	llm: Small_LLM_Model,
	functions: FunctionDefinition,
	prompt: str
) -> str:
	token_batch = llm.encode(prompt)
	token_ids = token_batch.tolist()[0]
	logits = llm.get_logits_from_input_ids(token_ids)
	current_text: str = prompt

	target_texts: list[str] = ['{"prompt": "' + prompt + '", ']
	current_text, token_ids = generate_constrained(
		llm,
		current_text,
		token_ids,
		target_texts
	)

	function_names = get_function_names(functions)
	target_texts = [
		'{"name": "' + function_name
		for function_name in function_names
	]
	current_text, token_ids = generate_constrained(
		llm,
		current_text,
		token_ids,
		target_texts
	)
	function: FunctionDefinition = get_function_definition(functions, current_text)
	target_texts = ['", ']
	current_text, token_ids = generate_constrained(
		llm,
		current_text,
		token_ids,
		target_texts
	)

	param_count: int = len(function.parameters)
	target_texts = ['"parameters": {']
	current_text, token_ids = generate_constrained(
		llm,
		current_text,
		token_ids,
		target_texts
	)
	i: int = 0
	for key, value in function.parameters.items():
		i += 1
		target_texts = ['"' + key + '": ']
		current_text, token_ids = generate_constrained(
			llm,
			current_text,
			token_ids,
			target_texts
		)
		if value.type == "number":
			...
		if value.type == "string":
			...
		if i < param_count:
			target_texts = [", "]
			current_text, token_ids = generate_constrained(
				llm,
				current_text,
				token_ids,
				target_texts
			)

	target_texts = ["}}"]
	current_text, token_ids = generate_constrained(
		llm,
		current_text,
		token_ids,
		target_texts
	)

	return current_text[len(prompt):]







def generate_json() -> list[dict]:
	results = []

	for test_case in test_cases:
		result = generate_function_call(...)
		results.append(result)

	return results