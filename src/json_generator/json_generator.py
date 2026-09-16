from llm_sdk import Small_LLM_Model
from models import FunctionDefinition
from constrained_decoder import (
    generate_constrained,
	generate_number,
	get_function_descriptions,
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
	context: str = prompt

	function_descriptions = get_function_descriptions(functions)
	target_texts = function_descriptions
	context, token_ids = generate_constrained(
		llm,
		context,
		token_ids,
		target_texts
	)

	function: FunctionDefinition = get_function_definition(functions, context, prompt)
	print(f"Founction choosen: {function.name}")

	target_texts: list[str] = ['{"prompt": "' + prompt + '", ']
	context, token_ids = generate_constrained(
		llm,
		context,
		token_ids,
		target_texts
	)

	target_texts = ['"name": "' + function.name + '", "parameters": {']
	context, token_ids = generate_constrained(
		llm,
		context,
		token_ids,
		target_texts
	)

	param_count: int = len(function.parameters)

	i: int = 0
	for key, value in function.parameters.items():
		i += 1
		target_texts = ['"' + key + '": ']
		context, token_ids = generate_constrained(
			llm,
			context,
			token_ids,
			target_texts
		)

		if value.type == "number":
			token_full_list = list(range(len(logits)))
			end_token_ids = [
				token_id
				for token_id in token_full_list
				if (llm.decode([token_id]) == " "
				or llm.decode([token_id]) == ",")
			]
			print(f"End token ids: {end_token_ids}")
			context, token_ids = generate_number(
				llm,
				context,
				token_ids,
				end_token_ids
			)

				
		if value.type == "string":
			...
		if i < param_count:
			target_texts = [", "]
			context, token_ids = generate_constrained(
				llm,
				context,
				token_ids,
				target_texts
			)

	target_texts = ["}}"]
	context, token_ids = generate_constrained(
		llm,
		context,
		token_ids,
		target_texts
	)

	return context







def generate_json() -> list[dict]:
	results = []

	for test_case in test_cases:
		result = generate_function_call(...)
		results.append(result)

	return results