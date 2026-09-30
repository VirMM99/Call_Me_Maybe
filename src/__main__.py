from llm_sdk import Small_LLM_Model
from .file_loader import load_fn_definitions, load_prompts
from .basemodels import FunctionDefinitionCheck, PromptItem
from .constrained_decode import constrained_decode_fn


def main() -> None:
    function_def_path: str = "data/input/functions_definition.json"
    function_call_path: str = "data/input/function_calling_tests.json"

    loaded_function: list[FunctionDefinitionCheck] = load_fn_definitions(function_def_path)
    test_prompts: list[PromptItem] = load_prompts(function_call_path)
    llm_model = Small_LLM_Model()
    list_fn_name: list[str] = []

    for function in loaded_function:
            list_fn_name.append(function.name)
    # We go for every element in test_prompts
    # every round test is a PromptItem diferent
    all_tokens_decoded: list[str] = []
    for test in test_prompts:
        prompts = test.prompt
        # we convert the tokens in numeric ID 
        input_ids = llm_model.encode(prompts).tolist()[0]
        logits = llm_model.get_logits_from_input_ids(input_ids)
        index_max_logit = constrained_decode_fn(logits, list_fn_name, llm_model)
        token_decoded = llm_model.decode(index_max_logit)
        while token_decoded != EOS:
            all_tokens_decoded.append(token_decoded)
            input_ids = input_ids + [index_max_logit]
            logits = llm_model.get_logits_from_input_ids(input_ids)
            index_max_logit = constrained_decode_fn(logits, list_fn_name, llm_model)
            token_decoded = llm_model.decode(index_max_logit)
    print (all_tokens_decoded)

    for test_prom in loaded_function:
        final_prompt = loaded_function + test_prom.prompt
        input_ids = llm_model.encode(final_prompt)


if __name__ == "__main__":
    main()
