from llm_sdk import Small_LLM_Model
from .file_loader import load_fn_definitions, load_prompts
from .basemodels import FunctionDefinitionCheck, PromptItem
from .constrained_decode import constrained_decode_fn


def main() -> None:
    function_def_path: str = "data/input/functions_definition.json"
    function_call_path: str = "data/input/function_calling_tests.json"

    loaded_function: list[FunctionDefinitionCheck] = load_fn_definitions(function_def_path)
    test_prompts: list[PromptItem] = load_prompts(function_call_path)
    llm_model: Small_LLM_Model = Small_LLM_Model()
    list_fn_name: list[str] = []

    for function in loaded_function:
            list_fn_name.append(function.name)

    # We go for every element in test_prompts
    # every round test is a PromptItem diferent
    all_tokens_decoded: list[str] = []
    for test in test_prompts:
        prompts: str = test.prompt
        generate_ids: list[int] = []
        selected_function: str | None = None
        # we convert the tokens in numeric ID 
        input_ids: list[int] = llm_model.encode(prompts).tolist()[0]

        while True:
            logits: list[float] = llm_model.get_logits_from_input_ids(input_ids)
            index_max_logit: int = constrained_decode_fn(
                                    logits,
                                    list_fn_name,
                                    loaded_function,
                                    llm_model,
                                    generate_ids,
                                    selected_function
                                )

            if index_max_logit == 128247:
                selected_function = llm_model.decode(generate_ids)
                print("Function selected:", selected_function)
                break
            input_ids.append(index_max_logit)
            generate_ids.append(index_max_logit)

        generate_ids = []
        while True:
            logits: list[float] = llm_model.get_logits_from_input_ids(input_ids)
            index_max_logit: int = constrained_decode_fn(
                                    logits,
                                    list_fn_name,
                                    loaded_function,
                                    llm_model,
                                    generate_ids,
                                    selected_function
                                )

            input_ids.append(index_max_logit)
            generate_ids.append(index_max_logit)

            decoded = llm_model.decode(generate_ids)
            print(decoded)

            if decoded.endswith("}"):
                break


if __name__ == "__main__":
    main()
