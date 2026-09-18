from ..llm_sdk import Small_LLM_Model
from .file_loader import load_fn_definitions, load_prompts
from .basemodels import FunctionDefinitionCheck, PromptItem


def main() -> None:
    function_def_path: str = "../data/functions_definition.json"
    function_call_path: str = "../data/function_calling_tests.json"

    loaded_function: list[FunctionDefinitionCheck] = load_fn_definitions(function_def_path)
    test_prompts: list[PromptItem] = load_prompts(function_call_path)
    llm_model = Small_LLM_Model()

    for test in test_prompts:
        prompts = test.prompt
        input_ids = llm_model.encode(prompts)
        logits = llm_model.get_logits_from_input_ids(input_ids)
        max_logits = llm_model.argmax(logits)
# Sí, pero esa última línea no la daría por correcta todavía: argmax es la operación que necesitamos conceptualmente, pero tenemos que usarla con la API concreta que os proporciona llm_sdk; no quiero hacerte cambiarla a ciegas. Como quieres que al terminar el for lo expliquemos todo, paramos aquí el for por hoy. La próxima vez retomamos comprobando esa llamada y después te explico línea por línea todo lo que has hecho, desde load_fn_definitions hasta los logits, con calma. ¡Buen trabajo hoy!
if __name__ == "__main__":
    main()
