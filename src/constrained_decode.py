from llm_sdk import Small_LLM_Model
import json
from .basemodels import FunctionDefinitionCheck


def get_allowed_tokens(
        text: str,
        llm_model: Small_LLM_Model,
        ) -> list [int]:
    """Return the token IDs corresponding to the given text."""
    return llm_model.encode(text).tolist()[0]


def constrained_decode_fn(
        logits: list[float],
        list_fn_name: list[str],
        function_definitions: list[FunctionDefinitionCheck],
        llm_model: Small_LLM_Model,
        generate_ids: list[int],
        selected_function: str | None = None
        ) -> int:

    list_ids: list[int] = []
    vocab_path: str = llm_model.get_path_to_vocab_file()

    with open(vocab_path, "r") as vocab_file:
        vocab: dict[str, int] = json.load(vocab_file)

    eos_token_id: int = 128247
    if not logits:
        raise ValueError("There is no Logits list.")

# generate_ids == token_ids → function finished → allow EOS.
# generate_ids is a prefix → allow the next function token.
# Otherwise → don't allow anything from that function.
    if selected_function is None:
        if not list_fn_name:
            raise ValueError(
                    "There is nothing inside"
                    " the list of functions.")

        for function_name in list_fn_name:
            token_ids: list[int] = llm_model.encode(
                function_name
            ).tolist()[0]

            if generate_ids == token_ids:
                list_ids.append(eos_token_id)
                print("FUNCTION:", function_name)
                print("TOKEN_IDS:", token_ids)
                print("GENERATED:", generate_ids)

            elif generate_ids == token_ids[:len(generate_ids)]:
                if len(generate_ids) < len(token_ids):
                    next_token: int = token_ids[len(generate_ids)]
                    list_ids.append(next_token)

        if not list_ids:
            raise ValueError("There is no list of IDs.")
    else:
        selected_definition = None

        for function in function_definitions:
            if function.name == selected_function:
                selected_definition = function
                break

        if selected_definition is None:
            raise ValueError("Selected function was not found.")
        print("Parameters:", selected_definition.parameters)

        if not generate_ids:
            list_ids = get_allowed_tokens("{", llm_model)
    # Is the token id in list_id?,
    # if NOT -inf (impossible to choose),
    # if YES we keep its logit
    for token_id in range (len(logits)):
        if token_id not in list_ids:
            logits[token_id] = float("-inf")
    # Choose the permitted token with the max logit
    index_max_logit: int = logits.index(max(logits))
    return index_max_logit
