from llm_sdk import Small_LLM_Model
import json
from .basemodels import FunctionDefinitionCheck


def get_allowed_tokens(
        text: str,
        llm_model: Small_LLM_Model,
        ) -> list [int]:
    """Return the token IDs corresponding to the given text."""
    return llm_model.encode(text).tolist()[0]


def get_param_names_types(
        function_def: FunctionDefinitionCheck,
        ) -> list[tuple[str, str]]:
    """Extract parameter name and types in order from
    function definition."""
    return [
        (name, param.type.value)
        for name, param in function_def.parameters.items()
    ]

def get_allowed_next_tokens(
        param_index: int,
        arg_state: str,
        param_names_and_types: list[tuple[str, str]],
        llm_model: Small_LLM_Model,
        vocab: dict[str, int],
        ) -> list[int]:
    """Return token IDs allowed based on explicit generation state."""

    if not param_names_and_types:
        return get_allowed_tokens("{}", llm_model)

    if arg_state == "start":
        return get_allowed_tokens("{", llm_model)

    if arg_state == "key":
        if param_index < len(param_names_and_types):
            param_name = param_names_and_types[param_index][0]
            return get_allowed_tokens(f'"{param_name}',
                                llm_model)
        return []

    if arg_state == "colon":
        return get_allowed_tokens(":", llm_model)
    if arg_state == "value":
        if param_index >= len(param_names_and_types):
            return []
        param_type = param_names_and_types[param_index][1]
        if param_type == "number":
            return [
                token_id
                for token, token_id in vocab.items()
                if token and (token[0].isdigit() or token[0] ==
                "-")
            ]
        if param_type == "string":
            return get_allowed_tokens('"', llm_model)
        return []

    if arg_state == "string_content":
        return [
            token_id
            for token, token_id in vocab.items()
            if token and '"' not in token and "\n" not in token
        ]
    if arg_state == "string_end":
        if param_index == len(param_names_and_types) - 1:
            return get_allowed_tokens("}", llm_model)
        else:
            return get_allowed_tokens(",", llm_model)
    if arg_state == "number_end":
        if param_index == len(param_names_and_types) - 1:
            return get_allowed_tokens("}", llm_model)
        else:
            return get_allowed_tokens(",", llm_model)

    if arg_state == "comma":
        return get_allowed_tokens(",", llm_model)
    if arg_state == "end":
        return get_allowed_tokens("}", llm_model)
    return []


def get_next_state(
        current_state: str,
        param_index: int,
        param_names_and_types: list[tuple[str, str]],
        token_text: str,
        ) -> tuple[int, str]:
    """Determine next (param_index, arg_state) based on current
    state and generate token."""
    num_params = len(param_names_and_types)
    if current_state == "start":
        return(0, "key")
    if current_state == "key":
        return(param_index, "colon")
    if current_state == "colon":
        return (param_index, "value")

    if current_state == "value":
        if param_index >= num_params:
            return(param_index, "end")
        param_type = param_names_and_types[param_index][1]
        if param_type == "string":
            if token_text == '"':
                return(param_index, "string_content")
            return (param_index, "value")
        if param_type == "number":
            if token_text and (token_text[-1].isdigit() or
                    token_text[-1] == "."):
                return(param_index, "number_end")
            return (param_index, "value")
        return (param_index, "value")

    if current_state == "string_content":
        if token_text == '"':
            return (param_index, "string_end")
        return (param_index, "string_content")
    if current_state == "string_end":
        if param_index == num_params - 1:
            return (param_index, "end")
        else:
            return (param_index + 1, "key")

    if current_state == "number_end":
        if param_index == num_params - 1:
            return (param_index, "end")
        else:
            return(param_index + 1, "key")

    if current_state == "comma":
        return(param_index, "key")

    if current_state == "end":
        return (param_index, "end")

    return (param_index, "end")


def constrained_decode_fn(
        logits: list[float],
        list_fn_name: list[str],
        function_definitions: list[FunctionDefinitionCheck],
        llm_model: Small_LLM_Model,
        generate_ids: list[int],
        selected_function: str | None = None,
        param_index: int = 0,
        arg_state: str = "start"
        ) -> tuple[int, int, str]:

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
        return index_max_logit, 0, "start"
    else:
        selected_definition = None
        for function in function_definitions:
            if function.name == selected_function:
                selected_definition = function
                break

        if selected_definition is None:
            raise ValueError("Selected function was not found.")
        # print("Parameters:", selected_definition.parameters)

        param_names_and_types = get_param_names_types(
            selected_definition)

        list_ids = get_allowed_next_tokens(
            param_index,
            arg_state,
            param_names_and_types,
            llm_model,
            vocab
        )


        if not list_ids:
            raise ValueError(
                "No allowed tokens for"
                f"param_index={param_index}, state={arg_state}"
            )
    # Is the token id in list_id?,
    # if NOT -inf (impossible to choose),
    # if YES we keep its logit
    for token_id in range (len(logits)):
        if token_id not in list_ids:
            logits[token_id] = float("-inf")
    # Choose the permitted token with the max logit
    index_max_logit: int = logits.index(max(logits))

    token_text = llm_model.decode([index_max_logit])
    next_param_index, next_arg_state = get_next_state(
        arg_state, param_index, param_names_and_types, token_text
    )
    return index_max_logit, next_param_index, next_arg_state
