from llm_sdk import Small_LLM_Model
import json
from .basemodels import FunctionDefinitionCheck

_vocab_cache: dict[str, dict[str, int]] = {}

def _load_vocab(vocab_path: str) -> dict[str, int]:
    if vocab_path not in _vocab_cache:
        with open(vocab_path, "r") as f:
            _vocab_cache[vocab_path] = json.load(f)
    return _vocab_cache[vocab_path]


def get_allowed_tokens(
        text: str,
        llm_model: Small_LLM_Model,
        ) -> list[int]:
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
        decoded_so_far: str,
        ) -> list[int]:
    """Return token IDs allowed based on explicit generation state."""

    if not param_names_and_types:
        return get_allowed_tokens("{}", llm_model)

    if arg_state == "start":
        return get_allowed_tokens("{", llm_model)

    if arg_state == "key":
        if param_index < len(param_names_and_types):
            param_name = param_names_and_types[param_index][0]
            full_key = f'"{param_name}":'
            encoded_key = llm_model.encode(full_key).tolist()[0]
            partial = llm_model.encode(
                        decoded_so_far
                    ).tolist()[0]
            if partial == encoded_key[:len(partial)]:
                if len(partial) == len(encoded_key):
                    return get_allowed_tokens(":", llm_model)
                next_token = encoded_key[len(partial)]
                return [next_token]
            return get_allowed_tokens(":", llm_model)
        return []

    if arg_state == "colon":
        return get_allowed_tokens(":", llm_model)

    if arg_state == "value":
        if param_index >= len(param_names_and_types):
            return []
        param_type = param_names_and_types[param_index][1]
        if param_type == "number":
            digit_tokens = [
                token_id
                for token, token_id in vocab.items()
                if token and (token[0].isdigit()
                            or token[0] == "-")
            ]
            if param_index == len(param_names_and_types) - 1:
                extra = get_allowed_tokens("}", llm_model)
            else:
                extra = get_allowed_tokens(",", llm_model)
            return digit_tokens + extra
        if param_type == "string":
            return get_allowed_tokens('"', llm_model)
        return []

    if arg_state == "string_content":
        return [
            token_id
            for token, token_id in vocab.items()
            if token and "\n" not in token and (
                '"' not in token or token.startswith('"')
            )
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
        decoded_so_far: str,
        llm_model: Small_LLM_Model
        ) -> tuple[int, str]:
    """Determine next (param_index, arg_state) based on current
    state, generate token and full decoded text."""
    num_params = len(param_names_and_types)

    if current_state == "start":
        return (0, "key")

    if current_state == "key":
        if param_index >= num_params:
            return(param_index, "end")
        param_name = param_names_and_types[param_index][0]
        full_key = f'"{param_name}":'
        if decoded_so_far.endswith(full_key):
            return (param_index, "value")
        return (param_index, "key")

    if current_state == "colon":
        return (param_index, "value")

    if current_state == "value":
        if param_index >= num_params:
            return (param_index, "end")
        param_type = param_names_and_types[param_index][1]
        if param_type == "string":
            if token_text == '"':
                return (param_index, "string_content")
            return (param_index, "value")
        if param_type == "number":
            if token_text == ",":
                return (param_index + 1, "key")
            if token_text == "}":
                return (param_index, "end")
            if token_text and (token_text[-1].isdigit()
                                or token_text[-1] == "."):
                return (param_index, "value")
            return (param_index, "value")
        return (param_index, "value")

    if current_state == "string_content":
        if token_text.startswith('"'):
            return(param_index, "string_end")
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
            return (param_index + 1, "key")

    if current_state == "comma":
        return (param_index, "key")

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
    vocab: dict[str, int] = _load_vocab(
            llm_model.get_path_to_vocad_file()
    )

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

            elif generate_ids == token_ids[:len(generate_ids)]:
                if len(generate_ids) < len(token_ids):
                    next_token: int = token_ids[len(generate_ids)]

                    list_ids.append(next_token)

        if not list_ids:
            raise ValueError("There is no list of IDs.")
        for token_id in range(len(logits)):
            if token_id not in list_ids:
                logits[token_id] = float("-inf")
        index_max_logit: int = logits.index(max(logits))
        return index_max_logit, 0, "start"
    else:
        selected_definition = None
        for function in function_definitions:
            if function.name == selected_function:
                selected_definition = function
                break

        if selected_definition is None:
            raise ValueError("Selected function was not found.")

        param_names_and_types = get_param_names_types(
            selected_definition)

        decoded_so_far: str = llm_model.decode(generate_ids)

        list_ids = get_allowed_next_tokens(
                    param_index,
                    arg_state,
                    param_names_and_types,
                    llm_model,
                    vocab,
                    decoded_so_far
                )

        if not list_ids:
            raise ValueError(
                "No allowed tokens for"
                f"param_index={param_index}, state={arg_state}"
            )
        for token_id in range(len(logits)):
            if token_id not in list_ids:
                logits[token_id] = float("-inf")
        selected_token: int = logits.index(max(logits))

        token_text = llm_model.decode([selected_token])
        next_param_index, next_arg_state = get_next_state(
            arg_state, param_index, param_names_and_types,
            token_text, decoded_so_far, llm_model
        )
        return selected_token, next_param_index, next_arg_state
