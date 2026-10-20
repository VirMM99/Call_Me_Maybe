from llm_sdk import Small_LLM_Model
import json
from .basemodels import FunctionDefinitionCheck

_vocab_cache: dict[str, dict[str, int]] = {}
_numeric_cache: dict[str, list[int]] = {}
_string_cache: dict[str, list[int]] = {}
_structural_cache: dict[str, list[int]] = {}


def _get_cached_tokens(text: str, llm_model: Small_LLM_Model) -> list[int]:
    if text not in _structural_cache:
        _structural_cache[text] = llm_model.encode(text).tolist()[0]
    return _structural_cache[text]


def _load_vocab(vocab_path: str) -> dict[str, int]:
    if vocab_path not in _vocab_cache:
        with open(vocab_path, "r") as f:
            _vocab_cache[vocab_path] = json.load(f)
    return _vocab_cache[vocab_path]


def _numeric_token_ids(
        vocab_path: str, vocab: dict[str, int]) -> list[int]:
    if vocab_path not in _numeric_cache:
        _numeric_cache[vocab_path] = [
            token_id
            for token, token_id in vocab.items()
            if token and all(
                c.isdigit() or c in ".-" for c in token
            )
        ]
    return _numeric_cache[vocab_path]

#LLM find the tokens that have '""
def _string_token_ids(
        vocab_path: str, vocab: dict[str, int]) -> list[int]:
    # We forced to clean the cache
    if vocab_path in _string_cache:
        del _string_cache[vocab_path]

    if vocab_path not in _string_cache:
        valid_ids = []
        for token, token_id in vocab.items():
            if not token:
                continue
            # Ġ (space)
            if token.count('\u0120') <= 2:
                continue
    
            if '"' not in token:
                valid_ids.append(token_id)
            else:
                parts = token.split('"')
                if len(parts) == 2:
                    after_quote = parts[1]
                    allowed_structural = set(", \n\t}")
                    if all(c in allowed_structural for c in after_quote):
                        valid_ids.append(token_id)

            _string_cache[vocab_path] = valid_ids
        return _string_cache[vocab_path]


def get_allowed_tokens(
        text: str,
        llm_model: Small_LLM_Model,
        ) -> list[int]:
    """Return the token IDs corresponding to the given text (cached)."""
    return _get_cached_tokens(text, llm_model)


def get_param_names_types(
        function_def: FunctionDefinitionCheck,
        ) -> list[tuple[str, str]]:
    """Extract parameter name and types in order from
    function definition."""
    return [
        (name, param.type.value)
        for name, param in function_def.parameters.items()
    ]


def _matched_key_tokens(
        key_ids: list[int],
        generate_ids: list[int],
        ) -> int:
    """How many leading tokens of key_ids already appear at the end
    of generate_ids."""
    for k in range(min(len(key_ids), len(generate_ids)), 0, -1):
        if generate_ids[-k:] == key_ids[:k]:
            return k
    return 0


def get_allowed_next_tokens(
        param_index: int,
        arg_state: str,
        param_names_and_types: list[tuple[str, str]],
        llm_model: Small_LLM_Model,
        vocab: dict[str, int],
        vocab_path: str,
        generate_ids: list[int],
        ) -> list[int]:
    """Return token IDs allowed based on explicit generation state."""
    num_params = len(param_names_and_types)

    if arg_state == "start":
        return get_allowed_tokens("{", llm_model)

    if arg_state == "key":
        if param_index >= num_params:
            return []
        param_name = param_names_and_types[param_index][0]
        key_ids = llm_model.encode(f'"{param_name}":').tolist()[0]

        matched = _matched_key_tokens(key_ids, generate_ids)
        if matched >= len(key_ids):
            return get_allowed_tokens(":", llm_model)
        return[key_ids[matched]]

    if arg_state == "value":
        if param_index >= num_params:
            return []
        param_type = param_names_and_types[param_index][1]
        if param_type == "string":
            return get_allowed_tokens('"', llm_model)
        if param_type in ("number", "integer"):
            terminator = "}" if param_index == num_params - 1 else ","
            return (
                _numeric_token_ids(vocab_path, vocab)
                + get_allowed_tokens (terminator, llm_model)
            )
        return []

    if arg_state == "string_content":
        quote_ids = get_allowed_tokens('"', llm_model)
        content_ids = _string_token_ids(vocab_path, vocab)

        # limited numbers of tokens generated
        quote_token_ids = set(quote_ids)
        content_ids = [
            token_id for token_id in content_ids
            if token_id not in quote_token_ids
        ]

        # generate_text = llm_model.decode(generate_ids)
        # # if the chain is too large, we force the close
        # current_string = generate_text.rsplit('"', 1)[-1]
        # if len(current_string) >= 100:
        #     return quote_ids
        return content_ids + quote_ids

    if arg_state == "sep":
        if param_index == num_params - 1:
            return get_allowed_tokens("}", llm_model)
        return get_allowed_tokens(",", llm_model)

    if arg_state == "end":
        eos_token_id: int = 128247
        # We let the key of final clousure 0 the last token in sequence(EOS)
        return get_allowed_tokens("}", llm_model) + [eos_token_id]
    return []


def get_next_state(
        current_state: str,
        param_index: int,
        param_names_and_types: list[tuple[str, str]],
        token_text: str,
        generate_ids: list[int],
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
        key_ids = llm_model.encode(f'"{param_name}":').tolist()[0]
        matched = _matched_key_tokens(key_ids, generate_ids)

        if matched >= len(key_ids):
            return(param_index, "value")
        return (param_index, "key")

    if current_state == "value":
        if param_index >= num_params:
            return (param_index, "end")

        param_type = param_names_and_types[param_index][1]
        if param_type == "string":
            quote_ids = llm_model.encode('"').tolist()[0]
            if generate_ids[-len(quote_ids):] == quote_ids:
                return (param_index, "string_content")
            return (param_index, "value")
        if param_type in ("number", "integer"):
            if token_text == "}":
                return (param_index, "end")
            if token_text == ",":
                return (param_index + 1, "key")
            return (param_index, "value")
        return (param_index, "value")

    if current_state == "end":
        return (param_index, "end")

    if current_state == "sep":
        if param_index == num_params - 1:
            return (param_index, "end")
        return (param_index + 1, "key")

    if current_state == "string_content":
        if '"' in token_text:
            if "," in token_text:
                if param_index >= num_params - 1:
                    return (param_index, "end")
                return (param_index + 1, "key")
            # if token only have quotation mark
            # then we go to sep, it can put the comma in the next step
            return(param_index, "sep")
        return (param_index, "string_content")
    return (param_index, "end")


def _apply_logit_mask(logits: list[float], allowed_ids: list[int]) -> int:
    """Common Helper: applies -inf 
    to unallowed logits and returns the best token."""
    if not logits:
        raise ValueError("There is no logits list")

    allowed: set[int] = set(allowed_ids)
    for token_id in range(len(logits)):
        if token_id not in allowed:
            logits[token_id] = float("-inf")

    return logits.index(max(logits))


def constrained_decode_function_name(
        logits: list[float],
        list_fn_name: list[str],
        llm_model: Small_LLM_Model,
        generate_ids: list[int]
        ) -> tuple[int, int, str]:
    """Responsibility 1: Decode and restrict only the function name."""
    if not list_fn_name:
        raise ValueError("There is nothing inside the list of functions")

    eos_token_id: int = 128247
    list_ids: list[int] = []

    for function_name in list_fn_name:
        token_ids: list[int] = llm_model.encode(function_name).tolist()[0]

        if generate_ids == token_ids:
            list_ids.append(eos_token_id)
        elif generate_ids == token_ids:
            if len(generate_ids) < len(token_ids):
                list_ids.append(token_ids[len(generate_ids)])

    if not list_ids:
        raise ValueError("There is no list of IDs.")

    selected_token = _apply_logit_mask(logits, list_ids)
    return selected_token, 0, "start"


def constrained_decode_parameters(
        logits: list[float],
        function_definitions: list[FunctionDefinitionCheck],
        llm_model: Small_LLM_Model,
        generate_ids: list[int],
        selected_function: str,
        param_index: int = 0,
        arg_state: str = "start"
        ) -> tuple[int, int, str]:
    """Responsibility 2: Decode and restrict JSON 
    parameters using the state machine."""
    vocab_path: str = llm_model.get_path_to_vocab_file()
    vocab: dict[str, int] = _load_vocab(vocab_path)

    # Print de prueba
    print("\n//// GENERATED IDS:")
    for i in generate_ids:
        print(llm_model.decode([i]), end=" ")
    # FIN DE PRUEBA

    selected_definition = None
    for function in function_definitions:
        if function.name == selected_function:
            selected_definition = function
            break

    if selected_definition is None:
        raise ValueError("Selected function was not found.")

    param_names_and_types = get_param_names_types(selected_definition)

    list_ids = get_allowed_next_tokens(
        param_index,
        arg_state,
        param_names_and_types,
        llm_model,
        vocab,
        vocab_path,
        generate_ids
    )
    if not list_ids:
        raise ValueError(
            f"No allowed tokens for param_index={param_index}, "
            f"state={arg_state}"
        )

    # Prueba debugueo: 1. We associate each token_id with their logit,
    # sorting them from major to minor
    # and we take the first 5
    top_5 = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)[:5]
    # 2. We go around them to print them
    print("\nTop 5 tokens más probables:")
    for token_id, score in top_5:
        token_text = llm_model.decode([token_id])
        print(f"Token ID: {token_id} | Score: {score:.2f} | Texto: {token_text!r}")
    # End of debug

    selected_token = _apply_logit_mask(logits, list_ids)
    token_text = llm_model.decode([selected_token])

    # This is the code to show Status
    print(
        f"state={arg_state}, "
        f"token_id={selected_token}, "
        f"token_text={token_text!r}"
    )

    new_generate_ids = generate_ids + [selected_token]
    next_param_index, next_arg_state = get_next_state(
        arg_state, param_index, param_names_and_types,
        token_text, new_generate_ids, llm_model
    )
    return selected_token, next_param_index, next_arg_state


