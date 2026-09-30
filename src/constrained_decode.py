from llm_sdk import Small_LLM_Model
import json

def constrained_decode_fn(
        logits: list[float],
        list_fn_name: list[str],
        llm_model: Small_LLM_Model,
        generate_ids: list[int]
        ) -> int:
    list_ids: list[int] = []
    vocab_path: str = llm_model.get_path_to_vocab_file()
    with open(vocab_path, "r") as vocab_file:
        vocab: dict[str, int] = json.load(vocab_file)
    eos_token_id: int | None = None
    for token, token_id in vocab.items():
        if token == "<|endoftext|>":
            eos_token_id = token_id
            break
    if eos_token_id is None:
        raise ValueError("EOS token was not found in vocabulary.")

    if not logits:
        raise ValueError("There is no Logits list.")

    if not list_fn_name:
        raise ValueError(
                "There is nothing inside"
                " the list of functions.")

    for function_name in list_fn_name:
        # token_id represent the position inside logits
        token_ids: list[int] = llm_model.encode(
            function_name).tolist()[0]
        # all the tokens of all the names(extend)
        if generate_ids == token_ids[:len(generate_ids)]:
            if len(generate_ids) < len(token_ids):
                next_token: int = token_ids[len(generate_ids)]
                list_ids.append(next_token)
    if not list_ids:
        raise ValueError("There is no list of IDs.")

    # Is the token id in list_id?,
    # if NOT -inf (impossible to choose),
    # if YES we keep its logit
    for token_id in range (len(logits)):
        if token_id not in list_ids:
            logits[token_id] = float("-inf")
    # Choose the permitted token with the max logit
    index_max_logit: int = logits.index(max(logits))
    return index_max_logit
