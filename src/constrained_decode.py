from llm_sdk import Small_LLM_Model


def constrained_decode_fn(
        logits: list[float],
        list_fn_name: list[str],
        llm_model = Small_LLM_Model()
        ) -> int:
    list_ids: list[int] = []


    for function_name in list_fn_name:
        # token_id represent the position inside logits
        token_ids = llm_model.encode(function_name).tolist()[0]
        # all the tokens of all the names(extend)
        list_ids.extend(token_ids)

    for token_id in range (len(logits)):
        if token_id not in list_ids:
            logits[token_id] = float("-inf")
    index_max_logit: int = logits.index(max(logits))
    return index_max_logit
