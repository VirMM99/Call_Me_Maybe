from llm_sdk import Small_LLM_Model

txt = "Hola"
model = Small_LLM_Model()

mesg_tokens = model.encode(txt)[0].tolist()
logits: list[float] = model.get_logits_from_input_ids(mesg_tokens)
token_id = logits.index(max(logits))
print(model.decode([token_id]))


# print(model.get_path_to_vocab_file())

