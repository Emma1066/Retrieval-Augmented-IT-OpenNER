from conversation import get_conv_template

def preprocess_instance(source, language):
    if language == "en":
        conv = get_conv_template("ner_as_qa_en")
    elif language == "zh":
        conv = get_conv_template("ner_as_qa_zh")
    else:
        raise ValueError(f"Unsupported language: {language}")
    for j, sentence in enumerate(source):
        value = sentence['value']
        if j == len(source) - 1: # why?
            value = None
        conv.append_message(conv.roles[j % 2], value)
    prompt = conv.get_prompt()
    return prompt

def get_response(responses):
    responses = [r.split('ASSISTANT:')[-1].strip() for r in responses]
    return responses

def get_conversations(examples, text, entity_type, language):
    if language == "zh":
        conversations = [{"from": "human", "value": f"文本：{text}"}, {"from": "gpt", "value": "我已读完这段文本。"}, {"from": "human", "value": f"文本中属于\"{entity_type}\"的实体有哪些？"}, {"from": "gpt", "value": "[]"}]
        if examples:
            example_turns = [{"from": "human", "value": f"以下是一些命名实体识别的例子：{examples}"}, {"from": "gpt", "value": "我已读完这些例子。"}]
            conversations = example_turns + conversations
    elif language == "en":
        conversations = [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]
        if examples:
            example_turns = [{"from": "human", "value": f"Here are some examples of named entity recognition: {examples}"}, {"from": "gpt", "value": "I've read these examples."}]
            conversations = example_turns + conversations
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    return conversations