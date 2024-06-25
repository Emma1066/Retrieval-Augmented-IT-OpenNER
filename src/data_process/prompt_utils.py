ROLE_TAG = "from"
CONTENT_TAG = "value"
USER_TAG = "human"
ASSISTANT_TAG = "gpt"

EXAMPLE_INTRO_PROMPT = {
    "en": "Here are some examples of named entity recognition:\n%s",
    "zh": "以下是一些命名实体识别的例子：\n%s"
}
EXAMPLE_RECEIVE_RESPONSE = {
    "en": "I've read these examples.",
    "zh": "我已读完这些例子。"
}
EXAMPLE_TEMPLATE = {
    "en": "Text: %s\nEntity: %s",
    "zh": "文本：%s\n实体：%s"
}