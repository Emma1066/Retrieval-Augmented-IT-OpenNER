import fire
import gradio as gr
from transformers import AutoTokenizer
from vllm import LLM
from functools import partial

from inference import inference
from utils import get_conversations

def get_input_examples(language):
    if language == "zh":
        examples = [
            ['文本：虎皮鸡爪配茄皇方便面，真的很好吃！\n实体：["虎皮鸡爪": "食物", "茄皇方便面": "食物"]', '奶酪博士家新出的奶酪小方很好吃。', '食物'],
            ['文本：小狗历是一款台历，它将两只小狗，小白和小鸡毛的故事画进了台历的页面中。\n实体：["小狗": "动物", "小白": "动物", "小鸡毛": "动物"]', 'B站up主，天下一场梦家里养了两只狗，分别名叫果冻和布丁', '动物']
        ]
    elif language == "en":
        examples = [
            ['Text: Did you know that there is a tunnel under Ocean Boulevard?\nEntity: ["Ocean Boulevard": "Location"]', 'Text: I rent a place on Cornelia Street.', 'Location'],
            ['Text: She wrote many love songs, such as Love Story and Lover. \nEntity: ["Love Story": "Love song", "Lover": "Love song"]', 'Can you write a rock song that sounds as good as Purple Rain?', 'Rock song']
        ]
    else:
        raise ValueError(f"Unsupported language: {language}")
    return examples

def main(
    model_path: str,
    tensor_parallel_size: int = 1,
    max_input_length: int = 2048,
    language: str = None,
):    
    if language not in ["zh", "en"]:
        raise ValueError(f"Unsupported language: {language}")

    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def evaluate(
        examples,
        text,
        entity_type,
        max_new_tokens=256,
        language: str = None,
    ):
        if len(tokenizer(text + entity_type)['input_ids']) > max_input_length:
            print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {max_input_length} tokens.")

        conversations = get_conversations(examples, text, entity_type, language)
        samples = [{"conversations": conversations}]

        output = inference(llm, samples, max_new_tokens=max_new_tokens, language=language)[0]
        yield output

    _evaluate = partial(evaluate, language=language)

    input_examples = get_input_examples(language)

    gr.Interface(
        fn=_evaluate,
        inputs=[
            gr.components.Textbox(lines=2, label="Examples", placeholder="Enter NER examples. (Leave it empty if there is no example)"),
            gr.components.Textbox(lines=2, label="Input", placeholder="Enter an input text."),
            gr.components.Textbox(lines=2, label="Entity type", placeholder="Enter an entity type."),
            gr.components.Slider(
                minimum=1, maximum=256, step=1, value=64, label="Max output tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        examples = input_examples,
        title="RA-IT-NER",
    ).queue().launch(server_name="0.0.0.0")

if __name__ == "__main__":
    fire.Fire(main)