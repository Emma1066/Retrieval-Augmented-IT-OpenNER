import fire
from vllm import LLM
from transformers import AutoTokenizer

from inference import inference
from utils import get_conversations

def main(
    model_path: str,
    max_new_tokens: int = 256,
    tensor_parallel_size: int = 1,
    max_input_length: int = 2048,
    language: str = None,
):    

    if language not in ["zh", "en"]:
        raise ValueError(f"Unsupported language: {language}")

    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    while True:
        try:
            examples = input('NER examples (If there is no example, just press ENTER): ')
            text = input('Input text: ')
            entity_type = input('Input entity type: ')
        except EOFError:
            text = entity_type = ''
        if not text:
            print("Exit...")
            break
        if len(tokenizer(text + entity_type)['input_ids']) > max_input_length:
            print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {max_input_length} tokens.")
            continue
        
        conversations = get_conversations(examples, text, entity_type, language)

        samples = [{"conversations": conversations}]

        output = inference(llm, samples, max_new_tokens=max_new_tokens, language=language)[0]
        print(output)


if __name__ == "__main__":
    fire.Fire(main)