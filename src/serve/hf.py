import fire
import torch
from transformers import pipeline

from utils import preprocess_instance, get_conversations

def main(
    model_path: str,
    max_new_tokens: int = 256,
    language: str = None,
):    

    if language not in ["zh", "en"]:
        raise ValueError(f"Unsupported language: {language}")

    generator = pipeline('text-generation', model=model_path, torch_dtype=torch.float16, device=0)
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
        
        conversations = get_conversations(examples, text, entity_type, language)

        sample = {"conversations": conversations}
        prompt = preprocess_instance(sample['conversations'], language)

        outputs = generator(prompt, max_length=max_new_tokens, return_full_text=False)
        print(outputs[0]['generated_text'])

if __name__ == "__main__":
    fire.Fire(main)