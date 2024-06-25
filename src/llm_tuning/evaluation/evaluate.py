'''
1. Run inference, generate predictions.
2. Compute f1 metric.
'''
import os
import json
import math
import re
import string
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Type
from conversation import *

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def entity_parser(text:str, format:str):
    if format == "conversation_sharegpt":
        # return: [mention1, mention2, ...]
        try:
            search = re.search(r'\[(.*?)\]', text)
            if search:
                text = search.group()
            else:
                text = '[]'
            items = json.loads(text)
            formatted_items = []
            for item in items:
                if isinstance(item, list) or isinstance(item, tuple):
                    item = tuple([normalize_answer(element) for element in item])
                else:
                    item = normalize_answer(item)
                if item not in formatted_items: # not include repeated answer
                    formatted_items.append(item)
            return formatted_items
        except Exception:
            return None
    elif format == "single_alpaca":
        # return: [(mention1, type1), (mention2, type2), ...]
        try:
            search = re.findall(r'\((.*?),(.*?)\)', text)
            formatted_items = [] # list of tuples
            for i in range(len(search)):
                assert len(search[i]) == 2
                curr_mention = normalize_answer(search[i][0])
                curr_type = normalize_answer(search[i][1])
                if (curr_mention, curr_type) not in formatted_items: # not include repeated answer
                    formatted_items.append((curr_mention, curr_type))
            return formatted_items
        except Exception:
            return None
    else:
        raise ValueError(f"Unrecognized format: {format}")

def format_number(input:float, decimal:int=2, percent:bool=True):
    # turn to percent
    if percent:
        output = input * 100

    # round off decimals
    output = round(output, decimal)

    return output


class NEREvaluator:
    def evaluate(self, preds: list, golds: list, format: str):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
        n_unparsed_gold, n_unparsed_pred = 0, 0
        for idx, (pred, gold) in enumerate(zip(preds, golds)):
            gold_tuples = entity_parser(gold, format)
            pred_tuples = entity_parser(pred, format)
            if idx < 3:
                print(f"gold: {gold_tuples}")
                print(f"pred: {pred_tuples}")
                print("")
            if gold_tuples is None:
                n_unparsed_gold += 1
                gold_tuples = []
            if pred_tuples is None:
                n_unparsed_pred += 1
                pred_tuples = []

            for t in pred_tuples:
                if t in gold_tuples:
                    n_correct += 1
                n_pos_pred += 1
            n_pos_gold += len(gold_tuples)
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            'precision': format_number(prec),
            'recall': format_number(recall),
            'f1': format_number(f1),
            'n_gold': n_pos_gold,
            'n_prediction': n_pos_pred,
            'n_correct': n_correct,
            "n_unparsed_gold": n_unparsed_gold,
            "n_unparsed_pred": n_unparsed_pred
        }


def write_json_file(target_file, datas):
    g = open(target_file, "w", encoding='utf-8')
    json.dump(datas, g, ensure_ascii=False, indent=2)
    g.close()

def write_jsonl_file(target_file, datas):
    g = open(target_file, "w", encoding='utf-8')
    for item in datas:
        g.write(json.dumps(item, ensure_ascii=False))
        g.write("\n")
    g.close()

def preprocess_instance(source, conversation_template):
    conv = get_conv_template(conversation_template)
    for j, sentence in enumerate(source):
        value = sentence['value']
        if j == len(source) - 1: # answer to be responded by LLM
            value = None
        conv.append_message(conv.roles[j % 2], value)
    prompt = conv.get_prompt()
    return prompt

# def get_response(responses):
#     responses = [r.split('ASSISTANT:')[-1].strip() for r in responses]
#     return responses

def get_prompts(data:List[dict], format:str, conversation_template):
    if format == "conversation_sharegpt":
        prompts = [preprocess_instance(example['conversations'], conversation_template) for example in data]
    elif format == "single_alpaca":
        prompts = [example['instruction'] for example in data]
    else:
        raise ValueError(f"Unrecognized format: {format}")
    return prompts

def get_labels(data:List[dict], format:str):
    if format == "conversation_sharegpt":
        labels = [example['conversations'][-1]['value'] for example in data]
    elif format == "single_alpaca":
        labels = [example['output'] for example in data]
    else:
        raise ValueError(f"Unrecognized format: {format}")
    return labels

def inference(
    args,
    model: Type[LLM],
    sampling_params: SamplingParams,
    examples: List[dict],
    max_new_tokens: int = 256,
):
    conversation_template = args.conversation_template
    prompts = get_prompts(examples, args.data_format, conversation_template)

    # lora
    if args.enable_lora:
        responses = model.generate(
            prompts, 
            sampling_params,
            lora_request=LoRARequest("adapter", 1, args.adapter_path)
            )
    # no lora
    else:
        responses = model.generate(
            prompts, 
            sampling_params
            )
    responses_corret_order = []
    response_set = {response.prompt: response for response in responses}
    for i, prompt in enumerate(prompts):
        assert prompt in response_set
        responses_corret_order.append(
            {"id":examples[i]["id"], "predict": response_set[prompt].outputs[0].text}
        )
    # outputs = get_response([output.outputs[0].text for output in responses])
    return responses_corret_order

def main(args):  
    data_path = f"{args.data_path}"
    model_path = f"{args.model_path}"
    output_dir = f"{args.output_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(data_path, 'r') as fh:
        examples = json.load(fh)
    if args.max_samples:
        examples = examples[:args.max_samples]

    # --- Inference ---
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=args.tensor_parallel_size, 
        enable_lora=args.enable_lora,
        gpu_memory_utilization=args.gpu_memory_utilization
        )
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p,
        max_tokens=args.max_tokens, 
        # stop=[args.stop]
        )

    golds = get_labels(examples, args.data_format)
    responses = inference(args, llm, sampling_params, examples)
    # add label
    assert len(responses) == len(golds)
    for i in range(len(responses)):
        responses[i]["label"] = golds[i]

    pred_path = os.path.join(output_dir, "generated_predictions.jsonl")
    write_jsonl_file(pred_path, responses)
    print(f"Generated predictions saved to: {pred_path}")

    # --- Compute f1 ---
    outputs = [x["predict"] for x in responses]
    eval_result = NEREvaluator().evaluate(outputs, golds, args.data_format)
    eval_path = os.path.join(output_dir, "ner_metircs.json")
    write_json_file(eval_path, eval_result)
    print(f'\nPrecision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--conversation_template", type=str, required=True)

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_format", type=str, required=True, choices=['conversation_sharegpt','single_alpaca'])
    parser.add_argument("--max_samples", type=int, default=None, help="debug")

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    args = parser.parse_args()

    args.enable_lora = False if args.adapter_path is None else True
    
    print('----------Evaluating--------------')
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    main(args)
    print('----------Finished!--------------')
