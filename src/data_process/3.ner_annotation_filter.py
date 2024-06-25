from typing import Dict, List
import os
from collections import Counter
import random
import copy
import json
from rich import progress
from tqdm import tqdm
import re
import sys

from file_utils import load_data, save_data

MAX_WORDS_NUM = 20
UNVALID_ENTITY_TYPES = [
    "实体", "实体类别", "其他", "无", "无类别", "未定义", "entity_type", "entity type"
    "entity", "else", "none", "na", "misc",
]
REMOVE_PUNCS_PAIRS = ["《》", "【】", "[]"]

def json2tuplelist(json_input:list[dict]) -> list[tuple]:
    output = [(list(x.keys())[0], list(x.values())[0]) for x in json_input]
    return output

def tuplelist2json(indata:list[tuple]) -> list[dict]:
    output = [{x[0]: x[1]} for x in indata]
    return output

def get_words_num(input_str:str, lang:str="zh") -> int:
    if lang == "zh":
        words_num = len(input_str)
    elif lang == "en":
        words_num = len(input_str.split(" "))
    
    return words_num

def remove_some_punctuations(input_str:str) -> str:
    for p_pair in REMOVE_PUNCS_PAIRS:
        if input_str.startswith(p_pair[0]) and input_str.endswith(p_pair[1]):
            input_str = input_str[1:-1] # remove punctuations
    
    return input_str
    

def filter_entity(
        input_path:str, 
        output_dir:str, 
        max_samples:int=None,
        lang:str="zh"
        ) -> None:

    print("------- Pre-filter entity before selection --------")
    print(input_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parsed_data_path = os.path.join(output_dir, "llm_response_parsed_filtered.jsonl")
    ent_statistic_path = os.path.join(output_dir, "entity_statistics.json")
    print(f"parsed_data_path: {parsed_data_path}")
    print(f"ent_statistic_path: {ent_statistic_path}")

    input_data = load_data(input_path)
    print(f"data_len = {len(input_data)}")
    
    if max_samples:
        input_data = input_data[:max_samples]
        print(f"Take head {max_samples} samples.")

    output_data = []

    all_selected_entities = []
    all_selected_types = []
    all_selected_pairs = []

    for i_item, item in progress.track(enumerate(input_data), total=len(input_data), description="Filter before selection."):
        prediction = item["prediction"]
        text = item["text"]

        # JSON to tuple list
        tuple_list = json2tuplelist(prediction)

        # [X] remove endless repeated preds
        try:
            unique_pairs = list(set(tuple_list))
        except Exception as e:
            print(e)
            print(tuple_list)
            # sys.exit()
            unique_pairs = []


        # [X] remove lengthy entity
        unique_pairs = [x for x in unique_pairs if get_words_num(x[0], lang=lang) <= MAX_WORDS_NUM]

        # [X] remove entity not in the passage
        unique_pairs = [x for x in unique_pairs if x[0].lower() in text.lower()]

        # [X] remove unvalid type
        unique_pairs = [x for x in unique_pairs if x[1].lower() not in UNVALID_ENTITY_TYPES]

        # [X] remove some punctuations: 《》, [], 【】
        unique_pairs = [(remove_some_punctuations(x[0]), x[1]) for x in unique_pairs]
        
        new_prediction = tuplelist2json(unique_pairs)
        if lang == "zh":
            curr_id = f"ner_zh_{i_item}"
        elif lang == "en":
            curr_id = f"ner_en_{i_item}"
        new_item = {"id":curr_id}
        for k in item.keys():
            new_item[k] = item[k]
        new_item["prediction"] = new_prediction
        output_data.append(new_item)

        # For counting final entity numbers
        all_selected_entities.extend([x[0].lower() for x in unique_pairs])
        all_selected_types.extend([x[1].lower() for x in unique_pairs])
        all_selected_pairs.extend([(x[0].lower(), x[1].lower()) for x in unique_pairs])
    
    save_data(parsed_data_path, output_data)
    print(f"Output data saved to {parsed_data_path}")

    # statistics of entity and types after selection
    total_entity_num = len(all_selected_entities)
    total_type_num = len(list(set(all_selected_types)))
    total_ent_type_pair_num = len(list(set(all_selected_pairs)))
    avg_pair_per_sample = total_ent_type_pair_num / len(output_data)
    ent_num_statistics = {
        "total_entity_num": total_entity_num,
        "total_type_num": total_type_num,
        "total_ent_type_pair_num": total_ent_type_pair_num,
        "avg_pair_per_sample": avg_pair_per_sample,
    }
    print("ent_num_statistics:\n" + json.dumps(ent_num_statistics, indent=2))

    save_data(ent_statistic_path, ent_num_statistics)
    print(f"Entity statistics saved to {ent_statistic_path}")

    # construct type to mention dictionary
    type2mentions: Dict[str, List[str]] = dict([(x, list()) for x in all_selected_types])
    for (ment, typ) in tqdm(all_selected_pairs, desc="Construct type2mentions"):
        type2mentions[typ].append(ment)
    # sort by type frequency
    type2mentions = dict(sorted(list(type2mentions.items()), key=lambda x: len(x[1]), reverse=True))
    save_data(os.path.join(output_dir, "type2mentions.json"), type2mentions, indent=None)

    type2freq = {x:len(y) for (x,y) in type2mentions.items()}
    save_data(os.path.join(output_dir, "type2freq.json"), type2freq, indent=2)



if __name__ == "__main__":
    # ------------------------------ zh ----------------------------------
    # filter before selection
    dataname = "sky_samples"
    lang = "zh"
    max_samples=50000
    model="gpt-3.5-turbo-0125"
    filter_entity(
        input_path="outputs/llm_api_calling/llm_annotation/gpt-3.5-turbo-0125/prompt_v0_json/sky_samples/llm_response_parsed.jsonl",
        output_dir=f"outputs/llm_api_calling/llm_annotation/{model}/prompt_v0_json/{dataname}",
        lang=lang,
        max_samples=max_samples
    )
