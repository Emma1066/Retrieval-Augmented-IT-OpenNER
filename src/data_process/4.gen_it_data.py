# # DEBUG
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9801))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"Unable to lauch debugger!")
#     print(e)

from typing import List
from tqdm import tqdm
import random
import ujson
import os

from file_utils import load_data, save_data, convert_format
from prompt_utils import ROLE_TAG, CONTENT_TAG, USER_TAG, ASSISTANT_TAG
from prompt_utils import EXAMPLE_INTRO_PROMPT, EXAMPLE_RECEIVE_RESPONSE, EXAMPLE_TEMPLATE

DATANAME_MAP = {
    'boson': "Boson",
    "cluener_sample_42_dev_split_half_into_test": "CLUE NER",
    "cmeee_sample_42_train_10k_dev_split_half_into_test": "CMeEE",
    "RenMin_sample_42_train_10k_dev_2k_test_2k": "RenMinRiBao 2014",
    "yidu_sample_42_train_split_dev_into_200": "Yidu",
    "msra_sample_42_train_10k_train_split_dev_into_2k_test_2k": "MSRA",
    "ontonotes4zh_sample_42_train_10k_dev_2k_test_2k": "Ontonotes 4",
    "weibo": "Weibo",
    "mit-movie": "MIT-Movie",
    "mit-restaurant": "MIT-Restaurant",
    "CrossNER_AI": "CrossNER AI",
    "CrossNER_literature": "CrossNER literature",
    "CrossNER_music": "CrossNER music",
    "CrossNER_politics": "CrossNER politics",
    "CrossNER_science": "CrossNER science"
}

def json2str(inputs:List[dict]):
    # print(inputs)
    outputs = [f"{{\"{list(x.keys())[0]}\": \"{list(x.values())[0]}\"}}" for x in inputs]
    outputs = ", ".join(outputs)
    outputs = f"[{outputs}]"

    return outputs

def gen_ner_singleturn_data(input_path:str, output_path:str, lang:str):
    """
    Generate All-in-one stype data, in the alpaca format.
    """
    ner_data_config = load_data("LLM-annotation/configs/prompt_config/ner_sft.json")[lang]["prompt_v0_allinone"]

    in_data = load_data(input_path)

    ner_data = []
    for i_item, item in tqdm(enumerate(in_data), ncols=100):
        text = item["text"]
        prediction = item["prediction"]

        instruction = ner_data_config["ner prompt"] % text
        output = json2str(prediction)

        if lang == "zh":
            curr_id = f"ner_zh_{i_item}"
        elif lang == "en":
            curr_id = f"ner_en_{i_item}"
        ner_data.append({
            "id": curr_id,
            "instruction": instruction,
            "output": output,
        })

    save_data(output_path, ner_data)
    print(f"ner data saved to {output_path}")

def get_negative_types(positive_types, type2freq, sample_prob):
    '''
    Sample negative types from the entire type set except the positive types.

    positive_types: entity types that exists in the passage.
    type2freq: all type to frequency.
    '''
    type2freq_filtered = dict() # remove positive types
    for curr_type in type2freq:
        if curr_type in positive_types:
            continue
        type2freq_filtered[curr_type] = type2freq[curr_type]

    # how many negatives?
    num_negative_types = 0
    for _ in positive_types:
        tmp_sample_flag = random.random() # [0,1)
        if tmp_sample_flag < sample_prob:
            num_negative_types += 1
    # debug small dataset
    if num_negative_types > len(type2freq_filtered):
        num_negative_types = len(type2freq_filtered)
        
    # what are negatives?
    type_pool = list(type2freq_filtered.keys())
    type_frequencies = list(type2freq_filtered.values())
    negative_types = random.sample(
        population=type_pool,
        k=num_negative_types,
        counts=type_frequencies
    ) # sample unique negatives with a probability proportional to the frequency

    return negative_types

def do_negative_sampling(entities, type2freq, sample_prob):
    '''
    Conduct negative sampling for entity types.

    entities: entity dict, {entity type: [entity mention 1, entity mention 2]}
    return:
        new entity dict with negative types added.
    '''
    positive_types = list(entities.keys())
    negative_types = get_negative_types(positive_types, type2freq, sample_prob)
    pos_neg_types = positive_types + negative_types
    random.shuffle(pos_neg_types)

    entities_w_ns = dict()
    for t in pos_neg_types:
        if t in entities:
            entities_w_ns[t] = entities[t]
        else:
            entities_w_ns[t] = []
    
    return entities_w_ns, len(negative_types)

def gen_ner_conversation_data(
        input_path:str, 
        output_path:str, 
        prompt_config_path:str,
        prompt_version:str,
        lang:str,
        negative_sampling:bool,
        sample_prob:float=0.5,
        entity_tag:str="prediction"
    ):
    """
    Generate Conversation-style data, in the sharegpt format.
    """
    print("\n ------------------ Generate conversation data -------------------")
    print(f"target path: {output_path}")
    indata = load_data(input_path)
    ner_data_config = load_data(prompt_config_path)[lang][prompt_version]
    # ----- Collect type2freq && Parse sentence, answers -------
    type2freq = dict()
    for _, item in enumerate(tqdm(indata, desc="Collect type2freq")):
        prediction = item[entity_tag] # json format

        type2mentions = dict()
        for pair in prediction:
            m, t = list(pair.items())[0]
            if t not in type2mentions:
                type2mentions[t] = []
            type2mentions[t].append(m)

        item["type2mentions"] = type2mentions

        for t, ms in type2mentions.items():
            if t not in type2freq:
                type2freq[t] = 0
            type2freq[t] += len(ms)

    # ----- Convert data format && Negative sampling ---------------
    data_converted = []
    total_num_negatives = 0
    total_num_queries = 0
    for i_item, item in enumerate(tqdm(indata, desc="alpaca single --> sharegpt conversation")):

        output_type2mentions = item["type2mentions"]
        text = item["text"]

        # Negative Sampling
        if negative_sampling:
            output_type2mentions, curr_num_negatives = do_negative_sampling(output_type2mentions, type2freq, sample_prob)
            total_num_negatives += curr_num_negatives
        total_num_queries += len(output_type2mentions)

        # Format convert: alpaca --> sharegpt
        # -------- Sharegpt: prompt v0 ------------
        role_tag = ROLE_TAG # "from"
        content_tag = CONTENT_TAG #"value"
        user_tag = USER_TAG # "human"
        assistant_tag = ASSISTANT_TAG # "gpt"

        # Different for zh/en/...
        if lang=="zh":
            slot_passage = "文本： %s"
            respond_to_text = "我已读完这段文本。"
        else:
            slot_passage = "Text: %s"
            respond_to_text = "I’ve read this text"

        slot_query_type = ner_data_config["ner prompt"]
        # -------------------------------

        # Construct conversations
        conversations = list()
        # first round: introduce text
        conversations.append({
            role_tag: user_tag,
            content_tag: slot_passage % text
        })
        conversations.append({
            role_tag: assistant_tag,
            content_tag: respond_to_text
        })
        # following rounds: querying pos && neg types
        for curr_type, curr_mentions in output_type2mentions.items():
            curr_query = slot_query_type % curr_type
            
            # Answer e.g., "[\"myPosition\", \"enemyPosition\"]"
            curr_answer = ", ".join([f"\"{x}\"" for x in curr_mentions])
            curr_answer = f"[{curr_answer}]"

            conversations.append({
                role_tag: user_tag,
                content_tag: curr_query
            })
            conversations.append({
                role_tag: assistant_tag,
                content_tag: curr_answer
            })


        data_converted.append({
            "id": item["id"],
            "conversations": conversations
        })

    outdir = os.path.dirname(output_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    save_data(output_path, data_converted)
    print(f"Conversation NER data saved to {output_path}")
    print(f"total_num_negatives = {total_num_negatives}")
    print(f"total_num_queries = {total_num_queries}")

def benchmark_test_set_to_conversation(
    input_path:str, 
    output_path:str, 
    label_path:str,
    lang:str,
    data_specific:bool=False,
    dataname:str=None,
    use_original_id:bool=False,
    ner_data_config_path:str=None,
):
    """Conversations, sharegpt"""
    print(f"{dataname}: test")
    indata = load_data(input_path)

    label_set = load_data(label_path)
    label_set = list(label_set.values())
    print(f"label set:\n{label_set}")

    ner_data_config = load_data(ner_data_config_path)[lang]["prompt_v0_conversation"]
    # ----- Collect type2freq && Parse sentence, answers -------
    type2freq = dict()
    for _, item in enumerate(tqdm(indata, desc="Collect type2freq")):
        label = item["label"]

        type2mentions = dict()
        for m, t in label.items():
            if t not in type2mentions:
                type2mentions[t] = []
            type2mentions[t].append(m)

        item["type2mentions"] = type2mentions

        for t, ms in type2mentions.items():
            if t not in type2freq:
                type2freq[t] = 0
            type2freq[t] += len(ms)

    # ----- Convert data format && Negative sampling ---------------
    data_converted = []
    for _, item in enumerate(tqdm(indata, desc="alpaca single --> sharegpt conversation")):
        output_type2mentions = item["type2mentions"]
        text = item["sentence"]

        # Format convert: alpaca --> sharegpt
        # -------- Sharegpt: prompt v0 ------------
        role_tag = "from"
        content_tag = "value"
        user_tag = "human"
        assistant_tag = "gpt"

        # Different for zh/en/...
        if lang=="zh":
            if data_specific:
                slot_passage = "数据集：%s\n 文本： %s"
            else:
                slot_passage = "文本： %s"
            respond_to_text = "我已读完这段文本。"
        else:
            if data_specific:
                slot_passage = "Dataset: %s\n Text: %s"
            else:
                slot_passage = "Text: %s"
            respond_to_text = "I’ve read this text"

        slot_query_type = ner_data_config["ner prompt"]
        # -------------------------------

        # Each type corresponds to a complete conversation.
        for curr_type in label_set:
            if curr_type in output_type2mentions:
                curr_mentions = output_type2mentions[curr_type]
            else:
                curr_mentions = []
    
            # Each type constructs a query conversation
            conversations = list()
            # first round: introduce text
            conversations.append({
                role_tag: user_tag,
                content_tag: slot_passage % (dataname,text) if data_specific else slot_passage % text
            })
            conversations.append({
                role_tag: assistant_tag,
                content_tag: respond_to_text
            })

            # get entity query and entity answer
            curr_query = slot_query_type % curr_type
            # Answer e.g., "[\"myPosition\", \"enemyPosition\"]"
            curr_answer = ", ".join([f"\"{x}\"" for x in curr_mentions])
            curr_answer = f"[{curr_answer}]"

            conversations.append({
                role_tag: user_tag,
                content_tag: curr_query
            })
            conversations.append({
                role_tag: assistant_tag,
                content_tag: curr_answer
            })

            curr_id = item["id"] if use_original_id else f"{dataname}_{len(data_converted)}"
            data_converted.append({
                "id": curr_id,
                "conversations": conversations
            })

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_data(output_path, data_converted)
    print(f"Conversation NER data saved to {output_path}")
    print(f"original_data_len = {len(indata)}")
    print(f"total_num_queries = {len(data_converted)}")


if __name__ == "__main__":
    # --------- Generate openNER instruction tuning training data ----------
    # Generate Sky-NER
    model = "gpt-3.5-turbo-0125"
    gen_ner_conversation_data(
        input_path=f"outputs/llm_api_calling/llm_annotation/{model}/prompt_v0_json/sky_samples/llm_response_parsed_filtered.jsonl",
        output_path=f"outputs/llm_api_calling/llm_annotation/{model}/prompt_v0_json/sky_samples/ner_it_data_conversation.json",
        prompt_config_path=f"configs/prompt_config/ner_sft.json",
        prompt_version="prompt_v0_conversation",
        lang="zh",
        negative_sampling=True,
        sample_prob=0.5,
    )

    # # -------- Generate openNER eval data ------------
    # # generate evaluation data - no example
    # # boson as an example
    # ner_prompt_config_path=f"configs/prompt_config/ner_sft.json"
    # benchmark_test_set_to_conversation(
    #     input_path=f"data/benchmark_data/boson/test.jsonl",
    #     label_path=f"data/benchmark_data/boson/abb2labelname.json",
    #     output_path=f"data/benchmark_it_data/regular/boson/test.json",
    #     lang="zh",
    #     use_original_id=True,
    #     ner_data_config_path=ner_prompt_config_path
    # )



