import os
from typing import List
from collections import Counter
import numpy as np
import random
import ujson
from rich import progress
import pandas as pd
from embeddings_utils import cosine_similarity
from multiprocessing import Process, Pool
from functools import partial
from copy import deepcopy
from tqdm import tqdm

from file_utils import load_data, save_data
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

def get_similar_examples_by_emb(
        query_emb, 
        query_sample,  
        src_emb, 
        src_data, 
        example_num:int=32
    )->None:
    src_emb_df = pd.DataFrame(columns=["embedding"])
    src_emb_df["embedding"] = list(src_emb)

    src_emb_df["similarity"] = src_emb_df.embedding.apply(lambda x: cosine_similarity(x, query_emb))

    cos_sims = src_emb_df["similarity"]
    sorted_idxes = np.argsort(cos_sims).tolist()
    sorted_idxes.reverse() # put most similar ones at head
    
    demos_selected = []
    cnt = 0
    while len(demos_selected) < example_num:
        curr_example = src_data[sorted_idxes[cnt]]
        if curr_example["id"]==query_sample["id"] and curr_example["text"]==query_sample["text"]: # do not include self
            continue
        demos_selected.append(src_data[sorted_idxes[cnt]])
        cnt += 1

    return demos_selected

def retrieve_similar_examples(
    src_data_path:str,
    src_emb_path:str,
    tar_data_path:str,
    tar_emb_path:str,
    save_similar_examples_path:str,
    example_num:int=32,
    save_ids_only:bool=True,
    max_samples:int=None,
    text_tag:str="text"
) -> None:
    src_data = load_data(src_data_path)
    src_emb = np.load(src_emb_path)
    tar_data = load_data(tar_data_path)
    tar_emb = np.load(tar_emb_path)
    print(f"\n------------ Retrieve similar examples ---------------")
    print(f"target_path: {save_similar_examples_path}")
    print(f"LEN(src_data) = {len(src_data)}")
    print(f"Shape(src_emb) = {src_emb.shape}")
    print(f"LEN(tar_data) = {len(tar_data)}")
    print(f"Shape(tar_emb) = {tar_emb.shape}")

    if max_samples is not None:
        tar_data = tar_data[:max_samples]

    out_data = []
    previous_query_text = ""
    previous_retrieved_examples = []
    for i_item, item in progress.track(enumerate(tar_data), total=len(tar_data)):
        query_emb = tar_emb[i_item]
        query_text = item[text_tag]
        
        if query_text == previous_query_text:
            retrieved_examples = deepcopy(previous_retrieved_examples)
        else:
            retrieved_examples = get_similar_examples_by_emb(query_emb, item, src_emb, src_data, example_num=example_num)
        
        new_item = deepcopy(item)

        if save_ids_only:
            retrieved_example_ids = [x["id"] for x in retrieved_examples]
            new_item[f"top_{example_num}_similar_examples"] = retrieved_example_ids
        else:
            new_item[f"top_{example_num}_similar_examples"] = retrieved_examples
    
        out_data.append(new_item)

        previous_query_text = query_text
        previous_retrieved_examples = retrieved_examples

    dirname = os.path.dirname(save_similar_examples_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_data(save_similar_examples_path, out_data)
    print(f"{len(out_data)} data saved to: {save_similar_examples_path}")

    # head_100_path = save_similar_examples_path.replace(".jsonl", "_head_100.jsonl")
    # save_data(head_100_path, tar_data[:100])
    # print(f"Head 100 data saved to: {head_100_path}")

def _get_text_from_conv(item:dict, lang:str) -> str:
    conv = item["conversations"]
    whole_input_text = conv[0]["value"]

    if lang=="zh":
        text_head="文本："
    elif lang=="en":
        text_head="Text: "
    else:
        raise ValueError(f"lang={lang}")

    mark_pos = whole_input_text.find(text_head)
    assert mark_pos >= 0

    passage_begin_pos = mark_pos + len(text_head)
    passage = whole_input_text[passage_begin_pos:]

    return passage

def retrieve_random_examples(
    src_data_path:str,
    tar_data_path:str,
    save_random_examples_path:str,
    lang:str,
    example_num:int=32,
    save_ids_only:bool=True,
    max_samples:int=None,
) -> None:
    src_data = load_data(src_data_path)
    tar_data = load_data(tar_data_path)
    print(f"\n------------ Retrieve random examples ---------------")
    print(f"LEN(src_data) = {len(src_data)}")
    print(f"LEN(tar_data) = {len(tar_data)}")

    if max_samples is not None:
        tar_data = tar_data[:max_samples]

    out_data = []
    previous_query_text = ""
    previous_retrieved_examples = []
    for i_item, item in progress.track(enumerate(tar_data), total=len(tar_data)):
        if "text" in item:
            query_text = item["text"]
        else:
            if "conversations" in item:
                query_text = _get_text_from_conv(item, lang=lang)
            else:
                raise ValueError(f"Cannot get text from item:\n{item}")
        
        if query_text == previous_query_text:
            retrieved_examples = deepcopy(previous_retrieved_examples)
        else:
            retrieved_examples = random.sample(src_data, example_num)
        
        new_item = {"id":item["id"], "text":query_text}

        if save_ids_only:
            retrieved_example_ids = [x["id"] for x in retrieved_examples]
            new_item[f"random_{example_num}_examples"] = retrieved_example_ids
        else:
            new_item[f"random_{example_num}_examples"] = retrieved_examples
    
        out_data.append(new_item)

        previous_query_text = query_text
        previous_retrieved_examples = retrieved_examples

    dirname = os.path.dirname(save_random_examples_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_data(save_random_examples_path, out_data)
    print(f"{len(out_data)} data saved to: {save_random_examples_path}")

def add_examp_to_multi_turn_conv_data(
        input_path:str, 
        example_path:str, 
        output_path:str, 
        example_num:int,
        example_tag:str,
        lang="zh",
        get_example_by_id:bool=True,
        example_id_path:str=None,
        max_samples:int=None
    ) -> None:
    r"""
    Add examples to the conversation training data.
        input_path: path of conversation data.
        example_path: path of similar examples.
        output_path: path of the example-augmented conversation data.
    """
    print(f"\n------------ Generate conversation data with similar examples ---------------")
    print(f"target_path: {output_path}")
    input_data = load_data(input_path)
    print(f"LEN(input_data) = {len(input_data)}")
    if max_samples is not None:
        input_data = input_data[:max_samples]

    # if get_example_by_id = True, this is the src example data
    # else. this is the target data with retrieved examples
    example_data = load_data(example_path)
    print(f"LEN(example_data) = {len(example_data)}")

    if get_example_by_id:
        print(f"Get similar example by ids.")
        example_ids = load_data(example_id_path)
        print(f"LEN(example_ids_data) = {len(example_ids)}")
        example_data = {x["id"]:x for x in example_data} # construct dict for indexing

    # prompt template of zh/en
    example_intro_prompt = EXAMPLE_INTRO_PROMPT[lang]
    example_receive_resp = EXAMPLE_RECEIVE_RESPONSE[lang]
    example_template = EXAMPLE_TEMPLATE[lang]

    role_tag = ROLE_TAG # "from"
    content_tag = CONTENT_TAG #"value"
    user_tag = USER_TAG # "human"
    assistant_tag = ASSISTANT_TAG # "gpt"

    previous_text = ""
    previous_example_turn = []
    for i_item, item in progress.track(enumerate(input_data), total=len(input_data)):
        curr_text = example_ids[i_item]["text"] # TODO: debug
        if i_item>0 and curr_text==previous_text:
            example_turn = previous_example_turn
        else:

            if not get_example_by_id:
                curr_examples = example_data[i_item][example_tag][:example_num]
            else:
                curr_example_ids = example_ids[i_item][example_tag][:example_num]
                curr_examples = [example_data[x] for x in curr_example_ids]

            example_str_ls = [example_template % (x["text"], x["entity"]) for x in curr_examples]
            example_str = "\n".join(example_str_ls)

            complete_example_content = example_intro_prompt % example_str

            example_turn = []
            example_turn.append({
                role_tag: user_tag,
                content_tag: complete_example_content
            })
            example_turn.append({
                role_tag: assistant_tag,
                content_tag: example_receive_resp
            })

        ori_conversation = item["conversations"]
        new_conversation = example_turn + ori_conversation

        item["conversations"] = new_conversation

        previous_text = curr_text
        previous_example_turn = example_turn
    
    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_data(output_path, input_data)
    print(f"Added examples to conversation data. Save to: {output_path}")


def get_adaptive_examples(
        ori_examples:List[dict], 
        example_num:int,
        strategy:str, 
        diverse_k:int=None, 
        bm25score_threshold:int=None,
        bm25scores:list=None
    ) -> None:
    '''
        ori_examples: Similar example Retrieved by NN.
        example_num: The number of the final used examples.

        bm25scores: BM25 score of each above example.
        bm25score_threshold: Discard the examples below this thresh.
        diverse_k: The number of examples in diverse KNN. The final used example is selected from this diverse_k examples.

        Intro of different strategies:
            - NN_bm25Rej: KNN with bm25 rejection. (1) Get k nearest examples. (2) Discard those having lower bm25 score than the bm25score_threshold.
            - diverseNN: Diverse nearest.
            - diverseNN_bm25Rej:  (1) Get k diverse NN. (2) Discard those having lower bm25 score than the bm25score_threshold.
            - diverseNN_bm25Rank: (1) Get K (large) NN. (2) Rank by bm25 scores. Get top-k. (3) Discard those having lower bm25 score than the bm25score_threshold.
    '''
    if "diverse" in strategy:
        if len(ori_examples) < diverse_k:
            raise ValueError(f"len(ori_examples)={len(ori_examples)} < diverse_k={diverse_k}")
        
    if strategy == "NN_bm25Rej":
        NN_examples = ori_examples[:example_num]
        NN_bm25scores = bm25scores[:example_num]
        keeped_NN = []
        for i in range(len(NN_examples)):
            if NN_bm25scores[i] >= bm25score_threshold:
                keeped_NN.append(NN_examples[i])
        return keeped_NN
    elif strategy == "diverseNN":
        large_NN = ori_examples[:diverse_k]
        diverse_NN = random.sample(large_NN, k=example_num)
        return diverse_NN
    elif strategy == "diverseNN_bm25Rej":
        large_NN = ori_examples[:diverse_k]
        large_NN_bm25scores = bm25scores[:diverse_k]

        sampled_indexes = random.sample(range(len(large_NN)), k=example_num)
        keeped_diverse_NN = []
        for i in sampled_indexes:
            if large_NN_bm25scores[i] >= bm25score_threshold:
                keeped_diverse_NN.append(large_NN[i])
        return keeped_diverse_NN
    elif strategy == "diverseNN_bm25Rank":
        large_NN = ori_examples[:diverse_k]
        large_NN_bm25scores = bm25scores[:diverse_k]

        sorted_ind_score_pair = sorted(list(enumerate(large_NN_bm25scores)), key=lambda x: x[1], reverse=True)
        sorted_ind_by_score = list(map(lambda x: x[0], sorted_ind_score_pair))

        keeped_diverse_ranked_NN = []
        for i in sorted_ind_by_score[:example_num]:
            if large_NN_bm25scores[i] >= bm25score_threshold:
                keeped_diverse_ranked_NN.append(large_NN[i])
        return keeped_diverse_ranked_NN
    else:
        raise ValueError(f"Unrecognized strategy: {strategy}")

def add_adaptive_examp_to_multi_turn_conv_data(
        in_data_path:str, 
        example_id_path:str,
        example_path:str, 
        out_data_path:str, 
        example_statistics_path:str,
        strategy:str,
        example_num:int,
        example_tag:str,
        lang="zh",
        max_samples:int=None,
        diverse_k:int=None,
        bm25score_threshold:int=20,
        bm25score_path:str=None
    ) -> None:
    '''
    The vanilla retrieval strategy.
        - sim_examp (NN): the vanilla nearest neighbors.
    Use more advanced retrieval strategy.
        - NN_bm25Rej
        - diverseNN
        - diverseNN_bm25Rej
        - diverseNN_bm25Rank
    '''
    print(f"\n------------ Generate conversation data with ADAPTIVE examples ---------------")
    print(f"in_data_path: {in_data_path}")
    print(f"out_data_path: {out_data_path}")
    input_data = load_data(in_data_path)
    print(f"LEN(input_data) = {len(input_data)}")
    
    print(f"Strategy: {strategy}")
    print(f"example_num: {example_num}")
    print(f"diverse_k: {diverse_k}")
    print(f"bm25score_threshold: {bm25score_threshold}")
    print(f"bm25score_path: {bm25score_path}")
    
    if max_samples is not None:
        input_data = input_data[:max_samples]
    
    # if get_example_by_id = True, this is the src example data
    # else. this is the target data with retrieved examples
    example_data = load_data(example_path)
    print(f"LEN(example_data) = {len(example_data)}")
    example_ids = load_data(example_id_path)
    print(f"LEN(example_ids_data) = {len(example_ids)}")

    id2examples = {x["id"]:x for x in example_data} # construct dict for indexing

    if "bm25" in strategy:
        bm25scores =  load_data(bm25score_path)
    else:
        bm25scores = None

    # prompt template of zh/en
    example_intro_prompt = EXAMPLE_INTRO_PROMPT[lang]
    example_receive_resp = EXAMPLE_RECEIVE_RESPONSE[lang]
    example_template = EXAMPLE_TEMPLATE[lang]

    role_tag = ROLE_TAG # "from"
    content_tag = CONTENT_TAG #"value"
    user_tag = USER_TAG # "human"
    assistant_tag = ASSISTANT_TAG # "gpt"
    
    out_data = []
    used_example_nums = []
    previous_text = ""
    previous_example_turn = []
    previous_ada_examples = []
    for i_item, item in tqdm(enumerate(input_data), total=len(input_data)):
        curr_text = example_ids[i_item]["text"]
        
        if i_item > 0 and curr_text == previous_text:
            example_turn = previous_example_turn
            curr_ada_examples = previous_ada_examples
        else:

            curr_example_ids = example_ids[i_item][example_tag]
            curr_examples = [id2examples[x] for x in curr_example_ids]

            if "bm25" in strategy:
                curr_examplescores = bm25scores[i_item][example_tag]
            else:
                curr_examplescores = None

            # Get adaptive examples with specific strategy
            curr_ada_examples = get_adaptive_examples(
                curr_examples, 
                example_num=example_num,
                strategy=strategy,
                diverse_k=diverse_k,
                bm25score_threshold=bm25score_threshold,
                bm25scores=curr_examplescores
            )

            example_turn = []
            if len(curr_ada_examples) > 0:
                example_str_ls = [example_template % (x["text"], x["entity"]) for x in curr_ada_examples]
                example_str = "\n".join(example_str_ls)

                complete_example_content = example_intro_prompt % example_str

                example_turn.append({
                    role_tag: user_tag,
                    content_tag: complete_example_content
                })
                example_turn.append({
                    role_tag: assistant_tag,
                    content_tag: example_receive_resp
                })

        used_example_nums.append(len(curr_ada_examples))

        ori_conversation = item["conversations"]
        new_conversation = example_turn + ori_conversation

        item_w_examples = deepcopy(item)
        item_w_examples["conversations"] = new_conversation
        out_data.append(item_w_examples)

        previous_example_turn = example_turn
        previous_text = curr_text
        previous_ada_examples = curr_ada_examples
    
    dirname = os.path.dirname(out_data_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_data(out_data_path, out_data)
    print(f"Added ADAPTIVE examples to conversation data. Save to: {out_data_path}")

    # Statistics of actually used examples
    avg_num = sum(used_example_nums) / len(used_example_nums)
    examp_num_counter = Counter(used_example_nums)
    examp_num_counter = dict(examp_num_counter)
    examp_num_counter = dict(sorted(examp_num_counter.items()))
    examp_num_counter["avg"] = avg_num
    save_data(example_statistics_path, examp_num_counter)
    print(f"Statistics of actual used examples: {example_statistics_path}")
    

def benchmark_test_set_to_rag_conversation(
    input_path:str, 
    label_info_path:str,
    retrieved_ids_path:str,
    retrieving_data_path:str,
    example_tag:str,
    example_num:int,
    output_path:str, 
    lang:str,
    data_specific:bool=False,
    dataname:str=None,
    use_original_id:bool=False,
    ner_data_config_path:str=None,
    text_tag_of_tar:str="sentence",
    text_tag_of_ref:str="text",
    entity_tag_of_tar:str="label",
    entity_tag_of_ref:str="entity",
):
    """
    Specific for test data 
        -> Each data to several 3-turn conversations.
        -> Each entity type to a single 3-turn conversation.
        -> 3 turns: (1) examples turn; (2) text turn; (3) entity query turn.
    
    Conversations, sharegpt
        Generate conversation-stype data with retreived examples. (RAFT/RAG)
    
    input_path: target data.
    retrieved_ids_path: retrieved example ids.
    retrieving_data_path: The original data used for retrieve.
    """
    print(f"\n\n{dataname}")
    indata = load_data(input_path)
    retrieved_ids = load_data(retrieved_ids_path)
    retrieving_data = load_data(retrieving_data_path)
    id2retrieving_data = {x["id"]:x for x in retrieving_data}
    # ner_data_config = load_data("LLM-annotation/configs/prompt_config/ner_sft.json")["prompt_v0_conversation"]
    ner_data_config = load_data(ner_data_config_path)[lang]["prompt_v0_conversation"]
    
    label_info = load_data(label_info_path)
    label_set = list(label_info.values())
    print(f"Labels: {label_set}")

    # ----- Collect type2freq && Parse sentence, answers -------
    type2freq = dict([(lb, 0) for lb in label_set])
    for _, item in enumerate(tqdm(indata, desc="Collect type2freq")):
        label = item[entity_tag_of_tar]

        type2mentions = dict([(lb, list()) for lb in label_set]) # include all types
        for m, t in label.items():
            if t == "misc":
                continue
            type2mentions[t].append(m)

        item["type2mentions"] = type2mentions

        for t, ms in type2mentions.items():
            type2freq[t] += len(ms)

    # ----- Convert data format && Negative sampling ---------------
    # prompt template of zh/en
    example_intro_prompt = EXAMPLE_INTRO_PROMPT[lang]
    example_receive_resp = EXAMPLE_RECEIVE_RESPONSE[lang]
    example_template = EXAMPLE_TEMPLATE[lang]

    # Format convert: alpaca --> sharegpt
    # -------- Sharegpt: prompt v0 ------------
    role_tag = "from"
    content_tag = "value"
    user_tag = "human"
    assistant_tag = "gpt"

    data_converted = []
    total_num_negatives = 0
    total_num_queries = 0
    for i_item, item in enumerate(tqdm(indata, desc="alpaca single --> sharegpt conversation")):
        output_type2mentions = item["type2mentions"]
        text = item[text_tag_of_tar]

        total_num_queries += len(output_type2mentions)

        # --- process examples ----
        curr_examp_ids = retrieved_ids[i_item][example_tag][:example_num]
        curr_examps = [id2retrieving_data[x] for x in curr_examp_ids]

        example_str_ls = [example_template % (x[text_tag_of_ref], convert_format(x[entity_tag_of_ref],target_format="json")) for x in curr_examps]
        example_str = "\n".join(example_str_ls)

        complete_example_content = example_intro_prompt % example_str

        if data_specific:
            if lang == "zh":
                complete_example_content = f"数据集：{dataname}\n{complete_example_content}"
            elif lang == "en":
                complete_example_content = f"Dataset: {dataname}\n{complete_example_content}"
            else:
                raise ValueError(f"lang={lang}")

        example_turn = []
        example_turn.append({
            role_tag: user_tag,
            content_tag: complete_example_content
        })
        example_turn.append({
            role_tag: assistant_tag,
            content_tag: example_receive_resp
        })

         # --- process text slots ----
        if lang=="zh":
            slot_passage = "文本： %s"
            respond_to_text = "我已读完这段文本。"
        else:
            slot_passage = "Text: %s"
            respond_to_text = "I’ve read this text"

        slot_query_type = ner_data_config["ner prompt"]

        # Each type corresponds to a complete conversation.
        for curr_type, curr_mentions in output_type2mentions.items():
            if len(curr_mentions)==0:
                total_num_negatives += 1

            curr_query = slot_query_type % curr_type
            
            # Answer e.g., "[\"myPosition\", \"enemyPosition\"]"
            curr_answer = ", ".join([f"\"{x}\"" for x in curr_mentions])
            curr_answer = f"[{curr_answer}]"

            # Construct conversations
            conversations = list()
            # 1-st round: give examples
            conversations.extend(example_turn)
            # 2-nd round: introduce text
            conversations.append({
                role_tag: user_tag,
                content_tag: slot_passage % text
            })
            conversations.append({  
                role_tag: assistant_tag,
                content_tag: respond_to_text
            })
            # 3-rd round: entity query
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
    print(f"total_num_negatives = {total_num_negatives}")
    print(f"total_num_queries = {total_num_queries}")

def benchmark_test_set_to_adaptive_rag_conversation(
    input_path:str, 
    label_info_path:str,
    retrieved_ids_path:str,
    retrieving_data_path:str,
    example_tag:str,
    example_num:int,
    output_path:str, 
    lang:str,
    data_specific:bool=False,
    dataname:str=None,
    use_original_id:bool=False,
    ner_data_config_path:str=None,
    text_tag_of_tar:str="sentence",
    text_tag_of_ref:str="text",
    entity_tag_of_tar:str="label",
    entity_tag_of_ref:str="entity",
    example_statistics_path:str=None,
    strategy:str=None,
    diverse_k:int=None,
    bm25score_threshold:int=20,
    bm25score_path:str=None
):
    """
    Convert benchmark test set to conversation data with adaptive RAG.
        - sim_examp (NN): the vanilla nearest neighbors.
    Use more advanced retrieval strategy.
        - NN_bm25Rej
        - diverseNN
        - diverseNN_bm25Rej
        - diverseNN_bm25Rank
    """
    print(f"\n\n{dataname}")
    indata = load_data(input_path)
    retrieved_ids = load_data(retrieved_ids_path)
    retrieving_data = load_data(retrieving_data_path)
    id2retrieving_data = {x["id"]:x for x in retrieving_data}
    # ner_data_config = load_data("LLM-annotation/configs/prompt_config/ner_sft.json")["prompt_v0_conversation"]
    ner_data_config = load_data(ner_data_config_path)[lang]["prompt_v0_conversation"]
    
    label_info = load_data(label_info_path)
    label_set = list(label_info.values())
    print(f"Labels: {label_set}")

    if "bm25" in strategy:
        bm25scores = load_data(bm25score_path)
    else:
        bm25scores = None

    # ----- Collect type2freq && Parse sentence, answers -------
    type2freq = dict([(lb, 0) for lb in label_set])
    for _, item in enumerate(tqdm(indata, desc="Collect type2freq")):
        label = item[entity_tag_of_tar]

        type2mentions = dict([(lb, list()) for lb in label_set]) # include all types
        for m, t in label.items():
            if t == "misc":
                continue
            type2mentions[t].append(m)

        item["type2mentions"] = type2mentions

        for t, ms in type2mentions.items():
            type2freq[t] += len(ms)

    # ----- Convert data format && Negative sampling ---------------
    # prompt template of zh/en
    example_intro_prompt = EXAMPLE_INTRO_PROMPT[lang]
    example_receive_resp = EXAMPLE_RECEIVE_RESPONSE[lang]
    example_template = EXAMPLE_TEMPLATE[lang]

    # Format convert: alpaca --> sharegpt
    # -------- Sharegpt: prompt v0 ------------
    role_tag = "from"
    content_tag = "value"
    user_tag = "human"
    assistant_tag = "gpt"

    data_converted = []
    used_example_nums = []
    total_num_negatives = 0
    total_num_queries = 0
    for i_item, item in enumerate(tqdm(indata, desc="alpaca single --> sharegpt conversation")):
        output_type2mentions = item["type2mentions"]
        text = item[text_tag_of_tar]

        total_num_queries += len(output_type2mentions)

        # --- process examples ----
        curr_examp_ids = retrieved_ids[i_item][example_tag]
        curr_examps = [id2retrieving_data[x] for x in curr_examp_ids]

        if "bm25" in strategy:
            curr_examplescores = bm25scores[i_item][example_tag]
        else:
            curr_examplescores = None

        # Get adaptive examples with specific strategy
        curr_ada_examples = get_adaptive_examples(
            curr_examps, 
            example_num=example_num,
            strategy=strategy,
            diverse_k=diverse_k,
            bm25score_threshold=bm25score_threshold,
            bm25scores=curr_examplescores
        )

        example_str_ls = [example_template % (x[text_tag_of_ref], convert_format(x[entity_tag_of_ref],target_format="json")) for x in curr_ada_examples]
        example_str = "\n".join(example_str_ls)

        complete_example_content = example_intro_prompt % example_str

        if data_specific:
            if lang == "zh":
                complete_example_content = f"数据集：{dataname}\n{complete_example_content}"
            elif lang == "en":
                complete_example_content = f"Dataset: {dataname}\n{complete_example_content}"
            else:
                raise ValueError(f"lang={lang}")

        example_turn = []
        if len(curr_ada_examples) > 0:
            example_turn.append({
                role_tag: user_tag,
                content_tag: complete_example_content
            })
            example_turn.append({
                role_tag: assistant_tag,
                content_tag: example_receive_resp
            })
        used_example_nums.append(len(curr_ada_examples))

         # --- process text slots ----
        if lang=="zh":
            slot_passage = "文本： %s"
            respond_to_text = "我已读完这段文本。"
        else:
            slot_passage = "Text: %s"
            respond_to_text = "I’ve read this text"

        slot_query_type = ner_data_config["ner prompt"]

        # Each type corresponds to a complete conversation.
        for curr_type, curr_mentions in output_type2mentions.items():
            if len(curr_mentions)==0:
                total_num_negatives += 1

            curr_query = slot_query_type % curr_type
            
            # Answer e.g., "[\"myPosition\", \"enemyPosition\"]"
            curr_answer = ", ".join([f"\"{x}\"" for x in curr_mentions])
            curr_answer = f"[{curr_answer}]"

            # Construct conversations
            conversations = list()
            # 1-st round: give examples
            conversations.extend(example_turn)
            # 2-nd round: introduce text
            conversations.append({
                role_tag: user_tag,
                content_tag: slot_passage % text
            })
            conversations.append({  
                role_tag: assistant_tag,
                content_tag: respond_to_text
            })
            # 3-rd round: entity query
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
    print(f"Conversation NER data with ada-example saved to {output_path}")
    print(f"total_num_negatives = {total_num_negatives}")
    print(f"total_num_queries = {total_num_queries}")

    # Statistics of actually used examples
    avg_num = sum(used_example_nums) / len(used_example_nums)
    examp_num_counter = Counter(used_example_nums)
    examp_num_counter = dict(examp_num_counter)
    examp_num_counter = dict(sorted(examp_num_counter.items()))
    examp_num_counter["avg"] = avg_num
    save_data(example_statistics_path, examp_num_counter)
    print(f"Statistics of actual used examples: {example_statistics_path}")



def retrieve_example_for_training_data():
    # For Chinese scenario
    # ------ For Sky-NER training data ------
    max_example_num = 128
    # retrieve similar examples
    retrieve_similar_examples(
        src_data_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_allinone.jsonl",
        src_emb_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_GTElargeEmb_text.npy",
        tar_data_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_allinone.jsonl",
        tar_emb_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_GTElargeEmb_text.npy",
        save_similar_examples_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_sim_examp_ids_{max_example_num}_GTElargeEmb_text.jsonl",
        example_num=max_example_num,
        save_ids_only=True,
    )

    # (1) add retrieved NN examples to vanilla IT conversations --> retrieval augmented IT conversations
    used_example_num=2
    add_examp_to_multi_turn_conv_data(
        input_path=f"data/it_data/sky_gpt3.5_5k_random_42/train.json",
        example_id_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_sim_examp_ids_{max_example_num}_GTElargeEmb_text.jsonl",
        example_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_allinone.jsonl",
        output_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_w_NN_{used_example_num}.json",
        example_num=used_example_num,
        example_tag=f"top_{max_example_num}_similar_examples",
        lang="zh",
        get_example_by_id=True,
    )

    # # (2) Use diverse retrieval strategies, including diverseNN, bm25 filtering, and so on
    # diverse_k=128
    # bm25score_threshold=20
    # for strategy in ["NN_bm25Rej", "diverseNN", "diverseNN_bm25Rej", "diverseNN_bm25Rank"]:
    #     if strategy=="NN_bm25Rej":
    #         strategy_mark=f"{strategy}_{used_example_num}_{bm25score_threshold}"
    #     if strategy=="diverseNN":
    #         strategy_mark=f"{strategy}_{used_example_num}_{diverse_k}"
    #     if strategy=="diverseNN_bm25Rej":
    #         strategy_mark=f"{strategy}_{used_example_num}_{diverse_k}_{bm25score_threshold}"
    #     if strategy=="diverseNN_bm25Rank":
    #         strategy_mark=f"{strategy}_{used_example_num}_{diverse_k}_{bm25score_threshold}"
    #     add_adaptive_examp_to_multi_turn_conv_data(
    #         input_path=f"data/it_data/sky_gpt3.5_5k_random_42/train.json",
    #         example_id_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_sim_examp_ids_{max_example_num}_GTElargeEmb_text.jsonl",
    #         example_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_allinone.jsonl",
    #         output_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_w_{strategy_mark}_GTElargeEmb_text.json",
    #         example_num=used_example_num,
    #         example_tag=f"top_{max_example_num}_similar_examples",
    #         lang="zh",
    #         get_example_by_id=True,
    #         example_statistics_path=f"data/it_data/sky_gpt3.5_5k_random_42/train_exampstat_{strategy_mark}_GTElargeEmb_text.json",
    #         strategy=strategy,
    #         diverse_k=diverse_k,
    #         bm25score_threshold=bm25score_threshold,
    #         bm25score_path=f"data/it_data/sky_gpt3.5_5k_random_42/bm25Score/train_simexamp_bmscores_{max_example_num}_GTElargeEmb_text.jsonl"
    #     )

def retrieve_outdomain_example_for_benchmarks():
    max_example_num=128
    ner_prompt_config_path = f"configs/prompt_config/ner_sft.json"
    
    setname="5k_random_42"

    lang="zh"
    outdomain_data_dir = f"data/it_data/sky_gpt3.5_{setname}"
    outdomain_dataname="skygpt3.5"

    # boson as an example
    dataname = "boson"

    bench_data_dir = f"data/benchmark_data/{dataname}"
    bench_data_rag_dir = f"data/benchmark_data_rag/{dataname}"
    bench_it_data_rag_dir = f"data/benchmark_it_data_rag/{dataname}"

    example_num=2
    
    # (1) ----- retrieve similar examples -------
    retrieve_similar_examples(
        src_data_path=f"{outdomain_data_dir}/train_allinone.jsonl",
        src_emb_path=f"{outdomain_data_dir}/train_GTElargeEmb_text.npy",
        tar_data_path=f"{bench_data_dir}/test.jsonl",
        tar_emb_path=f"{bench_data_rag_dir}/embeddings/test_GTElargeEmb_text.npy",
        save_similar_examples_path=f"{bench_data_rag_dir}/sim_examp_ids_from_outdomain/test_simexample_ids_{max_example_num}_{outdomain_dataname}_{setname}_GTElargeEmb_text.jsonl",
        example_num=max_example_num,
        save_ids_only=True,
        text_tag="sentence"
    )

    # Generate instruction data with retrieved examples
    benchmark_test_set_to_rag_conversation(
        input_path=f"{bench_data_dir}/test.jsonl", 
        label_info_path=f"{bench_data_dir}/abb2labelname.json",
        retrieved_ids_path=f"{bench_data_rag_dir}/sim_examp_ids_from_outdomain/test_simexample_ids_{max_example_num}_{outdomain_dataname}_{setname}_GTElargeEmb_text.jsonl",
        retrieving_data_path=f"{outdomain_data_dir}/train_allinone.jsonl",
        example_tag=f"top_{max_example_num}_similar_examples",
        example_num=example_num,
        text_tag_of_tar="sentence",
        text_tag_of_ref="text",
        entity_tag_of_tar="label",
        entity_tag_of_ref="entity",
        output_path=f"{bench_it_data_rag_dir}/outdomain/test_w_NN_{example_num}_{outdomain_dataname}_{setname}_GTElargeEmb_text.json",
        lang=lang,
        data_specific=False,
        dataname=DATANAME_MAP[dataname],
        use_original_id=True,
        ner_data_config_path=ner_prompt_config_path,
    )


    # # (2) ------ retrieve with example filtering ------
    # # Note: Please DO generate bm25 scores first before applying the following strategies
    # diverse_k=128
    # bm25score_threshold=20
    # for strategy in ["NN_bm25Rej", "diverseNN", "diverseNN_bm25Rej", "diverseNN_bm25Rank"]:
    #     if strategy=="NN_bm25Rej":
    #         strategy_mark=f"{strategy}_{example_num}_{bm25score_threshold}"
    #     if strategy=="diverseNN":
    #         strategy_mark=f"{strategy}_{example_num}_{diverse_k}"
    #     if strategy=="diverseNN_bm25Rej":
    #         strategy_mark=f"{strategy}_{example_num}_{diverse_k}_{bm25score_threshold}"
    #     if strategy=="diverseNN_bm25Rank":
    #         strategy_mark=f"{strategy}_{example_num}_{diverse_k}_{bm25score_threshold}"
    #     benchmark_test_set_to_adaptive_rag_conversation(
    #         input_path=f"{bench_data_dir}/test.jsonl", 
    #         label_info_path=f"{bench_data_dir}/abb2labelname.json",
    #         retrieved_ids_path=f"{bench_data_rag_dir}/sim_examp_ids_from_outdomain/test_simexample_ids_{max_example_num}_{outdomain_dataname}_{setname}_GTElargeEmb_text.jsonl",
    #         retrieving_data_path=f"{outdomain_data_dir}/train_allinone.jsonl",
    #         example_tag=f"top_{max_example_num}_similar_examples",
    #         example_num=example_num,
    #         text_tag_of_tar="sentence",
    #         text_tag_of_ref="text",
    #         entity_tag_of_tar="label",
    #         entity_tag_of_ref="entity",
    #         output_path=f"{bench_it_data_rag_dir}/outdomain/test_w_{strategy_mark}_{example_num}_{outdomain_dataname}_{setname}.json",
    #         lang=lang,
    #         data_specific=False,
    #         dataname=DATANAME_MAP[dataname],
    #         use_original_id=True,
    #         ner_data_config_path=ner_prompt_config_path,
    #         example_statistics_path=f"{bench_it_data_rag_dir}/outdomain/exampstat_test_w_{strategy_mark}_{example_num}_{outdomain_dataname}_{setname}.json",
    #         strategy=strategy,
    #         diverse_k=diverse_k,
    #         bm25score_threshold=bm25score_threshold,
    #         bm25score_path=f"{bench_data_rag_dir}/bm25Score_outdomain/test_simexample_bmscores_128_{outdomain_dataname}_{setname}_GTElargeEmb_text.jsonl"
    #     )


    # # (3) ------ retrieve random example ------
    # retrieve_random_examples(
    #     src_data_path=f"{outdomain_data_dir}/train_allinone.jsonl",
    #     tar_data_path=f"{bench_data_dir}/test.jsonl",
    #     save_random_examples_path=f"{bench_data_rag_dir}/rand_examp_ids_from_outdomain/test_randexample_ids_{max_example_num}_{outdomain_dataname}_{setname}.jsonl",
    #     lang=lang,
    #     example_num=max_example_num,
    #     save_ids_only=True,
    # )
    # # Generate instruction data with retrieved examples
    # example_num=2
    # benchmark_test_set_to_rag_conversation(
    #     input_path=f"{bench_data_dir}/test.jsonl", 
    #     label_info_path=f"{bench_data_dir}/abb2labelname.json",
    #     retrieved_ids_path=f"{bench_data_rag_dir}/rand_examp_ids_from_outdomain/test_randexample_ids_{max_example_num}_{outdomain_dataname}_{setname}.jsonl",
    #     retrieving_data_path=f"{outdomain_data_dir}/train_allinone.jsonl",
    #     example_tag=f"top_{max_example_num}_similar_examples",
    #     example_num=example_num,
    #     output_path=f"{bench_it_data_rag_dir}/outdomain/test_w_rand_{example_num}_{outdomain_dataname}_{setname}.json",
    #     lang=lang,
    #     data_specific=False,
    #     dataname=DATANAME_MAP[dataname],
    #     use_original_id=True,
    #     ner_data_config_path=ner_prompt_config_path,
    # )

def retrieve_indomain_example_for_benchmarks():
    # Generate instruction data with retrieved examples
    ner_prompt_config_path = f"configs/prompt_config/ner_sft.json"

    max_example_num=8
    example_num=2
    
    lang="zh"

    setname="8_random_42"
    
    # boson as an example
    dataname = "boson"
    bench_data_dir = f"data/benchmark_data/{dataname}"
    bench_data_rag_dir = f"data/benchmark_data_rag/{dataname}"
    bench_it_data_rag_dir = f"data/benchmark_it_data_rag/{dataname}"
    indomain_data_dir = f"data/benchmark_data/{dataname}"
    indomain_dataname="train"

    # Generate instruction data with retrieved examples
    benchmark_test_set_to_rag_conversation(
        input_path=f"{bench_data_dir}/test.jsonl", 
        label_info_path=f"{bench_data_dir}/abb2labelname.json",
        retrieved_ids_path=f"{bench_data_rag_dir}/sim_examp_ids_from_sampled_train/test_simexample_ids_{max_example_num}_{indomain_dataname}_{setname}_GTElargeEmb_text.jsonl",
        retrieving_data_path=f"{indomain_data_dir}/train.jsonl",
        example_tag=f"top_{max_example_num}_similar_examples",
        example_num=example_num,
        text_tag_of_tar="sentence",
        text_tag_of_ref="sentence",
        entity_tag_of_tar="label",
        entity_tag_of_ref="label",
        output_path=f"{bench_it_data_rag_dir}/indomain/test_w_NN_{example_num}_{indomain_dataname}_{setname}_GTElargeEmb_text.json",
        lang=lang,
        data_specific=False,
        dataname=DATANAME_MAP[dataname],
        use_original_id=True,
        ner_data_config_path=ner_prompt_config_path,
    )



if __name__ == "__main__":
    # retrieve_example_for_training_data()

    # retrieve_outdomain_example_for_benchmarks()

    retrieve_indomain_example_for_benchmarks()