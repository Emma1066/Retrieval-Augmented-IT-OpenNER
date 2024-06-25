from typing import List, Dict

import os
import json
import torch
from tqdm import tqdm
import pandas as pd
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

def copy_file_to_path(data, dir, path):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    if os.path.exists(path):
        raise ValueError(f"Path already exists: {path}")
    save_data(path, data)

def load_data(path, json_format=None):
    if path.endswith(".txt"):
        data = [eval(x.strip()) for x in open(path, "r", encoding="utf-8").readlines()]    
    elif path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
    elif path.endswith(".jsonl"):
        with open(path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
    else:
        raise ValueError(f"Wrong path for query data: {path}")

    return data

def save_data(path, data, json_format=None):
    if path.endswith(".txt"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data[:-1]:
                f.write(str(item) + "\n")
            f.write(str(data[-1]))
    elif path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=4, ensure_ascii=False))
    elif path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + "\n")
    else:
        raise ValueError(f"Wrong path for prompts saving: {path}")
    
def json2dict(json_list):
    d = dict()
    for item in json_list:
        k = list(item.keys())[0]
        v = item[k]
        d[k] = v
    return d

def dict2json(in_dict):
    out_json = []
    for k, v in in_dict.items():
        out_json.append(
            {k: v}
        )
    return out_json

def convert_format(data, target_format):
    if target_format == "dict":
        if isinstance(data, dict):
            data_new = data
        elif isinstance(data, list):
            data_new = json2dict(data)
        else:
            raise TypeError(f"Cannot handle type(data)={type(data)}")
    elif target_format == "json":
        if isinstance(data, list):
            data_new = data
        elif isinstance(data, dict):
            data_new = dict2json(data)
        else:
            raise TypeError(f"Cannot handle type(data)={type(data)}")
    else:
        raise NotImplementedError(f"convert_format() Not implemented for target_format={target_format}")

    return data_new

def format_json2str(in_json):
    out_str = "["
    for i_item, item in enumerate(in_json):
        k = list(item.keys())[0]
        v = item[k]
        out_str += "{\"%s\": \"%s\"}" % (k, v)        
        if i_item < len(in_json)-1:
            out_str += ", "
    out_str += "]"

    return out_str


def fetch_indexed_embs(train_embs, demo_data):
    '''
    train_embs: embs of whole training set.
    demo_data: demo set selected from training set.
    '''
    idxes = [x["idx"] for x in demo_data]
    idxes = torch.tensor(idxes).long()
    train_embs = torch.from_numpy(train_embs)
    demo_embs = torch.index_select(train_embs, dim=0, index=idxes)
    demo_embs = demo_embs.numpy()

    return demo_embs


# gold and pred, including count, rate, p/r/f1
def compute_metrics(args, data_responses:List[Dict], write_metric:bool=True):
    '''
    columns to be collected:
        ["Type", "Gold count", "Gold rate", "Pred count", "Pred rate", "Prec", "Rec", "F1"]
    '''
    
    id2label = args.id2label
    # Each class has one record
    type2record = {}
    for lb in id2label:
        type2record[lb] = {"Type":lb, "Gold count":0, "Gold rate":0, "Pred count":0, "Pred rate":0, "Correct count":0, "Prec":0, "Rec":0, "F1":0}
    
    for i_item, item in enumerate(tqdm(data_responses, desc="compute metric")):

        curr_label = item["label"]
        if isinstance(curr_label, str):
            curr_label = eval(curr_label)
        if isinstance(curr_label, list):
            curr_label = json2dict(curr_label)
        curr_pred = item["prediction"]
        if isinstance(curr_pred, str):
            curr_pred = eval(curr_pred)
        if isinstance(curr_pred, list):
            curr_pred = json2dict(curr_pred)

        # remove "" prediction
        if "" in curr_pred:
            del curr_pred[""]
        
        for tmp_mention, tmp_type in curr_label.items():
            type2record[tmp_type]["Gold count"] += 1
        
        ood_type_preds = []
        ood_mention_preds = []
        for tmp_mention, tmp_type in curr_pred.items():
            # ood type
            if tmp_type not in id2label:
                ood_type_preds.append({tmp_mention:tmp_type})
                continue
            type2record[tmp_type]["Pred count"] +=1
            # ood mention
            if tmp_mention not in item["text"]:
                ood_mention_preds.append({tmp_mention:tmp_type})
                continue
            # compare with gold label
            if tmp_mention in curr_label and tmp_type == curr_label[tmp_mention]:
                type2record[tmp_type]["Correct count"] += 1

        # print ood
        # if len(ood_type_preds)>0:
        #     logger.info(f"OOD Type predictions:\n{ood_type_preds}")
        # if len(ood_mention_preds)>0:
        #     logger.info(f"OOD mention predictions:\n{ood_mention_preds}")
    
    # compute overall metrics
    n_gold_tot = sum([x["Gold count"] for x in type2record.values()])
    n_pred_tot = sum([x["Pred count"] for x in type2record.values()])
    n_correct_tot = sum([x["Correct count"] for x in type2record.values()])
    prec_tot = n_correct_tot / n_pred_tot if n_pred_tot else 0
    rec_tot = n_correct_tot / n_gold_tot if n_gold_tot else 0
    if prec_tot and rec_tot:
        f1_tot = 2*prec_tot*rec_tot / (prec_tot+rec_tot)
    else:
        f1_tot = 0
    # prec_tot = round(prec_tot,4)*100
    prec_tot = Decimal(prec_tot*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
    # rec_tot = round(rec_tot,4)*100
    rec_tot = Decimal(rec_tot*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
    # f1_tot = round(f1_tot,4)*100
    f1_tot = Decimal(f1_tot*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")

    # compute metric of each class
    for k in type2record:
        gold_count = type2record[k]["Gold count"]
        pred_count = type2record[k]["Pred count"]
        correct_count = type2record[k]["Correct count"]
        
        gold_rate = gold_count / n_gold_tot if n_gold_tot else 0
        pred_rate = pred_count / n_pred_tot if n_pred_tot else 0
        # gold_rate = round(gold_rate,4)*100
        gold_rate = Decimal(gold_rate*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
        # pred_rate = round(pred_rate,4)*100
        pred_rate = Decimal(pred_rate*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")

        prec = correct_count / pred_count if pred_count else 0
        rec = correct_count / gold_count if gold_count else 0
        if prec and rec:
            f1 = 2*prec*rec / (prec+rec)
        else:
            f1 = 0
        # prec = round(prec,4)*100
        prec = Decimal(prec*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
        # rec = round(rec,4)*100
        rec = Decimal(rec*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
        # f1 = round(f1,4)*100
        f1 = Decimal(f1*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")

        type2record[k]["Gold rate"] = gold_rate
        type2record[k]["Pred rate"] = pred_rate
        type2record[k]["Prec"] = prec
        type2record[k]["Rec"] = rec
        type2record[k]["F1"] = f1

    type2record["Total"] = {"Type":"ToTal", "Gold count":n_gold_tot, "Gold rate":100, "Pred count":n_pred_tot, "Pred rate":100, "Correct count":n_correct_tot, "Prec":prec_tot, "Rec":rec_tot, "F1":f1_tot}

    df_metrics = pd.DataFrame(list(type2record.values()))
    logger.info(f"===== Metrics =====\n{df_metrics}")
    # save to files
    if write_metric:
        ner_metric_path = args.ner_metric_path
        df_metrics.to_excel(ner_metric_path, index=False)

def mask_key(key:str) -> str:
    chunk_len = len(key) // 3
    key = key[:chunk_len] + "*" * chunk_len + key[2*chunk_len:]
    
    return key

def print_used_api_keys(api_keys:List[Dict]) -> None:
    for key in api_keys:
        key["key"] = mask_key(key["key"])
    logger.info(json.dumps(api_keys, indent=2, ensure_ascii=False))