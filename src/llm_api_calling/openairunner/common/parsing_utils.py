import logging, logging.config
from typing import List, Dict
from argparse import Namespace
import re

from ..common.file_utils import json2dict

logger = logging.getLogger(__name__)

def response_2_prediction_of_dict_json(args, query, response, return_form="json"):
    ENTITY_TAG_EN = "entity 1"
    TYPE_TAG_EN = "type of entity 1"
    ENTITY_TAG_ZH = "实体1"
    TYPE_TAG_ZH = "实体1的类别"
    # if return empty answer
    if response in ["", "[]", "[{}]", "A: []", "{}"]:
        prediction = [] if return_form == "json" else {}
        return prediction

    sent = query

    # Replace Chinese punctuation with English punctuation
    punc_zh2en = {'，': ',', '。': '.', '：': ':'}
    response_punctransed = response.translate(str.maketrans(punc_zh2en))
    response_punctransed = response_punctransed.replace("\n", "")

    matched_list = re.findall(r'\[(.*?)\]', response_punctransed)
    if len(matched_list) == 0:
        # ---------- dict matching --------------
        matched_list = re.findall(r'\{(.*?)\}', response_punctransed)
        if len(matched_list) != 0:
            prediction = []
            contain_tag = False
            for matched_item in matched_list:
                matched_item = "{" + matched_item + "}"

                matched_item = matched_item.replace("null", "\"O\"")
                try:
                    eval_matched_item = eval(matched_item)
                except Exception as e_parse_dict:
                    logger.info(f"===== Error: Cannot match []. Fail to parse to dict item. ======")
                    logger.info(f"resp: {response}")
                    logger.info(f"matched_item: {matched_item}")
                    logger.info(e_parse_dict)
                    continue

                if isinstance(eval_matched_item, dict):
                    if (ENTITY_TAG_EN in eval_matched_item and TYPE_TAG_EN in eval_matched_item) or (ENTITY_TAG_ZH in eval_matched_item and TYPE_TAG_ZH in eval_matched_item):
                        contain_tag = True
                    # {"entity 1": "Oleg Shatskiku", "type of entity 1": "PERSON"}
                    if contain_tag:
                        tmp_ment = list(eval_matched_item.values())[0]
                        tmp_type = list(eval_matched_item.values())[1]
                        prediction.append({tmp_ment:tmp_type})
                    # {xxx:xxx, xxx:xxx, ...}
                    else:
                        # tmp_ment = list(eval_matched_item.keys())[0]
                        # tmp_type = list(eval_matched_item.values())[0]
                        # prediction.append({tmp_ment:tmp_type})
                        for k, v in eval_matched_item.items():
                            prediction.append({k:v})

            if len(prediction)>0:
                if return_form=="dict":
                    prediction=json2dict(prediction)
                return prediction
        # ---------- ------------ --------------
        logger.info(f"===== Error: Cannot match []. Fail to find valid dict answer. ======")
        logger.info(f"Text: {sent}")
        logger.info(f"Response: \n{response}")
        prediction = [] if return_form in ["json"] else {}
        return prediction
    else:
        try:
            ans_str = '[' + matched_list[-1] + ']'

            ans_str = ans_str.replace("null", "\"O\"")
            
            ans_eval = eval(ans_str)

            if len(ans_eval)==0:
                prediction = ans_eval
                if return_form == "dict":
                    prediction = json2dict(prediction)
                return prediction

            # Process the following format：
            #   [{"entity 1": "Oleg Shatskiku", "type of entity 1": "PERSON"}, ...]
            #   [{"实体1": "新华社", "实体1的类别": "机构名称"}, ...]
            if (ENTITY_TAG_EN in ans_eval[0] and TYPE_TAG_EN in ans_eval[0]) or (ENTITY_TAG_ZH in ans_eval[0] and TYPE_TAG_ZH in ans_eval[0]):
                prediction = []
                for tmp in ans_eval:
                    tmp_ment = list(tmp.values())[0]
                    tmp_type = list(tmp.values())[1]
                    prediction.append({tmp_ment:tmp_type})
                if return_form=="dict":
                    prediction = json2dict(prediction)
                return prediction
            
            # Handling two possible effective output formats：
            # 1： [{XX:XX, XX:XX, XX:XX}]
            # 2： [{XX:XX}, {XX:XX}, {XX:XX}]
            
            if len(ans_eval) == 1 and len(ans_eval[0]) > 1: # 1： [{XX:XX, XX:XX, XX:XX}]
                prediction_w_o = [
                    {k: v} for k,v in ans_eval[0].items()
                ]
            else: # 2： [{XX:XX}, {XX:XX}, {XX:XX}]
                # prediction_w_o = {list(item.keys())[0]: list(item.values())[0] for item in ans_eval}
                prediction_w_o = ans_eval
            # remove answers that whose type being "O" (null)
            prediction = []
            for item in prediction_w_o:
                k, v = list(item.items())[0]
                if v != "O":
                    prediction.append(item)
        except Exception as e_parse_list:
            # Possible missing individual ','} 'may result in incomplete JSON parsing
            # Attempt to match partially parseable dicts
            # ---------- dict matching --------------
            matched_list = re.findall(r'\{(.*?)\}', response_punctransed)
            if len(matched_list) != 0:
                prediction = []
                contain_tag = False
                for matched_item in matched_list:
                    matched_item = "{" + matched_item + "}"

                    matched_item = matched_item.replace("null", "\"O\"")
                    try:
                        eval_matched_item = eval(matched_item)
                    except Exception as e_parse_dict:
                        logger.info(f"===== Error: Matched [] but fail to parse. Fail to parse to dict item. ======")
                        logger.info(f"resp: {response}")
                        logger.info(f"matched_item: {matched_item}")
                        logger.info(e_parse_dict)
                        continue

                    if isinstance(eval_matched_item, dict):
                        if (ENTITY_TAG_EN in eval_matched_item and TYPE_TAG_EN in eval_matched_item) or (ENTITY_TAG_ZH in eval_matched_item and TYPE_TAG_ZH in eval_matched_item):
                            contain_tag = True
                        # {"entity 1": "Oleg Shatskiku", "type of entity 1": "PERSON"}
                        if contain_tag:
                            tmp_ment = list(eval_matched_item.values())[0]
                            tmp_type = list(eval_matched_item.values())[1]
                            prediction.append({tmp_ment:tmp_type})
                        # {xxx:xxx, xxx:xxx, ...}
                        else:
                            # tmp_ment = list(eval_matched_item.keys())[0]
                            # tmp_type = list(eval_matched_item.values())[0]
                            # prediction.append({tmp_ment:tmp_type})
                            for k, v in eval_matched_item.items():
                                prediction.append({k:v})

                if len(prediction)>0:
                    if return_form=="dict":
                        prediction=json2dict(prediction)
                    return prediction
            # ---------- ------------ --------------

            logger.info(f"===== Error: Matched [] but fail to parse. Fail to find valid dict answer. ======")
            logger.info(f"Text: {sent}")
            logger.info(f"Response: \n{response}")
            logger.info(f"Error traceback: {e_parse_list}")
            prediction = [] if return_form in ["json"] else {}
            return prediction
    
    if return_form=="dict":
        prediction=json2dict(prediction)
    return prediction

def response_2_prediction(args, query, response, resp_idx=None, question=None, return_form="dict", complete_form="question", return_responded_qa=False):
    if complete_form == "question":
        if return_form in ["dict", "json"]:
            prediction = response_2_prediction_of_dict_json(args, query, response, return_form=return_form)
        elif return_form == "list":
            raise NotImplementedError
        elif return_form == "tuple list":
            raise NotImplementedError
        else:
            raise ValueError(f"Unrecognized return_form: {return_form}")
        return prediction
    else:
        raise ValueError(f"Unrecognized complete_form={complete_form}")

