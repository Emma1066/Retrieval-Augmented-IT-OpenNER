
import json
import time
import logging
import os

from tqdm import tqdm
import argparse

from copy import deepcopy

from openairunner.common.log_utils import get_logger
from openairunner.common.file_utils import load_data, save_data, print_used_api_keys, compute_metrics
from openairunner.common.parsing_utils import response_2_prediction
from openairunner.common.openai_utils import run_llm, set_api_key
from openairunner.common.constants import model_list, ROOT_PATH, LOG_CONFIG_PATH, dataset_language_map

logger = logging.getLogger(__name__)

PASSAGE_TAGS = ["text", "sentence"]

class LLMAsker(object):
    def __init__(self, args) -> None:
        self.args = args

    def construct_message(self, text:str, prompt_config:dict) -> list[dict]:
        if self.args.llm_using == "llm_annotation":
            prompt = prompt_config["ner prompt"] % text
        elif self.args.llm_using == "llm_evaluation":
            labels_str = ", ".join(self.args.id2label)
            prompt = prompt_config["ner prompt"] % (labels_str, text)

        message=[
                {"role": "system", "content": prompt_config["system prompt"]},
                {"role": "user", "content": prompt},
        ]

        return message

    def generate_responses_batch(self, verbose:bool=True):
        # log settings
        log_path = os.path.join(self.args.output_dir, "ask_llm.log")
        if self.args.clear_log:
            if os.path.exists(log_path):
                with open(log_path, "a+", encoding="utf-8") as f:
                    f.truncate(0)
            else:
                print("Clearing log failed. Log file does not exits. Start new log.")
        logger = get_logger(
            __name__,
            log_path=log_path,
            log_config_path=LOG_CONFIG_PATH,
        )
        logger.info(">>>>>>> do_asking ... ...")

        # get data
        input_data = load_data(self.args.input_path)
        if self.args.max_samples is not None:
            input_data = input_data[:self.args.max_samples]

        result_path = self.args.result_path
        assert result_path.endswith(".jsonl")
        token_count_path = self.args.token_count_path

        prompt_config = load_data(self.args.prompt_config_path)[self.args.lang][self.args.prompt_version]
        return_format = prompt_config["return format"]
        logger.info(f"return_format: {return_format}")

        bar = tqdm(input_data, ncols=150)
        cnt_prompt_tokens, cnt_completion_tokens = 0,0
        avg_prompt_tokens, avg_completion_tokens = 0,0
        cnt_actual_queries = 0

        start_idx = 0
        if self.args.resume:
            if not os.path.exists(result_path):
                logger.warning(f"Previous results not exist: {result_path}")
                logger.warning("Run from scratch!")
            else:
                pre_res = load_data(result_path)
                start_idx = len(pre_res)
                logger.info(f"> Resume from last run. start_idx = {start_idx}")


        # Get the tag for input text.
        tmp_query = input_data[0]
        if sum([tag in tmp_query for tag in PASSAGE_TAGS]) != 1:
            raise ValueError(f"Confilct passage tag. You should specify only one tag from the valid tags for the input text. The valid tag list: {PASSAGE_TAGS}")
        for tag in PASSAGE_TAGS:
            if tag in tmp_query:
                passage_tag = tag
                break
            
        with open(result_path, "ab", buffering=0) as realtime_f:
            for i_query, query in enumerate(bar):
                bar.set_description("Query LLM NER")

                # if not started from the first sample
                if i_query < start_idx:
                    continue
                
                input_passage = query[passage_tag] # text to conduct NER

                # if input_passage is empty, do not run llm
                if input_passage == "":
                    logger.info("Empty input text, NOT run llm.")
                    response_item = {
                        "text": input_passage,
                        "response":"",
                        "prediction":[]
                    }
                    if self.args.llm_using == "llm_evaluation":
                        response_item["label"] = query["label"]

                    realtime_f.write((json.dumps(response_item, ensure_ascii=False)+"\n").encode("utf-8"))
                    continue
                    
                message = self.construct_message(input_passage, prompt_config)

                cnt_actual_queries += 1
                flag, response = run_llm(
                    message,
                    openai_key=self.args.api_key,
                    model_name=self.args.model,
                    temperature=self.args.temperature,
                    stop=self.args.stop
                )
                if not flag:
                    content = ""
                    curr_p_toks = 0
                    curr_c_toks = 0
                else:
                    curr_p_toks = response.usage.prompt_tokens
                    curr_c_toks = response.usage.completion_tokens
                    content = response.choices[0].message.content

                cnt_prompt_tokens += curr_p_toks
                cnt_completion_tokens += curr_c_toks
                avg_prompt_tokens = cnt_prompt_tokens / cnt_actual_queries
                avg_completion_tokens = cnt_completion_tokens / cnt_actual_queries

                parsed_content = response_2_prediction(
                    self.args, 
                    input_passage, 
                    content,
                    return_form=return_format
                )
                if i_query <= 5 or (verbose and i_query % 50 == 0):
                    msg_resp = deepcopy(message)
                    msg_resp.append(
                        {"role":"assistant","content":content,"parsed_content":parsed_content},
                    )
                    logger.info(f"\nConversation {i_query}: \n" + json.dumps(msg_resp, indent=2, ensure_ascii=False))

                response_item = {
                    "text": input_passage,
                    "response":content,
                    "prediction":parsed_content
                }
                if self.args.llm_using == "llm_evaluation":
                    response_item["label"] = query["label"]

                realtime_f.write((json.dumps(response_item, ensure_ascii=False)+"\n").encode("utf-8"))

                # monitor token count
                bar.set_postfix(
                    curr_p_toks=curr_p_toks,
                    curr_c_toks=curr_c_toks,
                    avg_toks=avg_prompt_tokens + avg_completion_tokens,
                    total_toks=cnt_prompt_tokens + cnt_completion_tokens
                )

        logger.info(f"response saved to: {result_path}")
        
        logger.info("Finished!")
        logger.info(f"prompt_tokens = {cnt_prompt_tokens}")
        logger.info(f"completion_tokens = {cnt_completion_tokens}")
        logger.info(f"total_tokens = {cnt_prompt_tokens + cnt_completion_tokens}\n")
        save_data(
            path=token_count_path,
            data={
                "prompt_tokens":cnt_prompt_tokens,
                "completion_tokens":cnt_completion_tokens,
                "total_tokens":cnt_prompt_tokens + cnt_completion_tokens,
                "avg_prompt_tokens":avg_prompt_tokens,
                "avg_completion_tokens":avg_completion_tokens,
                "avg_total_tokens":avg_prompt_tokens + avg_completion_tokens,
            }
        )
        logger.info(f"token_count saved to: {token_count_path}")

        print_used_api_keys(self.args.api_key)

    def parse_responses_batch(self):
        # log settings        
        log_path = os.path.join(self.args.output_dir, "parse.log")
        if self.args.clear_log:
            if os.path.exists(log_path):
                with open(log_path, "a+", encoding="utf-8") as f:
                    f.truncate(0)
            else:
                print("Clearing log failed. Log file does not exits. Start new log.")
        logger = get_logger(
            __name__,
            log_path=log_path,
            log_config_path=LOG_CONFIG_PATH,
        )
        logger.info(">>>>>>> do_parsing ... ...")

        # get dataset
        result_path = self.args.result_path
        assert result_path.endswith(".jsonl")

        prompt_config = load_data(self.args.prompt_config_path)[self.args.lang][self.args.prompt_version]
        return_format = prompt_config["return format"]
        logger.info(f"return_format: {return_format}")

        results = load_data(result_path)
        if self.args.max_samples is not None:
            results = results[:self.args.max_samples]

        bar = tqdm(results, ncols=100, desc="parsing")
        for i_item, item in enumerate(bar):
            text, response = item["text"], item["response"]
            prediction  = response_2_prediction(
                None,
                query=text,
                response=response,
                return_form=return_format
            )

            item["prediction"] = prediction

            if i_item <= 3:
                logger.info(f"\n{i_item}\n" + json.dumps(item, indent=2, ensure_ascii=False))
        
        # parsed_result_path = result_path.replace(".jsonl", "_parsed.jsonl")
        parsed_result_path = self.args.parsed_result_path
        save_data(parsed_result_path, results)

        return results

    def compute_ner_metric(self):
        # log settings        
        log_path = os.path.join(self.args.output_dir, "compute_metric.log")
        if self.args.clear_log:
            if os.path.exists(log_path):
                with open(log_path, "a+", encoding="utf-8") as f:
                    f.truncate(0)
            else:
                print("Clearing log failed. Log file does not exits. Start new log.")
        logger = get_logger(
            __name__,
            log_path=log_path,
            log_config_path=LOG_CONFIG_PATH,
        )
        logger.info(">>>>>>> do_compute_metric ... ...")

        # get dataset
        input_data = load_data(self.args.input_path)
        results = load_data(self.args.parsed_result_path)
        if self.args.max_samples is not None:
            input_data = input_data[:self.args.max_samples]
            results = results[:self.args.max_samples]

        data_responses = []
        for item_data, item_result in zip(input_data, results):
            data_responses.append({
                "text": item_data["sentence"],
                "label": item_data["label"],
                "prediction": item_result["prediction"]
            })
        
        compute_metrics(self.args, data_responses=data_responses, write_metric=True)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--input_path", default=None, type=str)
    parser.add_argument("--label_info_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--lang", default=None, type=str)

    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--max_len", default=1024, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)

    parser.add_argument("--prompt_config_path", default=None, type=str)
    parser.add_argument("--prompt_version", default=None, type=str)
    
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_samples", default=None, type=int)

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--do_asking", action="store_true")
    parser.add_argument("--do_parsing", action="store_true")
    parser.add_argument("--do_compute_metric", action="store_true")
    parser.add_argument("--llm_using", type=str, choices=["llm_annotation", "llm_evaluation"])

    parser.add_argument("--clear_log", action="store_true")

    return parser

def make_settings(args):
    # model settings
    args.stop = None
    args.api_key = set_api_key(model_name=args.model)

    # dataset language
    if args.lang is None:
        args.lang = dataset_language_map[args.dataname] # zh/en
    
    # path settings
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.result_path = os.path.join(output_dir, "llm_response.jsonl")
    args.parsed_result_path = os.path.join(output_dir, "llm_response_parsed.jsonl")
    args.ner_metric_path = os.path.join(output_dir, "ner_metric.xlsx")
    args.token_count_path = os.path.join(output_dir, "token_count.json")

    # label set
    if args.llm_using == "llm_evaluation":
        abb2lname = json.load(open(args.label_info_path, "r", encoding="utf-8"))
        args.id2label = list(abb2lname.values())

    return args


def main():
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    
    # prepare settings
    args = make_settings(args)

    logger.info("-------------------------------")
    logger.info("---------- Ask LLM ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    llm_asker = LLMAsker(args)
    
    if args.do_asking:
        llm_asker.generate_responses_batch()
    if args.do_parsing:
        llm_asker.parse_responses_batch()
    if args.do_compute_metric:
        llm_asker.compute_ner_metric()


if __name__ == "__main__":
    main()
