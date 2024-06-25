import json
import logging, logging.config
import os
import argparse

from openairunner.common.log_utils import get_logger
from openairunner.common.file_utils import load_data, save_data, compute_metrics
from openairunner.common.parsing_utils import two_stage_majority_voting
from openairunner.common.constants import model_list, ROOT_PATH

logger = logging.getLogger(__name__)

def main(args):
    # load response data
    data_response = load_data(args.response_path)

    # compute evaluation results
    compute_metrics(args, data_response)

    logger.info(f"Prediction data saved to: {args.pred_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_info_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    parser.add_argument("--model", default=None, type=str)

    parser.add_argument("--prompt_config_path", default=None, type=str)
    parser.add_argument("--prompt_version", default=None, type=str)
    
    args = parser.parse_args()

    output_dir = args.output_dir
    args.log_path = os.path.join(output_dir, "compute_metrics.log")
    args.result_path = os.path.join(output_dir, "llm_response.jsonl")

    # label set
    abb2lname = json.load(open(args.label_info_path, "r", encoding="utf-8"))
    args.id2label = list(abb2lname.values())

    logger.info("\n\n---------- Compute Evaluation Results ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)