import logging, logging.config
import json
import sys

from .constants import LOG_CONFIG_PATH

def get_logger(name, log_path:str, log_config_path:str=LOG_CONFIG_PATH) -> logging.Logger:
    f"""
    Gets a standard logger with pre-defined file handler and stream handler.
    """
    config_dict = json.load(open(LOG_CONFIG_PATH))
    config_dict['handlers']['file_handler']['filename'] =  log_path
    # config_dict['handlers']['file_handler']['level'] = "INFO"
    config_dict['root']['level'] = "INFO"
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger