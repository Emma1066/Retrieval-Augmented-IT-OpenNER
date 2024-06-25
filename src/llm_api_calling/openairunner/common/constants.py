model_list = {
    "gpt-3.5-turbo-0125": {"abbr":"gpt35tb0125", "publisher":"openai", "max_tokens":4096*4},
    "moonshot-v1-8k": {"abbr":"mst", "publisher":"moonshot", "max_tokens":8000},
    "glm-4": {"abbr":"glm4", "publisher":"zhipu", "max_tokens":128000},
    "claude-3-haiku-20240307": {"abbr":"claude_haiku", "publisher":"anthropic", "max_tokens":200000}
}

dataset_language_map = {
    "weibo": "zh",
    "msra": "zh",
    "ontonotes4zh": "zh",
}


# Fill in your own OpenAI API keys. 
# If your key does not need to set API base, set "set_base" to False. Otherwise, set "set_base" to True, and fill the "api_base" with your corresponding base.
my_openai_api_keys = {
    "openai":[
        {"key":"your_key", "base_url":"your_base"},
    ],
    "anthropic":[
        {"key":"your_key", "base_url":"your_base"},
    ],
    "moonshot":[
        {"key":"your_key", "base_url":"your_base"},
    ],
    "zhipu":[
        {"key":"your_key", "base_url":"your_base"},
    ]
}

ROOT_PATH = ""
LOG_CONFIG_PATH = "configs/log_config/log_config.json"
