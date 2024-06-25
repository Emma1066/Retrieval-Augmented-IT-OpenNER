import sys
import time
import random
import openai
import logging
import tiktoken

from .constants import model_list, my_openai_api_keys

logger = logging.getLogger(__name__)


def set_api_key(model_name):
    publisher = model_list[model_name]["publisher"]
    keys = my_openai_api_keys[publisher]
    return keys

def run_llm(
        messages, 
        openai_key=None, 
        model_name=None,
        temperature=0,
        stop=None,
        max_tokens=None,
        stream=False,
        return_text=True
):
    # TODO: Call different interface
    agent = SafeOpenai(openai_key, model_name=model_name)

    flag, response = agent.chat(
        model=model_name, 
        messages=messages, 
        temperature=temperature, 
        stop=stop,
        max_tokens=max_tokens,
        stream=stream,
        )
    # if flag:
    #     response = response.choices[0].message.content
    return flag, response

class SafeOpenai:
    def __init__(self, keys=None, model_name=None, start_id=None):
        if keys is None:
            raise "Please provide OpenAI Key."

        self.keys = keys

        if start_id is None:
            start_id = random.sample(list(range(len(keys))), 1)[0]
            
        self.key_id = start_id
        self.key_id = self.key_id % len(self.keys)
        current_key = self.keys[self.key_id % len(self.keys)]
        self.client = openai.OpenAI(api_key=current_key["key"], base_url=current_key["base_url"])

    def set_next_api_key(self):
        self.key_id = (self.key_id + 1) % len(self.keys)
        current_key = self.keys[self.key_id]
        self.client = openai.OpenAI(api_key=current_key["key"], base_url=current_key["base_url"])
        
    def chat(self, 
            model, 
            messages, 
            temperature, 
            stop=None,
            max_tokens=None,
            stream=False,
            sleep_seconds=3, 
            max_read_timeout=1, 
            max_http=1,
            DEBUG=False,
            stream_max_time=50000):
        
        # print(f"\nopenai.api_key = {self.client.api_key}")
        # print(f"openai.base_url  = {self.client.base_url}")
        while True:
            try:
                if not stream:
                    response = self.client.chat.completions.create(
                        model=model, 
                        messages=messages, 
                        temperature=temperature,
                        max_tokens=max_tokens,

                    )
                    return (True, response)
                else:
                    raise NotImplementedError
            except openai.APIConnectionError as e:
                #Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                # return (False, f"Failed to connect to OpenAI API: {e}")
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except openai.RateLimitError as e:
                #Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                # return (False, f"OpenAI API request exceeded rate limit: {e}")
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except Exception as e:
                print(f"Uncategoried err: {e}")
                # return (False, f"Uncategoried err: {e}")

                # safe content
                words_for_safety = ["risk", "不安全", "敏感", "安全"]
                if any([x.lower() in str(e).lower() for x in words_for_safety]):
                    return (False, f"{e}")
                # invalid output
                words_for_output= ["invalid Unicode output"]
                if any([x.lower() in str(e).lower() for x in words_for_output]):
                    return (False, f"{e}")

                self.set_next_api_key()
                time.sleep(sleep_seconds)
