import copy
import hashlib
import json
import os
import pickle
import re
from abc import ABC, abstractmethod
from typing import Tuple

import litellm
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI

os.environ["LANGFUSE_SECRET_KEY"] = os.environ["AIDEVS3_LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_PUBLIC_KEY"] = os.environ["AIDEVS3_LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_HOST"] = os.environ["AIDEVS3_LANGFUSE_HOST"]

langfuse_context.configure(
    secret_key=os.getenv('AIDEVS3_LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('AIDEVS3_LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('AIDEVS3_LANGFUSE_HOST'),
)

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

def cache_llm_request(func, messages, **kwargs):
    # Generate a unique hash from the function name and kwargs
    kwargs['messages'] = messages
    kwargs_str = json.dumps(kwargs, sort_keys=True)  # Convert kwargs to a consistent JSON string
    hash_obj = hashlib.md5((func.__name__ + kwargs_str).encode())  # Create MD5 hash with function name
    hash_filename = os.path.join("tmp", "models", f"{hash_obj.hexdigest()}.pkl")
    parent_dor = os.path.dirname(hash_filename)
    if not os.path.exists(parent_dor):
        os.makedirs(parent_dor)

    # Check if the file exists
    if os.path.exists(hash_filename):
        with open(hash_filename, 'rb') as file:
            # print("Cache found. Loading result.")
            return pickle.load(file)  # Load and return the cached result
    else:
        # print("No cache found. Executing function.")
        if "seed" in kwargs.keys():
            kwargs.pop("seed")
        result = func(**kwargs)  # Execute the function with kwargs
        with open(hash_filename, 'wb') as file:
            pickle.dump(result, file)  # Save the result to the cache
        return result  # Return the result

class __BaseModel(ABC):

    def __init__(self, model, debug=False, **kwargs):
        self.debug = debug
        self.model = model
        langfuse_context.update_current_trace(
            user_id="jrosa"
        )

    @abstractmethod
    # shoud return input_tokens, output_token and response
    def _chat(self, messages, **kwargs) -> str:
        pass

    def chat(self, messages, **kwargs):
        if self.debug:
            new_messages = json.dumps(messages)
            new_messages = re.sub(r'("url"\s*:\s*)"(.*?)"', r'\1"BASE64_IMAGE"', new_messages)
            print("*" * 20)
            print("MESSAGES: ", new_messages)
            print("*" * 20)
        response = cache_llm_request(self._chat, messages, **kwargs)
        if self.debug:
            print("RESPONSE: ", response)
            print("*" * 20)
        return response

    def ping(self):
        old_debug = self.debug
        self.debug = False
        messages = [
            {
                'role': 'user',
                'content': 'ping',
            },
        ]
        self.chat(messages)
        self.debug = old_debug


class OpenAiWhisper(__BaseModel):

    def __init__(self, **kwargs):
        model = "whisper-1"
        super().__init__(model, **kwargs)
        self.client = OpenAI()

    def _chat(self, messages) -> Tuple[int, int, str]:
        raise NotImplementedError()

    @observe(capture_input=False, capture_output=False, as_type="generation")
    def transcript(self, file, language="pl"):
        from tinytag import TinyTag
        import math

        tag = TinyTag.get(file)
        seconds = math.ceil(tag.duration)

        langfuse_context.update_current_observation(
            input=f"FILE: {file}",
            model=self.model,
        )
        with open(file, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language
            )
            langfuse_context.update_current_observation(
                usage={
                    "input": seconds,
                },
                output=transcription.text
            )
        return transcription.text


class __OpenAIModel(__BaseModel):
    def __init__(self, model, formats=None, **kwargs):
        super().__init__(model, **kwargs)
        self.client = OpenAI()
        self.formats = formats

    def _chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages, **kwargs
        )
        return response.choices[0].message.content


class OpenAi4oMini(__OpenAIModel):
    def __init__(self, **kwargs):
        model = "gpt-4o-mini"
        formats = []
        super().__init__(model, formats=formats, **kwargs)


class OpenAi4o(__OpenAIModel):
    def __init__(self, **kwargs):
        model = "gpt-4o"
        formats = []
        super().__init__(model, formats=formats, **kwargs)


class OpenAi35Turbo(__OpenAIModel):
    def __init__(self, **kwargs):
        model = "gpt-3.5-turbo"
        super().__init__(model, **kwargs)
        self.client = OpenAI()


class __OLLAMAModel(__OpenAIModel):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required, but unused
        )
        self.ping()


class Gemma2P2B(__OLLAMAModel):

    def __init__(self, **kwargs):
        super(Gemma2P2B, self).__init__("gemma2:2b", **kwargs)


class Gemma2P9B(__OLLAMAModel):

    def __init__(self, **kwargs):
        super(Gemma2P9B, self).__init__("gemma2:9b", **kwargs)


class Llama32P1B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(Llama32P1B, self).__init__("llama3.2:1b", **kwargs)


class Llama31P8B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(Llama31P8B, self).__init__("llama3.1:8b", **kwargs)


class Qwen25P3B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(Qwen25P3B, self).__init__("qwen2.5:3b", **kwargs)


class MistralP7B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(MistralP7B, self).__init__("mistral", **kwargs)


class BielikP11B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(BielikP11B, self).__init__("hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q5_K_M", **kwargs)


class Phi35P38B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(Phi35P38B, self).__init__("phi3.5:latest", **kwargs)


class LlavaP7B(__OLLAMAModel):

    def __init__(self, model="llava:7b", **kwargs):
        formats = [(672, 672), (336, 1344), (1344, 336)]
        super().__init__(model, formats=formats, **kwargs)


class LlavaP13B(LlavaP7B):
    def __init__(self, **kwargs):
        model = "llava:13b"
        super().__init__(model, **kwargs)


class Llama32visionP11B(__OLLAMAModel):
    def __init__(self, **kwargs):
        model = "llama3.2-vision"
        formats = []
        super().__init__(model, formats=formats, **kwargs)

class Clause35Sonet(__BaseModel):

    def __init__(self, **kwargs):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["AIDEVS3_ANTHROPIC_API_KEY"]
        super().__init__("claude-3-5-sonnet-20241022", **kwargs)
    def _chat(self, messages, **kwargs) -> str:
        response = litellm.completion(model=self.model, messages=messages)
        return response.choices[0].message.content