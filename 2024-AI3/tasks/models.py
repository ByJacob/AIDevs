import os
from abc import ABC, abstractmethod
from typing import Tuple

from langfuse.decorators import langfuse_context
from langfuse.openai import OpenAI

langfuse_context.configure(
    secret_key=os.getenv('AIDEVS3_LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('AIDEVS3_LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('AIDEVS3_LANGFUSE_HOST'),
)


class __BaseModel(ABC):

    def __init__(self, model, debug=False, **kwargs):
        self.debug = debug
        self.model = model

    @abstractmethod
    # shoud return input_tokens, output_token and response
    def _chat(self, messages) -> Tuple[int, int, str]:
        pass

    def chat(self, messages, **kwargs):
        if self.debug:
            print("*" * 20)
            print("MESSAGES: ", messages)
            print("*" * 20)
        response = self._chat(messages)
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


class __OLLAMAModel(__BaseModel):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required, but unused
        )

    def _chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content


class Gemma2P2B(__OLLAMAModel):

    def __init__(self, **kwargs):
        super(Gemma2P2B, self).__init__("gemma2:2b", **kwargs)


class Llama32P1B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(Llama32P1B, self).__init__("llama3.2:1b", **kwargs)


class Llama32P8B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(Llama32P8B, self).__init__("llama3.1:8b", **kwargs)


class Qwen25P3B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(Qwen25P3B, self).__init__("qwen2.5:3b", **kwargs)


class MistralP7B(__OLLAMAModel):
    def __init__(self, **kwargs):
        super(MistralP7B, self).__init__("mistral", **kwargs)
