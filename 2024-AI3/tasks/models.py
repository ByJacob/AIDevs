from abc import ABC, abstractmethod

import ollama


class __BaseModel(ABC):

    def __init__(self, debug=False, **kwargs):
        self.debug = debug

    @abstractmethod
    def _chat(self, messages):
        pass

    def chat(self, messages):
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
        super().__init__(**kwargs)
        self.model = model

    def _chat(self, messages) -> str:
        try:
            response = ollama.chat(model=self.model, messages=messages)
            return response['message']['content']
        except ollama.ResponseError as e:
            print('Error:', e.error)
            raise e


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
