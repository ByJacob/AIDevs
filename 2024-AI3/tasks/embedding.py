import json
import os
from abc import abstractmethod, ABC

from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI
from openai.types import Embedding

langfuse_context.configure(
    secret_key=os.getenv('AIDEVS3_LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('AIDEVS3_LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('AIDEVS3_LANGFUSE_HOST'),
)


class __BaseModel(ABC):

    def __init__(self, embedding_model, debug=False, **kwargs):
        self.debug = debug
        self.embedding_model = embedding_model
        langfuse_context.update_current_trace(
            user_id="jrosa"
        )

    @abstractmethod
    # shoud return input_tokens, output_token and response
    def _embedding(self, text, **kwargs) -> Embedding:
        pass

    def embedding(self, text, **kwargs) -> Embedding:
        if self.debug:
            print("*" * 20)
            print("MESSAGES: ", text)
            print("*" * 20)
        response = self._embedding(text, **kwargs)
        return response

    def ping(self):
        old_debug = self.debug
        self.debug = False
        self.embedding("PING")
        self.debug = old_debug


class __OpenAIModel(__BaseModel):

    def __init__(self, model, formats=None, **kwargs):
        super().__init__(model, **kwargs)
        self.client = OpenAI()
        self.formats = formats

    @observe(capture_input=False, capture_output=False, as_type="generation")
    def _embedding(self, text, **kwargs) -> Embedding:
        langfuse_context.update_current_observation(
            input=text,
            model=self.embedding_model,
        )
        response = self.client.embeddings.create(input=text, model=self.embedding_model)
        langfuse_context.update_current_observation(
            usage={
                "input": response.usage.prompt_tokens,
            }
        )
        return response.data[0]


class OpenAiTextEmbedding3Large(__OpenAIModel):

    def __init__(self, **kwargs):
        model = "text-embedding-3-large"
        super().__init__(model, **kwargs)


class __OLLAMAModel(__OpenAIModel):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required, but unused
        )
        self.ping()


class NomicEmbedText(__OLLAMAModel):

    def __init__(self, **kwargs):
        super(NomicEmbedText, self).__init__("nomic-embed-text", **kwargs)
