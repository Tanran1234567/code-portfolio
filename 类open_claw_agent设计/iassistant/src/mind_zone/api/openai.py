#!/usr/bin/env python
# -*-coding: utf-8-*-

from hashlib import md5
from typing import Any, AsyncGenerator, Generator, Optional, Self, Sequence, Union

import llama_index.core.instrumentation as instrument
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.instrumentation.events.llm import LLMPredictEndEvent, LLMPredictStartEvent
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.llms.llm import Model
from llama_index.core.program.utils import FlexibleModel
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.types import Model, TokenAsyncGen, TokenGen
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai.utils import to_openai_message_dicts
from llama_index.llms.openai_like import OpenAILike
from openai._streaming import AsyncStream, Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from mind_zone.api.types import APIClient
from mind_zone.api.utils import (
    astream_chat_response_to_tokens,
    astream_completion_response_to_tokens,
    raise_for_unimplemented,
    stream_chat_response_to_tokens,
    stream_completion_response_to_tokens,
)
from mind_zone.utils import format_messages

dispatcher = instrument.get_dispatcher(__name__)


class OpenAIClient(APIClient):
    """OpenAI兼容接口"""

    _client_id: Optional[str] = PrivateAttr()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._client_id = None

    @classmethod
    def class_name(cls) -> str:
        return "OpenAIClient"

    @classmethod
    def get_client_id(
        cls,
        api_key: str = "",
        base_url: str = "",
        api_version: str = "",
        model_name: str = "",
        model_type: str = "",
        **kwargs: Any,  # pyright: ignore
    ):
        hash_str = "key={}&url={}&version={}&model={}&type={}".format(
            api_key or "",
            base_url or "",
            api_version or "",
            model_name or "",
            model_type or "",
        )
        return md5(hash_str.encode("utf-8")).hexdigest()

    @property
    def client_id(self):
        if self._client_id is None:
            model_name = getattr(self, "model") if hasattr(self, "model") else getattr(self, "model_name")
            if isinstance(self, OpenAILikeEmbedding):
                model_type = "embedding"
            elif getattr(self, "is_chat_model", False):
                model_type = "chat"
            elif getattr(self, "is_function_calling_model", False):
                model_type = "function"
            else:
                model_type = ""
            self._client_id = self.get_client_id(
                api_key=self.api_key,
                base_url=self.api_base,
                api_version=self.api_version,
                model_name=model_name,
                model_type=model_type,
            )
        return self._client_id

    @property
    def base_url(self):
        return self.api_base

    @classmethod
    def _get_init_kwargs(cls, **kwargs: Any) -> dict[str, Any]:
        init_kwargs = {}
        init_kwargs["api_key"] = kwargs.pop("api_key", None)
        init_kwargs["api_base"] = kwargs.pop("api_base", None) or kwargs.pop("base_url", None)
        init_kwargs["api_version"] = kwargs.pop("api_version", None)

        if issubclass(cls, OpenAILike):
            init_kwargs["model"] = kwargs.pop("model", None) or kwargs.pop("model_name", None)
            init_kwargs["is_chat_model"] = kwargs.pop("is_chat_model", False)
            init_kwargs["is_function_calling_model"] = kwargs.pop("is_function_calling_model", False)
            init_kwargs["context_window"] = kwargs.pop("context_window", 32768)
            init_kwargs["max_tokens"] = kwargs.pop("max_tokens", None)
        else:
            init_kwargs["model_name"] = kwargs.pop("model", None) or kwargs.pop("model_name", None)
        init_kwargs.update(kwargs)

        if not init_kwargs.get("timeout"):
            init_kwargs["timeout"] = 180
        return init_kwargs

    @classmethod
    def from_dict(cls, **kwargs: Any) -> Self:
        return super().from_dict(cls._get_init_kwargs(**kwargs))

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:  # pyright: ignore
        raise TypeError(f"'complete' method NOT supported by client")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any  # pyright: ignore
    ) -> CompletionResponseGen:
        raise TypeError(f"'stream_complete' method NOT supported by client")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:  # pyright: ignore
        raise TypeError(f"'chat' method NOT supported by client")

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:  # pyright: ignore
        raise TypeError(f"'stream_chat' method NOT supported by client")

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any  # pyright: ignore
    ) -> CompletionResponse:
        raise TypeError(f"'acomplete' method NOT supported by client")

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any  # pyright: ignore
    ) -> CompletionResponseAsyncGen:
        raise TypeError(f"'astream_complete' method NOT supported by client")

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:  # pyright: ignore
        raise TypeError(f"'achat' method NOT supported by client")

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any  # pyright: ignore
    ) -> ChatResponseAsyncGen:
        raise TypeError(f"'astream_chat' method NOT supported by client")

    def get_query_embedding(self, query: str) -> Embedding:  # pyright: ignore
        raise TypeError(f"'get_query_embedding' method NOT supported by client")

    def get_text_embedding(self, text: str) -> Embedding:  # pyright: ignore
        raise TypeError(f"'get_text_embedding' method NOT supported by client")

    def get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any  # pyright: ignore
    ) -> list[Embedding]:
        raise TypeError(f"'get_text_embedding_batch' method NOT supported by client")

    async def aget_query_embedding(self, query: str) -> Embedding:  # pyright: ignore
        raise TypeError(f"'aget_query_embedding' method NOT supported by client")

    async def aget_text_embedding(self, text: str) -> Embedding:  # pyright: ignore
        raise TypeError(f"'aget_text_embedding' method NOT supported by client")

    async def aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False  # pyright: ignore
    ) -> list[Embedding]:
        raise TypeError(f"'aget_text_embedding_batch' method NOT supported by client")


class OpenAIChatClient(OpenAILike, OpenAIClient):
    """OpenAI Chat接口"""

    @classmethod
    def class_name(cls) -> str:
        return "OpenAIChatClient"

    def _use_chat_completions(self, kwargs: dict[str, Any]) -> bool:
        if "use_chat_completions" in kwargs:
            return kwargs.pop("use_chat_completions")
        return self.metadata.is_chat_model

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(**kwargs)
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        if (
            self.is_chat_model
            and model_kwargs.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is None
        ):
            if "extra_body" not in model_kwargs:
                model_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            elif "chat_template_kwargs" not in model_kwargs["extra_body"]:
                model_kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                model_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = False
        return model_kwargs

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        if "use_chat_completions" not in kwargs:
            kwargs["use_chat_completions"] = True

        return super().complete(prompt, formatted=formatted, **kwargs)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        if "use_chat_completions" not in kwargs:
            kwargs["use_chat_completions"] = True

        return super().stream_complete(prompt, formatted=formatted, **kwargs)

    def chat(
        self, messages: Sequence[Union[ChatMessage, dict[str, Any]]], use_native: bool = False, **kwargs: Any
    ) -> Union[ChatCompletion, ChatResponse]:
        if use_native and not any(isinstance(x, ChatMessage) for x in messages):
            client = self._get_client()
            return client.chat.completions.create(messages=messages, **self._get_model_kwargs(**kwargs))

        return super().chat(format_messages(messages), **kwargs)

    def stream_chat(
        self, messages: Sequence[Union[ChatMessage, dict[str, Any]]], use_native: bool = False, **kwargs: Any
    ) -> Union[Stream[ChatCompletionChunk], ChatResponseGen]:
        if use_native and not any(isinstance(x, ChatMessage) for x in messages):
            client = self._get_client()
            return client.chat.completions.create(messages=messages, stream=True, **self._get_model_kwargs(**kwargs))

        return super().stream_chat(format_messages(messages), **kwargs)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any  # pyright: ignore
    ) -> CompletionResponse:
        if not self.metadata.is_chat_model and "use_chat_completions" not in kwargs:
            kwargs["use_chat_completions"] = True

        return await super().acomplete(prompt, formatted=formatted, **kwargs)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any  # pyright: ignore
    ) -> CompletionResponseAsyncGen:
        if not self.metadata.is_chat_model and "use_chat_completions" not in kwargs:
            kwargs["use_chat_completions"] = True

        return await super().astream_complete(prompt, formatted=formatted, **kwargs)

    async def achat(
        self, messages: Sequence[Union[ChatMessage, dict[str, Any]]], use_native: bool = False, **kwargs: Any
    ) -> Union[ChatCompletion, ChatResponse]:
        if use_native and not any(isinstance(x, ChatMessage) for x in messages):
            client = self._get_aclient()
            return await client.chat.completions.create(messages=messages, **self._get_model_kwargs(**kwargs))

        return await super().achat(format_messages(messages), **kwargs)

    async def astream_chat(
        self, messages: Sequence[Union[ChatMessage, dict[str, Any]]], use_native: bool = False, **kwargs: Any
    ) -> Union[AsyncStream[ChatCompletionChunk], ChatResponseAsyncGen]:
        if use_native and not any(isinstance(x, ChatMessage) for x in messages):
            client = self._get_aclient()
            return await client.chat.completions.create(
                messages=messages, stream=True, **self._get_model_kwargs(**kwargs)
            )

        return await super().astream_chat(format_messages(messages), **kwargs)

    def _separate_kwargs(self, prompt: BasePromptTemplate, **kwargs: Any):
        prompt_args = {k: v for k, v in kwargs.items() if k in prompt.template_vars}
        # NOTE: 没有过滤非法参数，可能导致OpenAI接口调用失败
        model_kwargs = {k: v for k, v in kwargs.items() if k not in prompt.template_vars}
        return prompt_args, model_kwargs

    @dispatcher.span
    def predict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
        prompt_args, model_kwargs = self._separate_kwargs(prompt, **kwargs)
        dispatcher.event(LLMPredictStartEvent(template=prompt, template_args=prompt_args))
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.chat(messages, **model_kwargs)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = self.complete(formatted_prompt, formatted=True, **model_kwargs)
            output = response.text
        parsed_output = self._parse_output(output)
        dispatcher.event(LLMPredictEndEvent(output=parsed_output))
        return parsed_output

    @dispatcher.span
    def stream(self, prompt: BasePromptTemplate, **kwargs: Any) -> TokenGen:
        prompt_args, model_kwargs = self._separate_kwargs(prompt, **kwargs)
        self._log_template_data(prompt, **prompt_args)

        dispatcher.event(LLMPredictStartEvent(template=prompt, template_args=prompt_args))
        if self.metadata.is_chat_model:
            messages = to_openai_message_dicts(self._get_messages(prompt, **prompt_args))
            chat_response = self.stream_chat(messages, use_native=True, **model_kwargs)
            stream_tokens = stream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = self.stream_complete(formatted_prompt, formatted=True, **model_kwargs)
            stream_tokens = stream_completion_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens

    @dispatcher.span
    async def apredict(self, prompt: BasePromptTemplate, **kwargs: Any) -> str:
        prompt_args, model_kwargs = self._separate_kwargs(prompt, **kwargs)
        self._log_template_data(prompt, **prompt_args)

        dispatcher.event(LLMPredictStartEvent(template=prompt, template_args=prompt_args))
        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = await self.achat(messages, **model_kwargs)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = await self.acomplete(formatted_prompt, formatted=True, **model_kwargs)
            output = response.text

        parsed_output = self._parse_output(output)
        dispatcher.event(LLMPredictEndEvent(output=parsed_output))
        return parsed_output

    @dispatcher.span
    async def astream(self, prompt: BasePromptTemplate, **kwargs: Any) -> TokenAsyncGen:
        prompt_args, model_kwargs = self._separate_kwargs(prompt, **kwargs)
        self._log_template_data(prompt, **prompt_args)

        dispatcher.event(LLMPredictStartEvent(template=prompt, template_args=prompt_args))
        if self.metadata.is_chat_model:
            messages = to_openai_message_dicts(self._get_messages(prompt, **prompt_args))
            chat_response = await self.astream_chat(messages, use_native=True, **model_kwargs)
            stream_tokens = await astream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = await self.astream_complete(formatted_prompt, formatted=True, **model_kwargs)
            stream_tokens = await astream_completion_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens

    def structured_predict(self, *args: Any, **kwargs: Any) -> Model:  # pyright: ignore
        raise_for_unimplemented("structured_predict")

    async def astructured_predict(self, *args: Any, **kwargs: Any) -> Model:  # pyright: ignore
        raise_for_unimplemented("astructured_predict")

    def stream_structured_predict(
        self, *args: Any, **kwargs: Any  # pyright: ignore
    ) -> Generator[Union[Model, "FlexibleModel"], None, None]:
        raise_for_unimplemented("stream_structured_predict")

    async def astream_structured_predict(
        self, *args: Any, **kwargs: Any  # pyright: ignore
    ) -> AsyncGenerator[Union[Model, "FlexibleModel"], None]:
        raise_for_unimplemented("astream_structured_predict")

    def predict_and_call(self, *args: Any, **kwargs: Any) -> "AgentChatResponse":  # pyright: ignore
        raise_for_unimplemented("predict_and_call")

    async def apredict_and_call(self, *args: Any, **kwargs: Any) -> "AgentChatResponse":  # pyright: ignore
        raise_for_unimplemented("apredict_and_call")


class OpenAIEmbeddingClient(OpenAILikeEmbedding, OpenAIClient):
    """OpenAI Embedding接口"""

    @classmethod
    def class_name(cls) -> str:
        return "OpenAIEmbeddingClient"

    @classmethod
    def _get_init_kwargs(cls, **kwargs: Any) -> dict[str, Any]:
        init_kwargs = super()._get_init_kwargs(**kwargs)

        if not init_kwargs.get("additional_kwargs"):
            init_kwargs["additional_kwargs"] = {}
        if not init_kwargs["additional_kwargs"].get("extra_body"):
            init_kwargs["additional_kwargs"]["extra_body"] = {}
        if "truncate_prompt_tokens" not in init_kwargs["additional_kwargs"]["extra_body"]:
            init_kwargs["additional_kwargs"]["extra_body"]["truncate_prompt_tokens"] = -1
        return init_kwargs
