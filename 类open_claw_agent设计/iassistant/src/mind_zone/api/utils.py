#!/usr/bin/env python
# -*-coding: utf-8-*-

import inspect
from typing import Union, cast

from llama_index.core.llms.llm import (
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.types import TokenAsyncGen, TokenGen
from openai._streaming import AsyncStream, Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


def raise_for_unimplemented(method: str):
    raise NotImplementedError(f"Function {method} is NOT implemented, please contact the maintainer to accomplish")


def build_token_generator(
    response_gen: Union[CompletionResponseGen, ChatResponseGen, CompletionResponseAsyncGen, ChatResponseAsyncGen],
) -> Union[TokenGen, TokenAsyncGen]:
    if inspect.isgenerator(response_gen):

        def gen() -> TokenGen:
            is_reasoning_delta = False
            for response in response_gen:
                if reasoning_delta := response.additional_kwargs.get("thinking_delta"):
                    if not is_reasoning_delta:
                        is_reasoning_delta = True
                        yield "<think>\n" + reasoning_delta
                    else:
                        yield reasoning_delta
                elif content_delta := response.delta:
                    if is_reasoning_delta:
                        is_reasoning_delta = False
                        yield "</think>" + (content_delta or "")
                    else:
                        yield content_delta or ""

        return gen

    elif inspect.isasyncgen(response_gen):

        async def gen() -> TokenAsyncGen:
            is_reasoning_delta = False
            async for response in response_gen:
                if reasoning_delta := response.additional_kwargs.get("thinking_delta"):
                    if not is_reasoning_delta:
                        is_reasoning_delta = True
                        yield "<think>\n" + reasoning_delta
                    else:
                        yield reasoning_delta
                elif content_delta := response.delta:
                    if is_reasoning_delta:
                        is_reasoning_delta = False
                        yield "</think>" + (content_delta or "")
                    else:
                        yield content_delta or ""

        return gen

    return lambda: (yield from ())


def convert_raw_generator(
    response_gen: Union[Stream[ChatCompletionChunk], AsyncStream[ChatCompletionChunk]],
) -> Union[TokenGen, TokenAsyncGen]:
    if isinstance(response_gen, Stream):

        def gen() -> TokenGen:
            is_reasoning_delta = False
            for response in response_gen:
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) <= 0:
                    continue

                delta = response.choices[0].delta
                content_delta = delta.content or ""
                raw_reasoning = getattr(delta, "reasoning", None)
                reasoning_delta = raw_reasoning if isinstance(raw_reasoning, str) else ""
                if reasoning_delta:
                    if not is_reasoning_delta:
                        is_reasoning_delta = True
                        yield "<think>\n" + reasoning_delta
                    else:
                        yield reasoning_delta
                elif content_delta:
                    if is_reasoning_delta:
                        is_reasoning_delta = False
                        yield "</think>" + content_delta
                    else:
                        yield content_delta

        return gen

    elif isinstance(response_gen, AsyncStream):

        async def gen() -> TokenAsyncGen:
            is_reasoning_delta = False
            async for response in response_gen:
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) <= 0:
                    continue

                delta = response.choices[0].delta
                content_delta = delta.content or ""
                raw_reasoning = getattr(delta, "reasoning", None)
                reasoning_delta = raw_reasoning if isinstance(raw_reasoning, str) else ""
                if reasoning_delta:
                    if not is_reasoning_delta:
                        is_reasoning_delta = True
                        yield "<think>\n" + reasoning_delta
                    else:
                        yield reasoning_delta
                elif content_delta:
                    if is_reasoning_delta:
                        is_reasoning_delta = False
                        yield "</think>" + content_delta
                    else:
                        yield content_delta

        return gen

    return lambda: (yield from ())


def stream_completion_response_to_tokens(completion_response_gen: CompletionResponseGen) -> TokenGen:
    """Refactor stream_completion_response_to_tokens to get reasoning contents."""
    return build_token_generator(completion_response_gen)()


def stream_chat_response_to_tokens(chat_response_gen: Union[Stream[ChatCompletionChunk], ChatResponseGen]) -> TokenGen:
    """Refactor stream_chat_response_to_tokens to get reasoning contents."""
    if isinstance(chat_response_gen, Stream):
        return convert_raw_generator(chat_response_gen)()
    else:
        return build_token_generator(chat_response_gen)()


async def astream_completion_response_to_tokens(completion_response_gen: CompletionResponseAsyncGen) -> TokenAsyncGen:
    """Refactor astream_completion_response_to_tokens to get reasoning contents."""
    return build_token_generator(completion_response_gen)()


async def astream_chat_response_to_tokens(
    chat_response_gen: Union[AsyncStream[ChatCompletionChunk], ChatResponseAsyncGen],
) -> TokenAsyncGen:
    """Refactor astream_chat_response_to_tokens to get reasoning contents."""
    if isinstance(chat_response_gen, AsyncStream):
        return convert_raw_generator(chat_response_gen)()
    else:
        return build_token_generator(chat_response_gen)()
