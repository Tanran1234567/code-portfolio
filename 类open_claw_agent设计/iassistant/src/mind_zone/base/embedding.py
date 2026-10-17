#!/usr/bin/env python
# -*-coding: utf-8-*-

import asyncio
import re
from typing import Any, List

import numpy as np
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from mind_zone.api import APIClient, OpenAIEmbeddingClient
from mind_zone.register import Registry
from mind_zone.utils import Logger

Registry.Component.register("embed.text.default")(OpenAIEmbeddingClient)


@Registry.Component.register("embed.text.text-api")
class TextAPIEmbedding(BaseEmbedding):
    """调用API服务计算Embedding"""

    embed_dim: int

    def __init__(self, embed_dim: int, client: APIClient, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(embed_dim=embed_dim, **kwargs)

        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "TextAPIEmbedding"

    async def _aget_text_embedding(self, text: str) -> list[float]:  # pyright: ignore
        # TODO: 实现API计算Embedding的逻辑
        return self._get_vector()

    async def _aget_query_embedding(self, query: str) -> list[float]:  # pyright: ignore
        return self._get_vector()

    def _get_query_embedding(self, query: str) -> list[float]:  # pyright: ignore
        """Get query embedding."""
        return self._get_vector()

    def _get_text_embedding(self, text: str) -> list[float]:  # pyright: ignore
        """Get text embedding."""
        return self._get_vector()


class BaseTokenEmbedding:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_tokens(self, text: str) -> list[int]:
        raise NotImplementedError

    def get_tokens_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.get_tokens(text) for text in texts]

    async def aget_tokens(self, text: str) -> list[int]:
        return asyncio.to_thread(self.get_tokens, text)

    async def aget_tokens_batch(self, texts: list[str]) -> list[list[int]]:
        return await asyncio.gather(*[self.aget_tokens(text) for text in texts])

    def get_token_embeddings(self, tokens: list[int]) -> list[list[float]]:
        raise NotImplementedError

    def get_token_embeddings_batch(self, tokens_list: list[list[int]]) -> list[list[list[float]]]:
        return [self.get_token_embeddings(tokens) for tokens in tokens_list]

    async def aget_token_embeddings(self, tokens: list[int]) -> list[list[float]]:
        return await asyncio.to_thread(self.get_token_embeddings, tokens)

    async def aget_token_embeddings_batch(self, tokens_list: list[list[int]]) -> list[list[list[float]]]:
        return await asyncio.gather(*[self.aget_token_embeddings(tokens) for tokens in tokens_list])


class VLLMTokenEmbedding(BaseTokenEmbedding):
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        import httpx

        self._base_url = base_url.rstrip("/")
        self._model = model_name
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._aclient = httpx.AsyncClient(timeout=timeout)

    def get_tokens(self, text: str) -> list[int]:
        response = self._client.post(
            f"{self._base_url}/tokenize",
            json={"model": self._model, "prompt": text, "add_special_tokens": True},
        )  # TODO: add_special_tokens
        response.raise_for_status()
        result = response.json()
        return result.get("tokens", [])

    async def aget_tokens(self, text: str) -> list[int]:
        response = await self._aclient.post(
            f"{self._base_url}/tokenize",
            json={"model": self._model, "prompt": text, "add_special_tokens": True},
        )
        response.raise_for_status()
        result = response.json()
        """
        {
            'count': 123, 
            'max_model_len': 123, 
            'tokens': [1, ...], 
            'token_strs': ['']
        }
        """
        Logger.debug(f"Tokenized text: \n {text}\n\n{result}")
        return result.get("tokens", [])

    def get_token_embeddings(self, tokens: list[int]) -> list[list[float]]:
        if not tokens:
            return []
        response = self._client.post(
            f"{self._base_url}/pooling",
            json={
                "model": self._model,
                "input": tokens,
                "task": "token_embed",
            },
        )
        response.raise_for_status()
        result = response.json()
        """
        {
            "id": "pool-bce4799aa29a20a4",
            "object": "list",
            "created": 1773910772,
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "data": [
                {
                    "index": 0,
                    "object": "pooling",
                    "data": [
                        [0.030055, -0.00605, 0.00605, ..., -0.00605, 0.03005, 0.00605]
                        ...
                        [0.030055, -0.00605, 0.00605, ..., -0.00605, 0.03005, 0.00605]
                    ]
        """
        embeddings = result["data"][0]["data"]
        assert len(embeddings) == len(tokens)
        return embeddings

    async def aget_token_embeddings(self, tokens: list[int]) -> list[list[float]]:
        if not tokens:
            return []
        response = await self._aclient.post(
            f"{self._base_url}/pooling",
            json={
                "model": self._model,
                "input": tokens,
                "task": "token_embed",
            },
        )
        response.raise_for_status()
        result = response.json()
        embeddings = result["data"][0]["data"]
        assert len(embeddings) == len(tokens)
        return embeddings

    def close(self) -> None:
        if self._client:
            self._client.close()

    async def aclose(self) -> None:
        if self._aclient:
            await self._aclient.close()


@Registry.Component.register("embed.text.late-chunk")
class LateChunkEmbedding(OpenAILikeEmbedding):
    """
    实现了延迟分块
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout: float = 30.0,
        token_embedding: BaseTokenEmbedding = None,
        max_context_length: int = 8192,
        **kwargs: Any,
    ):
        """
        创建一个 LateChunkEmbedding 对象。
        Args:
            base_url: OpenAI Embedding 的基础 URL。
            model_name: 使用的模型名称。
            timeout: 请求超时时间。
            token_embedding: 用于获取 token embedding 的对象。
            max_context_length: 最大上下文长度。
        """
        super().__init__(
            model_name=model_name,
            api_base=base_url,
            timeout=timeout,
            **kwargs,
        )
        if token_embedding:
            self._token_embedding = token_embedding
        else:
            # openai 兼容的 api_base 格式为http://ip:port/v1
            # vllm pooling 和 tokenize 接口使用方式为http://ip:port/pooling
            token_embedding_base_url = re.sub(r"/v1(/.*)?$", "", base_url)
            self._token_embedding = VLLMTokenEmbedding(
                model_name=model_name,
                base_url=token_embedding_base_url,
                timeout=timeout,
                **kwargs,
            )
        self._max_context_length = max_context_length

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        tokens_list = self._token_embedding.get_tokens_batch(texts)

        chunks, positions = self._merge_tokens(tokens_list)

        chunk_embeddings = self._token_embedding.get_token_embeddings_batch(chunks)

        text_embeddings = self._split_embeddings(positions, chunk_embeddings)

        return text_embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[list[float]]:
        if not texts:
            return []

        tokens_list = await self._token_embedding.aget_tokens_batch(texts)
        Logger.debug(f"length of tokens_list: {len(tokens_list)}")

        chunks, positions = self._merge_tokens(tokens_list)
        Logger.debug(f"length of chunks: {len(chunks)}")

        chunk_embeddings = await self._token_embedding.aget_token_embeddings_batch(chunks)
        Logger.debug(f"length of chunk_embeddings: {len(chunk_embeddings)}")

        text_embeddings = self._split_embeddings(positions, chunk_embeddings)
        Logger.debug(f"length of text_embeddings: {len(text_embeddings)}")

        return text_embeddings

    def _merge_tokens(self, tokens_list: list[list[int]]) -> tuple[list[list[int]], list[tuple[int, int, int]]]:
        chunks = []
        positions = []
        current_chunk = []
        current_chunk_idx = 0
        current_chunk_len = 0

        for tokens in tokens_list:
            if not tokens:
                positions.append((current_chunk_idx, current_chunk_len, current_chunk_len))
                continue

            if len(current_chunk) + len(tokens) > self._max_context_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = []
                current_chunk_idx += 1
                current_chunk_len = 0

            start_offset = current_chunk_len
            current_chunk.extend(tokens)
            current_chunk_len = len(current_chunk)

            positions.append((current_chunk_idx, start_offset, current_chunk_len))

        if current_chunk:
            chunks.append(current_chunk)

        return chunks, positions

    def _split_embeddings(
        self,
        positions: list[tuple[int, int, int]],
        chunk_embeddings: list[list[list[float]]],
    ) -> list[list[float]]:
        text_embeddings = []
        for chunk_idx, start_offset, end_offset in positions:
            if start_offset == end_offset:
                text_embeddings.append([])
            else:
                embeddings = chunk_embeddings[chunk_idx][start_offset:end_offset]
                avg_embedding = np.mean(embeddings, axis=0).tolist()
                text_embeddings.append(avg_embedding)

        return text_embeddings
