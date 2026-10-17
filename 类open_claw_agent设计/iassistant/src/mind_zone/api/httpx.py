#!/usr/bin/env python
# -*-coding: utf-8-*-

import typing
from contextlib import asynccontextmanager, contextmanager
from hashlib import md5
from typing import Any, Optional

import httpx
from httpx import URL, USE_CLIENT_DEFAULT, Request, Response
from httpx._client import UseClientDefault
from httpx._types import (
    AuthTypes,
    CookieTypes,
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestExtensions,
    RequestFiles,
    TimeoutTypes,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from mind_zone.api.types import APIClient


class HTTPxClient(APIClient):
    """HTTPx兼容接口"""

    base_url: str = Field(description="The endpoint of service to use.")
    token: Optional[str] = Field(default=None, description="The access token to call the service.")
    default_headers: Optional[dict[str, str]] = Field(default=None, description="The default headers for API requests.")
    timeout: float = Field(default=30.0, description="The timeout, in seconds, for API requests.", ge=0)
    reuse_client: bool = Field(
        default=True,
        description=(
            "Reuse the OpenAI client between requests. When doing anything with large "
            "volumes of async API calls, setting this to false can improve stability."
        ),
    )

    _client_id: Optional[str] = PrivateAttr()
    _client: Optional[httpx.Client] = PrivateAttr()
    _aclient: Optional[httpx.AsyncClient] = PrivateAttr()

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        default_headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
        reuse_client: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            token=token,
            default_headers=default_headers,
            timeout=timeout,
            reuse_client=reuse_client,
            **kwargs,
        )

        self._client_id = None
        self._client = None
        self._aclient = None

    @classmethod
    def class_name(cls) -> str:
        return "HTTPxClient"

    @classmethod
    def get_client_id(cls, base_url: str = "", token: str = "", **kwargs: Any):  # pyright: ignore
        hash_str = "http_url={}&token={}".format(base_url or "", token or "")
        return md5(hash_str.encode("utf-8")).hexdigest()

    @property
    def client_id(self):
        if self._client_id is None:
            self._client_id = self.get_client_id(self.base_url, self.token)
        return self._client_id

    def _get_credential_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "headers": self.default_headers,
            "auth": None,
            "timeout": self.timeout,
        }

    def _get_client(self) -> httpx.Client:
        if not self.reuse_client:
            return httpx.Client(**self._get_credential_kwargs())

        if self._client is None:
            self._client = httpx.Client(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self) -> httpx.AsyncClient:
        if not self.reuse_client:
            return httpx.AsyncClient(**self._get_credential_kwargs())

        if self._aclient is None:
            self._aclient = httpx.AsyncClient(**self._get_credential_kwargs())
        return self._aclient

    def __getattribute__(self, name: str):
        method_list = ["request", "stream", "send", "get", "options", "head", "post", "put", "patch", "delete", "close"]
        if name in method_list:

            def sync_fn(*args: Any, **kwargs: Any):
                client = object.__getattribute__(self, "_get_client")()
                return getattr(client, name)(*args, **kwargs)

            return sync_fn

        elif name in [f"a{x}" for x in method_list]:

            async def async_fn(*args: Any, **kwargs: Any):
                client = object.__getattribute__(self, "_get_aclient")()
                if name == "aclose":
                    return await getattr(client, name)(*args, **kwargs)
                else:
                    return await getattr(client, name[1:])(*args, **kwargs)

            return async_fn
        else:
            return object.__getattribute__(self, name)

    def request(
        self,
        method: str,  # pyright: ignore
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    @contextmanager
    def stream(
        self,
        method: str,  # pyright: ignore
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> typing.Iterator[Response]:
        pass

    def send(
        self,
        request: Request,  # pyright: ignore
        *,
        stream: bool = False,  # pyright: ignore
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
    ) -> Response:
        pass

    def get(
        self,
        url: URL | str,  # pyright: ignore
        *,
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    def options(
        self,
        url: URL | str,  # pyright: ignore
        *,
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    def head(
        self,
        url: URL | str,  # pyright: ignore
        *,
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    def post(
        self,
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    def put(
        self,
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    def patch(
        self,
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    def delete(
        self,
        url: URL | str,  # pyright: ignore
        *,
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    def close(self) -> None:
        pass

    async def arequest(
        self,
        method: str,  # pyright: ignore
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    @asynccontextmanager
    async def astream(
        self,
        method: str,  # pyright: ignore
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> typing.AsyncIterator[Response]:
        pass

    async def asend(
        self,
        request: Request,  # pyright: ignore
        *,
        stream: bool = False,  # pyright: ignore
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
    ) -> Response:
        pass

    async def aget(
        self,
        url: URL | str,  # pyright: ignore
        *,
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    async def aoptions(
        self,
        url: URL | str,  # pyright: ignore
        *,
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    async def ahead(
        self,
        url: URL | str,  # pyright: ignore
        *,
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    async def apost(
        self,
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    async def aput(
        self,
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    async def apatch(
        self,
        url: URL | str,  # pyright: ignore
        *,
        content: RequestContent | None = None,  # pyright: ignore
        data: RequestData | None = None,  # pyright: ignore
        files: RequestFiles | None = None,  # pyright: ignore
        json: typing.Any | None = None,  # pyright: ignore
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    async def adelete(
        self,
        url: URL | str,  # pyright: ignore
        *,
        params: QueryParamTypes | None = None,  # pyright: ignore
        headers: HeaderTypes | None = None,  # pyright: ignore
        cookies: CookieTypes | None = None,  # pyright: ignore
        auth: AuthTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,  # pyright: ignore
        extensions: RequestExtensions | None = None,  # pyright: ignore
    ) -> Response:
        pass

    async def aclose(self) -> None:
        pass
