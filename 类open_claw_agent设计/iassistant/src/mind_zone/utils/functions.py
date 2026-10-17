#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import socket
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import httpx
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import BaseComponent, Document, NodeWithScore, TextNode
from llama_index.llms.openai.utils import from_openai_message_dict

from mind_zone.utils.logging import Logger


def get_host_addr():
    _socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ip_addr = None
    try:
        _socket.connect(("8.8.8.8", 80))
        ip_addr = _socket.getsockname()[0]
    except Exception:
        ip_addr = "127.0.0.1"
    finally:
        _socket.close()

    return ip_addr


def infer_media_type(file_ext: str, return_type: Optional[str] = None):
    if return_type and return_type.upper() == "MIME":
        type_dict = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }
        return type_dict.get(file_ext, "image/jpeg")
    else:
        type_dict = {
            ".aac": "audio",
            ".mp3": "audio",
            ".oga": "audio",
            ".wav": "audio",
            ".weba": "audio",
            ".avi": "video",
            ".mp4": "video",
            ".mpeg": "video",
            ".ogv": "video",
            ".webm": "video",
            ".bmp": "image",
            ".gif": "image",
            ".jpeg": "image",
            ".jpg": "image",
            ".png": "image",
            ".svg": "image",
            ".tif": "image",
            ".tiff": "image",
            ".webp": "image",
        }
        return type_dict.get(file_ext, "file")


def download_file(base_url: str) -> str:
    download_info = None
    with httpx.Client() as client:
        response = client.get(base_url)
        download_info = response.json()

    file_name = base_url.split("?")[-1].rsplit("/", 1)[-1]
    if not download_info:
        Logger.error("Download file {} failed".format(file_name))

    with httpx.Client() as client:
        response = client.get(download_info["data"])
        file_path = Path(os.getenv("MZ_CACHE_DIR", "/tmp")) / file_name
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path


async def upload_file(base_url: str, *, file_path: Union[str, Path], purpose: str) -> str:
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        base_url = "http://" + base_url
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    upload_info = None
    with httpx.Client() as client:
        response = client.get(base_url, params={"name": file_path.name, "purpose": purpose})
        upload_info = response.json().get("data", {})

    if not upload_info:
        Logger.error("Upload file {} failed".format(file_path.name))

    async with httpx.AsyncClient() as client:
        await client.post(
            upload_info["uploadUrl"],
            data=upload_info["authParams"],
            files={"file": (upload_info["newFileName"], open(str(file_path), "rb"))},
        )
        return upload_info["relativePath"]


def format_messages(messages: Sequence[Union[ChatMessage, dict[str, Any]]]) -> list[ChatMessage]:
    fmt_messages = []
    for msg in messages:
        if isinstance(msg, ChatMessage):
            fmt_messages.append(msg)
            continue

        if not isinstance(msg, dict) or not msg.get("role") or not isinstance(msg.get("content"), (str, list)):
            raise ValueError("Invalid message format as OpenAI chat API")

        if isinstance(msg["content"], list) and any(x.get("type") != "text" for x in msg["content"]):
            raise RuntimeError("ChatMessage only support text messages, use native API for other input types")

        fmt_messages.append(from_openai_message_dict(msg))
    return fmt_messages


def cast_from_dict(data: dict[str, Any], type_name: Optional[str] = None) -> BaseComponent:
    type_name = type_name or data.get("class_name")
    match type_name:
        case "NodeWithScore":
            return NodeWithScore.from_dict(data, node=TextNode.from_dict(data["node"]))
        case "Document":
            return Document.from_dict(data)
        case _:
            raise TypeError(f"Class {type_name} has not been defined yet.")


def clean_extra_mark(text: str, mark: str = "markdown"):
    if text.count("```") >= 2:
        text = text.rsplit("```", 1)[0].replace("```", "").strip()
    if text.startswith(mark):
        text = text[len(mark) :].strip()
    return text


def truncate_repr(obj: Any, max_size: int = 64, depth: int = 0) -> Any:
    if not obj or isinstance(obj, (int, float, bool)):
        return obj

    new_obj = None
    if isinstance(obj, (str, bytes)):
        if len(obj) > max_size:
            return obj[:max_size] + ("..." if isinstance(obj, str) else b"...")
        return obj

    elif isinstance(obj, (set, list, tuple)):
        new_obj = []
        for item in obj:
            new_obj.append(truncate_repr(item, max_size=max(10, int(0.75 * max_size)), depth=depth + 1))
            if depth > 0 and len(new_obj) == 2:
                new_obj.append("...")
                break
        return new_obj

    elif isinstance(obj, dict):
        new_obj = type(obj)()
        for key, val in obj.items():
            if depth > 0 and len(new_obj) == 2:
                new_obj[key] = "..."
                break
            new_obj[key] = truncate_repr(val, max_size=max(10, int(0.75 * max_size)), depth=depth + 1)
        return new_obj

    else:
        return deepcopy(obj)
