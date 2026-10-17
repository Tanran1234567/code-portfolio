#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
from pathlib import Path
from typing import Any, Optional

import httpx
from fsspec import AbstractFileSystem
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.core.schema import MediaResource
from llama_index.readers.file import (
    DocxReader,
    ImageReader,
    PandasCSVReader,
    PandasExcelReader,
    PDFReader,
    PptxReader,
    VideoAudioReader,
)

from mind_zone.api import HTTPResponse, HTTPxClient
from mind_zone.base.utils import get_num_pages
from mind_zone.register import Registry
from mind_zone.utils import infer_media_type


def get_metadata(*, file_path: str):
    metadata = default_file_metadata_func(file_path)
    exclude_keys = ["file_path", "creation_date", "last_modified_date", "last_accessed_date"]
    metadata = {k: v for k, v in metadata.items() if k not in exclude_keys}
    return metadata


Registry.Component.register("read_file.pdf.default")(PDFReader)
Registry.Component.register("read_file.docx.default")(DocxReader)
Registry.Component.register("read_file.pptx.default")(PptxReader)
Registry.Component.register("read_file.ppt.default")(PptxReader)
Registry.Component.register("read_file.pptm.default")(PptxReader)
Registry.Component.register("read_file.gif.default")(ImageReader)
Registry.Component.register("read_file.png.default")(ImageReader)
Registry.Component.register("read_file.jpg.default")(ImageReader)
Registry.Component.register("read_file.jpeg.default")(ImageReader)
Registry.Component.register("read_file.webp.default")(ImageReader)
Registry.Component.register("read_file.mp3.default")(VideoAudioReader)
Registry.Component.register("read_file.mp4.default")(VideoAudioReader)
Registry.Component.register("read_file.csv.default")(PandasCSVReader)
Registry.Component.register("read_file.xls.default")(PandasExcelReader)
Registry.Component.register("read_file.xlsx.default")(PandasExcelReader)


@Registry.Component.register("read_file.md.default")
@Registry.Component.register("read_file.txt.default")
class PlainTextReader(BaseReader):
    """解析文本文件"""

    def load_data(
        self, file: Path, extra_info: Optional[dict] = None, fs: Optional[AbstractFileSystem] = None, **kwargs: Any
    ) -> list[Document]:
        """Parses the file."""
        if fs:
            with fs.open(str(file)) as f:
                text = f.read()
        else:
            with open(str(file), "r", **kwargs) as f:
                text = f.read()
        metadata = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)

        return [Document(text_resource=MediaResource(text=text), metadata=metadata or {})]


@Registry.Component.register(f"read_file.pdf.text-api")
@Registry.Component.register(f"read_file.docx.text-api")
@Registry.Component.register(f"read_file.pptx.text-api")
@Registry.Component.register(f"read_file.ppt.text-api")
@Registry.Component.register(f"read_file.pptm.text-api")
@Registry.Component.register(f"read_file.xls.text-api")
@Registry.Component.register(f"read_file.xlsx.text-api")
class TextAPIReader(BaseReader):
    """调用API解析文件"""

    def load_data(
        self, api_url: str, input_file: Path, extra_info: Optional[dict] = None, **kwargs: Any
    ) -> list[Document]:
        media_type = infer_media_type(input_file.suffix)
        with httpx.Client(timeout=180) as client:
            response = client.post(
                api_url,
                data=dict(
                    file_type=input_file.suffix,
                    return_meta=kwargs.get("return_meta", False),
                    return_text=kwargs.get("return_text", True),
                    return_image=kwargs.get("return_image", False),
                ),
                files={media_type: (input_file.name, input_file.open("rb"))},
            )
            return self._extract_content(response, extra_info)

    def _extract_content(self, response: HTTPResponse, extra_info: Optional[dict] = None) -> list[Document]:
        resp_json = response.json()
        if resp_json["status"] != 0:
            raise RuntimeError("cannot read content from file")

        metadata = extra_info or {}
        if resp_json.get("metadata"):
            metadata.update(resp_json["metadata"])

        return [Document(text_resource=MediaResource(text=resp_json["content"]), metadata=metadata)]


@Registry.Component.register(f"read_file.pdf.mineru")
@Registry.Component.register(f"read_file.png.mineru")
@Registry.Component.register(f"read_file.jpeg.mineru")
@Registry.Component.register(f"read_file.jpg.mineru")
class MinerUReader(BaseReader):
    """调用API解析文件"""

    def __init__(self, client: HTTPxClient, backend: str, **kwargs: Any) -> None:  # pyright: ignore
        self._client = client
        self._client.timeout = 180
        self._backend = backend
        self._batch_size = 20

    def load_data(self, input_file: Path, extra_info: Optional[dict] = None, **kwargs: Any) -> list[Document]:
        assert (kwargs.get("backend") or self._backend) == self._backend, "Backend {} NOT supported by {}".format(
            kwargs["backend"], type(self).__name__
        )
        content = None
        output_dir = Path(os.getenv("MZ_CACHE_DIR", "/tmp"), "mineru")
        output_dir.mkdir(parents=True, exist_ok=True)
        parse_opts = self._get_parse_opts(**kwargs)

        if (
            input_file.suffix != ".pdf"
            or input_file.stat().st_size < 512000
            or (num_pages := get_num_pages(input_file)) < self._batch_size
        ):
            content = self._parse_content(input_file, output_dir, parse_opts)
        else:
            content = ""
            for i in range((num_pages + self._batch_size - 1) // self._batch_size):
                start_page_id = i * self._batch_size
                end_page_id = (i + 1) * self._batch_size
                md_content = self._parse_content(
                    input_file, output_dir, parse_opts, start_page_id=start_page_id, end_page_id=end_page_id
                )
                content += "\n\n" * int(md_content.startswith("#")) + md_content

        return [Document(text_resource=MediaResource(text=content), metadata=extra_info)]

    def _get_parse_opts(self, **kwargs: Any):
        parse_opts = dict(
            parse_method=kwargs.get("parse_method", "auto"),
            formula_enable=kwargs.get("formula_enable", False),
            table_enable=kwargs.get("table_enable", True),
            lang_list=kwargs.get("lang_list", ["ch"]),
            return_md=kwargs.get("return_md", False),
            return_images=kwargs.get("return_images", False),
            return_middle_json=kwargs.get("return_middle_json", False),
            return_model_output=kwargs.get("return_model_output", False),
            return_content_list=kwargs.get("return_content_list", False),
        )
        return parse_opts

    def _parse_content(
        self,
        input_file: Path,
        output_dir: Path,
        parse_opts: dict[str, Any],
        start_page_id: int = 0,
        end_page_id: int = 99999,
    ):
        response = self._client.post(
            self._client.base_url,
            json={
                "output_dir": str(output_dir),
                "backend": self._backend,
                **parse_opts,
                "start_page_id": start_page_id,
                "end_page_id": end_page_id,
            },
            files=[("files", (input_file.name, input_file.open("rb")))],
        )

        if response.status_code != 200:
            raise RuntimeError("cannot parse file using MinerU API")
        content = response.json()["results"].get(input_file.stem, {}).get("md_content", "")
        return content
