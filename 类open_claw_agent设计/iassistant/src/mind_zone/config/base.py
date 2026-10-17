#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
from pathlib import Path
from typing import Any, Union

import yaml

from mind_zone.config.utils import load_applications, load_prompts
from mind_zone.utils import Logger


class Config:
    """全局配置类"""

    def __init__(
        self,
        *,
        task_mgr_config: dict[str, Any],
        opensearch_config: dict[str, Any],
        milvus_config: dict[str, Any],
    ):
        self.task_mgr = task_mgr_config
        self.opensearch = opensearch_config
        self.milvus = milvus_config

    def get_config(self, db_type: str = "opensearch") -> Union[str, dict[str, Any]]:
        if any(x == "opensearch" for x in db_type.split(".")):
            return {
                "endpoint": "https://{}:{}".format(self.opensearch["host"], self.opensearch["port"]),
                "http_auth": (self.opensearch["username"], os.getenv("MZ_OPENSEARCH_PASSWD")),
                "http_compress": True,
                "use_ssl": True,
                "verify_certs": False,
            }
        elif any(x == "milvus" for x in db_type.split(".")):
            return {"endpoint": "http://{}:{}".format(self.milvus["host"], self.milvus["port"])}
        else:
            Logger.error(f"Cannot determine the type of vector database from {db_type}")
        return

    @classmethod
    def from_yaml(cls, conf_file: Union[str, Path]):
        if not isinstance(conf_file, Path):
            conf_file = Path(conf_file)

        with open(conf_file, "r", encoding="utf-8") as f:
            conf_data = yaml.safe_load(f.read())

        env = os.getenv("MZ_ENV", "dev")
        instance = Config(
            task_mgr_config=conf_data.get("task-mgr", {}).get(env, {}),
            opensearch_config=conf_data.get("opensearch", {}).get(env, {}),
            milvus_config=conf_data.get("milvus", {}).get(env, {}),
        )

        load_prompts(conf_file.with_name("prompts"))
        load_applications(conf_file.with_name("applications"))
        return instance
