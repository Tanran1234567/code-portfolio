#!/usr/bin/env python
# -*-coding: utf-8-*-

from pathlib import Path
from typing import Union

import yaml
from llama_index.core.prompts import PromptTemplate

from mind_zone.register import Registry
from mind_zone.utils import Logger


def load_prompts(root_path: Union[str, Path]):
    if not isinstance(root_path, Path):
        root_path = Path(root_path)

    for file in list(root_path.glob("*.yaml")) + list(root_path.glob("*.yml")):
        with open(file, "r", encoding="utf-8") as f:
            for data in yaml.safe_load_all(f.read()):
                name = data.get("name")
                if not name or name in Registry.Template:
                    Logger.error(f"Template '{name}' already exists, please check")
                    continue

                if not data.get("template"):
                    Logger.error(f"Template '{name}' is empty, please check")
                    continue

                Registry.Template[name] = PromptTemplate(data["template"])
    return


def load_applications(root_path: Union[str, Path]):
    if not isinstance(root_path, Path):
        root_path = Path(root_path)

    for file in list(root_path.glob("*.yaml")) + list(root_path.glob("*.yml")):
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f.read())
            name = data.get("name")
            if not name or name in Registry.Application:
                Logger.error(f"Application '{name}' already exists, please check")
                continue

            Registry.Application[name] = data.get("config", {})
    return
