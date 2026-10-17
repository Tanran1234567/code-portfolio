#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import sys
from pathlib import Path

import yaml

WORK_DIR = Path(__file__).parent.parent.absolute()
PY_VERSION = next(WORK_DIR.joinpath(".venv/lib").iterdir()).name
sys.path.insert(0, str(WORK_DIR / ".venv" / "lib" / PY_VERSION / "site-packages"))

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=WORK_DIR / "config" / ".env.{}".format(os.getenv("MZ_ENV", "dev")),
    override=True,
    verbose=True,
)
os.environ["MZ_SVC_NAME"] = "iassistant"
os.environ["MZ_CACHE_DIR"] = str(WORK_DIR / ".cache")
Path(os.getenv("MZ_CACHE_DIR")).mkdir(parents=True, exist_ok=True)

from mind_zone.config import load_prompts
from mind_zone.state import *
from mind_zone.tasks import *

load_prompts(WORK_DIR / "config" / "prompts")
Broker.flush_all()

with open(WORK_DIR / "config" / "config.yaml", "r", encoding="utf-8") as f:
    conf_data = yaml.safe_load(f.read())
    get_task_manager(
        **conf_data.get("task-mgr", {}).get(os.getenv("MZ_ENV", "dev"), {}), password=os.getenv("MZ_MONGODB_PASSWD")
    )
