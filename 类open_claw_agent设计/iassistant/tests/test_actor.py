#!/usr/bin/env python
# -*-coding: utf-8-*-

import logging
import os
import random
import sys
import time

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PY_VERSION = os.path.basename(os.listdir(os.path.join(WORK_DIR, "../../..", ".venv/lib"))[0])
sys.path.insert(0, os.path.abspath(os.path.join(WORK_DIR, "../../..", f".venv/lib/{PY_VERSION}/site-packages")))

import dramatiq
from dotenv import load_dotenv
from dramatiq.middleware import CurrentMessage

load_dotenv(dotenv_path=os.path.join(WORK_DIR, "../../../config/.env"), override=True, verbose=True)

from broker import Broker

Broker.flush_all()
print(__file__, dramatiq.get_broker())


@dramatiq.actor(max_retries=0)
def process(item_id: int):
    """
    An actor that simulates work. Even when run concurrently, it will always log the correct message_id.
    """
    # Get the message for the CURRENT execution context
    message_id = CurrentMessage.get_current_message().message_id

    logging.info(f"START: Processing item {item_id} (Message ID: {message_id})")
    time.sleep(random.randint(1, 5))
    logging.info(f"END: Finished item {item_id} (Message ID: {message_id})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    for i in range(10):
        process.send(i)
