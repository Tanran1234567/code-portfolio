#!/usr/bin/env python
# -*-coding: utf-8-*-

import logging
import os


def init_logger(name: str = "mind-zone", level: str = "INFO"):
    logger = logging.getLogger(name)
    level = (level or "INFO").upper()
    logger.setLevel(level)
    logger.propagate = False

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(level)

        if os.getenv("MZ_ENV", "dev") == "dev":
            message_fmt = "\033[4;32m%(levelname)s\t[%(asctime)s - %(filename)s:%(lineno)s]\033[0m  %(message)s"
        else:
            message_fmt = "%(levelname)s\t[%(asctime)s - %(filename)s:%(lineno)s]  %(message)s"
        handler.setFormatter(logging.Formatter(fmt=message_fmt, datefmt="%b %d %H:%M:%S"))
        logger.addHandler(handler)

    return logger


Logger = init_logger(os.getenv("MZ_SVC_NAME", "mind-zone"), os.getenv("MZ_LOG_LEVEL"))
