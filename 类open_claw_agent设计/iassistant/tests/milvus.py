#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import sys

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PY_VERSION = os.path.basename(os.listdir(os.path.join(WORK_DIR, "..", ".venv/lib"))[0])
sys.path.insert(0, os.path.abspath(os.path.join(WORK_DIR, "..", f".venv/lib/{PY_VERSION}/site-packages")))


from pymilvus import MilvusClient

client = MilvusClient("http://127.0.0.1:30230")
print(client)
print(client.query("iassistant_rag_index", filter='doc_id IN ["test-pdf-12345#chunk_0"]'))
