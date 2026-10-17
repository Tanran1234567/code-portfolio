#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import sys

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PY_VERSION = os.path.basename(os.listdir(os.path.join(WORK_DIR, "..", ".venv/lib"))[0])
sys.path.insert(0, os.path.abspath(os.path.join(WORK_DIR, "..", f".venv/lib/{PY_VERSION}/site-packages")))

from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url=f"http://127.0.0.1:30000/v1",
)

# model = "Qwen/Qwen3.5-122B-A10B-FP8"
# messages = [{"role": "user", "content": "数一数单词'strawberry'中有几个'r'字母"}]

# response = client.chat.completions.create(
#     model=model,
#     messages=messages,
#     extra_body={"chat_template_kwargs": {"enable_thinking": True}, "separate_reasoning": True},
# )
# print(response.choices[0].message)

client = OpenAI(
    api_key="",
    base_url="http://127.0.0.1:30080/v1",
)

response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-0.6B",
    input="你!" * 50000,
    extra_body={"truncate_prompt_tokens": -1},
)
print(response)
