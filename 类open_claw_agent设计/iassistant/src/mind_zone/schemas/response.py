#!/usr/bin/env python
# -*-coding: utf-8-*-

import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Generator, Optional, Union

from mind_zone.utils import Logger


class __InnerManager__:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Double-Checked Locking
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._err_dict = OrderedDict()
        return cls._instance

    def register(self, *, name: str, code: int, msg: str) -> None:
        if name in self._err_dict:
            Logger.error(f"Error {name} has been registered")
            return

        if any(d["code"] == code for d in self._err_dict.values()):
            tmp_name = next(k for k, v in self._err_dict.items() if v["code"] == code)
            Logger.error(f"Error code {code} has been registered by {tmp_name}")
            return
        self._err_dict[name] = {"code": code, "msg": msg}

    def __call__(
        self, err_name: str, success: bool = True, data: Optional[dict[str, Any]] = None, stream: bool = False
    ) -> Union[dict[str, Any], Callable[[], Generator]]:
        if err_name not in self._err_dict:
            Logger.error(f"Error {err_name} has NOT been registered")
            return

        code = self._err_dict[err_name]["code"]
        data_dict = {"err_code": code}
        if code != 0:
            data_dict["err_msg"] = self._err_dict[err_name]["msg"]
        if data:
            data_dict["data"] = data

        if not stream:
            data_dict["success"] = success
            return data_dict

        def wrap_fn():
            data_dict["finish"] = True
            yield json.dumps(data_dict, ensure_ascii=False) + "\n"

        return wrap_fn


FormatOutput = __InnerManager__()
FormatOutput.register(name="no-error", code=0, msg="")
# TODO: 补充错误信息
"""错误码：1xx，任务队列"""
FormatOutput.register(name="collect-res", code=101, msg="收集Workflow事件结果失败")
"""错误码：2xx，文件处理"""
FormatOutput.register(name="parse-file", code=201, msg="文件解析失败")
FormatOutput.register(name="file-not-found", code=202, msg="文件未找到")
"""错误码：3xx，服务调用"""
FormatOutput.register(name="call-llm-api", code=301, msg="调用API错误")
FormatOutput.register(name="split-doc", code=302, msg="文档分段错误")
FormatOutput.register(name="metadata", code=303, msg="提取元信息错误")
FormatOutput.register(name="embedding", code=304, msg="计算Embedding错误")
FormatOutput.register(name="store-doc", code=305, msg="文档存储错误")
FormatOutput.register(name="retrieval", code=306, msg="文档检索错误")
FormatOutput.register(name="postprocess", code=307, msg="检索结果后处理错误")
FormatOutput.register(name="vstore-del", code=308, msg="向量数据库删除数据失败")
FormatOutput.register(name="query-opt", code=309, msg="查询优化失败")
"""错误码：4xx，模型生成"""
"""错误码：5xx，结果合成"""
FormatOutput.register(name="json-format", code=502, msg="JSON格式错误")
"""错误码：9xx，结果合成"""
FormatOutput.register(name="runtime-err", code=901, msg="运行时错误")
FormatOutput.register(name="type-err", code=902, msg="类型错误")
FormatOutput.register(name="value-err", code=903, msg="值错误")
FormatOutput.register(name="index-err", code=904, msg="索引错误")
FormatOutput.register(name="os-err", code=905, msg="操作系统错误")
FormatOutput.register(name="syntax-err", code=906, msg="句法错误")
FormatOutput.register(name="other-err", code=999, msg="其它类型错误")
