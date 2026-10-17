#!/usr/bin/env python
# -*-coding: utf-8-*-

from abc import abstractmethod

from mind_zone.register import Registry


class PreProcessor:
    """预处理模块"""

    @abstractmethod
    def process(self, text: str) -> str:
        raise NotImplementedError()


@Registry.Component.register("BuiltinPreProcessor")
class BuiltinPreProcessor(PreProcessor):
    """内置预处理模块"""

    def __init__(self) -> None:
        super().__init__()

    def process(self, text: str) -> str:
        return super().process(text)


@Registry.Component.register("TextAPIPreProcessor")
class TextAPIPreProcessor(PreProcessor):
    """调用API预处理输入文本"""

    def __init__(self) -> None:
        super().__init__()

    def process(self, text: str) -> str:
        return super().process(text)
