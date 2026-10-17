#!/usr/bin/env python
# -*-coding: utf-8-*-

from llama_index.core.extractors import (
    BaseExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)

from mind_zone.register import Registry

Registry.Component.register("extract.title.default")(TitleExtractor)
Registry.Component.register("extract.keyword.default")(KeywordExtractor)
Registry.Component.register("extract.summary.default")(SummaryExtractor)
Registry.Component.register("extract.question.default")(QuestionsAnsweredExtractor)


class TextAPIExtractor(BaseExtractor):
    """调用API提取文档元数据"""

    pass
