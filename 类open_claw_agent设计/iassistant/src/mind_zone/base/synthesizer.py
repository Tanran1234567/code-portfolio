#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Optional, Type

from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.utils import is_chat_model
from llama_index.core.response_synthesizers import (
    Accumulate,
    BaseSynthesizer,
    CompactAndRefine,
    Generation,
    Refine,
    SimpleSummarize,
    TreeSummarize,
)
from llama_index.core.types import BaseModel

from mind_zone.base.prompts import (
    CHAT_TEXT_QA_PROMPT,
    CHAT_TREE_SUMMARIZE_PROMPT,
    DEFAULT_SIMPLE_INPUT_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_TREE_SUMMARIZE_PROMPT,
)
from mind_zone.register import Registry


@Registry.Component.register("synthesize.text.default")
@Registry.Component.register("synthesize.text.summarize")
class RespondBySummarize(SimpleSummarize):

    def __init__(
        self,
        llm: Optional[LLM] = None,
        prompt_helper: Optional[PromptHelper] = None,
        custom_template: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        default_text_qa_conditionals = [(is_chat_model, CHAT_TEXT_QA_PROMPT)]
        if not custom_template:
            text_qa_template = SelectorPromptTemplate(
                default_template=DEFAULT_TEXT_QA_PROMPT, conditionals=default_text_qa_conditionals
            )
        else:
            text_qa_template = SelectorPromptTemplate(
                default_template=PromptTemplate(custom_template, prompt_type=PromptType.QUESTION_ANSWER),
                conditionals=default_text_qa_conditionals,
            )

        super().__init__(llm=llm, prompt_helper=prompt_helper, text_qa_template=text_qa_template, streaming=streaming)


@Registry.Component.register("synthesize.text.generation")
class RespondWithoutContext(Generation):

    def __init__(
        self,
        llm: Optional[LLM] = None,
        prompt_helper: Optional[PromptHelper] = None,
        custom_template: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        if not custom_template:
            simple_template = DEFAULT_SIMPLE_INPUT_PROMPT
        else:
            simple_template = PromptTemplate(custom_template, prompt_type=PromptType.SIMPLE_INPUT)

        super().__init__(llm=llm, prompt_helper=prompt_helper, simple_template=simple_template, streaming=streaming)


@Registry.Component.register("synthesize.text.tree-summarize")
class RespondByTreeSummarize(TreeSummarize):

    def __init__(
        self,
        llm: Optional[LLM] = None,
        prompt_helper: Optional[PromptHelper] = None,
        custom_template: Optional[str] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        streaming: bool = False,
        use_async: bool = False,
        verbose: bool = False,
    ) -> None:
        default_tree_summarize_conditionals = [(is_chat_model, CHAT_TREE_SUMMARIZE_PROMPT)]
        if not custom_template:
            summary_template = SelectorPromptTemplate(
                default_template=DEFAULT_TREE_SUMMARIZE_PROMPT, conditionals=default_tree_summarize_conditionals
            )
        else:
            summary_template = SelectorPromptTemplate(
                default_template=PromptTemplate(custom_template, prompt_type=PromptType.SUMMARY),
                conditionals=default_tree_summarize_conditionals,
            )

        super().__init__(
            llm=llm,
            prompt_helper=prompt_helper,
            summary_template=summary_template,
            output_cls=output_cls,
            streaming=streaming,
            use_async=use_async,
            verbose=verbose,
        )


# TODO: 下面几种类型需要改造
Registry.Component.register("synthesize.text.refine")(Refine)
Registry.Component.register("synthesize.text.compact-refine")(CompactAndRefine)
Registry.Component.register("synthesize.text.accumulate")(Accumulate)


@Registry.Component.register("synthesize.text.text-api")
class TextAPISynthesizer(BaseSynthesizer):
    """调用API服务生成"""

    pass
