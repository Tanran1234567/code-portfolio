#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Optional

from llama_index.core.prompts.base import PromptTemplate

from mind_zone.types import BaseEnum, SkipDict
from mind_zone.utils import Logger


class UserGroupType(BaseEnum):
    GUEST = "guest"
    STANDARD = "standard"


class L1IntentType(BaseEnum):
    DIRECT = "direct"
    RAG = "rag"
    CHATBI = "chatbi"


# TODO: 改到配置文件中
__INTENT_TREE__ = SkipDict(
    direct=dict(
        _desc="无需检索，直接回答：闲聊问候、文本创作、逻辑推理、格式转换",
        chitchat={
            "_desc": "闲聊、问候、情绪表达，不涉及知识库",
            "template": PromptTemplate(
                "你是友好亲切的对话助手，用自然轻松的语气回应用户。\n"
                "保持简短，不过度解释，营造舒适的对话氛围。\n\n"
                "用户说：{query_str}"
            ),
        },
        text_writing={
            "_desc": "写邮件、写文案、内容创作",
            "template": PromptTemplate(
                "你是专业写作助手。\n"
                "请根据以下需求创作内容，注意识别所需文体（邮件/报告/文案等），"
                "确保语言流畅、结构清晰、符合使用场景。\n\n"
                "写作需求：{query_str}\n\n"
                "创作内容："
            ),
        },
        reasoning={
            "_desc": "逻辑推理、数学计算，LLM 直接回答",
            "template": PromptTemplate(
                "你是逻辑推理助手。\n"
                "请对以下问题进行系统分析，逐步展示推理过程，"
                "确保每个结论都有明确的依据，最终给出清晰可靠的答案。\n\n"
                "问题：{query_str}\n\n"
                "分析过程："
            ),
        },
        beyond_scope={
            "_desc": "超出服务范围，无法处理",
            "template": PromptTemplate(
                "非常抱歉，您的问题超出了当前服务范围，我暂时无法回答。\n"
                "建议您：\n"
                "1. 尝试换个方式重新描述问题\n"
                "2. 联系人工客服获取专业帮助\n\n"
                "您的问题：{query_str}"
            ),
        },
    ),
    rag=dict(
        _desc="需要检索知识库或私有数据：事实查询、产品信息、政策条款、订单账户数据、流程文档、状态追踪",
        fact_lookup={
            "_desc": "产品参数、技术规格、政策条款等事实性查询，答案在知识库中",
            "template": PromptTemplate(
                "你是专业的知识库问答助手。\n"
                "请严格依据以下检索内容回答用户问题，不得编造或推测检索内容之外的信息。\n"
                "若检索内容中没有相关答案，请明确告知用户'当前知识库中未找到相关信息'。\n"
                "[检索内容]\n{context_str}\n\n"
                "[用户问题]\n{query_str}\n\n"
                "请给出准确、简洁的回答："
            ),
        },
        comparison={
            "_desc": "多个对象的对比问题，如 A 和 B 哪个更好、有什么区别",
            "template": PromptTemplate(
                "你是结构化分析助手。\n"
                "请依据以下检索内容，对用户的对比问题进行系统分析。\n"
                "要求：以表格或分点形式呈现，每项对比结论须有检索内容支撑，不得主观臆断。\n\n"
                "[检索内容]\n{context_str}\n\n"
                "[对比问题]\n{query_str}\n\n"
                "请输出结构化对比结果："
            ),
        },
        # data_fetch={
        #     "_desc": "用户私有数据查询，如订单、账户余额、历史记录",
        #     "guest_template": PromptTemplate(
        #         "您好，您查询的内容需要登录后才能访问。\n"
        #         "请先完成登录，或联系客服获取帮助。\n\n"
        #         "您的问题：{query_str}"
        #     ),
        #     "standard_template": PromptTemplate(
        #         "你是账户数据查询助手。\n"
        #         "请直接从以下用户数据中提取并呈现查询结果，不得推测或补充不存在的数据。\n"
        #         "若数据中无对应记录，请明确告知用户。\n\n"
        #         "[用户数据]\n{context_str}\n\n"
        #         "[查询请求]\n{query_str}\n\n"
        #         "查询结果："
        #     ),
        # },
        # status_track={
        #     "_desc": "物流、审批、任务等实时状态追踪",
        #     "guest_template": PromptTemplate(
        #         "您好，状态查询功能需要登录后才能使用。\n"
        #         "请先完成登录，或联系客服获取帮助。\n\n"
        #         "您的问题：{query_str}"
        #     ),
        #     "standard_template": PromptTemplate(
        #         "你是状态追踪助手。\n"
        #         "请从以下数据中提取当前状态信息，清晰呈现状态节点和关键时间。\n"
        #         "若数据中无对应记录，请明确告知用户。\n\n"
        #         "[状态数据]\n{context_str}\n\n"
        #         "[查询]\n{query_str}\n\n"
        #         "当前状态："
        #     ),
        # },
        process_guide={
            "_desc": "操作步骤、办理流程类问题",
            "template": PromptTemplate(
                "你是操作流程引导助手。\n"
                "请依据以下流程文档，用编号步骤清晰呈现操作流程，"
                "每步说明具体操作要点和注意事项。\n\n"
                "[流程文档]\n{context_str}\n\n"
                "[用户问题]\n{query_str}\n\n"
                "操作步骤："
            ),
        },
    ),
)


def init_intent_data():
    data_dict = dict(
        L1={
            "names": list(__INTENT_TREE__.keys()),
            "descs": [x["_desc"] for x in __INTENT_TREE__.values()],
        },
        L2={},
    )
    for l1 in __INTENT_TREE__.keys():
        data_dict["L2"][l1] = {
            "names": [k for k in __INTENT_TREE__[l1].keys() if k != "_desc"],
            "descs": [v["_desc"] for k, v in __INTENT_TREE__[l1].items() if k != "_desc"],
        }
    return data_dict


__INTENT_DATA__ = init_intent_data()


def get_intent_name(l1_index: Optional[int] = None, l1_intent: Optional[str] = None, l2_index: Optional[int] = None):
    assert (l1_index is None) != (l1_intent is None)
    if l1_intent is None:
        l1_intent = __INTENT_DATA__["L1"]["names"][l1_index]

    if l2_index is None:
        return l1_intent
    return __INTENT_DATA__["L2"][l1_intent]["names"][l2_index]


def get_intent_choices(l1_intent: Optional[str] = None):
    if l1_intent is None:
        return __INTENT_DATA__["L1"]["descs"]

    return __INTENT_DATA__["L2"][l1_intent]["descs"]


def get_intent_template(l1_intent: str, l2_intent: str, user_group: UserGroupType) -> PromptTemplate:
    template = __INTENT_TREE__.get(f"{l1_intent}.{l2_intent}.{user_group}_template")
    template = template or __INTENT_TREE__.get(f"{l1_intent}.{l2_intent}.template")
    if not template:
        Logger.warning(
            f"NO template for intent '{l1_intent}.{l2_intent}' and user group '{user_group}', fallback to beyond_scope"
        )
        return __INTENT_TREE__["direct.beyond_scope.template"]

    return template
