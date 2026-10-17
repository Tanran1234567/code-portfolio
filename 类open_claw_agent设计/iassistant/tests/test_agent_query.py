#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

pytest.importorskip("llama_index")

from llama_index.core.workflow import Context, StopEvent, Workflow

import mind_zone.agent.base as base_module
from mind_zone.agent.base import AgentWorker
from mind_zone.agent.rag_query import RAGQueryFlow
from mind_zone.schemas import QueryAnswerRequest, QueryIntent, QueryRequestEvent


def test_decompose_empty_query():
    sub_queries = AgentWorker._decompose_query("")

    assert len(sub_queries) == 0


def test_decompose_single_query():
    sub_queries = AgentWorker._decompose_query("介绍一下系统功能")

    assert len(sub_queries) == 1
    assert "系统功能" in sub_queries[0]


@pytest.mark.parametrize(
    "query",
    [
        "介绍系统功能；说明使用流程",
        "介绍系统功能;说明使用流程",
        "介绍系统功能\n说明使用流程",
        "介绍系统功能另外说明使用流程",
        "介绍系统功能同时说明使用流程",
        "介绍系统功能并且说明使用流程",
        "介绍系统功能以及说明使用流程",
    ],
)
def test_decompose_splits_two_meaningful_tasks(query):
    sub_queries = AgentWorker._decompose_query(query)

    assert_two_task_contract(sub_queries, first_keywords=("系统功能",), second_keywords=("使用流程",))


def test_decompose_ignores_blank_parts():
    sub_queries = AgentWorker._decompose_query("介绍系统功能； ；说明使用流程")

    assert_two_task_contract(sub_queries, first_keywords=("系统功能",), second_keywords=("使用流程",))


def test_refine_empty():
    assert not AgentWorker._refine_query_text("")
    assert not AgentWorker._refine_query_text(None)


def test_refine_nfkc_and_whitespace():
    refined = AgentWorker._refine_query_text("　介绍　系统　")
    multiline = AgentWorker._refine_query_text("介绍系统\n\n说明流程")

    assert "介绍" in refined and "系统" in refined
    assert "　" not in refined
    assert "  " not in refined
    assert multiline.count("\n") == 1
    assert all(part.strip() for part in multiline.splitlines())


def test_refine_preserves_newlines_for_decompose():
    raw = "介绍系统功能\n说明使用流程"
    refined = AgentWorker._refine_query_text(raw)
    sub_queries = AgentWorker._decompose_query(refined)

    assert_two_task_contract(sub_queries, first_keywords=("系统功能",), second_keywords=("使用流程",))


def test_expand_polite_prefix():
    first = AgentWorker._expand_query_text("请帮我介绍一下系统功能")
    second = AgentWorker._expand_query_text("能否说明使用流程")

    assert not first.startswith("请帮我")
    assert "系统功能" in first
    assert not second.startswith("能否")
    assert "使用流程" in second


def test_expand_parenthetical():
    expanded = AgentWorker._expand_query_text("部署流程（Docker）")

    assert "部署流程" in expanded
    assert "Docker" in expanded


def test_expand_multiline():
    expanded = AgentWorker._expand_query_text("请帮我介绍A\n请问说明B")
    lines = expanded.splitlines()

    assert len(lines) == 2
    assert "A" in lines[0]
    assert "B" in lines[1]
    assert all(not line.startswith(("请帮我", "请问")) for line in lines)


def test_pipeline_refine_expand_decompose_contract():
    raw = "请帮我介绍功能；说明流程（后台）"
    sub_queries = preprocess_query(raw)

    assert_two_task_contract(sub_queries, first_keywords=("功能",), second_keywords=("流程", "后台"))


class LLMPreprocessStub:
    """Test double for future LLM-based query preprocessing."""

    @staticmethod
    def refine(query):
        assert query
        return "请概述系统能力；请解释后台操作步骤（Docker）"

    @staticmethod
    def expand(query):
        assert "系统能力" in query
        return "概述系统核心功能；解释后台使用流程，包含 Docker 部署线索"

    @staticmethod
    def decompose(query):
        assert "后台使用流程" in query
        return ["概述系统核心能力", "解释后台使用流程，包含 Docker"]


def test_pipeline_contract_allows_llm_rephrasing():
    stub = LLMPreprocessStub()
    refined = stub.refine("介绍系统功能；说明后台流程")
    expanded = stub.expand(refined)
    sub_queries = stub.decompose(expanded)

    assert_two_task_contract(sub_queries, first_keywords=("系统",), second_keywords=("后台", "流程"))
    assert sub_queries != ["介绍系统功能", "说明后台流程"]


@pytest.mark.asyncio
async def test_handle_request_runs_each_decomposed_subquery(monkeypatch):
    worker = AgentWorker.__new__(AgentWorker)
    Workflow.__init__(worker, timeout=1)
    request = QueryAnswerRequest(app_name="test-app", text="介绍系统功能；说明后台流程")
    ctx = Context(worker)
    sub_queries = ["概述系统核心能力", "解释后台使用流程，包含 Docker"]
    routed_queries = []

    async def fake_route_query(query):
        return "direct", "mock"

    async def fake_run_single_flow(ctx_arg, request_arg, ev):
        assert ctx_arg is ctx
        assert request_arg is request
        routed_queries.append(ev.query)
        return StopEvent(result={"query": ev.query})

    monkeypatch.setattr(base_module, "get_intent_template", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_route_query", fake_route_query)
    monkeypatch.setattr(worker, "_run_single_flow", fake_run_single_flow)
    await ctx.store.set("request", request)
    await ctx.store.set("sub_queries", sub_queries)

    result = await worker.handle_request(
        ctx,
        QueryRequestEvent(
            intent=QueryIntent(l1="direct", l2="mock"),
            query=request.text,
            context=[],
            template=None,
            stream=False,
        ),
    )

    assert len(routed_queries) == 2
    assert_two_task_contract(routed_queries, first_keywords=("系统",), second_keywords=("后台", "流程"))
    assert [item["query"] for item in result.result] == routed_queries


def test_format_chunks_from_dict_node():
    chunks = [
        {
            "score": 0.9,
            "node": {
                "id_": "chunk-1",
                "text": "检索片段内容",
                "metadata": {"file_id": "file-1", "bundle_id": "bundle-1"},
            },
        }
    ]

    assert RAGQueryFlow._format_chunks(chunks) == [
        {
            "index": 0,
            "doc_id": "chunk-1",
            "score": 0.9,
            "file_id": "file-1",
            "bundle_id": "bundle-1",
            "text": "检索片段内容",
        }
    ]


def test_format_chunks_from_flat_dict():
    chunks = [{"node_id": "chunk-1", "score": 0.8, "text": "检索片段内容"}]

    assert RAGQueryFlow._format_chunks(chunks) == [
        {
            "index": 0,
            "doc_id": "chunk-1",
            "score": 0.8,
            "file_id": None,
            "bundle_id": None,
            "text": "检索片段内容",
        }
    ]


def preprocess_query(raw_query):
    refined = AgentWorker._refine_query_text(raw_query)
    expanded = AgentWorker._expand_query_text(refined)
    return AgentWorker._decompose_query(expanded)


def assert_two_task_contract(sub_queries, *, first_keywords, second_keywords):
    assert len(sub_queries) == 2
    assert all(isinstance(query, str) and query.strip() == query and query for query in sub_queries)
    assert any(has_keywords(query, first_keywords) for query in sub_queries)
    assert any(has_keywords(query, second_keywords) for query in sub_queries)


def has_keywords(text, keywords):
    return all(keyword in text for keyword in keywords)
