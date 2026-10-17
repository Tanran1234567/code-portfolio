#!/usr/bin/env python
# -*-coding: utf-8-*-

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate, PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

############################################
# Tree
############################################
DEFAULT_SUMMARY_PROMPT_TMPL = (
    "Write a summary of the following. Try to use only the "
    "information provided. "
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)

DEFAULT_SUMMARY_PROMPT = PromptTemplate(DEFAULT_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY)

# insert prompts
DEFAULT_INSERT_PROMPT_TMPL = (
    "Context information is below. It is provided in a numbered list "
    "(1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Given the context information, here is a new piece of "
    "information: {new_chunk_text}\n"
    "Answer with the number corresponding to the summary that should be updated. "
    "The answer should be the number corresponding to the "
    "summary that is most relevant to the question.\n"
)
DEFAULT_INSERT_PROMPT = PromptTemplate(DEFAULT_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT)


# # single choice
DEFAULT_QUERY_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
    "Provide choice in the following format: 'ANSWER: <number>' and explain why "
    "this summary was selected in relation to the question.\n"
)
DEFAULT_QUERY_PROMPT = PromptTemplate(DEFAULT_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT)

# multiple choice
DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
    "Some choices are given below. It is provided in a numbered "
    "list (1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices "
    "(no more than {branching_factor}, ranked by most relevant to least) that "
    "are most relevant to the question: '{query_str}'\n"
    "Provide choices in the following format: 'ANSWER: <numbers>' and explain why "
    "these summaries were selected in relation to the question.\n"
)
DEFAULT_QUERY_PROMPT_MULTIPLE = PromptTemplate(
    DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL, prompt_type=PromptType.TREE_SELECT_MULTIPLE
)


DEFAULT_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
DEFAULT_REFINE_PROMPT = PromptTemplate(DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "上下文信息如下。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "根据上下文信息而不是先验知识，回答提问。\n"
    "提问：{query_str}\n"
    "答案："
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)

DEFAULT_TREE_SUMMARIZE_TMPL = (
    "下面是多个来源的上下文信息。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "根据多个来源的信息而不是先验知识，回答提问。\n"
    "提问：{query_str}\n"
    "答案："
)
DEFAULT_TREE_SUMMARIZE_PROMPT = PromptTemplate(DEFAULT_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY)


############################################
# Keyword Table
############################################
DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "Some text is provided below. Given the text, extract up to {max_keywords} "
    "keywords from the text. Avoid stopwords."
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
DEFAULT_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)


# NOTE: the keyword extraction for queries can be the same as
# the one used to build the index, but here we tune it to see if performance is better.
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


############################################
# Structured Store
############################################
DEFAULT_SCHEMA_EXTRACT_TMPL = (
    "We wish to extract relevant fields from an unstructured text chunk into "
    "a structured schema. We first provide the unstructured text, and then "
    "we provide the schema that we wish to extract. "
    "-----------text-----------\n"
    "{text}\n"
    "-----------schema-----------\n"
    "{schema}\n"
    "---------------------\n"
    "Given the text and schema, extract the relevant fields from the text in "
    "the following format: "
    "field1: <value>\nfield2: <value>\n...\n\n"
    "If a field is not present in the text, don't include it in the output."
    "If no fields are present in the text, return a blank string.\n"
    "Fields: "
)
DEFAULT_SCHEMA_EXTRACT_PROMPT = PromptTemplate(DEFAULT_SCHEMA_EXTRACT_TMPL, prompt_type=PromptType.SCHEMA_EXTRACT)

# NOTE: taken from langchain and adapted
# https://github.com/langchain-ai/langchain/blob/v0.0.303/libs/langchain/langchain/chains/sql_database/prompt.py
DEFAULT_TEXT_TO_SQL_TMPL = (
    "Given an input question, first create a syntactically correct {dialect} "
    "query to run, then look at the results of the query and return the answer. "
    "You can order the results by a relevant column to return the most "
    "interesting examples in the database.\n\n"
    "Never query for all the columns from a specific table, only ask for a "
    "few relevant columns given the question.\n\n"
    "Pay attention to use only the column names that you can see in the schema "
    "description. "
    "Be careful to not query for columns that do not exist. "
    "Pay attention to which column is in which table. "
    "Also, qualify column names with the table name when needed. "
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run\n"
    "SQLResult: Result of the SQLQuery\n"
    "Answer: Final answer here\n\n"
    "Only use tables listed below.\n"
    "{schema}\n\n"
    "Question: {query_str}\n"
    "SQLQuery: "
)

DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_TMPL,
    prompt_type=PromptType.TEXT_TO_SQL,
)

DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL = """\
Given an input question, first create a syntactically correct {dialect} \
query to run, then look at the results of the query and return the answer. \
You can order the results by a relevant column to return the most \
interesting examples in the database.

Pay attention to use only the column names that you can see in the schema \
description. Be careful to not query for columns that do not exist. \
Pay attention to which column is in which table. Also, qualify column names \
with the table name when needed.

IMPORTANT NOTE: you can use specialized pgvector syntax (`<->`) to do nearest \
neighbors/semantic search to a given vector from an embeddings column in the table. \
The embeddings value for a given row typically represents the semantic meaning of that row. \
The vector represents an embedding representation \
of the question, given below. Do NOT fill in the vector values directly, but rather specify a \
`[query_vector]` placeholder. For instance, some select statement examples below \
(the name of the embeddings column is `embedding`):
SELECT * FROM items ORDER BY embedding <-> '[query_vector]' LIMIT 5;
SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5;
SELECT * FROM items WHERE embedding <-> '[query_vector]' < 5;

You are required to use the following format, \
each taking one line:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use tables listed below.
{schema}


Question: {query_str}
SQLQuery: \
"""

DEFAULT_TEXT_TO_SQL_PGVECTOR_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL,
    prompt_type=PromptType.TEXT_TO_SQL,
)

# NOTE: by partially filling schema, we can reduce to a QuestionAnswer prompt
# that we can feed to ur table
DEFAULT_TABLE_CONTEXT_TMPL = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided context information below. "
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}"
)

DEFAULT_TABLE_CONTEXT_QUERY = (
    "Provide a high-level description of the table, "
    "as well as a description of each column in the table. "
    "Provide answers in the following format:\n"
    "TableDescription: <description>\n"
    "Column1Description: <description>\n"
    "Column2Description: <description>\n"
    "...\n\n"
)

DEFAULT_TABLE_CONTEXT_PROMPT = PromptTemplate(DEFAULT_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT)

# NOTE: by partially filling schema, we can reduce to a refine prompt
# that we can feed to ur table
DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided some context information below. "
    "{context_msg}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
DEFAULT_REFINE_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)


############################################
# Knowledge-Graph Table
############################################
DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)
DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_KG_TRIPLET_EXTRACT_TMPL,
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
)

DEFAULT_DYNAMIC_EXTRACT_TMPL = (
    "Extract up to {max_knowledge_triplets} knowledge triplets from the given text. "
    "Each triplet should be in the form of (head, relation, tail) with their respective types.\n"
    "---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Relation Types: {allowed_relation_types}\n"
    "\n"
    "Use these types as a starting point, but introduce new types if necessary based on the context.\n"
    "\n"
    "GUIDELINES:\n"
    "- Output in JSON format: [{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
    "- Use the most complete form for entities (e.g., 'United States of America' instead of 'USA')\n"
    "- Keep entities concise (3-5 words max)\n"
    "- Break down complex phrases into multiple triplets\n"
    "- Ensure the knowledge graph is coherent and easily understandable\n"
    "---------------------\n"
    "EXAMPLE:\n"
    "Text: Tim Cook, CEO of Apple Inc., announced the new Apple Watch that monitors heart health. "
    "UC Berkeley researchers studied the benefits of apples.\n"
    "Output:\n"
    "[{{'head': 'Tim Cook', 'head_type': 'PERSON', 'relation': 'CEO_OF', 'tail': 'Apple Inc.', 'tail_type': 'COMPANY'}},\n"
    " {{'head': 'Apple Inc.', 'head_type': 'COMPANY', 'relation': 'PRODUCES', 'tail': 'Apple Watch', 'tail_type': 'PRODUCT'}},\n"
    " {{'head': 'Apple Watch', 'head_type': 'PRODUCT', 'relation': 'MONITORS', 'tail': 'heart health', 'tail_type': 'HEALTH_METRIC'}},\n"
    " {{'head': 'UC Berkeley', 'head_type': 'UNIVERSITY', 'relation': 'STUDIES', 'tail': 'benefits of apples', 'tail_type': 'RESEARCH_TOPIC'}}]\n"
    "---------------------\n"
    "Text: {text}\n"
    "Output:\n"
)

DEFAULT_DYNAMIC_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_DYNAMIC_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

DEFAULT_DYNAMIC_EXTRACT_PROPS_TMPL = (
    "Extract up to {max_knowledge_triplets} knowledge triplets from the given text. "
    "Each triplet should be in the form of (head, relation, tail) with their respective types and properties.\n"
    "---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Entity Properties: {allowed_entity_properties}\n"
    "Relation Types: {allowed_relation_types}\n"
    "Relation Properties: {allowed_relation_properties}\n"
    "\n"
    "Use these types as a starting point, but introduce new types if necessary based on the context.\n"
    "\n"
    "GUIDELINES:\n"
    "- Output in JSON format: [{{'head': '', 'head_type': '', 'head_props': {{...}}, 'relation': '', 'relation_props': {{...}}, 'tail': '', 'tail_type': '', 'tail_props': {{...}}}}]\n"
    "- Use the most complete form for entities (e.g., 'United States of America' instead of 'USA')\n"
    "- Keep entities concise (3-5 words max)\n"
    "- Break down complex phrases into multiple triplets\n"
    "- Ensure the knowledge graph is coherent and easily understandable\n"
    "---------------------\n"
    "EXAMPLE:\n"
    "Text: Tim Cook, CEO of Apple Inc., announced the new Apple Watch that monitors heart health. "
    "UC Berkeley researchers studied the benefits of apples.\n"
    "Output:\n"
    "[{{'head': 'Tim Cook', 'head_type': 'PERSON', 'head_props': {{'prop1': 'val', ...}}, 'relation': 'CEO_OF', 'relation_props': {{'prop1': 'val', ...}}, 'tail': 'Apple Inc.', 'tail_type': 'COMPANY', 'tail_props': {{'prop1': 'val', ...}}}},\n"
    " {{'head': 'Apple Inc.', 'head_type': 'COMPANY', 'head_props': {{'prop1': 'val', ...}}, 'relation': 'PRODUCES', 'relation_props': {{'prop1': 'val', ...}}, 'tail': 'Apple Watch', 'tail_type': 'PRODUCT', 'tail_props': {{'prop1': 'val', ...}}}},\n"
    " {{'head': 'Apple Watch', 'head_type': 'PRODUCT', 'head_props': {{'prop1': 'val', ...}}, 'relation': 'MONITORS', 'relation_props': {{'prop1': 'val', ...}}, 'tail': 'heart health', 'tail_type': 'HEALTH_METRIC', 'tail_props': {{'prop1': 'val', ...}}}},\n"
    " {{'head': 'UC Berkeley', 'head_type': 'UNIVERSITY', 'head_props': {{'prop1': 'val', ...}}, 'relation': 'STUDIES', 'relation_props': {{'prop1': 'val', ...}}, 'tail': 'benefits of apples', 'tail_type': 'RESEARCH_TOPIC', 'tail_props': {{'prop1': 'val', ...}}}}]\n"
    "---------------------\n"
    "Text: {text}\n"
    "Output:\n"
)

DEFAULT_DYNAMIC_EXTRACT_PROPS_PROMPT = PromptTemplate(
    DEFAULT_DYNAMIC_EXTRACT_PROPS_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

############################################
# HYDE
##############################################
HYDE_TMPL = (
    "Please write a passage to answer the question\n"
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)
DEFAULT_HYDE_PROMPT = PromptTemplate(HYDE_TMPL, prompt_type=PromptType.SUMMARY)


############################################
# Simple Input
############################################
DEFAULT_SIMPLE_INPUT_TMPL = "{query_str}"
DEFAULT_SIMPLE_INPUT_PROMPT = PromptTemplate(DEFAULT_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT)


############################################
# JSON Path
############################################
DEFAULT_JSON_PATH_TMPL = (
    "We have provided a JSON schema below:\n"
    "{schema}\n"
    "Given a task, respond with a JSON Path query that "
    "can retrieve data from a JSON value that matches the schema.\n"
    "Provide the JSON Path query in the following format: 'JSONPath: <JSONPath>'\n"
    "You must include the value 'JSONPath:' before the provided JSON Path query."
    "Example Format:\n"
    "Task: What is John's age?\n"
    "Response: JSONPath: $.John.age\n"
    "Let's try this now: \n\n"
    "Task: {query_str}\n"
    "Response: "
)
DEFAULT_JSON_PATH_PROMPT = PromptTemplate(DEFAULT_JSON_PATH_TMPL, prompt_type=PromptType.JSON_PATH)


############################################
# Choice Select
############################################
DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "以下为一份文档列表。每个文档都有一个编号及其内容摘要。同时提供一个相关问题。\n"
    "请按照相关程度从高到低的顺序，列出回答该问题需要参考的文档编号以及相关性评分。\n"
    "根据你认为该文档与问题的相关程度给出1-10相关性评分。\n"
    "【注意】不要包含与问题不相关的任何文档。\n"
    "【示例格式】\n"
    "Document 1:\n<文档1内容摘要>\n\n"
    "Document 2:\n<文档2内容摘要>\n\n"
    "...\n\n"
    "Document 10:\n<文档10内容摘要>\n\n"
    "问题：<问题>\n"
    "答案：\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "现在让我们试试这个：\n\n"
    "{context_str}\n"
    "问题：{query_str}\n"
    "答案：\n"
)
DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT)


############################################
# Structured Choice Select
############################################
STRUCTURED_CHOICE_SELECT_PROMPT_TMPL = (
    "A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Let's try this now: \n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "Answer:\n"
)
STRUCTURED_CHOICE_SELECT_PROMPT = PromptTemplate(
    STRUCTURED_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)


############################################
# RankGPT Rerank template
############################################
RANKGPT_RERANK_PROMPT_TMPL = (
    "搜索查询：{query}。\n"
    "请根据与搜索查询的相关性，对上述{num}个段落进行排序。"
    "段落应按相关性降序列出，标识符置于方括号内，最相关的段落排在最前。"
    "输出格式应为：[ ] > [ ]，例如：[1] > [2]。\n"
    "请仅返回排序结果，无需任何额外说明或解释。"
)
RANKGPT_RERANK_PROMPT = PromptTemplate(RANKGPT_RERANK_PROMPT_TMPL, prompt_type=PromptType.RANKGPT_RERANK)


############################################
# JSONalyze Query Template
############################################
DEFAULT_JSONALYZE_PROMPT_TMPL = (
    "You are given a table named: '{table_name}' with schema, "
    "generate SQLite SQL query to answer the given question.\n"
    "Table schema:\n"
    "{table_schema}\n"
    "Question: {question}\n\n"
    "SQLQuery: "
)
DEFAULT_JSONALYZE_PROMPT = PromptTemplate(DEFAULT_JSONALYZE_PROMPT_TMPL, prompt_type=PromptType.TEXT_TO_SQL)


############################################
# ChatMessage Template
############################################
# text qa prompt
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "你是一个在全世界都值得信赖的专家问答系统。\n"
        "始终使用提供的上下文信息而不是先验知识来回答提问。\n"
        "必须遵循的一些规则：\n"
        "1. 永远不要在答案中直接引用给定的上下文。\n"
        "2. 避免使用“基于上下文，……”或“上下文信息……”之类的语句。"
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "上下文信息如下。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "根据上下文信息而不是先验知识，回答提问。\n"
            "提问：{query_str}\n"
            "答案："
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# Tree Summarize
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "下面是多个来源的上下文信息。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "根据多个来源的信息而不是先验知识，回答提问。\n"
            "提问：{query_str}\n"
            "答案："
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS)


# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "You are an expert Q&A system that strictly operates in two modes "
            "when refining existing answers:\n"
            "1. **Rewrite** an original answer using the new context.\n"
            "2. **Repeat** the original answer if the new context isn't useful.\n"
            "Never reference the original answer or context directly in your answer.\n"
            "When in doubt, just repeat the original answer.\n"
            "New Context: {context_msg}\n"
            "Query: {query_str}\n"
            "Original Answer: {existing_answer}\n"
            "New Answer: "
        ),
        role=MessageRole.USER,
    )
]


CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)


# Table Context Refine Prompt
CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
    ChatMessage(content="{query_str}", role=MessageRole.USER),
    ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
    ChatMessage(
        content=(
            "We have provided a table schema below. "
            "---------------------\n"
            "{schema}\n"
            "---------------------\n"
            "We have also provided some context information below. "
            "{context_msg}\n"
            "---------------------\n"
            "Given the context information and the table schema, "
            "refine the original answer to better "
            "answer the question. "
            "If the context isn't useful, return the original answer."
        ),
        role=MessageRole.USER,
    ),
]
CHAT_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS)


############################################
# History Query Transform
############################################
HISTORY_QUERY_TRANSFORM_TMPL = (
    "你是一个查询改写助手。下面是一段对话历史和一个新的用户提问。\n"
    "请根据对话历史，将用户的最新提问改写为一个独立、完整、自包含的查询语句，"
    "使其在不依赖上下文的情况下也能被准确理解。\n"
    "如果提问已经是完整且清晰的，直接返回原提问即可。\n"
    "只需输出改写后的查询语句，不要添加任何解释或多余内容。\n"
    "\n"
    "对话历史：\n"
    "{history_str}\n"
    "\n"
    "最新提问：{query_str}\n"
    "改写后的查询："
)
DEFAULT_HISTORY_QUERY_TRANSFORM_PROMPT = PromptTemplate(
    HISTORY_QUERY_TRANSFORM_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)
