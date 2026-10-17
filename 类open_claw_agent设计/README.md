# 智能助手与任务分解评测

## 项目用途

面向自然语言问答与知识库检索，提供文档入库、检索增强生成（RAG）、查询处理及后台任务管理。文件夹还包含子任务分解、步骤规划和查询优化的多版本测试集与已有评测汇报。

## 按文件夹阅读

| 路径 | 作用 |
|---|---|
| `iassistant/src/main.py` | FastAPI 应用入口，初始化服务、挂载路由与异常处理 |
| `iassistant/src/mind_zone/agent/` | 直接问答、意图路由、RAG 索引与检索等流程；部分智能体仍为骨架 |
| `iassistant/src/mind_zone/base/` | 文件读取、切分、向量化、检索、后处理和答案合成组件 |
| `iassistant/src/mind_zone/routers/` | 文件、分块、查询与任务接口 |
| `iassistant/src/mind_zone/state/`、`tasks/` | MongoDB 状态管理、Dramatiq/Redis 异步任务与进度跟踪 |
| `iassistant/config/` | 服务、模型、应用和提示词配置模板 |
| `iassistant/tests/` | 服务接口、异步任务和监控相关测试/示例 |
| `测试集/` 与根目录 JSONL | 任务分解、步骤规划、查询优化的测试数据及版本记录 |
| `评测汇报_子任务与步骤分解测试汇总.md` | 既有实验汇报，包含错误分析与指标解释 |

## 体现的实现能力

- 用 FastAPI 与 LlamaIndex 组织请求处理、检索、重排/后处理和生成链路。
- 用任务队列与持久化状态拆开耗时文档处理和在线请求，支持进度查询与取消。
- 接入 Milvus/OpenSearch、模型服务和 OpenTelemetry 监控。
- 通过多版本评测数据与分类分析定位规划能力短板，而不只比较答案字符串。

## 已有实验记录与范围

原汇报记录：子任务分解决策准确率 99.94%（5,000 条）；步骤分解 LLM 评判正确率 68.11%（已生成的 4,449 条，计划为 5,000 条）。这些是历史报告中的数值，本次没有复跑。参考答案主要由 LLM 生成，非逐条人工确认；步骤评判也使用 LLM。

当前源码中 `ChatBIFlow` 和 `ReActAgent` 仍有骨架实现；汇报提及的 `eval_orchestrator_decompose.py` 等完整评测脚本未包含在该快照中。不能把报告描述的全部编排能力视为当前仓库已可完整运行。

## 环境准备

`pyproject.toml` 要求 Python 3.12。需要自行准备 Redis、MongoDB、Nacos、Milvus/OpenSearch 与兼容模型服务。

```bash
cd iassistant
uv sync
# 参考 config/env.example 配置自己的环境变量和 config/ 下服务参数
PYTHONPATH=src uv run uvicorn main:app --host 127.0.0.1 --port 8080
```

以上为 Bash 示例，运行前必须完成外部依赖配置。公开副本中的地址、用户名、命名空间为示例值。Docker 镜像配置改为本地基础镜像：先使用 `Dockerfile.base` 构建 `iassistant-base:py3.12`，再按目标平台构建应用镜像。
