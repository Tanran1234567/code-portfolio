# Diplomat | 多智能体外交谈判平台

## 项目用途

搭建一个可配置外交场景的 Web 平台：多位代表按轮次生成谈判发言，主席管理会议阶段，记忆模块压缩历史，评估模块按规则输出评分与反馈。前端可以观察请求、发言和结果，并导出演示结果。

## 按文件夹阅读

| 路径 | 作用 |
|---|---|
| `backend/app/main.py` | FastAPI 服务入口 |
| `backend/app/diplomat/` | 先生成态度与策略，再生成公开发言 |
| `backend/app/chair/` | 主席消息、阶段提示与裁定 |
| `backend/app/memory/` | 每轮发言摘要，供后续轮次使用 |
| `backend/app/evaluator/` | 分项评分、总分与反馈 |
| `backend/app/llm_client.py` | 兼容模型接口、环境变量配置与请求封装 |
| `backend/app/models.py`、`storage.py` | 数据模型与结果存储 |
| `frontend/src/App.jsx`、`api.js` | React 页面、流程调度与 API 交互 |
| `dataset/`、`frontend/src/worldviews/` | 场景、规则、背景等演示配置 |
| `doc/main.md` | 系统结构、流程和接口设计 |
| `tools/` | JSON 演示结果转换为 PDF 的工具 |

## 核心链路与实现特点

场景配置 → 主席开场 → 代表态度/策略/发言 → 轮次摘要 → 多轮谈判与裁定 → 评分/反馈 → 导出。

前后端分离，HTTP 层与业务服务分层；Pydantic 约束输入输出；提示词独立为 JSON 配置。无 API key 时部分模块回退到 stub，便于联调，但 stub 不代表真实模型推理能力。

记忆模块已生成文本摘要，而 `alliances/conflicts/concessions` 等结构化信号当前仍返回空列表；不能描述为已完成联盟关系自动抽取。

## 本地启动

后端与前端分别运行：

```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm ci
npm run dev
```

复制 `.env.example` 为 `.env`，再填写自己的模型配置。前端依赖以 `package-lock.json` 为准；Vite 7 的运行环境要求见原 `doc/main.md`。本次未调用真实模型接口，也未执行完整前后端联调。
