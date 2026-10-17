# Code Portfolio | 代码项目作品集

作者：**Tanran1234567** · 整理日期：2026-10-17

这个仓库按原始文件夹组织了 4 组项目，展示智能体应用、临床时序建模、全栈开发、计算机视觉与算法实践。作者确认这些项目由本人完成；已有框架、算法和教学素材的来源说明保留在原文件中。

**[项目介绍 PDF](docs/project-portfolio.pdf)** · [公开版本说明](PUBLICATION_NOTES.md) · [来源与许可](THIRD_PARTY_NOTICES.md)

| 文件夹 | 项目在做什么 | 建议先看 |
|---|---|---|
| [类open_claw_agent设计](类open_claw_agent设计/README.md) | 智能助手/RAG 服务、异步文档任务及任务分解评测材料 | `iassistant/src/mind_zone/agent/rag_query.py` |
| [creatinine_ward_v3_handoff_20260908](creatinine_ward_v3_handoff_20260908/README.md) | 病房电子病历时序表征与未来 48 小时肌酐 AKI 风险研究 | `code/clin_jepa/evaluation/creatinine_aki_v3_probe.py` |
| [Diplomat-main](Diplomat-main/README.md) | 多外交官谈判、主席调度、共享记忆与评分的 Web 平台 | `backend/app/diplomat/service.py`、`frontend/src/App.jsx` |
| [python_fourth](python_fourth/README.md) | Python、OpenCV、深度学习、目标检测、语义分割和五子棋实践 | `五子棋/wuziqi.py`、`yolo_od/yolo11_od.py` |

## 建议阅读顺序

1. **Diplomat**：前后端完整，适合先了解系统功能与模块划分。
2. **智能助手**：查看 RAG 流程、后台任务管理与评测分析。
3. **Creatinine Ward**：查看数据管线、建模方法和实验边界。
4. **Python 实践**：查看基础训练及独立算法、视觉应用。

## 本次整理的验证

- 275 个 Python 文件通过语法解析；33 个 JSON 文件、52,564 行 JSONL 均可解析。
- 五子棋通过空棋盘落点、五连胜负判断、非胜局判断及深度 1 搜索合法落点的基础检查。
- 未启动依赖外部数据库、模型接口或 GPU 的完整系统；未重新训练模型或复跑原实验。
- 源码统计包含注释、空行及已有框架代码，不能直接作为原创工作量或完成度的证明。

## 运行与数据

各项目环境独立，详见各文件夹 README。公开版本不含真实密钥、VPN 配置、患者级数据、虚拟环境、模型权重和大型外部数据集。环境模板保留，服务地址已改为本地示例值；使用前需要配置自己的服务。
