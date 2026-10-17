# Creatinine Ward v3 | 病房时序表征与肌酐风险建模

## 项目用途

基于 Clin-JEPA 思路，对住院电子病历按 24 小时整理成状态、干预和人口学文本，再用表征学习与线性探针研究：**当前处于 R0 的患者，在未来 48 小时是否出现血清肌酐定义的 AKI**。这是研究性实现，不是可直接部署的临床诊断工具。

## 按文件夹阅读

| 路径 | 作用 |
|---|---|
| `code/clin_jepa/data_hosp/` | 病房队列筛选、观察/干预抽取、日粒度聚合、轨迹文本和 v3 标签构造 |
| `code/clin_jepa/model/` | 动作条件 Transformer 预测器和时间特征等组件 |
| `code/clin_jepa/training/` | 编码、训练、采样、损失与训练阶段实现 |
| `code/clin_jepa/evaluation/` | embedding 预计算、rollout、探针和评价指标 |
| `code/configs/` | 数据、训练与评估配置 |
| `code/scripts/`、`code/tests/` | 运行脚本、完整性校验和测试 |
| `code/docs/` | 架构与训练说明 |
| `reports/` | 仅保留汇总层面的 v3 主任务报告 |
| `data/README.md` | 公开版本的数据排除与复现要求 |

## 主要流程

授权数据 → 无 ICU 病房队列 → 24h 分桶 → state/action/static 文本 → Qwen3-8B + LoRA 编码 → 线性 logistic 探针 → 风险评分与统计评价。

原实验固定 3,000 人，train/val/test 为 2,100/450/450。主探针将过去 72h 的 state 和 action 表征分别均值汇聚，再与 static 表征拼接。扩展方案加入动作条件 predictor 的未来两步 rollout 表征。

## 已有实验结果

以下来自随附交接总览及主任务报告，**本次未重跑**：

| 特征 | 测试 AUROC | 95% CI | AUPRC |
|---|---:|---|---:|
| 过去 72h history | 0.705 | 0.67-0.74 | 0.252 |
| history + full48 rollout | 0.713 | 0.68-0.75 | 0.259 |

两个区间重叠，不能声称 rollout 有可靠增益。full48 使用记录中的未来干预作为条件，不能等同于仅依赖当前可用信息的实时前瞻预测。标签只使用血清肌酐、未纳入尿量，2/3 期合并，队列也不是自然患病率样本。

## 复现条件

需自行获得相应数据访问授权、准备 MIMIC-IV 数据、Qwen 权重及训练 checkpoint/embedding 缓存；本仓库不提供这些资源。建议 Linux/GPU 环境，依赖见 `code/pyproject.toml` 与 `code/requirements.txt`，具体数据入口见 `code/scripts/run_creatinine_ward_v1.sh`，v3 下游流程见 `code/BITAHUB_CREATININE_WARD_V3_DOWNSTREAM.md`。

原交接材料中提到的部分文档（例如 `code/docs/evaluation.md`）不在快照中，原文所列 `data/` 文件也已从公开版本移除。新增 `code/README.md` 用于说明当前发布范围。

保留 `code/LICENSE` 的 MIT 许可与原署名。论文框架与算法来源不因作品集整理而改变。

