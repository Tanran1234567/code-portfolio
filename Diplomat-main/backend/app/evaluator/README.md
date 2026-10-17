## 评分者模块（Evaluator）

本模块负责 **在会议结束后，根据 rubrics 对 A/B/C/D 的表现评分并给出反馈**。

---

### 接收什么输入（HTTP 请求）

#### Endpoint

- **POST** `/api/v1/scenarios/{scenario_id}/evaluations`
- 可选 Header：`X-Debug-Prompts: 1`（开启后端返回 prompt/debug 信息）

#### Request Body（JSON）

```json
{
  "scenario_id": "string",
  "rubrics": [{ "id": "r1", "title": "…", "weight": 1.0 }],
  "speeches": [
    { "round": 1, "speaker": "A", "content": "…", "timestamp": "…" }
  ]
}
```

说明：

- `speeches` 期望包含 **全轮次日志**
- `rubrics` 为专家提供的评分规则（结构目前不做强约束）

---

### 负责什么输出（HTTP 响应 + 副作用）

#### Response Body（JSON）

示例：

```json
{
  "scores": { "A": 82, "B": 76, "C": 88, "D": 71 },
  "feedback": { "A": "string", "B": "string", "C": "string", "D": "string" },
  "debug": {
    "system": "string",
    "user": "string",
    "model": "deepseek/deepseek-v3.2",
    "temperature": 0.2
  }
}
```

---

### 你接下来要做什么（TODO）

- 增强评分准确性

---
