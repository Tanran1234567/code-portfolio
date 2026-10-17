## 记忆模块（Memory / Summary）

本模块负责 **在每轮结束后生成 RoundSummary（共享记忆）**，供下一轮外交官使用，以控制上下文长度。

---

### 接收什么输入（HTTP 请求）

#### Endpoint

- **POST** `/api/v1/scenarios/{scenario_id}/round-summaries`
- 可选 Header：`X-Debug-Prompts: 1`（开启后端返回 prompt/debug 信息）

#### Request Body（JSON）

```json
{
  "round": 2,
  "speeches": [
    { "round": 2, "speaker": "A", "content": "…", "timestamp": "…" }
  ]
}
```

说明：

- `speeches` **应当是本轮 A/B/C/D 的全部发言**（由调度器/前端组装）

---

### 负责什么输出（HTTP 响应 + 副作用）

#### Response Body（JSON）

- 正常返回：`RoundSummaryResponse`
  - `round`: number
  - `summary`: string
  - `signals`: `{alliances, conflicts, concessions, escalation_risk}`
  - `debug`（可选，仅当 `X-Debug-Prompts: 1`）：`{system,user,model,temperature}`

示例：

```json
{
  "round": 2,
  "summary": "string",
  "signals": {
    "alliances": ["A-D"],
    "conflicts": ["B-C"],
    "concessions": ["D"],
    "escalation_risk": "medium"
  },
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

- **结构化 signals 抽取**：从 speeches/summaries 里抽取联盟/冲突/让步/升级风险
- **摘要质量**：保证摘要可作为下一轮共享输入（去冗余、保留关键立场变化与信号）

---
