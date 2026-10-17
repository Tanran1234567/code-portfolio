import './App.css'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  generateAgentSpeech,
  generateChairMessage,
  generateEvaluation,
  generateRoundSummary,
  getScenarioBackground,
  getScenarioIssues,
  getScenarioSecrets,
  listDatasetScenarios,
  nowIso,
} from './api'
import General from './worldviews/General.json'

function App() {
  const [scenarioId, setScenarioId] = useState('demo-001')
  const [rounds, setRounds] = useState(2)
  const [scenarioOptions, setScenarioOptions] = useState([]) // string[]
  const [scenarioOptionsLoading, setScenarioOptionsLoading] = useState(false)

  const [worldBackground, setWorldBackground] = useState(General.world_background || '')
  const [countryProfiles, setCountryProfiles] = useState(General.countries || {})
  // notes 文本：用于 textarea 展示/编辑（兼容旧数据结构）
  const [countrySecrets, setCountrySecrets] = useState(General.secrets || {})
  // 完整结构化 secrets（含 decision_factors 等）：用于注入到外交官 prompt 的 {secret_json}
  const [countrySecretsFull, setCountrySecretsFull] = useState(null) // Record<string, any> | null
  const [issuesText, setIssuesText] = useState(General.issues_text || '')
  const agents = useMemo(() => Object.keys(countryProfiles || {}), [countryProfiles])
  const [rubrics, setRubrics] = useState([
    { id: 'r1', title: '是否对己方信息和立场有准确把握？', weight: 1.0 },
    { id: 'r2', title: '是否对他方立场和谈判全局有准确把握？', weight: 1.0 },
    { id: 'r3', title: '是否有效表达了己方在谈判中的诉求？', weight: 1.0 },
    { id: 'r4', title: '是否有效推动了谈判的进展？', weight: 1.0 },
    { id: 'r5', title: '是否体现了优秀的谈判手段和技巧？', weight: 1.0 },
    { id: 'r6', title: '是否有效回应或打击了谈判对手？', weight: 1.0 },
    { id: 'r7', title: '是否维护或提升了己方的外交形象？', weight: 1.0 },
    { id: 'r8', title: '是否创新性地提出方案或解决分歧？', weight: 1.0 },
    { id: 'r9', title: '是否体现了优秀的语言表达和沟通能力？', weight: 1.0 },
    { id: 'r10', title: '是否体现了得体的谈判礼仪？', weight: 1.0 },
  ])

  const [speeches, setSpeeches] = useState([]) // Speech[]
  const [summaries, setSummaries] = useState([]) // RoundSummary[]
  const [adjudication, setAdjudication] = useState(null) // Chair adjudication JSON
  const [evaluation, setEvaluation] = useState(null)
  const [showPrivateSpeeches, setShowPrivateSpeeches] = useState(false)
  const [calls, setCalls] = useState([]) // {id, title, url, method, request, response, error, tsStart, tsEnd}[]
  const [debugPrompts, setDebugPrompts] = useState(true)
  const [runSnapshot, setRunSnapshot] = useState(null) // 本次运行的输入/结果快照（用于导出）

  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [log, setLog] = useState([])
  const [ruleQuestion, setRuleQuestion] = useState('')
  const [ruleAnswer, setRuleAnswer] = useState('')
  const abortRef = useRef({ aborted: false })
  const callSeq = useRef(0)

  async function loadScenarioFromDataset({ silent = false } = {}) {
    const sid = String(scenarioId || '').trim()
    if (!sid) return

    try {
      // 1) 优先从后端 API 加载（如果你同时启动了 backend）
      // 2) 若后端不可用，则回退到 Vite dev server 暴露的 /dataset/{scenario_id}.json
      let raw = null
      try {
        const [bg, iss, sec] = await Promise.all([
          getScenarioBackground(sid),
          getScenarioIssues(sid),
          // 使用 full=1 拉取完整 secrets（含 decision_factors 等），用于注入 prompt
          getScenarioSecrets(sid, { full: true }),
        ])
        raw = { background: bg, issues: iss?.issues || [], secrets: sec, scenario_id: sid }
      } catch {
        // fallback: 直接读取 /dataset/{sid}.json（由 vite.config.js dev middleware 提供）
        const resp = await fetch(`/dataset/${encodeURIComponent(sid)}.json`)
        if (!resp.ok) throw new Error(`读取 /dataset/${sid}.json 失败：HTTP ${resp.status}`)
        raw = await resp.json()
      }

      const bgObj = raw?.background || {}
      setWorldBackground(String(bgObj?.world_background || ''))

      // countries: dataset 新版是 {国家名: "profile文本"}；后端接口是 {国家名: {profile}}
      /** @type {Record<string, string>} */
      const profiles = {}
      const countries = bgObj?.countries || {}
      for (const [k, v] of Object.entries(countries || {})) {
        profiles[k] = typeof v === 'string' ? v : String(v?.profile || '')
      }
      setCountryProfiles(profiles)

      // secrets:
      // - 后端 full=1 返回 dataset 原始对象（含 decision_factors 等）
      // - dataset 文件本身也可能是 {国家:{notes,...}} 或 {国家:"notes字符串"}
      /** @type {Record<string, any>} */
      const secretsFullOut = {}
      /** @type {Record<string, string>} */
      const secretsOut = {}
      const secObj = raw?.secrets || {}
      for (const [k, v] of Object.entries(secObj || {})) {
        if (v && typeof v === 'object') {
          secretsFullOut[k] = v
          secretsOut[k] = String(v?.notes || '')
        } else {
          // 若只有 notes 字符串，也包装成对象，便于后续与 decision_factors 合并
          secretsFullOut[k] = { notes: String(v || '') }
          secretsOut[k] = String(v || '')
        }
      }
      setCountrySecrets(secretsOut)
      setCountrySecretsFull(secretsFullOut)

      const issuesArr = Array.isArray(raw?.issues) ? raw.issues : Array.isArray(raw?.issues?.issues) ? raw.issues.issues : []
      const text = issuesArr
        .map((it, idx) => `${idx + 1}. ${String(it?.title || '').trim() || `议题 ${idx + 1}`}`)
        .join('\n')
      setIssuesText(text)
      if (!silent) setError('')
    } catch (e) {
      if (silent) return
      if (e instanceof ApiError) {
        setError(`加载 dataset 失败：${e.message}\n${JSON.stringify(e.info, null, 2)}`)
      } else {
        setError(`加载 dataset 失败：${String(e?.message || e)}`)
      }
    }
  }

  async function refreshScenarioOptions({ silent = false } = {}) {
    setScenarioOptionsLoading(true)
    try {
      const list = await listDatasetScenarios()
      setScenarioOptions(Array.isArray(list) ? list.map((x) => String(x)) : [])
      if (!silent) setError('')
    } catch (e) {
      // 后端未启动时不影响使用：保留为空列表，继续走手填 scenarioId / General.json
      setScenarioOptions([])
      if (!silent) {
        if (e instanceof ApiError) {
          setError(`加载 scenario 列表失败：${e.message}\n${JSON.stringify(e.info, null, 2)}`)
        } else {
          setError(`加载 scenario 列表失败：${String(e?.message || e)}`)
        }
      }
    } finally {
      setScenarioOptionsLoading(false)
    }
  }

  // 初次加载：尝试用后端 dataset 覆盖默认 General（后端未启动时静默忽略）
  useEffect(() => {
    refreshScenarioOptions({ silent: true })
    loadScenarioFromDataset({ silent: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const callsByTime = useMemo(() => {
    const list = Array.isArray(calls) ? [...calls] : []
    // 以调用开始时间为主键；同一毫秒内保持稳定
    list.sort((a, b) => {
      const ta = Number(a?.tsStart ?? 0)
      const tb = Number(b?.tsStart ?? 0)
      if (ta !== tb) return ta - tb
      return String(a?.id ?? '').localeCompare(String(b?.id ?? ''))
    })
    return list
  }, [calls])

  function pushLog(line) {
    setLog((l) => [...l, `[${new Date().toLocaleTimeString()}] ${line}`])
  }

  function getAgentSecretPayload(agentId) {
    const full = countrySecretsFull?.[agentId]
    if (full !== undefined && full !== null) return full
    return countrySecrets?.[agentId] ?? ''
  }

  /**
   * 解析“二选一”输出（例如：同意/不同意、申请/不申请）。
   * 规则：先判否定前缀，再判肯定前缀；无法判定则返回 null。
   * @param {any} text
   * @param {string} yesPrefix
   * @param {string} noPrefix
   * @returns {boolean|null}
   */
  function decideYesNo(text, yesPrefix, noPrefix) {
    const t = String(text || '').trim()
    if (!t) return null
    if (t.startsWith(noPrefix)) return false
    if (t.startsWith(yesPrefix)) return true
    return null
  }

  function stripDebug(value) {
    if (Array.isArray(value)) return value.map(stripDebug)
    if (value && typeof value === 'object') {
      const out = {}
      for (const [k, v] of Object.entries(value)) {
        if (k === 'debug') continue
        out[k] = stripDebug(v)
      }
      return out
    }
    return value
  }

  function downloadJson(data, filename) {
    const text = JSON.stringify(data, null, 2)
    const blob = new Blob([text], { type: 'application/json;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  function exportSpeechesJson() {
    const snap = runSnapshot || null
    const payload = stripDebug({
      scenario_id: snap?.scenario_id ?? scenarioId,
      exported_at: nowIso(),
      rounds: snap?.rounds ?? Number(rounds || 0),
      speeches: snap?.speeches ?? speeches,
    })
    const safeScenario = String(payload.scenario_id || 'scenario').replace(/[\\/:*?"<>|]+/g, '_')
    downloadJson(payload, `diplomat_speeches_${safeScenario}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`)
  }

  function exportFullResultJson() {
    const snap = runSnapshot || null
    const payload = stripDebug({
      scenario_id: snap?.scenario_id ?? scenarioId,
      exported_at: nowIso(),
      input: snap?.input ?? null,
      speeches: snap?.speeches ?? speeches,
      round_summaries: snap?.round_summaries ?? summaries,
      adjudication: snap?.adjudication ?? adjudication,
      evaluation: snap?.evaluation ?? evaluation,
    })
    const safeScenario = String(payload.scenario_id || 'scenario').replace(/[\\/:*?"<>|]+/g, '_')
    downloadJson(payload, `diplomat_result_${safeScenario}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`)
  }

  const issues = useMemo(() => {
    const lines = String(issuesText || '')
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)

    return lines.map((line, idx) => {
      // 支持：1. xxx / 1、xxx / 1) xxx / 1）xxx
      const title = line.replace(/^\s*\d+\s*[.、)）]\s*/, '').trim()
      return {
        issue_id: `i-${idx + 1}`,
        title: title || line,
        description: '',
      }
    })
  }, [issuesText])

  const promptList = useMemo(() => {
    const items = []
    for (const c of callsByTime) {
      const dbg = c?.response?.debug
      if (!dbg) continue

      const isDiplomat = typeof c.url === 'string' && c.url.includes('/agents/') && c.url.includes('/speeches')
      const isMemory = typeof c.url === 'string' && c.url.includes('/round-summaries')
      const isEval = typeof c.url === 'string' && c.url.includes('/evaluations')

      let kind = 'other'
      let agentId = null
      let round = c?.request?.round ?? null
      if (isDiplomat) {
        kind = 'diplomat'
        agentId = c?.request?.agent_id ?? null
      } else if (isMemory) {
        kind = 'memory'
      } else if (isEval) {
        kind = 'evaluator'
      }

      items.push({
        id: c.id,
        kind,
        agentId,
        round,
        title: c.title,
        url: c.url,
        debug: dbg,
        tsStart: c?.tsStart ?? null,
      })
    }

    // 按时间升序（更符合“回放/时间线”阅读）
    items.sort((a, b) => {
      const ta = Number(a.tsStart ?? 0)
      const tb = Number(b.tsStart ?? 0)
      if (ta !== tb) return ta - tb
      const ra = Number(a.round ?? 999999)
      const rb = Number(b.round ?? 999999)
      if (ra !== rb) return ra - rb
      return String(a.agentId ?? '').localeCompare(String(b.agentId ?? ''))
    })
    return items
  }, [callsByTime])

  const diplomatPromptsByAgent = useMemo(() => {
    /** @type {Record<string, any[]>} */
    const map = {}
    for (const p of promptList) {
      if (p.kind !== 'diplomat') continue
      const a = p.agentId || '?'
      if (!map[a]) map[a] = []
      map[a].push(p)
    }
    return map
  }, [promptList])

  const stageMeta = useMemo(() => {
    const mk = (key, label, order, group) => ({ key, label, order, group })
    return [
      mk('stage_notice', '主席提示/计时', 10, '主持'),
      mk('opening_statement', '开场陈述（1分钟）', 20, '开场'),
      mk('negotiation_round_1', '第一轮谈判（30分钟）', 30, '谈判'),
      mk('consultation_request', '自由磋商申请', 40, '自由磋商'),
      mk('consultation_vote', '自由磋商投票', 50, '自由磋商'),
      mk('consultation_private_message', '自由磋商私聊（私密）', 60, '自由磋商'),
      mk('mid_press_statement', '中场新闻发布：陈述（2分钟）', 70, '中场发布'),
      mk('mid_press_question', '中场新闻发布：提问', 80, '中场发布'),
      mk('mid_press_answer', '中场新闻发布：回答', 90, '中场发布'),
      mk('negotiation_round_2', '第二轮谈判（20分钟）', 100, '谈判'),
      mk('issue_outcome_report', '主席逐项询问：各方汇报成果', 110, '裁定前询问'),
      mk('adjudicate', '主席裁定（按议题）', 120, '裁定'),
      mk('final_press_statement', '最终新闻发布：陈述（2分钟）', 130, '最终发布'),
      mk('final_press_question', '最终新闻发布：提问', 140, '最终发布'),
      mk('final_press_answer', '最终新闻发布：回答', 150, '最终发布'),
    ]
  }, [])

  const stageLabelMap = useMemo(() => {
    const m = {}
    for (const s of stageMeta) m[s.key] = s.label
    return m
  }, [stageMeta])

  const pipelineSpeeches = useMemo(() => {
    const items = Array.isArray(speeches) ? speeches : []
    const filtered = showPrivateSpeeches ? items : items.filter((s) => (s?.visibility || 'public') !== 'private')
    const withStage = filtered.map((s) => ({ ...s, stage: s?.stage || 'unknown' }))
    return withStage.sort((a, b) => {
      const ta = Date.parse(String(a?.timestamp || '')) || 0
      const tb = Date.parse(String(b?.timestamp || '')) || 0
      if (ta !== tb) return ta - tb
      const oa = stageMeta.find((x) => x.key === a.stage)?.order ?? 9999
      const ob = stageMeta.find((x) => x.key === b.stage)?.order ?? 9999
      if (oa !== ob) return oa - ob
      const ra = Number(a.round ?? 999999)
      const rb = Number(b.round ?? 999999)
      if (ra !== rb) return ra - rb
      return String(a.speaker ?? '').localeCompare(String(b.speaker ?? ''))
    })
  }, [speeches, showPrivateSpeeches, stageMeta])

  const pipelineGroups = useMemo(() => {
    /** @type {Record<string, any[]>} */
    const map = {}
    for (const s of pipelineSpeeches) {
      const key = s.stage || 'unknown'
      if (!map[key]) map[key] = []
      map[key].push(s)
    }
    return map
  }, [pipelineSpeeches])

  const speechesByTime = useMemo(() => {
    const list = Array.isArray(speeches) ? [...speeches] : []
    list.sort((a, b) => {
      const ta = Date.parse(String(a?.timestamp || '')) || 0
      const tb = Date.parse(String(b?.timestamp || '')) || 0
      if (ta !== tb) return ta - tb
      const ra = Number(a?.round ?? 999999)
      const rb = Number(b?.round ?? 999999)
      if (ra !== rb) return ra - rb
      return String(a?.speaker ?? '').localeCompare(String(b?.speaker ?? ''))
    })
    return list
  }, [speeches])

  function buildBackground() {
    /** @type {Record<string, {profile: string}>} */
    const countries = {}
    for (const a of agents) {
      countries[a] = { profile: countryProfiles[a] || '' }
    }
    return {
      scenario_id: scenarioId,
      world_background: worldBackground,
      countries,
    }
  }

  function buildIssues() {
    return { issues }
  }

  function resetRunData() {
    setSpeeches([])
    setSummaries([])
    setAdjudication(null)
    setEvaluation(null)
    setCalls([])
    setLog([])
    setError('')
    setRunSnapshot(null)
    abortRef.current.aborted = false
  }

  function startCall({ title, url, method, request }) {
    const id = `${Date.now()}-${++callSeq.current}`
    const item = { id, title, url, method, request, response: null, error: null, tsStart: Date.now(), tsEnd: null }
    setCalls((c) => [item, ...c])
    return id
  }

  function finishCall(id, { response, error }) {
    setCalls((c) =>
      c.map((x) =>
        x.id === id ? { ...x, response: response ?? null, error: error ?? null, tsEnd: Date.now() } : x
      )
    )
  }

  async function askChairRule() {
    const q = String(ruleQuestion || '').trim()
    if (!q) return
    const headers = debugPrompts ? { 'X-Debug-Prompts': '1' } : undefined
    const req = { stage: 'rule_qa', round: 1, question: q, constraints: { note: '仅回答规则/流程问题；议题实质问题不回答。' } }
    const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/chair/messages`
    const callId = startCall({ title: '主席规则答疑', url, method: 'POST', request: req })
    try {
      const out = await generateChairMessage(scenarioId, req, { headers })
      finishCall(callId, { response: out })
      setRuleAnswer(out?.message || '')
    } catch (e) {
      finishCall(callId, { error: e instanceof ApiError ? e.info : String(e?.message || e) })
      setRuleAnswer('')
      setError(String(e?.message || e))
    }
  }

  async function runAll() {
    setRunning(true)
    setError('')
    abortRef.current.aborted = false
    try {
      const background = buildBackground()
      const issuesObj = buildIssues()
      const headers = debugPrompts ? { 'X-Debug-Prompts': '1' } : undefined
      const maxRounds = Math.max(1, Math.min(2, Number(rounds || 2))) // 赛制默认两轮，前端先按两轮流程跑

      const stageConstraints = {
        opening_statement: { speaking_time_minutes: 1, target_chars: '150-180', max_chars: 180 },
        // 谈判阶段改造：同一轮允许多次发言，但需满足“累计不超过总时间/字数”。
        // 经验值：中文约 180 字/分钟（与 opening_statement 150-180 字/分钟保持一致）。
        negotiation_round_1: {
          timebox_minutes: 30,
          per_speaker_minutes: 3,
          // 兼容旧字段：max_chars 视为“每位代表团本轮总字数上限”
          max_chars: 540,
          // 新字段：明确总预算与分段策略（每轮每方最多发言 N 次）
          total_speaker_minutes: 3,
          total_max_chars: 540,
          max_turns_per_speaker: 3,
          chars_per_minute: 180,
          consult_once: true,
          consult_max_minutes: 3,
        },
        mid_press_statement: { speaking_time_minutes: 2, max_chars: 360 },
        mid_press_question: { max_chars: 60 },
        mid_press_answer: { max_chars: 120 },
        negotiation_round_2: {
          timebox_minutes: 20,
          per_speaker_minutes: 3,
          max_chars: 540,
          total_speaker_minutes: 3,
          total_max_chars: 540,
          max_turns_per_speaker: 3,
          chars_per_minute: 180,
          consult_once: true,
          consult_max_minutes: 3,
        },
        final_press_statement: { speaking_time_minutes: 2, max_chars: 360 },
        final_press_question: { max_chars: 60 },
        final_press_answer: { max_chars: 120 },
        consultation_request: { max_chars: 80 },
        consultation_vote: { max_chars: 80 },
        consultation_private_message: { max_chars: 240, consult_max_minutes: 3 },
        issue_outcome_report: { max_chars: 360 },
      }

      const countChars = (text) => String(text || '').length
      const estimateMinutesFromChars = (chars, charsPerMinute) => {
        const cpm = Number(charsPerMinute || 180)
        if (!Number.isFinite(cpm) || cpm <= 0) return 0
        const c = Number(chars || 0)
        if (!Number.isFinite(c) || c <= 0) return 0
        return c / cpm
      }

      /** @param {string} title @param {any} request @param {string} url */
      async function callChair(title, request, url) {
        const callId = startCall({ title, url, method: 'POST', request })
        try {
          const out = await generateChairMessage(scenarioId, request, { headers })
          finishCall(callId, { response: out })
          return out
        } catch (e) {
          finishCall(callId, { error: e instanceof ApiError ? e.info : String(e?.message || e) })
          throw e
        }
      }

      /** @param {string} agentId @param {any} payload @param {string} url @param {string} title */
      async function callDiplomat(agentId, payload, url, title) {
        const callId = startCall({ title, url, method: 'POST', request: payload })
        try {
          const out = await generateAgentSpeech(scenarioId, agentId, payload, { headers })
          finishCall(callId, { response: out })
          return out
        } catch (e) {
          finishCall(callId, { error: e instanceof ApiError ? e.info : String(e?.message || e) })
          throw e
        }
      }

      // 记录本次运行的输入快照，避免用户中途改配置导致导出不一致
      setRunSnapshot({
        scenario_id: scenarioId,
        rounds: maxRounds,
        input: stripDebug({
          background,
          issues: issuesObj,
          secrets: countrySecrets,
          rubrics,
        }),
        speeches: [],
        round_summaries: [],
        adjudication: null,
        evaluation: null,
      })

      let allSpeeches = []
      let allSummaries = []

      // 0) 主席开场：宣布流程与时间规则
      pushLog('主席：宣布流程与时间规则')
      {
        const req = {
          stage: 'stage_notice',
          round: 1,
          issues: issuesObj,
          speeches: [],
          constraints: {
            flow: [
              '开场陈述：每方 1 分钟（建议 150-180 字）',
              '第一轮谈判：30 分钟（主席计时；每方单次不超过 2-3 分钟）',
              '每轮可申请一次自由磋商：需半数以上代表团同意 + 主席同意；不超过 3 分钟',
              '中场新闻发布：每方 2 分钟，随后可一问一答',
              '第二轮谈判：20 分钟（多数比赛仅两轮）',
              '主席按议题逐项裁定是否达成及内容',
              '最终新闻发布：每方 2 分钟，可回答提问',
            ],
            time_warning_minutes: 5,
          },
        }
        const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/chair/messages`
        const out = await callChair('主席阶段提示（开场）', req, url)
        if (out?.message) {
          const speech = { round: 1, speaker: '主席', content: out.message || '', stage: 'stage_notice', visibility: 'public', timestamp: nowIso() }
          allSpeeches = [...allSpeeches, speech]
          setSpeeches(allSpeeches)
          setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
        }
      }

      // 1) 开场陈述：每方 1 分钟（150-180 字）
      pushLog('开场陈述：每方 1 分钟（建议 150-180 字）')
      {
        /** @type {any[]} */
        const openingSpeeches = []
        for (const agentId of agents) {
          if (abortRef.current.aborted) throw new Error('已中止')
          const payload = {
            agent_id: agentId,
            round: 1,
            stage: 'opening_statement',
            constraints: stageConstraints.opening_statement,
            secret: getAgentSecretPayload(agentId),
            background,
            issues: issuesObj,
            speech_history: openingSpeeches,
            shared_memory: allSummaries,
          }
          pushLog(`开场陈述：${agentId}…`)
          const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`
          const out = await callDiplomat(agentId, payload, url, `开场陈述（${agentId}）`)
          const speech = { round: 1, speaker: agentId, content: out.speech || '', stage: 'opening_statement', visibility: 'public', timestamp: nowIso() }
          openingSpeeches.push(speech)
          allSpeeches = [...allSpeeches, speech]
          setSpeeches(allSpeeches)
          setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
        }
      }

      // 2) 两轮谈判（第一轮 30 分钟 + 中场新闻发布 + 第二轮 20 分钟）
      for (let r = 1; r <= maxRounds; r++) {
        if (abortRef.current.aborted) throw new Error('已中止')
        const negoStage = r === 1 ? 'negotiation_round_1' : 'negotiation_round_2'
        pushLog(`开始第 ${r} 轮谈判（${negoStage}）`)

        /** @type {any[]} */
        const roundSpeeches = []

        // 允许“一轮谈判多次发言”：在同一轮内对 agents 做多轮调度；每方的累计时间/字数不超过总预算。
        const cfg = stageConstraints[negoStage] || {}
        const charsPerMinute = Number(cfg.chars_per_minute || 180)
        const totalCharsBudget = Number(cfg.total_max_chars || cfg.max_chars || 0)
        const totalMinutesBudget = Number(cfg.total_speaker_minutes || cfg.per_speaker_minutes || 0)
        const maxTurns = Math.max(1, Math.min(10, Number(cfg.max_turns_per_speaker || 1)))
        const turnsByAgent = Object.fromEntries(agents.map((a) => [a, 0]))

        // 预算足够小则退化为单次（避免极端配置导致“每次可用字数=0”）
        const basePerTurnChars =
          totalCharsBudget > 0 ? Math.max(30, Math.ceil(totalCharsBudget / maxTurns)) : Number(cfg.max_chars || 540)

        while (true) {
          if (abortRef.current.aborted) throw new Error('已中止')
          let anySpoke = false

          for (const agentId of agents) {
            if (abortRef.current.aborted) throw new Error('已中止')

            const mySpeeches = roundSpeeches.filter((s) => s.speaker === agentId && s.stage === negoStage)
            const usedChars = mySpeeches.reduce((sum, s) => sum + countChars(s?.content), 0)
            const usedMinutes = mySpeeches.reduce(
              (sum, s) => sum + estimateMinutesFromChars(countChars(s?.content), charsPerMinute),
              0
            )

            const remainingChars = Math.max(0, totalCharsBudget - usedChars)
            const remainingMinutes = Math.max(0, totalMinutesBudget - usedMinutes)
            const remainingCharsByTime = Math.max(0, Math.floor(remainingMinutes * charsPerMinute))

            if (turnsByAgent[agentId] >= maxTurns) continue
            if (totalCharsBudget > 0 && remainingChars <= 0) continue
            if (totalMinutesBudget > 0 && remainingMinutes <= 0) continue

            // 每次发言的上限：不超过剩余预算；同时控制在“分段目标字数”附近，鼓励多轮发言。
            const perTurnMax = Math.max(
              30,
              Math.min(basePerTurnChars, remainingChars || basePerTurnChars, remainingCharsByTime || basePerTurnChars)
            )

            const payload = {
              agent_id: agentId,
              round: r,
              stage: negoStage,
              constraints: {
                ...cfg,
                // max_chars 是外交官 prompt 的硬约束：这里传“本次最多可用字数”
                max_chars: perTurnMax,
                total_max_chars: totalCharsBudget,
                total_speaker_minutes: totalMinutesBudget,
                remaining_total_chars: remainingChars,
                remaining_minutes: Number(remainingMinutes.toFixed(2)),
                turn_index: turnsByAgent[agentId] + 1,
                max_turns_per_speaker: maxTurns,
              },
              secret: getAgentSecretPayload(agentId),
              background,
              issues: issuesObj,
              // 同轮内带“已发生的发言”，便于后发言者参考本轮上下文
              speech_history: roundSpeeches,
              shared_memory: allSummaries,
            }

            pushLog(`调用外交官 ${agentId}…（round=${r} turn=${turnsByAgent[agentId] + 1}/${maxTurns}）`)
            const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`
            const out = await callDiplomat(agentId, payload, url, `谈判发言（${agentId}, round=${r}, turn=${turnsByAgent[agentId] + 1}）`)
            const speech = {
              round: r,
              speaker: agentId,
              content: out.speech || '',
              stage: negoStage,
              visibility: 'public',
              timestamp: nowIso(),
            }
            roundSpeeches.push(speech)
            turnsByAgent[agentId] = (turnsByAgent[agentId] || 0) + 1
            anySpoke = true

            allSpeeches = [...allSpeeches, speech]
            setSpeeches(allSpeeches)
            setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
            pushLog(`外交官 ${agentId} 完成`)
          }

          if (!anySpoke) break
        }

        // 主席：每轮结束前 5 分钟提示（按赛制要求，模拟一次）
        {
          const chairUrl = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/chair/messages`
          const chairReq = {
            stage: 'stage_notice',
            round: r,
            issues: issuesObj,
            speeches: [],
            constraints: {
              kind: 'time_warning',
              round: r,
              time_remaining_minutes: 5,
              instruction: '请各代表团把握最后 5 分钟，聚焦尚未解决的议题并提出可执行条款。',
            },
          }
          const chairOut = await callChair(`主席：5分钟提示（round=${r}）`, chairReq, chairUrl)
          if (chairOut?.message) {
            const speech = { round: r, speaker: '主席', content: chairOut.message || '', stage: 'stage_notice', visibility: 'public', timestamp: nowIso() }
            allSpeeches = [...allSpeeches, speech]
            setSpeeches(allSpeeches)
            setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
          }
        }

        // 自由磋商（每轮最多一次）：任一方可申请，需半数以上同意 + 主席同意；最长 3 分钟
        pushLog(`第 ${r} 轮：自由磋商申请（可选）`)
        {
          /** @type {{agentId: string, text: string}[]} */
          const requests = []
          for (const agentId of agents) {
            if (abortRef.current.aborted) throw new Error('已中止')
            const payload = {
              agent_id: agentId,
              round: r,
              stage: 'consultation_request',
              constraints: stageConstraints.consultation_request,
              secret: getAgentSecretPayload(agentId),
              background,
              issues: issuesObj,
              speech_history: roundSpeeches,
              shared_memory: allSummaries,
            }
            const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`
            const out = await callDiplomat(agentId, payload, url, `自由磋商申请（${agentId}, round=${r}）`)
            const text = String(out.speech || '').trim()
            const speech = { round: r, speaker: agentId, content: text, stage: 'consultation_request', visibility: 'public', timestamp: nowIso() }
            allSpeeches = [...allSpeeches, speech]
            setSpeeches(allSpeeches)
            setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
            requests.push({ agentId, text })
          }

          const applicants = requests.filter((x) => decideYesNo(x.text, '申请', '不申请') === true)
          if (applicants.length) {
            const proposer = applicants[0].agentId
            pushLog(`自由磋商：收到申请（提议方：${proposer}），开始投票`)

            let yes = 0
            /** @type {{agentId: string, vote: string}[]} */
            const votes = []
            for (const voter of agents) {
              if (abortRef.current.aborted) throw new Error('已中止')
              const payload = {
                agent_id: voter,
                round: r,
                stage: 'consultation_vote',
                constraints: { ...stageConstraints.consultation_vote, proposer },
                secret: getAgentSecretPayload(voter),
                background,
                issues: issuesObj,
                speech_history: [{ speaker: proposer, content: applicants[0].text }],
                shared_memory: allSummaries,
              }
              const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(voter)}/speeches`
              const out = await callDiplomat(voter, payload, url, `自由磋商投票（${voter}→${proposer}, round=${r}）`)
              const text = String(out.speech || '').trim()
              const speech = { round: r, speaker: voter, content: text, stage: 'consultation_vote', visibility: 'public', timestamp: nowIso() }
              allSpeeches = [...allSpeeches, speech]
              setSpeeches(allSpeeches)
              setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
              votes.push({ agentId: voter, vote: text })
              if (decideYesNo(text, '同意', '不同意') === true) yes++
            }

            const threshold = Math.ceil(agents.length / 2)
            const approvedByDelegations = yes >= threshold

            // 主席审核（这里按赛制：达到门槛即同意；主席可在 LLM 输出中给出提醒）
            const chairUrl = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/chair/messages`
            const chairReq = {
              stage: 'stage_notice',
              round: r,
              issues: issuesObj,
              speeches: [],
              constraints: {
                kind: 'consultation_decision',
                proposer,
                votes,
                yes,
                threshold,
                decision: approvedByDelegations ? '同意' : '不同意',
                consult_max_minutes: 3,
              },
            }
            const chairOut = await callChair(`主席：自由磋商审核（round=${r}）`, chairReq, chairUrl)
            if (chairOut?.message) {
              const speech = { round: r, speaker: '主席', content: chairOut.message || '', stage: 'stage_notice', visibility: 'public', timestamp: nowIso() }
              allSpeeches = [...allSpeeches, speech]
              setSpeeches(allSpeeches)
              setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
            }

            if (approvedByDelegations) {
              // 规则升级：私下磋商仅限“提出者 + 一名接受者”之间进行，其他代表团不可见且不参与对话。
              // 接受者选择策略（无需额外交互）：从投票“同意”的代表团中取第一个（排除提出者）。若无，则跳过私聊。
              const acceptor = votes.find((x) => x.agentId !== proposer && decideYesNo(x.vote, '同意', '不同意') === true)?.agentId || null
              if (!acceptor) {
                pushLog('自由磋商获批，但未找到接受者（无同意方），跳过私聊')
              } else {
                pushLog(`自由磋商获批：仅 ${proposer} ↔ ${acceptor} 进行 2-3 分钟私密对话（其他代表团不可见）`)

                // 用多轮短消息模拟“2-3 分钟对话”，并把上下文仅提供给双方
                const totalMinutes = 3
                const turns = 4 // A→B→A→B
                const perTurnMax = Math.max(60, Math.floor((totalMinutes * 180) / turns)) // 经验值：180字/分钟

                /** @type {any[]} */
                const privateHistory = []
                const pair = [proposer, acceptor]
                for (let t = 0; t < turns; t++) {
                  if (abortRef.current.aborted) throw new Error('已中止')
                  const speaker = pair[t % 2]
                  const counterpart = speaker === proposer ? acceptor : proposer

                  const payload = {
                    agent_id: speaker,
                    round: r,
                    stage: 'consultation_private_message',
                    constraints: {
                      ...stageConstraints.consultation_private_message,
                      // 明确“仅对话双方可见”的语境与对手信息，避免模型写成对公众发言
                      proposer,
                      acceptor,
                      counterpart,
                      // 硬约束：单条不超字数；对话总时长提示（2-3分钟）
                      max_chars: perTurnMax,
                      consult_total_minutes: totalMinutes,
                      dialog_turn: t + 1,
                      dialog_total_turns: turns,
                    },
                    secret: getAgentSecretPayload(speaker),
                    background,
                    issues: issuesObj,
                    // 只给双方私聊历史，不把私聊泄露给 roundSpeeches / 其他 agent
                    speech_history: privateHistory,
                    shared_memory: allSummaries,
                  }
                  const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(speaker)}/speeches`
                  const out = await callDiplomat(speaker, payload, url, `自由磋商私聊（${speaker}↔${counterpart}, round=${r}, turn=${t + 1}）`)
                  const speech = {
                    round: r,
                    speaker,
                    content: out.speech || '',
                    stage: 'consultation_private_message',
                    visibility: 'private',
                    timestamp: nowIso(),
                  }
                  privateHistory.push(speech)
                  allSpeeches = [...allSpeeches, speech]
                  setSpeeches(allSpeeches)
                  setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
                }
              }
            }
          } else {
            pushLog('自由磋商：无人申请，跳过')
          }
        }

        pushLog(`第 ${r} 轮：生成记忆摘要`)
        const memPayload = { round: r, speeches: roundSpeeches }
        const memUrl = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/round-summaries`
        const memCallId = startCall({ title: `记忆摘要生成（round=${r}）`, url: memUrl, method: 'POST', request: memPayload })
        let summary
        try {
          summary = await generateRoundSummary(scenarioId, memPayload, { headers })
          finishCall(memCallId, { response: summary })
        } catch (e) {
          finishCall(memCallId, { error: e instanceof ApiError ? e.info : String(e?.message || e) })
          throw e
        }
        allSummaries = [...allSummaries, summary]
        setSummaries(allSummaries)
        setRunSnapshot((s) => (s ? { ...s, round_summaries: allSummaries } : s))
        pushLog(`第 ${r} 轮摘要完成`)

        // 仅第一轮后进行中场新闻发布
        if (r === 1 && maxRounds >= 2) {
          pushLog('中场新闻发布：每方 2 分钟 + 一问一答')

          const chairUrl = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/chair/messages`
          const chairReq = {
            stage: 'stage_notice',
            round: 1,
            issues: issuesObj,
            speeches: [],
            constraints: { notice: '进入中场新闻发布。每方 2 分钟陈述谈判进展与立场；每方陈述后允许其他方或观察员进行一问一答。' },
          }
          const chairOut = await callChair('主席阶段提示（中场新闻发布）', chairReq, chairUrl)
          if (chairOut?.message) {
            const speech = { round: 1, speaker: '主席', content: chairOut.message || '', stage: 'stage_notice', visibility: 'public', timestamp: nowIso() }
            allSpeeches = [...allSpeeches, speech]
            setSpeeches(allSpeeches)
            setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
          }

          /** @type {any[]} */
          const pressStatements = []
          for (let idx = 0; idx < agents.length; idx++) {
            const agentId = agents[idx]
            if (abortRef.current.aborted) throw new Error('已中止')

            // 陈述
            {
              const payload = {
                agent_id: agentId,
                round: 1,
                stage: 'mid_press_statement',
                constraints: stageConstraints.mid_press_statement,
                secret: getAgentSecretPayload(agentId),
                background,
                issues: issuesObj,
                speech_history: pressStatements,
                shared_memory: allSummaries,
              }
              const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`
              const out = await callDiplomat(agentId, payload, url, `中场发布陈述（${agentId}）`)
              const speech = { round: 1, speaker: agentId, content: out.speech || '', stage: 'mid_press_statement', visibility: 'public', timestamp: nowIso() }
              pressStatements.push(speech)
              allSpeeches = [...allSpeeches, speech]
              setSpeeches(allSpeeches)
              setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
            }

            // 一问一答：默认由下一方提问（简化实现）
            const asker = agents[(idx + 1) % agents.length]
            if (asker && asker !== agentId) {
              // 提问
              let qSpeech = null
              {
                const payload = {
                  agent_id: asker,
                  round: 1,
                  stage: 'mid_press_question',
                  constraints: stageConstraints.mid_press_question,
                  secret: getAgentSecretPayload(asker),
                  background,
                  issues: issuesObj,
                  speech_history: [{ speaker: agentId, content: pressStatements[pressStatements.length - 1]?.content || '' }],
                  shared_memory: allSummaries,
                }
                const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(asker)}/speeches`
                const out = await callDiplomat(asker, payload, url, `中场发布提问（${asker}→${agentId}）`)
                qSpeech = { round: 1, speaker: asker, content: out.speech || '', stage: 'mid_press_question', visibility: 'public', timestamp: nowIso() }
                allSpeeches = [...allSpeeches, qSpeech]
                setSpeeches(allSpeeches)
                setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
              }
              // 回答
              {
                const payload = {
                  agent_id: agentId,
                  round: 1,
                  stage: 'mid_press_answer',
                  constraints: stageConstraints.mid_press_answer,
                  secret: getAgentSecretPayload(agentId),
                  background,
                  issues: issuesObj,
                  speech_history: [{ speaker: asker, content: qSpeech?.content || '' }],
                  shared_memory: allSummaries,
                }
                const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`
                const out = await callDiplomat(agentId, payload, url, `中场发布回答（${agentId}→${asker}）`)
                const aSpeech = { round: 1, speaker: agentId, content: out.speech || '', stage: 'mid_press_answer', visibility: 'public', timestamp: nowIso() }
                allSpeeches = [...allSpeeches, aSpeech]
                setSpeeches(allSpeeches)
                setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
              }
            }
          }
        }
      }

      // 3) 主席最终裁定（按议题逐项裁定）
      // 3) 主席逐项询问各方成果（按议题）
      pushLog('主席：按议题逐项询问各方谈判成果')
      {
        const issuesList = Array.isArray(issuesObj?.issues) ? issuesObj.issues : []
        for (const issue of issuesList) {
          if (abortRef.current.aborted) throw new Error('已中止')
          const chairUrl = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/chair/messages`
          const chairReq = {
            stage: 'stage_notice',
            round: maxRounds,
            issues: issuesObj,
            speeches: [],
            constraints: {
              kind: 'issue_inquiry',
              issue_id: issue?.issue_id,
              issue_title: issue?.title,
              instruction:
                '请各代表团就该议题逐项汇报：达成/部分达成/未达成，并尽量用可执行条款描述。主席随后将据此与公开发言进行裁定。',
            },
          }
          const chairOut = await callChair(`主席：逐项询问（${issue?.title || issue?.issue_id || '议题'}）`, chairReq, chairUrl)
          if (chairOut?.message) {
            const speech = { round: maxRounds, speaker: '主席', content: chairOut.message || '', stage: 'stage_notice', visibility: 'public', timestamp: nowIso() }
            allSpeeches = [...allSpeeches, speech]
            setSpeeches(allSpeeches)
            setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
          }

          /** @type {any[]} */
          const issueReports = []
          for (const agentId of agents) {
            if (abortRef.current.aborted) throw new Error('已中止')
            const payload = {
              agent_id: agentId,
              round: maxRounds,
              stage: 'issue_outcome_report',
              constraints: { ...stageConstraints.issue_outcome_report, issue_id: issue?.issue_id, issue_title: issue?.title },
              secret: getAgentSecretPayload(agentId),
              background,
              issues: issuesObj,
              speech_history: issueReports,
              shared_memory: allSummaries,
            }
            const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`
            const out = await callDiplomat(agentId, payload, url, `成果汇报（${agentId}｜${issue?.title || issue?.issue_id || ''}）`)
            const speech = {
              round: maxRounds,
              speaker: agentId,
              content: out.speech || '',
              stage: 'issue_outcome_report',
              visibility: 'public',
              timestamp: nowIso(),
            }
            issueReports.push(speech)
            allSpeeches = [...allSpeeches, speech]
            setSpeeches(allSpeeches)
            setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
          }
        }
      }

      pushLog('主席：按议题逐项裁定谈判成果')
      {
        const req = {
          stage: 'adjudicate',
          round: maxRounds,
          background,
          issues: issuesObj,
          speeches: allSpeeches.filter((s) => s.visibility !== 'private'),
          constraints: { adjudication_basis: '仅以公开发言中明确同意/承诺为准；含糊则部分达成或未达成。' },
        }
        const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/chair/messages`
        const out = await callChair('主席裁定（按议题）', req, url)
        setAdjudication(out?.adjudication || null)
        setRunSnapshot((s) => (s ? { ...s, adjudication: out?.adjudication || null } : s))
        if (out?.message) {
          const speech = { round: maxRounds, speaker: '主席', content: out.message || '', stage: 'adjudicate', visibility: 'public', timestamp: nowIso() }
          allSpeeches = [...allSpeeches, speech]
          setSpeeches(allSpeeches)
          setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
        }
      }

      // 4) 最后新闻发布：每方 2 分钟 + 一问一答（简化为下一方提问）
      pushLog('最后新闻发布：每方 2 分钟 + 一问一答')
      {
        /** @type {any[]} */
        const finalStatements = []
        for (let idx = 0; idx < agents.length; idx++) {
          const agentId = agents[idx]
          if (abortRef.current.aborted) throw new Error('已中止')

          // 陈述
          {
            const payload = {
              agent_id: agentId,
              round: maxRounds,
              stage: 'final_press_statement',
              constraints: stageConstraints.final_press_statement,
              secret: getAgentSecretPayload(agentId),
              background,
              issues: issuesObj,
              speech_history: finalStatements,
              shared_memory: allSummaries,
            }
            const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`
            const out = await callDiplomat(agentId, payload, url, `最终发布陈述（${agentId}）`)
            const speech = { round: maxRounds, speaker: agentId, content: out.speech || '', stage: 'final_press_statement', visibility: 'public', timestamp: nowIso() }
            finalStatements.push(speech)
            allSpeeches = [...allSpeeches, speech]
            setSpeeches(allSpeeches)
            setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
          }

          const asker = agents[(idx + 1) % agents.length]
          if (asker && asker !== agentId) {
            let qSpeech = null
            // 提问
            {
              const payload = {
                agent_id: asker,
                round: maxRounds,
                stage: 'final_press_question',
                constraints: stageConstraints.final_press_question,
                secret: getAgentSecretPayload(asker),
                background,
                issues: issuesObj,
                speech_history: [{ speaker: agentId, content: finalStatements[finalStatements.length - 1]?.content || '' }],
                shared_memory: allSummaries,
              }
              const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(asker)}/speeches`
              const out = await callDiplomat(asker, payload, url, `最终发布提问（${asker}→${agentId}）`)
              qSpeech = { round: maxRounds, speaker: asker, content: out.speech || '', stage: 'final_press_question', visibility: 'public', timestamp: nowIso() }
              allSpeeches = [...allSpeeches, qSpeech]
              setSpeeches(allSpeeches)
              setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
            }
            // 回答
            {
              const payload = {
                agent_id: agentId,
                round: maxRounds,
                stage: 'final_press_answer',
                constraints: stageConstraints.final_press_answer,
                secret: getAgentSecretPayload(agentId),
                background,
                issues: issuesObj,
                speech_history: [{ speaker: asker, content: qSpeech?.content || '' }],
                shared_memory: allSummaries,
              }
              const url = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`
              const out = await callDiplomat(agentId, payload, url, `最终发布回答（${agentId}→${asker}）`)
              const aSpeech = { round: maxRounds, speaker: agentId, content: out.speech || '', stage: 'final_press_answer', visibility: 'public', timestamp: nowIso() }
              allSpeeches = [...allSpeeches, aSpeech]
              setSpeeches(allSpeeches)
              setRunSnapshot((s) => (s ? { ...s, speeches: allSpeeches } : s))
            }
          }
        }
      }

      // 5) 评分（仅对代表团发言评分，不含主席）
      pushLog('开始评分…')
      const evalPayload = {
        scenario_id: scenarioId,
        rubrics,
        speeches: allSpeeches.filter((s) => agents.includes(s.speaker)),
      }
      const evalUrl = `/api/v1/scenarios/${encodeURIComponent(scenarioId)}/evaluations`
      const evalCallId = startCall({ title: '评分（evaluation）', url: evalUrl, method: 'POST', request: evalPayload })
      let evalOut
      try {
        evalOut = await generateEvaluation(scenarioId, evalPayload, { headers })
        finishCall(evalCallId, { response: evalOut })
      } catch (e) {
        finishCall(evalCallId, { error: e instanceof ApiError ? e.info : String(e?.message || e) })
        throw e
      }
      setEvaluation(evalOut)
      setRunSnapshot((s) => (s ? { ...s, evaluation: evalOut } : s))
      pushLog('评分完成')
    } catch (e) {
      if (e instanceof ApiError) {
        setError(`接口错误：${e.message}\n${JSON.stringify(e.info, null, 2)}`)
      } else {
        setError(String(e?.message || e))
      }
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brandTitle">Diplomat 流程控制台</div>
        </div>
        <div className="topbarActions">
          <label className="toggle">
            <input type="checkbox" checked={debugPrompts} onChange={(e) => setDebugPrompts(e.target.checked)} />
            <span>显示 prompt</span>
          </label>
          <button
            className="btn"
            onClick={exportSpeechesJson}
            disabled={running || (runSnapshot?.speeches?.length ?? speeches.length) === 0}
            title="导出 speeches（不包含任何 debug 信息）"
          >
            导出 speeches JSON
          </button>
          <button
            className="btn"
            onClick={exportFullResultJson}
            disabled={
              running ||
              ((runSnapshot?.speeches?.length ?? speeches.length) === 0 &&
                (runSnapshot?.round_summaries?.length ?? summaries.length) === 0 &&
                !(runSnapshot?.evaluation ?? evaluation))
            }
            title="导出本次运行的输入+结果（自动剔除 debug 字段）"
          >
            导出完整结果 JSON
          </button>
          <button className="btn" onClick={resetRunData} disabled={running}>
            重置
          </button>
          <button
            className="btn primary"
            onClick={runAll}
            disabled={running || !scenarioId.trim() || Number(rounds || 0) <= 0}
          >
            {running ? '运行中…' : '一键运行'}
          </button>
          <button className="btn danger" onClick={() => (abortRef.current.aborted = true)} disabled={!running}>
            中止
          </button>
        </div>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>配置</h2>

          <div className="row">
            <label>
              <div className="label">scenario_id</div>
              <input value={scenarioId} onChange={(e) => setScenarioId(e.target.value)} placeholder="例如 demo-001" />
            </label>
            <label>
              <div className="label">轮数</div>
              <input
                type="number"
                min={1}
                max={20}
                value={rounds}
                onChange={(e) => setRounds(Number(e.target.value))}
              />
            </label>
          </div>

          <div className="row" style={{ marginTop: 8, alignItems: 'flex-end' }}>
            <label style={{ flex: 1 }}>
              <div className="label">从 dataset 选择 scenario_id</div>
              <select
                value={scenarioOptions.includes(String(scenarioId || '')) ? String(scenarioId || '') : ''}
                onChange={(e) => {
                  const v = String(e.target.value || '').trim()
                  if (!v) return
                  setScenarioId(v)
                }}
                disabled={scenarioOptionsLoading || scenarioOptions.length === 0}
              >
                <option value="">
                  {scenarioOptionsLoading
                    ? '加载中…'
                    : scenarioOptions.length
                      ? '请选择…'
                      : '（无列表：请启动后端或手动输入）'}
                </option>
                {scenarioOptions.map((sid) => (
                  <option key={sid} value={sid}>
                    {sid}
                  </option>
                ))}
              </select>
            </label>
            <button className="btn" onClick={() => refreshScenarioOptions()} disabled={running || scenarioOptionsLoading}>
              刷新列表
            </button>
            <button
              className="btn"
              onClick={() => loadScenarioFromDataset()}
              disabled={running || !String(scenarioId || '').trim()}
              title="自动加载：优先后端；否则 /dataset/{scenario_id}.json"
            >
              加载
            </button>
          </div>

          <div className="block">
            <div className="label">世界观 Background.world_background</div>
            <textarea value={worldBackground} onChange={(e) => setWorldBackground(e.target.value)} rows={4} />
          </div>

          <div className="block">
            <div className="label">各国 Profile（A/B/C/D）</div>
            <div className="row4">
              {agents.map((a) => (
                <label key={a}>
                  <div className="labelMini">{a}</div>
                  <textarea
                    value={countryProfiles[a] || ''}
                    onChange={(e) => setCountryProfiles((p) => ({ ...p, [a]: e.target.value }))}
                    rows={4}
                  />
                </label>
              ))}
            </div>
          </div>

          <div className="block">
            <div className="label">各国 Secret（私密信息，仅供本方使用，不应公开分享）</div>
            <div className="row4">
              {agents.map((a) => (
                <label key={`${a}-secret`}>
                  <div className="labelMini">{a}</div>
                  <textarea
                    value={countrySecrets?.[a] || ''}
                    onChange={(e) => {
                      const v = e.target.value
                      setCountrySecrets((p) => ({ ...(p || {}), [a]: v }))
                      // 同步更新完整结构化 secrets 的 notes 字段，保留 decision_factors 等其他字段
                      setCountrySecretsFull((p) => {
                        const cur = p?.[a]
                        const base = cur && typeof cur === 'object' ? cur : { notes: String(cur || '') }
                        return { ...(p || {}), [a]: { ...base, notes: v } }
                      })
                    }}
                    rows={4}
                    placeholder="例如：底线、优先级、可交换筹码、不得公开的信息…"
                  />
                </label>
              ))}
            </div>
          </div>

          <div className="block">
            <div className="label">Issues（编号列表，每行一个议题）</div>
            <textarea
              value={issuesText}
              onChange={(e) => setIssuesText(e.target.value)}
              rows={8}
            />
            <div className="hint">
              格式示例：1. 议题A{'\n'}2. 议题B{'\n'}（会自动解析为后端所需的 Issues JSON）
            </div>
            <details className="hint" style={{ marginTop: 8 }}>
              <summary style={{ cursor: 'pointer' }}>查看解析后的 Issues JSON</summary>
              <pre className="json" style={{ marginTop: 8 }}>{JSON.stringify({ issues }, null, 2)}</pre>
            </details>
          </div>

          <div className="block">
            <div className="label">Rubrics（JSON 数组）</div>
            <textarea
              value={JSON.stringify(rubrics, null, 2)}
              onChange={(e) => {
                try {
                  const obj = JSON.parse(e.target.value || '[]')
                  setRubrics(Array.isArray(obj) ? obj : [])
                } catch {
                  // ignore
                }
              }}
              rows={7}
            />
          </div>

          <div className="block">
            <div className="label">向主席询问规则/流程（不会回答议题裁定或谈判内容）</div>
            <textarea value={ruleQuestion} onChange={(e) => setRuleQuestion(e.target.value)} rows={3} placeholder="例如：自由磋商需要多少代表团同意？每轮能申请几次？" />
            <div className="row" style={{ marginTop: 8 }}>
              <button className="btn" onClick={askChairRule} disabled={running || !String(ruleQuestion || '').trim()}>
                询问主席
              </button>
            </div>
            {ruleAnswer ? <pre className="json" style={{ marginTop: 8 }}>{ruleAnswer}</pre> : null}
          </div>
        </section>

        <section className="panel">
          <h2>运行日志</h2>
          {error ? (
            <pre className="error">{error}</pre>
          ) : (
            <pre className="log">{log.length ? log.join('\n') : '点击“一键运行”开始。'}</pre>
          )}

          <h2 style={{ marginTop: 16 }}>所有 Agent 的 Prompt 列表</h2>
          {!debugPrompts ? (
            <div className="hint">未开启“显示 prompt”，后端不会返回 prompt。打开右上角开关后重新运行即可。</div>
          ) : Object.keys(diplomatPromptsByAgent).length === 0 ? (
            <div className="hint">暂无（运行后会按 A/B/C/D 聚合显示每次外交官调用的 prompt）。</div>
          ) : (
            <div className="promptList">
              {agents.map((a) => (
                <details key={a} className="promptGroup" open>
                  <summary className="promptGroupSummary">
                    <span className="promptGroupTitle">外交官 {a}</span>
                    <span className="promptGroupMeta">{(diplomatPromptsByAgent[a] || []).length} 次</span>
                  </summary>
                  <div className="promptGroupBody">
                    {(diplomatPromptsByAgent[a] || []).map((p) => (
                      <details key={p.id} className="promptItem">
                        <summary className="promptItemSummary">
                          <span>round={p.round ?? '-'}</span>
                          <span className="hint">model: {p.debug?.model || 'unknown'} / temp: {String(p.debug?.temperature ?? '')}</span>
                        </summary>
                        <div className="callPrompt">
                          <div className="callCols">
                            <div>
                              <div className="resultTitle">system</div>
                              <pre className="json">{p.debug?.system || ''}</pre>
                            </div>
                            <div>
                              <div className="resultTitle">user</div>
                              <pre className="json">{p.debug?.user || ''}</pre>
                            </div>
                          </div>
                        </div>
                      </details>
                    ))}
                  </div>
                </details>
              ))}
            </div>
          )}

          <div className="rowActions">
            <h2 style={{ margin: 0 }}>调用上下文（每次请求的完整 payload）</h2>
            <button className="btn" onClick={() => setCalls([])} disabled={running || calls.length === 0}>
              清空
            </button>
          </div>
          <div className="calls">
            {callsByTime.length === 0 ? (
              <div className="hint">运行后会记录每一次 POST 的请求/响应（包含 speech_history、shared_memory 等完整上下文）。</div>
            ) : (
              callsByTime.map((c) => (
                <details key={c.id} className="callItem">
                  <summary className="callSummary">
                    <span className="callTitle">{c.title}</span>
                    <span className="callMeta">
                      {c.method} {c.url}
                      {c.error ? <span className="badge err">错误</span> : c.response ? <span className="badge ok">OK</span> : null}
                    </span>
                  </summary>
                  <div className="callBody">
                    <div className="callCols">
                      <div>
                        <div className="resultTitle">Request</div>
                        <pre className="json">{JSON.stringify(c.request, null, 2)}</pre>
                      </div>
                      <div>
                        <div className="resultTitle">{c.error ? 'Error' : 'Response'}</div>
                        <pre className="json">{JSON.stringify(c.error || c.response, null, 2)}</pre>
                      </div>
                    </div>
                    {c.response?.debug ? (
                      <div className="callPrompt">
                        <div className="resultTitle">Prompt（后端实际使用）</div>
                        <div className="callCols">
                          <div>
                            <div className="resultTitle">system</div>
                            <pre className="json">{c.response.debug.system || ''}</pre>
                          </div>
                          <div>
                            <div className="resultTitle">user</div>
                            <pre className="json">{c.response.debug.user || ''}</pre>
                          </div>
                        </div>
                        <div className="hint">model: {c.response.debug.model || 'unknown'} / temperature: {String(c.response.debug.temperature ?? '')}</div>
                      </div>
                    ) : null}
                  </div>
                </details>
              ))
            )}
          </div>

          <h2 style={{ marginTop: 16 }}>结果</h2>
          <div className="resultGrid">
            <div className="resultCard">
              <div className="resultTitle">流程视图（按阶段聚合）</div>
              <div className="pipelineToolbar">
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={showPrivateSpeeches}
                    onChange={(e) => setShowPrivateSpeeches(e.target.checked)}
                  />
                  <span>显示私聊/私密内容</span>
                </label>
                <div className="hint" style={{ margin: 0 }}>
                  说明：默认仅展示 public 发言；“自由磋商私聊”会被标记为 private。
                </div>
              </div>

              <div className="pipeline">
                {stageMeta
                  .filter((m) => (pipelineGroups[m.key] || []).length > 0)
                  .map((m) => {
                    const items = pipelineGroups[m.key] || []
                    return (
                      <details key={m.key} className="stageCard" open>
                        <summary className="stageSummary">
                          <span className="stageTitle">{m.label}</span>
                          <span className="stageMeta">
                            <span className="badge ok">{items.length} 条</span>
                          </span>
                        </summary>
                        <div className="stageBody">
                          {items.map((s, idx) => {
                            const isPrivate = (s?.visibility || 'public') === 'private'
                            return (
                              <div key={`${m.key}-${idx}-${s.timestamp || ''}`} className={`speechItem ${isPrivate ? 'private' : ''}`}>
                                <div className="speechHeader">
                                  <div className="speechLeft">
                                    <span className="speakerTag">{s.speaker}</span>
                                    <span className="stageTag">{stageLabelMap[s.stage] || s.stage}</span>
                                    {isPrivate ? <span className="privateTag">private</span> : null}
                                  </div>
                                  <div className="speechRight">
                                    <span className="hint" style={{ margin: 0 }}>
                                      round={String(s.round ?? '-')}{s.timestamp ? ` · ${new Date(s.timestamp).toLocaleString()}` : ''}
                                    </span>
                                  </div>
                                </div>
                                <pre className="speechText">{String(s.content || '').trim() || '（空）'}</pre>
                              </div>
                            )
                          })}
                        </div>
                      </details>
                    )
                  })}

                {pipelineSpeeches.length === 0 ? <div className="hint">暂无（运行后这里会按阶段显示完整流程）。</div> : null}
              </div>
            </div>

            <details className="resultCard">
              <summary className="resultTitle" style={{ cursor: 'pointer' }}>主席裁定（Adjudication）</summary>
              <pre className="json">{adjudication ? JSON.stringify(adjudication, null, 2) : '暂无'}</pre>
            </details>

            <details className="resultCard">
              <summary className="resultTitle" style={{ cursor: 'pointer' }}>Round Summaries（{summaries.length}）</summary>
              <pre className="json">{summaries.length ? JSON.stringify(summaries, null, 2) : '暂无'}</pre>
            </details>

            <details className="resultCard">
              <summary className="resultTitle" style={{ cursor: 'pointer' }}>Evaluation</summary>
              <pre className="json">{evaluation ? JSON.stringify(evaluation, null, 2) : '暂无'}</pre>
            </details>

            <details className="resultCard">
              <summary className="resultTitle" style={{ cursor: 'pointer' }}>Speeches 原始 JSON（{speeches.length}）</summary>
              <pre className="json">{speeches.length ? JSON.stringify(speechesByTime, null, 2) : '暂无'}</pre>
            </details>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
