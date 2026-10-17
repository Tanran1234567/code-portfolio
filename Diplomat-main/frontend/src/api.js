export class ApiError extends Error {
  /** @param {string} message @param {any} info */
  constructor(message, info) {
    super(message)
    this.name = 'ApiError'
    this.info = info
  }
}

/**
 * @template T
 * @param {string} url
 * @param {RequestInit & { json?: any }} [init]
 * @returns {Promise<T>}
 */
export async function apiFetch(url, init = {}) {
  const headers = new Headers(init.headers || {})
  if (init.json !== undefined) headers.set('Content-Type', 'application/json')

  const resp = await fetch(url, {
    ...init,
    headers,
    body: init.json !== undefined ? JSON.stringify(init.json) : init.body,
  })

  const text = await resp.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!resp.ok) throw new ApiError(`HTTP ${resp.status} ${resp.statusText}`, data)
  return /** @type {T} */ (data)
}

export function nowIso() {
  return new Date().toISOString()
}

/**
 * @param {string} scenarioId
 */
export function getScenarioBackground(scenarioId) {
  return apiFetch(`/api/v1/scenarios/${encodeURIComponent(scenarioId)}/background`)
}

/**
 * @param {string} scenarioId
 */
export function getScenarioIssues(scenarioId) {
  return apiFetch(`/api/v1/scenarios/${encodeURIComponent(scenarioId)}/issues`)
}

/**
 * @param {string} scenarioId
 * @param {{ full?: boolean }} [opts]
 */
export function getScenarioSecrets(scenarioId, opts = {}) {
  const full = opts?.full ? '1' : '0'
  return apiFetch(`/api/v1/scenarios/${encodeURIComponent(scenarioId)}/secrets?full=${full}`)
}

export function listDatasetScenarios() {
  return apiFetch(`/api/v1/datasets/scenarios`)
}

/**
 * @param {string} scenarioId
 * @param {string} agentId
 * @param {any} payload
 */
export function generateAgentSpeech(scenarioId, agentId, payload, opts = {}) {
  return apiFetch(`/api/v1/scenarios/${encodeURIComponent(scenarioId)}/agents/${encodeURIComponent(agentId)}/speeches`, {
    method: 'POST',
    json: payload,
    headers: opts.headers,
  })
}

/**
 * @param {string} scenarioId
 * @param {any} payload
 */
export function generateRoundSummary(scenarioId, payload, opts = {}) {
  return apiFetch(`/api/v1/scenarios/${encodeURIComponent(scenarioId)}/round-summaries`, {
    method: 'POST',
    json: payload,
    headers: opts.headers,
  })
}

/**
 * @param {string} scenarioId
 * @param {any} payload
 */
export function generateEvaluation(scenarioId, payload, opts = {}) {
  return apiFetch(`/api/v1/scenarios/${encodeURIComponent(scenarioId)}/evaluations`, {
    method: 'POST',
    json: payload,
    headers: opts.headers,
  })
}

/**
 * 主席/主持人：阶段提示、规则答疑、最终裁定
 * @param {string} scenarioId
 * @param {any} payload
 */
export function generateChairMessage(scenarioId, payload, opts = {}) {
  return apiFetch(`/api/v1/scenarios/${encodeURIComponent(scenarioId)}/chair/messages`, {
    method: 'POST',
    json: payload,
    headers: opts.headers,
  })
}

