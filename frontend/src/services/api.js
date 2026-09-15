import { demoApi, demoMode, subscribeDemo } from './demo'

async function request(path, options) {
  const response = await fetch(path, options)
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.detail || `请求失败 (${response.status})`)
  }
  return response.json()
}

const realApi = {
  listDatasets: () => request('/api/datasets'),
  listAlgorithms: () => request('/api/algorithms'),
  listExperiments: () => request('/api/experiments'),
  getExperiment: (id) => request(`/api/experiments/${id}`),
  createExperiment: (config) => request('/api/experiments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  }),
}

export const experimentApi = demoMode ? demoApi : realApi

export function subscribeToExperiment(id, onRecord, onError) {
  if (demoMode) return subscribeDemo(id, onRecord)
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  let socket = new WebSocket(`${protocol}//${window.location.host}/api/ws/experiments/${id}`)
  let pollTimer = null
  let closed = false

  const close = () => {
    closed = true
    if (socket) {
      socket.onclose = null
      socket.close()
      socket = null
    }
    if (pollTimer) {
      window.clearInterval(pollTimer)
      pollTimer = null
    }
  }

  const deliver = (record) => {
    onRecord(record)
    if (['completed', 'failed'].includes(record.status)) close()
  }

  const startPolling = () => {
    if (closed || pollTimer) return
    pollTimer = window.setInterval(async () => {
      try {
        deliver(await experimentApi.getExperiment(id))
      } catch (error) {
        onError(error)
      }
    }, 800)
  }

  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data)
      if (payload.type === 'experiment') deliver(payload.experiment)
    } catch (error) {
      onError(error)
    }
  }
  socket.onerror = () => {
    if (socket) {
      socket.onclose = null
      socket.close()
      socket = null
    }
    startPolling()
  }
  socket.onclose = () => {
    socket = null
    startPolling()
  }

  return close
}
