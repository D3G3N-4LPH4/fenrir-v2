import { useEffect, useRef, useState, useCallback } from 'react'
import type {
  AvailableStrategy,
  BotConfigSurface,
  BotStatus,
  FeedEvent,
  OuroborosStats,
  Position,
  StrategyStatus,
} from './types'
import StatusBar from './components/StatusBar'
import BotControls from './components/BotControls'
import PositionsTable from './components/PositionsTable'
import EventFeed from './components/EventFeed'
import BrainStats from './components/BrainStats'
import SettingsPanel from './components/SettingsPanel'
import StrategiesPanel from './components/StrategiesPanel'
import './App.css'

const API = '/api'
const WS_URL = `ws://localhost:8000/ws/updates`
const API_KEY = import.meta.env.VITE_API_KEY ?? ''
const POLL_MS = 5000
const MAX_EVENTS = 150

function App() {
  const [status, setStatus] = useState<BotStatus | null>(null)
  const [positions, setPositions] = useState<Position[]>([])
  const [events, setEvents] = useState<FeedEvent[]>([])
  const [strategies, setStrategies] = useState<StrategyStatus[]>([])
  const [ouroboros, setOuroboros] = useState<OuroborosStats | null>(null)
  const [brainStats, setBrainStats] = useState<Record<string, unknown> | null>(null)
  const [config, setConfig] = useState<BotConfigSurface | null>(null)
  const [availableStrategies, setAvailableStrategies] = useState<AvailableStrategy[]>([])
  const [centerTab, setCenterTab] = useState<'positions' | 'strategies'>('positions')
  const [wsConnected, setWsConnected] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const headers: Record<string, string> = API_KEY ? { 'X-API-Key': API_KEY } : {}

  // ── REST polling ──────────────────────────────────────────────────
  const poll = useCallback(async () => {
    try {
      const [statusRes, posRes] = await Promise.all([
        fetch(`${API}/bot/status`, { headers }),
        fetch(`${API}/bot/positions`, { headers }),
      ])
      if (statusRes.ok) setStatus(await statusRes.json())
      if (posRes.ok) {
        const data = await posRes.json()
        setPositions(data.positions ?? [])
      }
    } catch {
      // silently ignore — WS connection badge shows health
    }

    try {
      const fullRes = await fetch(`${API}/bot/full-status`, { headers })
      if (fullRes.ok) {
        const full = await fullRes.json()
        if (full.ai_brain) setBrainStats(full.ai_brain)
        if (full.ouroboros) setOuroboros(full.ouroboros)
        if (full.strategies) setStrategies(full.strategies)
      }
    } catch {
      // full-status only available when bot is running
    }

    try {
      const cfgRes = await fetch(`${API}/bot/config`, { headers })
      if (cfgRes.ok) {
        const data = await cfgRes.json()
        if (data.config) setConfig(data.config)
      }
    } catch {
      // config unavailable until the bot is configured/started — ignore
    }

    try {
      const stratRes = await fetch(`${API}/bot/strategies/available`, { headers })
      if (stratRes.ok) {
        const data = await stratRes.json()
        if (data.strategies) setAvailableStrategies(data.strategies)
      }
    } catch {
      // ignore — strategies tab shows empty until reachable
    }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  const toggleStrategy = async (strategyId: string, enable: boolean) => {
    const action = enable ? 'enable' : 'disable'
    const res = await fetch(`${API}/bot/strategies/${strategyId}/${action}`, {
      method: 'POST',
      headers,
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail ?? `Failed to ${action} (${res.status})`)
    }
    await poll()  // refresh live state
  }

  const saveConfig = async (patch: Partial<BotConfigSurface>) => {
    const res = await fetch(`${API}/bot/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(patch),
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail ?? `Save failed (${res.status})`)
    }
    const data = await res.json()
    if (data.config) setConfig(data.config)
  }

  useEffect(() => {
    poll()
    const interval = setInterval(poll, POLL_MS)
    return () => clearInterval(interval)
  }, [poll])

  // ── WebSocket ─────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(
      API_KEY ? `${WS_URL}?key=${API_KEY}` : WS_URL,
      API_KEY ? [`authorization.${API_KEY}`] : undefined,
    )
    wsRef.current = ws

    ws.onopen = () => setWsConnected(true)

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data as string)

        // System lifecycle messages from the server
        if ('event' in msg && !('event_type' in msg)) {
          if (msg.event === 'bot_started' || msg.event === 'bot_stopped') {
            poll()
          }
          return
        }

        // TradeEvent from EventBus
        if ('event_type' in msg) {
          const ev: FeedEvent = {
            id: `${msg.timestamp}-${Math.random()}`,
            event_type: msg.event_type,
            category: msg.category,
            severity: msg.severity,
            timestamp: msg.timestamp,
            token_symbol: msg.token_symbol ?? null,
            token_address: msg.token_address ?? null,
            strategy_id: msg.strategy_id ?? null,
            message: msg.message,
            data: msg.data ?? {},
          }
          setEvents(prev => [ev, ...prev].slice(0, MAX_EVENTS))

          // Refresh positions on trade events
          if (['BUY_EXECUTED', 'SELL_EXECUTED', 'OUROBOROS_DETECTED'].includes(msg.event_type)) {
            poll()
          }
        }
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      setWsConnected(false)
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => ws.close()
  }, [poll])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  // ── Strategy control ──────────────────────────────────────────────
  const pauseStrategy = async (id: string) => {
    await fetch(`${API}/bot/strategies/${id}/pause`, { method: 'POST', headers })
    poll()
  }

  const resumeStrategy = async (id: string) => {
    await fetch(`${API}/bot/strategies/${id}/resume`, { method: 'POST', headers })
    poll()
  }

  // ── Bot start/stop ────────────────────────────────────────────────
  const startBot = async (mode: string) => {
    await fetch(`${API}/bot/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify({ mode }),
    })
    setTimeout(poll, 500)
  }

  const stopBot = async () => {
    await fetch(`${API}/bot/stop`, { method: 'POST', headers })
    setTimeout(poll, 500)
  }

  const isRunning = status?.status === 'running'

  return (
    <div className="layout">
      <StatusBar
        status={status}
        wsConnected={wsConnected}
        ouroboros={ouroboros}
      />

      <div className="main-grid">
        <div className="left-col">
          <BotControls
            status={status}
            strategies={strategies}
            onStart={startBot}
            onStop={stopBot}
            onPause={pauseStrategy}
            onResume={resumeStrategy}
          />
          <BrainStats stats={brainStats} />
          <SettingsPanel config={config} onSave={saveConfig} />
        </div>

        <div className="center-col">
          <div className="center-tabs">
            <button
              className={centerTab === 'positions' ? 'center-tab active' : 'center-tab'}
              onClick={() => setCenterTab('positions')}
            >
              Positions
            </button>
            <button
              className={centerTab === 'strategies' ? 'center-tab active' : 'center-tab'}
              onClick={() => setCenterTab('strategies')}
            >
              Strategies
            </button>
          </div>
          {centerTab === 'positions' ? (
            <PositionsTable positions={positions} isRunning={isRunning} />
          ) : (
            <StrategiesPanel
              strategies={availableStrategies}
              isRunning={isRunning}
              onToggle={toggleStrategy}
            />
          )}
        </div>

        <div className="right-col">
          <EventFeed events={events} />
        </div>
      </div>
    </div>
  )
}

export default App
