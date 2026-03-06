import { useEffect, useRef, useState, useCallback } from 'react'
import type { BotStatus, FeedEvent, OuroborosStats, Position, StrategyStatus } from './types'
import StatusBar from './components/StatusBar'
import BotControls from './components/BotControls'
import PositionsTable from './components/PositionsTable'
import EventFeed from './components/EventFeed'
import BrainStats from './components/BrainStats'
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
  const [wsConnected, setWsConnected] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const headers = API_KEY ? { 'X-API-Key': API_KEY } : {}

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
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

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
        </div>

        <div className="center-col">
          <PositionsTable positions={positions} isRunning={isRunning} />
        </div>

        <div className="right-col">
          <EventFeed events={events} />
        </div>
      </div>
    </div>
  )
}

export default App
