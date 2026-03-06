import { useEffect, useRef } from 'react'
import type { FeedEvent } from '../types'

interface Props {
  events: FeedEvent[]
}

const SEV_CLASS: Record<string, string> = {
  debug:    'ev-debug',
  info:     'ev-info',
  warning:  'ev-warn',
  critical: 'ev-crit',
}

const TYPE_ICON: Record<string, string> = {
  TOKEN_DETECTED:    '🔍',
  AI_DECISION:       '🧠',
  AI_OVERRIDE:       '⚡',
  BUY_EXECUTED:      '🟢',
  SELL_EXECUTED:     '🔴',
  TRADE_FAILED:      '⚠️',
  BUDGET_EXHAUSTED:  '💸',
  OUROBOROS_DETECTED:'🐍',
  ERROR:             '❌',
}

function fmtTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return '--:--:--'
  }
}

export default function EventFeed({ events }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const autoScroll = useRef(true)

  // Auto-scroll to top (newest events at top)
  useEffect(() => {
    if (autoScroll.current && containerRef.current) {
      containerRef.current.scrollTop = 0
    }
  }, [events])

  return (
    <div className="ef-container">
      <div className="ef-header">
        <span className="panel-title" style={{ marginBottom: 0 }}>Live Events</span>
        <span className="badge badge-muted">{events.length}</span>
      </div>

      <div
        className="ef-list"
        ref={containerRef}
        onScroll={e => {
          const el = e.currentTarget
          autoScroll.current = el.scrollTop < 40
        }}
      >
        {events.length === 0 && (
          <div className="ef-empty">Waiting for events…</div>
        )}
        {events.map(ev => (
          <div
            key={ev.id}
            className={`ef-item ${SEV_CLASS[ev.severity] ?? 'ev-info'}`}
          >
            <div className="ef-row1">
              <span className="ef-icon">{TYPE_ICON[ev.event_type] ?? '●'}</span>
              <span className="ef-type">{ev.event_type.replace(/_/g, ' ')}</span>
              {ev.token_symbol && (
                <span className="ef-symbol">${ev.token_symbol}</span>
              )}
              <span className="ef-time">{fmtTime(ev.timestamp)}</span>
            </div>
            {ev.message && (
              <div className="ef-msg">{ev.message}</div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <style>{`
        .ef-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
        .ef-header {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          border-bottom: 1px solid var(--border);
          flex-shrink: 0;
        }
        .ef-list {
          flex: 1;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
        }
        .ef-empty {
          padding: 24px 16px;
          color: var(--text-muted);
          font-size: 12px;
          text-align: center;
        }
        .ef-item {
          padding: 8px 14px;
          border-bottom: 1px solid var(--border);
          border-left: 3px solid transparent;
          transition: background var(--transition);
        }
        .ef-item:hover { background: var(--card-hover); }

        .ev-debug { border-left-color: var(--text-muted); }
        .ev-info  { border-left-color: var(--blue); }
        .ev-warn  { border-left-color: var(--amber); background: rgba(245,158,11,0.03); }
        .ev-crit  { border-left-color: var(--accent); background: rgba(124,58,237,0.04); }

        .ef-row1 {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
        }
        .ef-icon   { font-size: 13px; line-height: 1; }
        .ef-type   { font-weight: 700; letter-spacing: 0.03em; flex: 1; color: var(--text-dim); font-size: 10px; text-transform: uppercase; }
        .ef-symbol { color: var(--accent); font-weight: 700; font-size: 12px; }
        .ef-time   { color: var(--text-muted); font-size: 10px; flex-shrink: 0; margin-left: auto; }
        .ef-msg {
          margin-top: 3px;
          font-size: 11px;
          color: var(--text-dim);
          line-height: 1.4;
          word-break: break-word;
        }
      `}</style>
    </div>
  )
}
