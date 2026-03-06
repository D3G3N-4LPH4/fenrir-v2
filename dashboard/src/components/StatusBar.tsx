import type { BotStatus, OuroborosStats } from '../types'

interface Props {
  status: BotStatus | null
  wsConnected: boolean
  ouroboros: OuroborosStats | null
}

function fmt(seconds: number | null): string {
  if (seconds === null) return '--'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

function pnlClass(pct: number) {
  if (pct > 0) return 'pnl-pos'
  if (pct < 0) return 'pnl-neg'
  return 'pnl-zero'
}

export default function StatusBar({ status, wsConnected, ouroboros }: Props) {
  const running = status?.status === 'running'
  const portfolio = status?.portfolio

  return (
    <header className="status-bar">
      {/* Left: brand + bot state */}
      <div className="sb-left">
        <span className="sb-brand">ᚠ FENRIR</span>

        <span className={`sb-dot ${running ? 'dot-green' : status?.status === 'error' ? 'dot-red' : 'dot-muted'}`} />
        <span className="sb-status-text">
          {status?.status?.toUpperCase() ?? 'CONNECTING…'}
        </span>

        {status?.mode && (
          <span className="badge badge-purple" style={{ marginLeft: 8 }}>
            {status.mode}
          </span>
        )}

        {status?.uptime_seconds != null && (
          <span className="sb-meta">up {fmt(status.uptime_seconds)}</span>
        )}
      </div>

      {/* Center: portfolio summary */}
      {portfolio && running && (
        <div className="sb-center">
          <span className="sb-kv">
            <span className="sb-k">positions</span>
            <span className="sb-v">{portfolio.num_positions}</span>
          </span>
          <span className="sb-sep" />
          <span className="sb-kv">
            <span className="sb-k">invested</span>
            <span className="sb-v">{portfolio.total_invested_sol.toFixed(4)} SOL</span>
          </span>
          <span className="sb-sep" />
          <span className="sb-kv">
            <span className="sb-k">pnl</span>
            <span className={`sb-v ${pnlClass(portfolio.total_pnl_pct)}`}>
              {portfolio.total_pnl_pct >= 0 ? '+' : ''}{portfolio.total_pnl_pct.toFixed(2)}%
              {' '}({portfolio.total_pnl_sol >= 0 ? '+' : ''}{portfolio.total_pnl_sol.toFixed(4)} SOL)
            </span>
          </span>
        </div>
      )}

      {/* Right: ws + ouroboros */}
      <div className="sb-right">
        {ouroboros && ouroboros.total_detections > 0 && (
          <span className="badge badge-amber" title="Ouroboros detections this session">
            🐍 {ouroboros.total_detections} ouroboros
          </span>
        )}
        <span className={`badge ${wsConnected ? 'badge-green' : 'badge-red'}`}>
          {wsConnected ? '⬤ LIVE' : '◌ OFFLINE'}
        </span>
      </div>

      <style>{`
        .status-bar {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 0 16px;
          height: 46px;
          background: var(--surface);
          border-bottom: 1px solid var(--border);
          flex-shrink: 0;
        }
        .sb-brand {
          font-size: 15px;
          font-weight: 700;
          color: var(--accent);
          letter-spacing: 0.05em;
          margin-right: 4px;
        }
        .sb-dot {
          width: 7px; height: 7px;
          border-radius: 50%;
          display: inline-block;
          margin-right: 6px;
        }
        .dot-green  { background: var(--green); box-shadow: 0 0 6px var(--green); }
        .dot-red    { background: var(--red);   box-shadow: 0 0 6px var(--red); }
        .dot-muted  { background: var(--text-muted); }
        .sb-status-text { font-weight: 600; font-size: 12px; }
        .sb-meta { color: var(--text-muted); font-size: 11px; margin-left: 8px; }
        .sb-left  { display: flex; align-items: center; gap: 4px; flex: 0 0 auto; }
        .sb-center { display: flex; align-items: center; gap: 10px; flex: 1; justify-content: center; }
        .sb-right  { display: flex; align-items: center; gap: 8px; flex: 0 0 auto; margin-left: auto; }
        .sb-kv    { display: flex; gap: 6px; align-items: baseline; }
        .sb-k     { color: var(--text-muted); font-size: 10px; text-transform: uppercase; }
        .sb-v     { font-size: 12px; font-weight: 600; }
        .sb-sep   { width: 1px; height: 14px; background: var(--border); }
      `}</style>
    </header>
  )
}
