import { useState } from 'react'
import type { BotStatus, StrategyStatus } from '../types'

interface Props {
  status: BotStatus | null
  strategies: StrategyStatus[]
  onStart: (mode: string) => void
  onStop: () => void
  onPause: (id: string) => void
  onResume: (id: string) => void
}

const MODES = ['simulation', 'conservative', 'aggressive', 'degen'] as const

export default function BotControls({ status, strategies, onStart, onStop, onPause, onResume }: Props) {
  const [selectedMode, setSelectedMode] = useState<string>('simulation')
  const running = status?.status === 'running'

  return (
    <>
      {/* Bot Start / Stop */}
      <div className="panel">
        <div className="panel-title">Bot Control</div>

        {!running ? (
          <div className="ctrl-start">
            <select
              value={selectedMode}
              onChange={e => setSelectedMode(e.target.value)}
              className="mode-select"
            >
              {MODES.map(m => (
                <option key={m} value={m}>{m.toUpperCase()}</option>
              ))}
            </select>
            <button className="btn-success" onClick={() => onStart(selectedMode)}>
              START
            </button>
          </div>
        ) : (
          <div className="ctrl-stop">
            <div className="running-badge">
              <span className="dot-pulse" />
              Running in {status?.mode?.toUpperCase()} mode
            </div>
            <button className="btn-danger" onClick={onStop}>
              STOP BOT
            </button>
          </div>
        )}

        {status?.error && (
          <div className="ctrl-error">{status.error}</div>
        )}
      </div>

      {/* Strategies */}
      {strategies.length > 0 && (
        <div className="panel">
          <div className="panel-title">Strategies</div>
          <div className="strategy-list">
            {strategies.map(s => (
              <div key={s.strategy_id} className="strategy-row">
                <div className="strategy-info">
                  <span className="strategy-name">{s.display_name}</span>
                  <div className="strategy-meta">
                    <span>{s.trades_today} trades</span>
                    <span className={s.win_rate_today >= 0.5 ? 'pnl-pos' : 'pnl-neg'}>
                      {(s.win_rate_today * 100).toFixed(0)}% win
                    </span>
                    <span className="pnl-pos">{s.budget_remaining.toFixed(3)} SOL left</span>
                  </div>
                </div>
                <div className="strategy-actions">
                  {!s.active ? (
                    <span className="badge badge-red">INACTIVE</span>
                  ) : s.paused ? (
                    <button className="btn-success" style={{ padding: '3px 10px', fontSize: '11px' }}
                      onClick={() => onResume(s.strategy_id)}>RESUME</button>
                  ) : (
                    <button className="btn-amber" style={{ padding: '3px 10px', fontSize: '11px' }}
                      onClick={() => onPause(s.strategy_id)}>PAUSE</button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        .ctrl-start {
          display: flex;
          gap: 8px;
          align-items: center;
        }
        .mode-select {
          font-family: var(--font);
          font-size: 12px;
          background: var(--surface);
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 6px 8px;
          flex: 1;
          cursor: pointer;
        }
        .ctrl-stop {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .running-badge {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: var(--text-dim);
        }
        .dot-pulse {
          width: 8px; height: 8px;
          border-radius: 50%;
          background: var(--green);
          box-shadow: 0 0 0 0 rgba(16,185,129,0.4);
          animation: pulse 2s infinite;
        }
        @keyframes pulse {
          0%   { box-shadow: 0 0 0 0   rgba(16,185,129,0.4); }
          70%  { box-shadow: 0 0 0 8px rgba(16,185,129,0); }
          100% { box-shadow: 0 0 0 0   rgba(16,185,129,0); }
        }
        .ctrl-error {
          margin-top: 8px;
          font-size: 11px;
          color: var(--red);
          background: var(--red-dim);
          border-radius: var(--radius);
          padding: 6px 8px;
          word-break: break-all;
        }
        .strategy-list { display: flex; flex-direction: column; gap: 10px; }
        .strategy-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 8px;
        }
        .strategy-info { display: flex; flex-direction: column; gap: 3px; }
        .strategy-name { font-size: 12px; font-weight: 600; }
        .strategy-meta { display: flex; gap: 8px; font-size: 10px; color: var(--text-muted); }
        .btn-amber { border-color: var(--amber); color: var(--amber); }
        .btn-amber:hover { background: var(--amber-dim); }
      `}</style>
    </>
  )
}
