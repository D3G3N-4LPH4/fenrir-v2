import { useState } from 'react'
import type { AvailableStrategy } from '../types'

interface Props {
  strategies: AvailableStrategy[]
  isRunning: boolean
  onToggle: (strategyId: string, enable: boolean) => Promise<void>
}

// A strategy is "on" when it's loaded and active (not paused).
export default function StrategiesPanel({ strategies, isRunning, onToggle }: Props) {
  const [pending, setPending] = useState<Record<string, boolean>>({})
  const [error, setError] = useState<string | null>(null)

  const toggle = async (s: AvailableStrategy) => {
    const enable = !(s.active)
    setPending(p => ({ ...p, [s.strategy_id]: true }))
    setError(null)
    try {
      await onToggle(s.strategy_id, enable)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Toggle failed')
    } finally {
      setPending(p => ({ ...p, [s.strategy_id]: false }))
    }
  }

  return (
    <div className="panel">
      <div className="panel-title">
        Strategies
        {!isRunning && <span className="strat-hint-inline"> · start the bot to switch</span>}
      </div>

      {!strategies.length && (
        <div className="strat-empty">No strategies reported.</div>
      )}

      <div className="strat-list">
        {strategies.map(s => {
          const state = !s.loaded ? 'inactive' : s.paused ? 'paused' : 'active'
          return (
            <div className="strat-row" key={s.strategy_id}>
              <div className="strat-info">
                <div className="strat-name-line">
                  <span className="strat-name">{s.display_name}</span>
                  <span className={`strat-badge strat-${state}`}>{state}</span>
                  {s.uses_market_data && <span className="strat-badge strat-mkt">market-data</span>}
                </div>
                <div className="strat-desc">{s.description || s.strategy_id}</div>
              </div>
              <label className="strat-toggle">
                <input
                  type="checkbox"
                  checked={s.active}
                  disabled={!isRunning || pending[s.strategy_id]}
                  onChange={() => toggle(s)}
                />
                <span className="strat-track" />
              </label>
            </div>
          )
        })}
      </div>

      {error && <div className="strat-err">{error}</div>}

      <style>{`
        .strat-hint-inline { font-size: 10px; color: var(--text-muted); font-weight: 400; }
        .strat-empty { font-size: 12px; color: var(--text-muted); padding: 8px 0; }
        .strat-list { display: flex; flex-direction: column; }
        .strat-row {
          display: flex; align-items: center; justify-content: space-between; gap: 12px;
          padding: 11px 0; border-bottom: 1px solid var(--border);
        }
        .strat-row:last-child { border-bottom: none; }
        .strat-info { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
        .strat-name-line { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
        .strat-name { font-size: 13px; color: var(--text); }
        .strat-desc { font-size: 10.5px; color: var(--text-muted); line-height: 1.35; }
        .strat-badge {
          font-size: 9px; letter-spacing: .04em; text-transform: uppercase;
          padding: 1px 6px; border-radius: 20px; border: 1px solid var(--border);
        }
        .strat-active { color: var(--green); border-color: var(--green); }
        .strat-paused { color: var(--amber); border-color: var(--amber); }
        .strat-inactive { color: var(--text-muted); }
        .strat-mkt { color: var(--blue); border-color: var(--blue); }
        .strat-toggle { position: relative; width: 34px; height: 18px; flex-shrink: 0; }
        .strat-toggle input { opacity: 0; width: 0; height: 0; }
        .strat-track {
          position: absolute; inset: 0; background: var(--surface);
          border: 1px solid var(--border); border-radius: 20px; cursor: pointer;
          transition: var(--transition);
        }
        .strat-track::before {
          content: ''; position: absolute; width: 12px; height: 12px; left: 2px; top: 2px;
          background: var(--text-muted); border-radius: 50%; transition: var(--transition);
        }
        .strat-toggle input:checked + .strat-track {
          background: var(--accent-glow); border-color: var(--accent);
        }
        .strat-toggle input:checked + .strat-track::before {
          transform: translateX(16px); background: var(--accent);
        }
        .strat-toggle input:disabled + .strat-track { opacity: .45; cursor: not-allowed; }
        .strat-err {
          margin-top: 10px; font-size: 11px; color: var(--red);
          background: var(--red-dim); border-radius: var(--radius); padding: 6px 8px;
        }
      `}</style>
    </div>
  )
}
