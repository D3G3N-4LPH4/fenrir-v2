import { useCallback, useEffect, useState } from 'react'
import type { DiscoveryConfigSurface, DiscoveryItem, DiscoveryScores } from '../types'

interface Props {
  apiBase: string
  apiKey: string
}

const CHAINS = ['solana', 'ethereum', 'bnb', 'base'] as const
const FILTERS = ['low_cap_alpha', 'mid_cap_momentum', 'high_cap'] as const
const FILTER_LABELS: Record<string, string> = {
  low_cap_alpha: 'Low Cap',
  mid_cap_momentum: 'Mid Cap',
  high_cap: 'High Cap',
}
const POLL_MS = 8000

function shortAddr(a: string): string {
  return a.length > 12 ? `${a.slice(0, 6)}…${a.slice(-4)}` : a
}

function fmtUsd(n: number): string {
  if (!n) return '—'
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`
  return `$${n.toFixed(0)}`
}

function fmtAge(mins: number): string {
  if (mins < 60) return `${Math.round(mins)}m`
  const h = mins / 60
  if (h < 24) return `${h.toFixed(0)}h`
  return `${(h / 24).toFixed(0)}d`
}

function scoreColor(v: number): string {
  if (v >= 60) return 'var(--green, #22c55e)'
  if (v >= 40) return 'var(--amber, #f59e0b)'
  return 'var(--red, #ef4444)'
}

const COMPONENTS: { key: keyof DiscoveryScores; label: string }[] = [
  { key: 'momentum', label: 'Mom' },
  { key: 'safety', label: 'Saf' },
  { key: 'liquidity', label: 'Liq' },
  { key: 'holder', label: 'Hld' },
  { key: 'community', label: 'Com' },
]

export default function DiscoveryPanel({ apiBase, apiKey }: Props) {
  const [items, setItems] = useState<DiscoveryItem[]>([])
  const [config, setConfig] = useState<DiscoveryConfigSurface | null>(null)
  const [chain, setChain] = useState<string>('')
  const [filter, setFilter] = useState<string>('')
  const [minScore, setMinScore] = useState<number>(0)
  const [loaded, setLoaded] = useState(false)

  const headers: Record<string, string> = apiKey ? { 'X-API-Key': apiKey } : {}

  const load = useCallback(async () => {
    const params = new URLSearchParams()
    if (chain) params.set('chain', chain)
    if (filter) params.set('filter', filter)
    if (minScore) params.set('min_score', String(minScore))
    try {
      const [dRes, cRes] = await Promise.all([
        fetch(`${apiBase}/discover?${params.toString()}`, { headers }),
        fetch(`${apiBase}/discover/config`, { headers }),
      ])
      if (dRes.ok) setItems((await dRes.json()).results ?? [])
      if (cRes.ok) setConfig(await cRes.json())
    } catch {
      /* transient — keep last results */
    } finally {
      setLoaded(true)
    }
  }, [apiBase, apiKey, chain, filter, minScore]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    load()
    const t = setInterval(load, POLL_MS)
    return () => clearInterval(t)
  }, [load])

  const disabled = config != null && !config.enabled

  return (
    <div className="dp-container">
      <div className="dp-header">
        <span className="panel-title" style={{ marginBottom: 0 }}>
          Discovery
        </span>
        <span className="badge badge-muted">{items.length}</span>
        {config?.running && <span className="badge badge-green">live</span>}
        <div className="dp-controls">
          <select value={chain} onChange={e => setChain(e.target.value)}>
            <option value="">All chains</option>
            {CHAINS.map(c => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <select value={filter} onChange={e => setFilter(e.target.value)}>
            <option value="">All filters</option>
            {FILTERS.map(f => (
              <option key={f} value={f}>
                {FILTER_LABELS[f]}
              </option>
            ))}
          </select>
          <label className="dp-minscore">
            min <b>{minScore}</b>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={minScore}
              onChange={e => setMinScore(Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      {disabled ? (
        <div className="dp-empty">
          Discovery is disabled. Set <code>DISCOVERY_ENABLED=true</code> (and{' '}
          <code>DISCOVERY_CHAINS</code>) and restart the bot.
        </div>
      ) : items.length === 0 ? (
        <div className="dp-empty">
          {loaded ? 'No tokens match the current filters yet…' : 'Loading…'}
        </div>
      ) : (
        <div className="dp-table-wrap">
          <table className="dp-table">
            <thead>
              <tr>
                <th>Token</th>
                <th>Matched</th>
                <th>MCap</th>
                <th>Liq</th>
                <th>Vol 24h</th>
                <th>Holders</th>
                <th>Age</th>
                <th className="dp-score-col">Score</th>
              </tr>
            </thead>
            <tbody>
              {items.map(it => (
                <tr key={`${it.chain}:${it.token_address}`}>
                  <td>
                    <div className="dp-token">
                      <span className="dp-symbol">${it.symbol}</span>
                      <span className={`badge chain-${it.chain}`}>{it.chain}</span>
                    </div>
                    <span className="dp-addr" title={it.token_address}>
                      {shortAddr(it.token_address)}
                    </span>
                  </td>
                  <td>
                    <div className="dp-filters">
                      {it.matched_filters.map(f => (
                        <span key={f} className="badge badge-muted">
                          {FILTER_LABELS[f] ?? f}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>{fmtUsd(it.market_cap_usd)}</td>
                  <td>{fmtUsd(it.liquidity_usd)}</td>
                  <td>{fmtUsd(it.volume_24h_usd)}</td>
                  <td>{it.holder_count?.toLocaleString() ?? '—'}</td>
                  <td>{fmtAge(it.age_minutes)}</td>
                  <td className="dp-score-col">
                    <div className="dp-overall">
                      <span className="dp-overall-num" style={{ color: scoreColor(it.scores.overall) }}>
                        {it.scores.overall.toFixed(0)}
                      </span>
                      <div className="dp-bar">
                        <div
                          className="dp-bar-fill"
                          style={{
                            width: `${it.scores.overall}%`,
                            background: scoreColor(it.scores.overall),
                          }}
                        />
                      </div>
                    </div>
                    <div className="dp-components">
                      {COMPONENTS.map(c => (
                        <span
                          key={c.key}
                          className="dp-comp"
                          title={`${c.label}: ${it.scores[c.key].toFixed(0)}`}
                        >
                          <span className="dp-comp-label">{c.label}</span>
                          <span className="dp-comp-bar">
                            <span
                              className="dp-comp-fill"
                              style={{
                                width: `${it.scores[c.key]}%`,
                                background: scoreColor(it.scores[c.key]),
                              }}
                            />
                          </span>
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <style>{`
        .dp-container { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
        .dp-header {
          display: flex; align-items: center; gap: 8px; padding: 12px 16px;
          border-bottom: 1px solid var(--border); flex-shrink: 0; flex-wrap: wrap;
        }
        .dp-controls { display: flex; align-items: center; gap: 8px; margin-left: auto; }
        .dp-controls select {
          background: var(--surface); color: var(--text); border: 1px solid var(--border);
          border-radius: 6px; padding: 4px 8px; font-size: 11px; font-family: var(--font);
        }
        .dp-minscore { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-muted); }
        .dp-minscore input { width: 84px; }
        .dp-empty { padding: 32px 16px; color: var(--text-muted); font-size: 12px; text-align: center; }
        .dp-empty code { background: var(--surface); padding: 1px 5px; border-radius: 4px; }
        .dp-table-wrap { overflow: auto; flex: 1; }
        .dp-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .dp-table th {
          padding: 8px 12px; text-align: left; font-size: 10px; font-weight: 700;
          letter-spacing: 0.07em; text-transform: uppercase; color: var(--text-muted);
          background: var(--surface); border-bottom: 1px solid var(--border);
          position: sticky; top: 0; z-index: 1;
        }
        .dp-table td {
          padding: 10px 12px; border-bottom: 1px solid var(--border);
          vertical-align: middle; font-variant-numeric: tabular-nums;
        }
        .dp-table tr:hover td { background: var(--card-hover); }
        .dp-token { display: flex; align-items: center; gap: 6px; }
        .dp-symbol { font-weight: 700; font-size: 13px; }
        .dp-addr { font-size: 10px; color: var(--text-muted); }
        .dp-filters { display: flex; gap: 4px; flex-wrap: wrap; }
        .dp-score-col { min-width: 150px; }
        .dp-overall { display: flex; align-items: center; gap: 8px; }
        .dp-overall-num { font-weight: 800; font-size: 15px; width: 24px; }
        .dp-bar { flex: 1; height: 6px; background: var(--surface); border-radius: 3px; overflow: hidden; }
        .dp-bar-fill { height: 100%; border-radius: 3px; }
        .dp-components { display: flex; gap: 6px; margin-top: 5px; }
        .dp-comp { display: flex; flex-direction: column; gap: 2px; align-items: center; }
        .dp-comp-label { font-size: 8px; color: var(--text-muted); letter-spacing: 0.04em; }
        .dp-comp-bar { width: 20px; height: 3px; background: var(--surface); border-radius: 2px; overflow: hidden; }
        .dp-comp-fill { display: block; height: 100%; }

        .chain-solana   { background: rgba(153,69,255,0.15); color: #b388ff; }
        .chain-ethereum { background: rgba(98,126,234,0.15); color: #8ea0ff; }
        .chain-bnb      { background: rgba(240,185,11,0.15); color: #f0b90b; }
        .chain-base     { background: rgba(0,82,255,0.15); color: #5b8dff; }
        .badge-green    { background: rgba(34,197,94,0.15); color: #22c55e; }
      `}</style>
    </div>
  )
}
