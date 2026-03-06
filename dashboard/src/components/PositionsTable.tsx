import type { Position } from '../types'

interface Props {
  positions: Position[]
  isRunning: boolean
}

function pnlClass(pct: number) {
  if (pct > 0)  return 'pnl-pos'
  if (pct < 0)  return 'pnl-neg'
  return 'pnl-zero'
}

function holdTime(entryTime: string): string {
  const secs = Math.floor((Date.now() - new Date(entryTime).getTime()) / 1000)
  if (secs < 60) return `${secs}s`
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins}m`
  return `${Math.floor(mins / 60)}h ${mins % 60}m`
}

function shortAddr(addr: string): string {
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`
}

export default function PositionsTable({ positions, isRunning }: Props) {
  return (
    <div className="pt-container">
      <div className="pt-header">
        <span className="panel-title" style={{ marginBottom: 0 }}>
          Open Positions
        </span>
        <span className="badge badge-muted">{positions.length}</span>
      </div>

      {positions.length === 0 ? (
        <div className="pt-empty">
          {isRunning ? 'No open positions — watching for launches…' : 'Bot is not running'}
        </div>
      ) : (
        <div className="pt-table-wrap">
          <table className="pt-table">
            <thead>
              <tr>
                <th>Token</th>
                <th>Strategy</th>
                <th>Invested</th>
                <th>PnL %</th>
                <th>PnL SOL</th>
                <th>Price</th>
                <th>Hold</th>
                <th>Flags</th>
              </tr>
            </thead>
            <tbody>
              {positions.map(pos => (
                <tr key={pos.token_address} className={pos.ouroboros_triggered ? 'row-ouroboros' : ''}>
                  <td>
                    <div className="token-cell">
                      <span className="token-symbol">${pos.token_symbol}</span>
                      <span className="token-addr" title={pos.token_address}>
                        {shortAddr(pos.token_address)}
                      </span>
                    </div>
                  </td>
                  <td>
                    <span className="badge badge-muted">{pos.strategy_id}</span>
                  </td>
                  <td>{pos.amount_sol_invested.toFixed(4)}</td>
                  <td className={pnlClass(pos.pnl_percent)}>
                    {pos.pnl_percent >= 0 ? '+' : ''}{pos.pnl_percent.toFixed(2)}%
                  </td>
                  <td className={pnlClass(pos.pnl_sol)}>
                    {pos.pnl_sol >= 0 ? '+' : ''}{pos.pnl_sol.toFixed(4)}
                  </td>
                  <td className="price-cell">
                    <div>{pos.current_price.toExponential(3)}</div>
                    <div className="price-peak" title="Peak price">
                      ▲ {pos.peak_price.toExponential(3)}
                    </div>
                  </td>
                  <td>{holdTime(pos.entry_time)}</td>
                  <td>
                    <div className="flags">
                      {pos.ouroboros_triggered && (
                        <span className="badge badge-amber" title={`Trail tightened to ${pos.trailing_stop_override_pct}%`}>
                          🐍 {pos.trailing_stop_override_pct}%
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <style>{`
        .pt-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          overflow: hidden;
        }
        .pt-header {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          border-bottom: 1px solid var(--border);
          flex-shrink: 0;
        }
        .pt-empty {
          padding: 32px 16px;
          color: var(--text-muted);
          font-size: 12px;
          text-align: center;
        }
        .pt-table-wrap {
          overflow: auto;
          flex: 1;
        }
        .pt-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 12px;
        }
        .pt-table th {
          padding: 8px 12px;
          text-align: left;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 0.07em;
          text-transform: uppercase;
          color: var(--text-muted);
          background: var(--surface);
          border-bottom: 1px solid var(--border);
          position: sticky;
          top: 0;
          z-index: 1;
        }
        .pt-table td {
          padding: 10px 12px;
          border-bottom: 1px solid var(--border);
          vertical-align: middle;
          font-variant-numeric: tabular-nums;
        }
        .pt-table tr:hover td { background: var(--card-hover); }
        .row-ouroboros td { background: rgba(245,158,11,0.04); }
        .row-ouroboros:hover td { background: rgba(245,158,11,0.08); }

        .token-cell { display: flex; flex-direction: column; gap: 2px; }
        .token-symbol { font-weight: 700; font-size: 13px; }
        .token-addr { font-size: 10px; color: var(--text-muted); font-family: var(--font); }

        .price-cell { display: flex; flex-direction: column; gap: 1px; }
        .price-peak { font-size: 10px; color: var(--text-muted); }

        .flags { display: flex; gap: 4px; flex-wrap: wrap; }
      `}</style>
    </div>
  )
}
