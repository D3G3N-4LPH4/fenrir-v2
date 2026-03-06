interface Props {
  stats: Record<string, unknown> | null
}

function Stat({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="stat-row">
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={color ? { color } : undefined}>{value}</span>
    </div>
  )
}

export default function BrainStats({ stats }: Props) {
  if (!stats) {
    return (
      <div className="panel">
        <div className="panel-title">AI Brain</div>
        <div className="bs-empty">—</div>
      </div>
    )
  }

  const evaluated = (stats.ai_entries_evaluated as number) ?? 0
  const bought    = (stats.ai_entries_bought as number)    ?? 0
  const skipped   = (stats.ai_entries_skipped as number)   ?? 0
  const timeouts  = (stats.ai_timeouts as number)          ?? 0
  const errors    = (stats.ai_errors as number)            ?? 0
  const overrides = (stats.ai_exits_overridden as number)  ?? 0
  const avgMs     = (stats.ai_avg_response_ms as number)   ?? 0
  const buyRate   = evaluated > 0 ? ((bought / evaluated) * 100).toFixed(0) : '0'

  // nested ai_analyst_report
  const analyst = (stats.ai_analyst_report as Record<string, unknown>) ?? {}
  const localModel = analyst.model_type === 'local'

  return (
    <div className="panel">
      <div className="panel-title">
        AI Brain {localModel && <span className="badge badge-purple" style={{ marginLeft: 6 }}>LOCAL</span>}
      </div>

      <div className="stat-grid">
        <Stat label="evaluated"  value={evaluated} />
        <Stat label="buy rate"   value={`${buyRate}%`} color={parseInt(buyRate) > 40 ? 'var(--green)' : 'var(--text-dim)'} />
        <Stat label="bought"     value={bought}    color="var(--green)" />
        <Stat label="skipped"    value={skipped}   color="var(--text-muted)" />
        <Stat label="timeouts"   value={timeouts}  color={timeouts > 0 ? 'var(--amber)' : 'var(--text-muted)'} />
        <Stat label="errors"     value={errors}    color={errors > 0 ? 'var(--red)' : 'var(--text-muted)'} />
        <Stat label="overrides"  value={overrides} color={overrides > 0 ? 'var(--accent)' : 'var(--text-muted)'} />
        <Stat label="avg resp"   value={`${avgMs.toFixed(0)}ms`}
              color={avgMs > 3000 ? 'var(--amber)' : avgMs > 1000 ? 'var(--text-dim)' : 'var(--green)'} />
      </div>

      <style>{`
        .bs-empty { color: var(--text-muted); font-size: 12px; }
        .stat-grid { display: flex; flex-direction: column; gap: 5px; }
        .stat-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 12px;
        }
        .stat-label { color: var(--text-muted); }
        .stat-value { font-weight: 600; font-variant-numeric: tabular-nums; }
      `}</style>
    </div>
  )
}
