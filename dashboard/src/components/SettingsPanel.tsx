import { useEffect, useState } from 'react'
import type { BotConfigSurface } from '../types'

interface Props {
  config: BotConfigSurface | null
  onSave: (patch: Partial<BotConfigSurface>) => Promise<void>
}

// Local edit buffer keeps typing responsive even if a poll tick lands mid-edit.
export default function SettingsPanel({ config, onSave }: Props) {
  const [form, setForm] = useState<BotConfigSurface | null>(config)
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [message, setMessage] = useState<{ text: string; kind: 'ok' | 'err' } | null>(null)

  // Adopt server config only while the user isn't mid-edit.
  useEffect(() => {
    if (!dirty) setForm(config)
  }, [config, dirty])

  if (!form) {
    return (
      <div className="panel">
        <div className="panel-title">Settings</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Waiting for config from the bot…
        </div>
      </div>
    )
  }

  const set = <K extends keyof BotConfigSurface>(key: K, value: BotConfigSurface[K]) => {
    setForm(f => (f ? { ...f, [key]: value } : f))
    setDirty(true)
  }

  const save = async () => {
    if (!form) return
    setSaving(true)
    setMessage(null)
    try {
      await onSave(form)
      setDirty(false)
      setMessage({ text: 'Settings saved.', kind: 'ok' })
    } catch (e) {
      setMessage({ text: e instanceof Error ? e.message : 'Save failed.', kind: 'err' })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-title">Settings</div>

      <div className="setting-row">
        <div className="setting-label">
          <span className="setting-name">Market scanner</span>
          <span className="setting-hint">MARKET_SCANNER_ENABLED</span>
        </div>
        <label className="toggle">
          <input
            type="checkbox"
            checked={form.market_scanner_enabled}
            onChange={e => set('market_scanner_enabled', e.target.checked)}
          />
          <span className="toggle-track" />
        </label>
      </div>

      <div className="setting-row">
        <div className="setting-label">
          <span className="setting-name">Global daily SOL limit</span>
          <span className="setting-hint">GLOBAL_DAILY_SOL_LIMIT · 0 = disabled</span>
        </div>
        <input
          type="number" step="0.1" min="0" className="setting-input"
          value={form.global_daily_sol_limit}
          onChange={e => set('global_daily_sol_limit', Number(e.target.value))}
        />
      </div>

      <div className="setting-row">
        <div className="setting-label">
          <span className="setting-name">Buy amount</span>
          <span className="setting-hint">BUY_AMOUNT_SOL per trade</span>
        </div>
        <input
          type="number" step="0.01" min="0" className="setting-input"
          value={form.buy_amount_sol}
          onChange={e => set('buy_amount_sol', Number(e.target.value))}
        />
      </div>

      <div className="setting-row">
        <div className="setting-label">
          <span className="setting-name">Mid-cap min USD</span>
          <span className="setting-hint">mid_cap_min_usd</span>
        </div>
        <input
          type="number" step="10000" min="0" className="setting-input"
          value={form.mid_cap_min_usd}
          onChange={e => set('mid_cap_min_usd', Number(e.target.value))}
        />
      </div>

      <div className="setting-row">
        <div className="setting-label">
          <span className="setting-name">Large-cap min USD</span>
          <span className="setting-hint">large_cap_min_usd</span>
        </div>
        <input
          type="number" step="50000" min="0" className="setting-input"
          value={form.large_cap_min_usd}
          onChange={e => set('large_cap_min_usd', Number(e.target.value))}
        />
      </div>

      <div className="setting-row">
        <div className="setting-label">
          <span className="setting-name">Scanner max positions</span>
          <span className="setting-hint">scanner_max_positions</span>
        </div>
        <input
          type="number" step="1" min="0" className="setting-input"
          value={form.scanner_max_positions}
          onChange={e => set('scanner_max_positions', Math.round(Number(e.target.value)))}
        />
      </div>

      <div className="setting-row setting-row-range">
        <div className="setting-label">
          <span className="setting-name">Confidence gate</span>
          <span className="setting-hint">ai_min_confidence_to_buy — {form.ai_min_confidence_to_buy.toFixed(2)}</span>
        </div>
        <input
          type="range" min="0" max="1" step="0.01" className="setting-range"
          value={form.ai_min_confidence_to_buy}
          onChange={e => set('ai_min_confidence_to_buy', Number(e.target.value))}
        />
      </div>

      <div className="setting-actions">
        <button className="btn-accent" onClick={save} disabled={saving || !dirty}>
          {saving ? 'SAVING…' : 'SAVE SETTINGS'}
        </button>
        {dirty && !saving && (
          <button onClick={() => { setForm(config); setDirty(false); setMessage(null) }}>
            REVERT
          </button>
        )}
      </div>

      {message && (
        <div className={message.kind === 'ok' ? 'setting-msg-ok' : 'setting-msg-err'}>
          {message.text}
        </div>
      )}

      <style>{`
        .setting-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          padding: 9px 0;
          border-bottom: 1px solid var(--border);
        }
        .setting-row:last-of-type { border-bottom: none; }
        .setting-row-range { flex-direction: column; align-items: stretch; gap: 6px; }
        .setting-label { display: flex; flex-direction: column; gap: 2px; }
        .setting-name { font-size: 12px; color: var(--text); }
        .setting-hint { font-size: 10px; color: var(--text-muted); }
        .setting-input {
          width: 96px;
          font-family: var(--font);
          font-size: 12px;
          background: var(--surface);
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 5px 8px;
          text-align: right;
        }
        .setting-input:focus { outline: none; border-color: var(--accent); }
        .setting-range { width: 100%; accent-color: var(--accent); }
        .toggle { position: relative; width: 34px; height: 18px; flex-shrink: 0; }
        .toggle input { opacity: 0; width: 0; height: 0; }
        .toggle-track {
          position: absolute; inset: 0;
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 20px;
          cursor: pointer;
          transition: var(--transition);
        }
        .toggle-track::before {
          content: '';
          position: absolute;
          width: 12px; height: 12px;
          left: 2px; top: 2px;
          background: var(--text-muted);
          border-radius: 50%;
          transition: var(--transition);
        }
        .toggle input:checked + .toggle-track { background: var(--accent-glow); border-color: var(--accent); }
        .toggle input:checked + .toggle-track::before { transform: translateX(16px); background: var(--accent); }
        .setting-actions { display: flex; gap: 8px; margin-top: 12px; }
        .setting-actions .btn-accent { flex: 1; }
        .setting-msg-ok {
          margin-top: 8px; font-size: 11px; color: var(--green);
          background: var(--green-dim); border-radius: var(--radius); padding: 6px 8px;
        }
        .setting-msg-err {
          margin-top: 8px; font-size: 11px; color: var(--red);
          background: var(--red-dim); border-radius: var(--radius); padding: 6px 8px;
        }
      `}</style>
    </div>
  )
}
