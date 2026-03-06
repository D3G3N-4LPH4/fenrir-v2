export interface Position {
  token_address: string
  token_symbol: string
  strategy_id: string
  entry_time: string
  entry_price: number
  current_price: number
  amount_tokens: number
  amount_sol_invested: number
  pnl_percent: number
  pnl_sol: number
  peak_price: number
  trailing_stop_override_pct: number | null
  ouroboros_triggered: boolean
}

export interface Portfolio {
  num_positions: number
  total_invested_sol: number
  current_value_sol: number
  total_pnl_sol: number
  total_pnl_pct: number
}

export interface BotStatus {
  status: 'stopped' | 'running' | 'error'
  mode: string | null
  uptime_seconds: number | null
  positions_count: number
  portfolio: Portfolio | null
  error: string | null
}

export interface AiBrainStats {
  ai_entries_evaluated: number
  ai_entries_bought: number
  ai_entries_skipped: number
  ai_timeouts: number
  ai_errors: number
  rule_fallbacks: number
  ai_exits_evaluated: number
  ai_exits_overridden: number
  ai_avg_response_ms: number
}

export interface StrategyStatus {
  strategy_id: string
  display_name: string
  active: boolean
  paused: boolean
  sol_spent_today: number
  budget_remaining: number
  positions_open: number
  trades_today: number
  win_rate_today: number
}

export interface OuroborosStats {
  positions_tracked: number
  total_detections: number
  currently_triggered: number
}

export interface FeedEvent {
  id: string
  event_type: string
  category: string
  severity: string
  timestamp: string
  token_symbol: string | null
  token_address: string | null
  strategy_id: string | null
  message: string
  data: Record<string, unknown>
}

export type WsMessage = FeedEvent | {
  event: string
  bot_status?: string
  mode?: string
  timestamp?: string
  strategy_id?: string
}
