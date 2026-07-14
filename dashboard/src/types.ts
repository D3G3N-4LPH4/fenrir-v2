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

export interface BotConfigSurface {
  market_scanner_enabled: boolean
  global_daily_sol_limit: number
  buy_amount_sol: number
  mid_cap_min_usd: number
  large_cap_min_usd: number
  scanner_max_positions: number
  ai_min_confidence_to_buy: number
}

export interface AvailableStrategy {
  strategy_id: string
  display_name: string
  description: string
  uses_market_data: boolean
  loaded: boolean
  active: boolean
  paused: boolean
}

export interface DiscoveryScores {
  overall: number
  momentum: number
  safety: number
  liquidity: number
  holder: number
  community: number
  risk: number
}

export interface DiscoveryItem {
  chain: string
  token_address: string
  symbol: string
  name: string
  market_cap_usd: number
  liquidity_usd: number
  volume_24h_usd: number
  holder_count: number | null
  age_minutes: number
  dex_id: string | null
  matched_filters: string[]
  scores: DiscoveryScores
  socials: { twitter: string | null; telegram: string | null; website: string | null }
  detected_at: string
}

export interface DiscoveryConfigSurface {
  enabled: boolean
  running?: boolean
  chains: string[]
  filters: string[]
  interval_seconds: number
  min_alert_score: number
}
