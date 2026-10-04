# FENRIR v2 — Multi-Tier AI Trading & Discovery Platform

Async Python system for memecoin trading and research. Two halves share one codebase:

1. **Trading bot** (Solana) — an autonomous AI decision engine that trades the **whole market**, not just launches:
   - **Low cap** — fresh [Pump.fun](https://pump.fun) launches, bought directly on the bonding curve
   - **Mid cap** — migrated coins (graduated off the curve to PumpSwap/Raydium)
   - **Large cap** — established $1M+ coins surfaced by a market scanner
2. **Discovery platform** (multi-chain) — a **scout** that sweeps Solana *and* EVM chains (Ethereum, BNB, Base, Robinhood), runs candidates through filters → scoring → read-only strategy playbooks → on-chain safety forensics, and pushes graded Telegram alerts. A drift-free backtester replays real signals to measure forward performance. This is the research loop; live positions are entered manually from its alerts.

Mid/large caps are traded on AMMs via Jupiter; fresh launches via direct on-chain pump.fun execution. Real-time WebSocket launch + block-zero ignition monitoring, a multi-source discovery sweep, Jito MEV protection, a pluggable strategy system, Ouroboros dump-recovery detection, market-geometry pre-scoring, and a live React web dashboard with in-place settings + strategy switching.

```bash
pip install -e ".[all]"
cp config/.env.example .env   # fill in your keys
python -m fenrir --mode simulation     # the trading bot
python tools/scout.py --chains solana robinhood   # the discovery scout
```

> **Live-capable.** The pump.fun buy/sell path is verified against the current on-chain program (v2 buyback fees, per-token fee resolution, Token-2022), and the Jupiter path is verified for migrated/established tokens. Every real send is simulate-guarded. Trade only with a dedicated wallet and funds you can lose.

---

## Features

### Trading

- **Multi-tier candidate discovery** — real-time WebSocket monitoring of Pump.fun launches (low cap) plus a periodic **market scanner** (Jupiter trending) that surfaces migrated mid-caps and established large-caps by market cap
- **Direct pump.fun bonding-curve buy/sell** — verified against the current on-chain program (v2 buyback fee accounts resolved per-token, Token-2022 support, live fee-recipient, dynamic priority fee)
- **Jupiter buy/sell for non-curve tokens** — migrated/established coins trade on AMMs; routed through Jupiter (`lite-api.jup.ag`) on both entry and exit
- Simulate-before-send guard on every live transaction
- Jito MEV bundle protection (optional)
- Multi-source price feeds with fallback

### AI Decision Engine

- Claude Brain — autonomous LLM evaluates every token before entry and monitors exits
- Supports cloud API (Anthropic / OpenRouter) or a **local abliterated model** via vLLM / llama.cpp
- Session memory: rolling decision history informs future decisions
- Historical memory: cross-session pattern learning (creator fingerprinting, liquidity profiles)
- Dynamic position sizing is rules-set — the brain can veto or shrink an entry, never size above the configured buy amount
- Mechanical exits (stop loss, take profit, trailing stop, max hold) execute without an LLM call — the brain cannot cancel a triggered exit

### Risk Management

- Stop loss, take profit, trailing stop, max hold time — all per-strategy configurable
- **Ouroboros detector** — identifies dump → fake recovery → second dump patterns and auto-tightens trailing stops per position
- **Market geometry analyzer** — pre-entry scoring across four axes (creator imprint, momentum geometry, liquidity depth, defense robustness) with auto-derived TradeParams
- Per-strategy SOL budget + max-position limits
- **Global daily SOL cap** — master safety valve on net live exposure across all strategies (defaults to 2.0 SOL/day; the bot refuses to start in any non-simulation mode with it at 0)
- **Pre-trade security filter** — mint/freeze authority, LP burn, holder concentration; fail-closed and on by default
- Configurable AI confidence threshold

### Discovery & Research (Scout)

- **Multi-chain sweep** — one pipeline over Solana and EVM chains (Ethereum, BNB, Base, Robinhood). Sources: DexScreener (boosted + latest paid profiles), GeckoTerminal (new / trending pools), pump.fun bonding curves, and a Robinhood-chain on-chain new-pair monitor
- **Filter → score → playbook tagging** — candidates pass tiered filters (low-cap alpha, mid-cap momentum, risk-on `degen_launch` / `volatility_breakout`), get a 0–100 score, and are run read-only through the signal strategies to tag which playbook(s) they fit, with cross-strategy confluence
- **On-chain safety forensics** — GoPlus (EVM honeypot / tax / LP-lock / blacklist), RugCheck (Solana), 0xPerceptor (Robinhood-chain deep checks), Solana holder-distribution forensics, bundled-launch / deployer-cluster detection, and platform-vault LP-custody detection (StonkFun / LaunchLab). Fail-open by design
- **Event-driven lanes** — block-zero curve-ignition watcher (websocket `logsSubscribe`, 10–50% of curve), graduation watch (50–85%), and a migration sniper — covering the pre-DexScreener window the poll path misses
- **Feedback loop** — gate-clearance tracker records entry prices and reports forward 1h/4h/24h hit-rates per filter/pattern; chart-pattern recognition (dependency-free ZigZag, 11 patterns) adds confluence
- **Drift-free backtester** — replays recorded samples through the *real* strategy `evaluate_token` + unified `Signal` + each strategy's own exits; win% / expectancy / profit-factor / Sharpe / max-drawdown, with confluence splits
- **Telegram alerts** — graded cards (symbol/chain/source, score, playbooks, metrics, tap-to-copy contract), multi-chat, with case-normalized 24h dedup

### Infrastructure

- Pluggable strategy system — 16 registered strategies (sniper family, graduation, migration-snipe, momentum, mean-reversion, reversal, volume-anomaly, volume-surge, narrative-tracker, range-rotation, flush-recovery, degen-ignition, volatility-breakout), easy to add more
- Event bus — decoupled alerting (log, Telegram, audit, WebSocket, AI health monitor)
- Merkle hash-chain audit trail — tamper-evident SQLite trade log
- Budget tracker — per-strategy daily spend limits + global cap
- FastAPI REST + WebSocket API
- Rich terminal dashboard (`--dashboard` flag)
- **React web dashboard** — live positions, event feed, AI stats, a **Settings panel** (change scanner / caps / buy amount / confidence gate live, no restart), and a **Strategies tab** (activate/pause any strategy on a running bot)

---

## Project Structure

```text
fenrir/
├── bot.py                   # Orchestrator — wires all components
├── config.py                # BotConfig dataclass + trading presets
├── ai/
│   ├── brain.py             # ClaudeBrain — autonomous decision engine
│   ├── decision_engine.py   # AITradingAnalyst (cloud API)
│   ├── local_backend.py     # LocalAITradingAnalyst (local vLLM/llama.cpp)
│   ├── market_geometry.py   # MarketGeometryAnalyzer — pre-entry scoring
│   └── memory.py            # Session + historical AI memory
├── core/
│   ├── positions.py         # Position tracking + exit condition logic
│   ├── dump_recovery.py     # PostDumpRecoveryDetector (Ouroboros pattern)
│   ├── budget.py            # Per-strategy SOL budget enforcement
│   ├── client.py            # Solana RPC client
│   ├── jupiter.py           # Jupiter swap engine
│   └── wallet.py            # Wallet management
├── data/
│   ├── audit.py             # Merkle hash-chain audit trail
│   ├── historical_memory.py # Cross-session SQLite learning store
│   ├── price_feed.py        # Multi-source price feeds
│   └── database.py          # Trade database
├── events/
│   ├── bus.py               # Async pub/sub event bus
│   ├── types.py             # TradeEvent definitions + factory helpers
│   └── adapters/            # log, telegram, audit, health monitor
├── strategies/              # 16 registered strategies (classic + market-data "playbook")
│   ├── base.py              # TradingStrategy ABC + TradeParams
│   ├── sniper.py            # Fast entry on new launches (+ conservative/degen)
│   ├── graduation.py        # Targets tokens approaching Raydium migration
│   ├── migration_snipe.py   # Snipes freshly-migrated (PumpSwap) tokens
│   ├── momentum.py          # Momentum / mean_reversion / reversal (market-data)
│   ├── volume_anomaly.py    # Volume-spike + volume_surge (higher-cap)
│   ├── narrative.py         # Narrative / social momentum
│   ├── range_rotation.py    # Community-coin rebalancing playbook
│   ├── flush_recovery.py    # Post-flush second-leg entries
│   ├── degen_ignition.py    # Risk-on earliness play
│   └── volatility_breakout.py # Vertical-move breakout (hardened, fail-closed)
├── discovery/               # Multi-chain scout: the research pipeline
│   ├── models.py            # TokenSnapshot, Chain enum, SafetySignals
│   ├── filters.py           # Tiered entry filters (+ risk-on tier)
│   ├── scoring.py           # 0–100 candidate scoring
│   ├── playbooks.py         # Read-only strategy tagging + confluence
│   ├── scanner.py           # Discovery sweep orchestration
│   ├── acceleration.py      # Momentum-transition accel tracker
│   ├── chart_patterns.py    # ZigZag + 11 chart patterns
│   ├── bundle_check.py      # Bundled-launch / deployer-cluster detection
│   ├── lp_vault.py / lp_lock_v4.py  # Platform-vault + v4 LP-lock reads
│   ├── solana_forensics.py  # Holder-distribution forensics
│   ├── pool_deepcheck.py    # Per-pool Swap-event qualification
│   ├── second_life.py       # SAPLING second-life gate
│   ├── seen.py / alerts.py  # Dedup + Telegram alert formatting
│   └── providers/           # dexscreener, geckoterminal, goplus, perceptor,
│                            #   rh_onchain, pumpfun, pumpfun_events
├── signals/                 # Unified Signal abstraction + noisy-OR aggregator
├── backtest/                # Drift-free replay backtester (engine, metrics, report)
├── evm/                     # EVM read path (adapters, evaluator, scanner, safety)
├── trading/
│   ├── engine.py            # Buy/sell execution (pump curve + Jupiter)
│   ├── monitor.py           # PumpFun WebSocket monitor (+ migration feed)
│   └── scanner.py           # MarketScanner — multi-tier Jupiter-trending discovery
├── protocol/
│   ├── pumpfun.py           # Pump.fun bonding curve protocol
│   └── jito.py              # Jito MEV bundle submission
└── ui/
    └── dashboard.py         # Rich terminal dashboard

api/
└── server.py                # FastAPI REST + WebSocket server

dashboard/                   # React web dashboard (Vite + TypeScript)
├── src/
│   ├── App.tsx              # Root layout, WS + polling state
│   ├── types.ts             # TypeScript interfaces
│   └── components/
│       ├── StatusBar.tsx    # Bot state, mode, live PnL, WS indicator
│       ├── BotControls.tsx  # Start/stop, strategy pause/resume
│       ├── PositionsTable.tsx # Live positions with Ouroboros badge
│       ├── EventFeed.tsx    # Real-time event stream
│       ├── BrainStats.tsx   # AI decision metrics
│       ├── SettingsPanel.tsx   # Live config (scanner, caps, confidence gate)
│       └── StrategiesPanel.tsx # Strategies tab — switch strategies live
└── package.json

tools/                       # Scout + research CLIs (see "Discovery Tools" below)
tests/                       # 1600+ tests (pytest + pytest-asyncio)
config/                      # default.json, devnet.json, .env.example
```

---

## Installation

**Requirements:** Python 3.10+ (3.12 recommended/tested), Node.js 18+ (for web dashboard)

```bash
# Clone
git clone https://github.com/D3G3N-4LPH4/fenrir-v2.git
cd fenrir-v2

# Python environment
pip install -e ".[all]"

# Copy and fill in environment variables
cp config/.env.example .env
```

---

## Configuration

`.env` — all settings can be set here or as environment variables:

```env
# Solana
SOLANA_RPC_URL=https://your-rpc-endpoint.com
SOLANA_WS_URL=wss://your-rpc-endpoint.com
WALLET_PRIVATE_KEY=your_base58_private_key

# AI (cloud)
OPENROUTER_API_KEY=sk-or-...

# AI (local — OBLITERATUS abliterated model)
AI_LOCAL_MODEL_ENABLED=false
AI_LOCAL_MODEL_URL=http://localhost:8000/v1/chat/completions
AI_LOCAL_MODEL_NAME=fenrir-brain

# Telegram alerts (optional; used by the bot and the scout)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_IDS=-100111,-100222   # comma-separated, preferred (multi-chat)
TELEGRAM_CHAT_ID=-100111            # legacy single-chat fallback

# Market scanner (mid/large-cap discovery) — off by default
MARKET_SCANNER_ENABLED=false
MID_CAP_MIN_USD=200000
LARGE_CAP_MIN_USD=1000000
SCANNER_MIN_LIQUIDITY_USD=50000

# Risk / execution
GLOBAL_DAILY_SOL_LIMIT=2.0      # master cap across all strategies; bot refuses non-sim start at 0
DYNAMIC_PRIORITY_FEE_ENABLED=false

# Web API
FENRIR_API_KEY=your_secret_key
FENRIR_DEV_MODE=false       # set true to skip auth locally
# Extra browser origins allowed to call the API (comma-separated) — e.g. a
# remote/Replit-hosted dashboard. localhost dev ports are always allowed.
# No wildcard: allow_credentials is on, so origins must be listed explicitly.
FENRIR_CORS_ORIGINS=https://your-repl.replit.app,https://your-repl.repl.co
```

Key `BotConfig` fields (can also be set in `config/default.json`):

| Field | Default | Description |
| --- | --- | --- |
| `mode` | `simulation` | `simulation` / `conservative` / `aggressive` / `degen` |
| `buy_amount_sol` | 0.1 | SOL per trade |
| `stop_loss_pct` | 25.0 | Exit if down this % |
| `take_profit_pct` | 100.0 | Exit if up this % |
| `trailing_stop_pct` | 15.0 | Trail from peak (auto-tightened by Ouroboros) |
| `ai_analysis_enabled` | true | Enable AI decision engine |
| `ai_min_confidence_to_buy` | 0.6 | Minimum AI confidence to enter |
| `ai_local_model_enabled` | false | Route AI to local model instead of cloud |
| `market_scanner_enabled` | false | Scan Jupiter trending for mid/large-cap candidates |
| `mid_cap_min_usd` / `large_cap_min_usd` | 200k / 1M | Market-cap tier thresholds (USD) |
| `global_daily_sol_limit` | 2.0 | Net-exposure cap across all strategies (bot refuses non-sim start at 0) |
| `dynamic_priority_fee_enabled` | false | Size priority fee from recent on-chain fees |

---

## Running

### Terminal bot

```bash
# Simulation mode (no real trades)
python -m fenrir --mode simulation

# With terminal dashboard
python -m fenrir --mode conservative --dashboard

# Multiple strategies
python -m fenrir --mode aggressive --strategies sniper graduation

# Override risk params
python -m fenrir --mode simulation --stop-loss 20 --take-profit 150
```

### API server + Web dashboard

```bash
# Terminal 1: API backend (port 8000). FENRIR_DEV_MODE=true skips API-key auth
# for local use; set FENRIR_API_KEY instead for anything exposed.
FENRIR_DEV_MODE=true python -m api.server        # or: uvicorn api.server:app --port 8000 --reload

# Terminal 2: Web dashboard (port 5173) — proxies /api -> :8000
cd dashboard
npm install
npm run dev
# Open http://localhost:5173 — Unchain (start), then the Positions / Strategies tabs
```

> **Note:** If using the local OBLITERATUS model, vLLM also defaults to port 8000.
> Run the API server on a different port: `uvicorn api.server:app --port 8001`
> and update `dashboard/vite.config.ts` proxy target accordingly.

### Remote dashboard (e.g. Replit) via an HTTPS tunnel

A dashboard served over **HTTPS** (like a Replit-hosted page) can't call an
`http://localhost` bot API — browsers block mixed content, and `wss://` needs TLS
too. Expose the local API over HTTPS with a tunnel:

```bash
# Install cloudflared (no signup) or ngrok, then:
python scripts/tunnel.py            # auto-detects provider, tunnels :8000
```

The script grabs the public `https://…` URL, adds it to `FENRIR_CORS_ORIGINS` in
`.env`, and prints the next steps. Then:

1. **(Re)start the API** so it reads the updated `.env` (`python -m api.server`).
2. Point the remote dashboard's API base at the tunnel URL (`https://…`) and its
   WebSocket at `wss://<same-host>/ws/updates`.
3. Send your `FENRIR_API_KEY` as the `X-API-Key` header (or `FENRIR_DEV_MODE=true`
   while testing).
4. Also add the **dashboard's own origin** to `FENRIR_CORS_ORIGINS` (the script
   adds the tunnel URL; the browser page's origin must be listed too).

> Quick-tunnel URLs are random per run. For a stable URL use a named Cloudflare
> tunnel or an ngrok reserved/static domain.

### Local model setup (OBLITERATUS)

Run the AI brain fully offline on your own GPU with safety guardrails removed:

```bash
# 1. Install and abliterate
pip install obliteratus
obliteratus obliterate meta-llama/Llama-3.1-8B-Instruct \
    --method advanced \
    --output-dir ./models/fenrir-brain

# 2. Serve with vLLM (OpenAI-compatible)
pip install vllm
vllm serve ./models/fenrir-brain \
    --port 8000 \
    --served-model-name fenrir-brain \
    --max-model-len 4096

# 3. Enable in .env
AI_LOCAL_MODEL_ENABLED=true
AI_LOCAL_MODEL_URL=http://localhost:8000/v1/chat/completions
```

The brain performs a health check on startup and falls back to the cloud API if the local server is unreachable.

---

## Strategies

| Strategy | ID | Description |
| --- | --- | --- |
| Sniper | `sniper` | Fast entry on new launches matching liquidity/mcap filters |
| Conservative / Degen Sniper | `sniper_conservative` / `sniper_degen` | Sniper tuned tighter / looser |
| Graduation | `graduation` | Targets tokens approaching Raydium migration (>70% bonding curve) |
| Migration Snipe | `migration_snipe` | Snipes freshly-migrated (PumpSwap) tokens |
| Momentum | `momentum` | Market-data momentum entries |
| Mean Reversion | `mean_reversion` | Oversold-bounce entries |
| Reversal | `reversal` | Market-data reversal signals |
| Volume Anomaly | `volume_anomaly` | Volume-spike signals |
| Volume Surge | `volume_surge` | Higher-cap volume-driven strategy |
| Narrative Tracker | `narrative_tracker` | Narrative / social momentum |
| Range Rotation | `range_rotation` | Community-coin rebalancing playbook |
| Flush Recovery | `flush_recovery` | Post-flush second-leg entries |
| Degen Ignition | `degen_ignition` | Risk-on earliness play (mcap $500–$30k, hot tape) |
| Volatility Breakout | `volatility_breakout` | Vertical-move breakout; fail-closed on unknown holders |

The **market-data (“playbook”) strategies** evaluate a `TokenSnapshot` and are what the scout tags candidates against read-only; they default to **disabled for live trading**. Each strategy has its own `budget_sol`, `max_concurrent_positions`, and can be paused/resumed — or **activated live on a running bot** — from the dashboard's Strategies tab or the API (market-data strategies build their data provider on demand).

---

## Discovery Tools

The scout research layer runs as standalone CLIs (typically on a cron / systemd cadence, read-only — they surface candidates, they don't trade):

| Tool | What it does |
| --- | --- |
| `tools/scout.py` | The core 10-min sweep: multi-chain sources → filters → scoring → playbook tags → safety → Telegram |
| `tools/curve_events.py` | Block-zero curve-ignition watcher (websocket `logsSubscribe`, 10–50% of curve) |
| `tools/grad_snipe.py` | pump.fun graduation/migration watcher (catches migrations within minutes) |
| `tools/rh_pair_watch.py` | Robinhood-chain on-chain new-pair fast lane |
| `tools/gate_tracker.py` | Gate-clearance tracker: `record` / `tick` / `report` forward hit-rates |
| `tools/wallet_watch.py` | Tracked-wallet buys as scout candidates (`--wallets`, `--state`) |
| `tools/hot_watch.py` | Acceleration fast-lane (needs `accel_history` seeded from live cycles) |
| `tools/coin_watch.py` | Per-coin time-series tracker + PVP set view (side-by-side volume rotation) |
| `tools/user_watch.py` / `tools/user_cases.py` | Track user-dropped contracts; turn shared coins into filter-tuning data |
| `tools/perceptor.py` | 0xPerceptor on-chain forensics CLI (`investigate` / `status` / `sweep` / `deepcheck`) |
| `tools/evaluate.py` | One-off evaluate a single address across the full pipeline |
| `tools/channel_poll.py` | Poll Telegram channels for candidate contracts |
| `tools/signal_backtest.py` / `tools/backtest.py` | Drift-free signal backtest + legacy backtest framework |
| `tools/evm_collect.py` | Forward-price sample collector for EVM backtesting |

> Several tools want your **private `SOLANA_RPC_URL`** for fast-polling (the public endpoint rate-limits). `.env` also needs `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_IDS` (comma-separated) for alerts.

---

## Risk Systems

### Ouroboros Detector

Detects the dump → fake recovery → second dump pattern that traps holders:

```text
Peak: 1.00 → Dump to 0.55 (-45%) → Bounce to 0.70 (+27%) → ALERT → Second dump to 0.20
```

On detection, the trailing stop for that position is automatically tightened from the strategy default (e.g. 15%) to 8%, kicking an early exit before the second dump. Fires an `OUROBOROS_DETECTED` event on the event bus.

### Market Geometry Analyzer

Runs four analysis modules on every token before AI evaluation:

| Axis | What it measures |
| --- | --- |
| Creator Imprint | Liquidity/mcap ratio, social links, description quality → rug vs holder pattern |
| Momentum Geometry | Launch liquidity profile → organic vs coordinated pump |
| Liquidity Depth | Bonding curve migration progress, absolute SOL floor |
| Defense Robustness | mcap/liquidity inflation ratio → sell wall risk |

Composite quality and risk scores (0–1) feed auto-derived `TradeParams` (position size, slippage, stops, take-profit) and an AI context block injected into every entry evaluation prompt.

---

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Server health + bot status |
| `POST` | `/bot/start` | Start bot with config |
| `POST` | `/bot/stop` | Stop bot |
| `GET` | `/bot/status` | Status + portfolio summary |
| `GET` | `/bot/full-status` | Full status: AI, strategies, budget, ouroboros, geometry |
| `GET` | `/bot/positions` | All open positions with Ouroboros flags |
| `POST` | `/bot/trade` | Manual buy/sell |
| `GET` | `/bot/strategies` | Runtime states of loaded strategies |
| `GET` | `/bot/strategies/available` | All registered strategies + live state (Strategies tab) |
| `POST` | `/bot/strategies/{id}/pause` \| `/resume` | Pause / resume a loaded strategy |
| `POST` | `/bot/strategies/{id}/enable` \| `/disable` | Activate (live-load) / deactivate a strategy |
| `GET` | `/bot/config` | Live config surface |
| `POST` | `/bot/config` | Apply a config patch live (no restart) or stage it |
| `WS` | `/ws/updates` | Real-time event stream |

Docs at `http://localhost:8000/docs` when the API server is running.

---

## Tech Stack

- **Python 3.10+** (3.12 recommended) — asyncio, aiohttp, websockets
- **Solana** — solana, solders, base58; **EVM** — keyless on-chain reads over JSON-RPC
- **Discovery data** — DexScreener, GeckoTerminal, Jupiter; **safety** — GoPlus, RugCheck, 0xPerceptor; keccak via pycryptodome
- **AI** — Claude via Anthropic / OpenRouter (with free-tier fallback); local via vLLM or llama.cpp (OBLITERATUS)
- **API** — FastAPI + Uvicorn
- **Storage** — SQLite (trades, audit chain, historical memory); JSON state for scout dedup / tracking
- **Frontend** — React 18 + TypeScript + Vite (JetBrains Mono, no chart libs)
- **Terminal UI** — Rich
- **Testing** — pytest + pytest-asyncio (1600+ tests); dashboard type-checked with `tsc`
- **Linting** — Ruff, Mypy, Pyright, pre-commit

---

## Development

```bash
# Install with dev extras
pip install -e ".[all]"

# Run tests
pytest

# Skip devnet integration tests (requires live Solana devnet)
pytest -m "not devnet"

# Lint + format
ruff check .
ruff format .

# Type check
mypy fenrir/
```

---

## License

MIT — see [LICENSE](LICENSE).
