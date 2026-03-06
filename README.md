# FENRIR v2 — Pump.fun Solana Trading Bot

Async Python trading bot for sniping memecoin launches on [Pump.fun](https://pump.fun) (Solana).
Autonomous AI decision engine, real-time WebSocket monitoring, Jito MEV protection, pluggable strategy system, Ouroboros dump-recovery detection, market geometry analysis, and a live web dashboard.

```bash
pip install -e ".[all]"
cp config/.env.example .env   # fill in your keys
python -m fenrir --mode simulation
```

---

## Features

### Trading

- WebSocket monitoring of Pump.fun token launches in real time
- Direct bonding curve buy/sell via on-chain program interaction
- Jupiter swap engine for liquidity routing
- Jito MEV bundle protection (optional)
- Multi-source price feeds with fallback

### AI Decision Engine

- Claude Brain — autonomous LLM evaluates every token before entry and monitors exits
- Supports cloud API (Anthropic / OpenRouter) or a **local abliterated model** via vLLM / llama.cpp
- Session memory: rolling decision history informs future decisions
- Historical memory: cross-session pattern learning (creator fingerprinting, liquidity profiles)
- Dynamic position sizing — AI can override the configured buy amount
- AI exit override — brain can hold through mechanical triggers with a hard-floor safety net

### Risk Management

- Stop loss, take profit, trailing stop, max hold time — all per-strategy configurable
- **Ouroboros detector** — identifies dump → fake recovery → second dump patterns and auto-tightens trailing stops per position
- **Market geometry analyzer** — pre-entry scoring across four axes (creator imprint, momentum geometry, liquidity depth, defense robustness) with auto-derived TradeParams
- Per-strategy SOL budget enforcement
- Configurable AI confidence threshold

### Infrastructure

- Pluggable strategy system — `sniper` and `graduation` built in, easy to add custom strategies
- Event bus — decoupled alerting (log, Telegram, audit, WebSocket, AI health monitor)
- Merkle hash-chain audit trail — tamper-evident SQLite trade log
- Budget tracker — per-strategy daily spend limits
- FastAPI REST + WebSocket API
- Rich terminal dashboard (`--dashboard` flag)
- **React web dashboard** — live positions, event feed, AI stats, strategy controls

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
├── strategies/
│   ├── base.py              # TradingStrategy ABC + TradeParams
│   ├── sniper.py            # Fast entry on new launches
│   └── graduation.py        # Targets tokens approaching Raydium migration
├── trading/
│   ├── engine.py            # Buy/sell execution
│   └── monitor.py           # PumpFun WebSocket monitor
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
│       └── BrainStats.tsx   # AI decision metrics
└── package.json

tools/                       # Backtesting framework
tests/                       # 246 tests (pytest + pytest-asyncio)
config/                      # default.json, devnet.json, .env.example
```

---

## Installation

**Requirements:** Python 3.12+, Node.js 18+ (for web dashboard)

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

# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Web API
FENRIR_API_KEY=your_secret_key
FENRIR_DEV_MODE=false       # set true to skip auth locally
```

Key `BotConfig` fields (can also be set in `config/default.json`):

| Field | Default | Description |
| --- | --- | --- |
| `mode` | `simulation` | `simulation` / `conservative` / `aggressive` / `degen` |
| `buy_amount_sol` | 0.1 | SOL per trade |
| `stop_loss_pct` | 25.0 | Exit if down this % |
| `take_profit_pct` | 100.0 | Exit if up this % |
| `trailing_stop_pct` | 15.0 | Trail from peak (auto-tightened by Ouroboros) |
| `ai_analysis_enabled` | false | Enable AI decision engine |
| `ai_min_confidence_to_buy` | 0.6 | Minimum AI confidence to enter |
| `ai_local_model_enabled` | false | Route AI to local model instead of cloud |

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
# Terminal 1: API backend (port 8000)
uvicorn api.server:app --port 8000 --reload

# Terminal 2: Web dashboard (port 5173)
cd dashboard
npm install
npm run dev
# Open http://localhost:5173
```

> **Note:** If using the local OBLITERATUS model, vLLM also defaults to port 8000.
> Run the API server on a different port: `uvicorn api.server:app --port 8001`
> and update `dashboard/vite.config.ts` proxy target accordingly.

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
| Graduation | `graduation` | Targets tokens approaching Raydium migration (>70% bonding curve) |

Each strategy has its own `budget_sol`, `max_concurrent_positions`, and can be paused/resumed independently via the API or web dashboard.

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
| `GET` | `/bot/strategies` | Strategy runtime states |
| `POST` | `/bot/strategies/{id}/pause` | Pause a strategy |
| `POST` | `/bot/strategies/{id}/resume` | Resume a strategy |
| `GET` | `/bot/config` | Current bot config |
| `WS` | `/ws/updates` | Real-time event stream |

Docs at `http://localhost:8000/docs` when the API server is running.

---

## Tech Stack

- **Python 3.12+** — asyncio, aiohttp, websockets
- **Solana** — solana, solders, base58
- **AI** — Claude via Anthropic / OpenRouter; local via vLLM or llama.cpp (OBLITERATUS)
- **API** — FastAPI + Uvicorn
- **Storage** — SQLite (trades, audit chain, historical memory)
- **Frontend** — React 18 + TypeScript + Vite (JetBrains Mono, no chart libs)
- **Terminal UI** — Rich
- **Testing** — pytest + pytest-asyncio (246 tests)
- **Linting** — Ruff, Mypy, pre-commit

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
