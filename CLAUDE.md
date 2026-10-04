# FENRIR v2 — Token Discovery Scout + Automated Trading Stack

## Project Overview
Multi-chain (Solana + Robinhood chain) token discovery feeding a Telegram scout, plus a full automated trading stack: the engine takes signals, sizes positions, executes Jupiter swaps and direct pump.fun bonding-curve buys, and manages positions (2s curve / 10s AMM polling). Simulation-first — `python -m fenrir` defaults to `--mode simulation`; live modes (`conservative`/`aggressive`/`degen`) are a deliberate startup flag.

The block-zero ignition lane (`tools/curve_watch.py`, 2-min) is the frozen measurement lane: rules frozen, precommitted n=100 gate-tracker records, circuit breaker at n=50 if 24h profit factor < 0.5. No new playbooks until it reports.

## Doctrine (do not regress these)

- **Rules in front of the brain.** The LLM (Claude via OpenRouter) may veto entries and initiate exits. It can never raise position size above the configured amount nor cancel a mechanical exit trigger (stop loss, take profit, trailing stop, max hold, wallet-sell signal). There is no override path — `OVERRIDE_HOLD` was removed.
- **Fail-closed AI.** AI timeout/error/unavailability never degrades into an unattended rule buy (`AI_FALLBACK_TO_RULES=false`).
- **Safety defaults.** `GLOBAL_DAILY_SOL_LIMIT` defaults to 2.0 SOL/day; the bot refuses to start in any non-simulation mode with it at 0. The pre-trade security filter (mint/freeze authority, LP burn, holder concentration) is fail-closed and on by default. Scout forensics are fail-open — correct for Telegram cards; they never gate live entries.
- **Measurement over narrative.** Gate tracker (`tools/gate_tracker.py`) stamps every alert with clearance price and forward returns. The README leads with hit rate / profit factor / drawdown, not features.

## Tech Stack
- Python 3.12+ / asyncio / aiohttp (never httpx for new client code — proxy env breaks it)
- Solana SDK (solana, solders, base58); Robinhood chain via raw JSON-RPC
- Claude via OpenRouter (decision engine); fail-closed
- SQLite (trade database); pytest + pytest-asyncio

## Key Directories
- `fenrir/` — package: `bot.py`, `config.py`, `trading/` (engine), `discovery/` (scout pipeline: filters, scoring, entry tiers, regime), `ai/` (brain), `strategies/`, `backtest/`, `filters/` (pre-trade security/market gates)
- `tools/` — operator tooling: `scout.py` (10-min discovery), `curve_watch.py` (2-min ignition lane), `gate_tracker.py` (measurement), `evaluate.py`, `channel_poll.py`, `user_watch.py`, `wallet_watch.py`
- `api/` + `dashboard/` — optional control plane / UI. Not required to run the bot.
- `config/` — `.env.example` documents every setting
- `tests/` — test suite; `tools/ci_gate.py` must pass before cutting a patch bundle

## Conventions
- Patches ship as `git am`-clean bundles against a stated base, with HANDOVER.md. Never bundle `.env`, secrets, or d3g3n's uncommitted files.
- Protected untracked files are never committed: `tools/group_scan.py`, `tools/curve_watch.py`, `api/server.py`, `dashboard/*`, `tools/telegram_notify.py`, `.env.bak-blank`, `.watch_state.json`, `docs/specs/`.
- Before cutting a bundle: `ruff format` (write mode) on the patch's changed Python files, then `tools/ci_gate.py`, then eyeball `git diff --cached --stat`.
- `pyright` in this sandbox needs `--pythonpath /home/hatch/workspace/fenrir-v2/.venv/bin/python`.
- Never run `tools/group_scan.py tick` by hand while its 2-min cron is live (getUpdates 409).
- d3g3n executes trades manually. Never trade, move funds, import keys, or enable live mode from this sandbox.

## Skills

### Solana / Blockchain / Web3
@skill ~/.claude/skills/skills/blockchain-developer/SKILL.md
@skill ~/.claude/skills/skills/web3-testing/SKILL.md

### Python & Async
@skill ~/.claude/skills/skills/async-python-patterns/SKILL.md
@skill ~/.claude/skills/skills/python-pro/SKILL.md
@skill ~/.claude/skills/skills/python-patterns/SKILL.md
@skill ~/.claude/skills/skills/python-testing-patterns/SKILL.md
@skill ~/.claude/skills/skills/python-development-python-scaffold/SKILL.md

### FastAPI & API Design
@skill ~/.claude/skills/skills/fastapi-pro/SKILL.md
@skill ~/.claude/skills/skills/fastapi-router-py/SKILL.md
@skill ~/.claude/skills/skills/api-patterns/SKILL.md
@skill ~/.claude/skills/skills/api-design-principles/SKILL.md
@skill ~/.claude/skills/skills/api-security-best-practices/SKILL.md

### Security
@skill ~/.claude/skills/skills/security-auditor/SKILL.md

### Trading & Risk Management
@skill ~/.claude/skills/skills/risk-manager/SKILL.md
@skill ~/.claude/skills/skills/risk-metrics-calculation/SKILL.md
@skill ~/.claude/skills/skills/backtesting-frameworks/SKILL.md

### AI / LLM Integration
@skill ~/.claude/skills/skills/ai-engineer/SKILL.md
@skill ~/.claude/skills/skills/llm-app-patterns/SKILL.md
@skill ~/.claude/skills/skills/llm-application-dev-ai-assistant/SKILL.md

### Database & Performance
@skill ~/.claude/skills/skills/database-optimizer/SKILL.md
@skill ~/.claude/skills/skills/sql-optimization-patterns/SKILL.md
@skill ~/.claude/skills/skills/performance-engineer/SKILL.md
@skill ~/.claude/skills/skills/performance-profiling/SKILL.md

### Monitoring & Observability
@skill ~/.claude/skills/skills/observability-engineer/SKILL.md
@skill ~/.claude/skills/skills/observability-monitoring-monitor-setup/SKILL.md
@skill ~/.claude/skills/skills/distributed-tracing/SKILL.md

### Error Handling & Reliability
@skill ~/.claude/skills/skills/error-handling-patterns/SKILL.md
@skill ~/.claude/skills/skills/temporal-python-pro/SKILL.md

### Deployment & CI/CD
@skill ~/.claude/skills/skills/deployment-engineer/SKILL.md
@skill ~/.claude/skills/skills/docker-expert/SKILL.md
@skill ~/.claude/skills/skills/cicd-automation-workflow-automate/SKILL.md

### Testing
@skill ~/.claude/skills/skills/testing-patterns/SKILL.md
@skill ~/.claude/skills/skills/e2e-testing-patterns/SKILL.md

### Architecture
@skill ~/.claude/skills/skills/architecture/SKILL.md
@skill ~/.claude/skills/skills/architecture-patterns/SKILL.md
@skill ~/.claude/skills/skills/architecture-decision-records/SKILL.md

### Task Orchestration
@skill ~/.claude/skills/skills/bullmq-specialist/SKILL.md
