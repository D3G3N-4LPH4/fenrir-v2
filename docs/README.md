# FENRIR - Pump.fun Trading Bot

AI-powered Solana memecoin trading bot for pump.fun launches with intelligent risk management.

## Features

- **Automated Trading** - Monitor pump.fun for new token launches and execute trades automatically
- **AI-Powered Decisions** - Claude Brain evaluates every token with session memory and portfolio awareness
- **Risk Management** - Stop loss, take profit, trailing stops, time-based exits with AI override capability
- **Trading Modes** - Simulation (paper trading), Conservative, Aggressive, Degen
- **Real-time Monitoring** - WebSocket-based live blockchain monitoring with polling fallback
- **Direct Bonding Curve** - Trades directly against pump.fun's bonding curve program (no DEX needed)
- **MEV Protection** - Optional Jito bundle submission to prevent front-running
- **Multi-Source Pricing** - Aggregated prices from Jupiter, Birdeye, and DexScreener
- **Portfolio Tracking** - Track positions, P&L, and performance metrics
- **REST API** - FastAPI backend for external control and monitoring
- **Backtesting** - Test strategies on historical data before risking real money

## Project Structure

```
fenrir/                     Main package
  config.py                 TradingMode, BotConfig
  logger.py                 FenrirLogger
  bot.py                    FenrirBot + CLI entry point
  core/                     Core infrastructure
    wallet.py               WalletManager
    client.py               SolanaClient (RPC interface)
    positions.py            Position, PositionManager
    jupiter.py              JupiterSwapEngine
  trading/                  Trade execution
    engine.py               TradingEngine (buy/sell via bonding curve)
    monitor.py              PumpFunMonitor (WebSocket + polling)
  protocol/                 On-chain protocols
    pumpfun.py              Pump.fun program interface
    jito.py                 Jito MEV protection
  ai/                       AI decision engine
    brain.py                ClaudeBrain (autonomous orchestrator)
    decision_engine.py      AITradingAnalyst (prompt building + LLM calls)
    memory.py               AISessionMemory (rolling decision history)
  data/                     Data layer
    price_feed.py           Multi-source price aggregation
    database.py             SQLite trade database
    analytics.py            Performance analytics
api/                        REST API
  server.py                 FastAPI backend
tests/                      Test suite
  test_core.py              Core module tests
tools/                      Utilities
  backtest.py               Backtesting framework
config/                     Configuration
  default.json              Example config
  .env.example              Environment variables template
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[all]"

# 2. Configure environment
cp config/.env.example .env
# Edit .env with your RPC URL and wallet key

# 3. Run in simulation mode (safe - no real trades)
python -m fenrir --mode simulation

# 4. Run with custom parameters
python -m fenrir --mode conservative --buy-amount 0.05 --stop-loss 15

# 5. Run with config file
python -m fenrir --config config/default.json
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SOLANA_RPC_URL` | Solana RPC endpoint (QuickNode, Helius) | Yes |
| `SOLANA_WS_URL` | Solana WebSocket endpoint | Yes |
| `WALLET_PRIVATE_KEY` | Base58-encoded private key | For live trading |
| `OPENROUTER_API_KEY` | OpenRouter API key for Claude Brain | For AI features |

## Trading Modes

| Mode | Risk | Description |
|------|------|-------------|
| `simulation` | None | Paper trading, no real transactions |
| `conservative` | Low | Small positions, strict stops |
| `aggressive` | Medium | Larger positions, wider stops |
| `degen` | High | Maximum risk, maximum reward |

## AI Brain

When enabled (`ai_analysis_enabled: true`), Claude Brain:

1. **Entry evaluation** - Analyzes every token launch with full context (session history, portfolio state, risk level)
2. **Exit management** - Can override mechanical triggers (stop-loss, take-profit) if it detects momentum
3. **Session memory** - Tracks recent decisions and outcomes, adjusting risk appetite based on win/loss streaks
4. **Safety floor** - Never overrides stop-loss when drawdown exceeds 1.5x the configured stop-loss percentage

## API Server

```bash
# Start API server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Endpoints
GET  /health          - Health check
POST /bot/start       - Start bot with config
POST /bot/stop        - Stop bot
GET  /bot/status      - Current status + portfolio
GET  /bot/positions   - Open positions
POST /bot/trade       - Manual buy/sell
WS   /ws/updates      - Real-time updates
```

## Backtesting

```python
from tools.backtest import BacktestEngine, BacktestConfig
from fenrir.config import BotConfig

engine = BacktestEngine()
engine.load_from_file("historical_data.json")

config = BacktestConfig(
    bot_config=BotConfig(buy_amount_sol=0.1, stop_loss_pct=25),
    starting_capital_sol=10.0,
)

results = engine.run_backtest(config)
print(engine.generate_report(results))
```

## Warning

This bot trades REAL money on-chain. Losses can be total.
Memecoins are extremely high risk. Most go to zero.
Use ONLY with funds you can afford to lose completely.
Not financial advice. Educational/experimental purposes only.
Always test in simulation mode first.
