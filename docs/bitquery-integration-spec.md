# Bitquery as a FENRIR data source — integration spec

**Status:** proposed (design only — not yet implemented)
**Author:** engineering
**Scope:** add Bitquery on-chain analytics to feed (1) the security gate's holder-distribution
check and (2) the smart-money tracker's early-buyer analytics.

---

## 1. Why Bitquery

FENRIR already covers price/liquidity/momentum (DexScreener, Jupiter, Birdeye), rug scoring
(RugCheck, on-chain security filter) and follow-the-wallet discovery (smart-money tracker via
RPC balance-delta polling). The one capability none of those give cheaply is **rich on-chain
analytics**: full holder distributions and *historical early-buyer* lists for a token.

Bitquery is a GraphQL API over indexed Solana data (DEXTrades, Transfers, BalanceUpdates,
TokenSupplyUpdates) that answers two questions FENRIR can't today:

1. **"How is this token's supply distributed across holders?"** — full top-N holder
   concentration, not just the RPC `getTokenLargestAccounts` top-20.
2. **"Who bought this token earliest, and are any of them wallets I track / known snipers?"**
   — historical first-N buyers with timestamps, in one query instead of walking every wallet.

Both are **decision-support** signals (they enrich the AI + gate), **not** hot-path execution
data. That framing drives every design choice below.

## 2. Non-goals / guardrails

- **Not on the latency-critical buy path.** Bitquery is a paid, rate-limited external GraphQL
  API with real-world latency (100s of ms–seconds). It must never sit between token detection
  and the entry decision for a fresh launch. It enriches; it never blocks a snipe.
- **Fail-open, flag-gated.** Off by default. Any Bitquery error/timeout degrades gracefully to
  the existing behaviour (RPC holder check; RPC smart-money polling). A Bitquery outage must
  never wedge trading — same contract as RugCheck (#36) and the model fallback (#41).
- **Bounded cost.** Cache aggressively (per-mint, short TTL), cap queries/cycle, and only query
  when the cheaper existing signal is inconclusive.

## 3. Bitquery API essentials

> ⚠️ **Validate exact schema in the Bitquery IDE before coding.** The field names below are
> representative of Bitquery's Solana V2 schema and are known to drift; treat them as intent,
> confirm against the live playground + docs at implementation time.

- **Endpoint:** `https://streaming.bitquery.io/eap` (V2 EAP, Solana) — GraphQL POST.
- **Auth:** OAuth2 **Bearer token** (generate from a Bitquery application; the legacy
  `X-API-KEY` header still works on some plans). Store as `BITQUERY_TOKEN` in `.env` — never
  commit it.
- **Cost model:** points-per-query; paid tiers. Historical/aggregated queries cost more than
  realtime. Assume a modest monthly budget → cache + cap are mandatory, not optional.

### 3.1 Holder distribution query (intent)

Aggregate current balances by owner for a mint, take the top N, sum their share of supply:

```graphql
query TopHolders($mint: String!, $topN: Int!) {
  Solana {
    BalanceUpdates(
      where: { BalanceUpdate: { Currency: { MintAddress: { is: $mint } } } }
      orderBy: { descendingByField: "balance" }
      limit: { count: $topN }
    ) {
      BalanceUpdate {
        Account { Address }
        balance: PostBalance(maximum: Block_Slot)
      }
    }
  }
}
```

Derive: `top10_pct = sum(top-10 balances) / total_supply * 100`. (Total supply from the existing
RPC `getTokenSupply`, or a `TokenSupplyUpdates` query.)

### 3.2 Early-buyer query (intent)

First N DEX buyers of a token, oldest first, with the buyer wallet + time:

```graphql
query EarlyBuyers($mint: String!, $n: Int!) {
  Solana {
    DEXTrades(
      where: { Trade: { Buy: { Currency: { MintAddress: { is: $mint } } } } }
      orderBy: { ascending: Block_Time }
      limit: { count: $n }
    ) {
      Block { Time }
      Trade { Buy { Account { Address } Amount } }
    }
  }
}
```

Derive: the ordered list of earliest buyers → intersect with the tracked/known-sniper wallet set.

## 4. FENRIR integration design

### 4.1 New module: `fenrir/data/bitquery.py`

```python
class BitqueryClient:
    def __init__(self, token: str, session, logger, timeout_s: float = 6.0): ...
    async def top_holder_pct(self, mint: str, top_n: int = 10) -> float | None: ...
    async def early_buyers(self, mint: str, n: int = 20) -> list[EarlyBuyer] | None: ...
    # lazy aiohttp session; per-mint TTL cache; returns None on any error (fail-open)
```

- Mirrors the shape of existing helpers (`JupiterSwapEngine`, `SecurityFilter`): lazy session,
  defensive parsing, `None` on failure.
- Small `@dataclass EarlyBuyer(address: str, amount: float, ts: datetime)`.
- Per-mint cache with a short TTL (holders ~60s, early-buyers effectively immutable → long TTL).

### 4.2 Config (opt-in, defaults off)

```
bitquery_enabled: bool = False
bitquery_token: str = ""            # env BITQUERY_TOKEN
bitquery_timeout_seconds: float = 6.0
bitquery_holder_topn: int = 10
bitquery_early_buyers_n: int = 20
```

### 4.3 Wiring #1 — holder distribution → security gate

In `fenrir/filters/security.py`, the top-10 holder check currently uses RPC
`getTokenLargestAccounts` (top-20, no owner aggregation → misses wallets holding across multiple
token accounts). Add Bitquery as a **more accurate primary** with RPC as fallback:

- If `bitquery_enabled`, call `top_holder_pct(mint)`; use it for the `max_top10_holder_pct` gate.
- On `None` (error/timeout/disabled) → fall back to the existing RPC path unchanged.
- Record the value in `result.details["top10_holder_pct"]` (already surfaced to the AI).

This strengthens the existing gate without changing its contract.

### 4.4 Wiring #2 — early-buyer analytics → smart-money tracker

When a token becomes a candidate (scanner or smart-money buy), optionally enrich it:

- `early_buyers(mint)` → intersect with `smart_money_wallets ∪ smart_money_priority_wallets`
  (and, later, a curated "known sniper" list).
- Attach to `token_data`: `early_smart_buyers: int`, `earliest_tracked_rank: int | None`
  (e.g. "a tracked wallet was the 3rd buyer").
- Surface in `_build_market_signal_context`: *"2 wallets you track were among the first 20
  buyers (earliest rank #3)"* — a strong conviction multiplier for the AI.
- Guardrail: only run this for **non-time-critical** candidates (scanner/smart-money paths,
  which already tolerate seconds of latency), never on the fresh-launch snipe path.

### 4.5 Optional wiring #3 — periodic top-holder drift (later)

For open positions, a slow background check (minutes) for a sudden jump in top-holder
concentration = a distribution/dump risk → feed the AI exit evaluator (same mechanism as the
smart-money sell-signal, #46). Deferred to a follow-up.

## 5. Testing

- Unit: mock the GraphQL POST; assert `top_holder_pct` / `early_buyers` parse representative
  responses, return `None` on HTTP error / malformed body / timeout, and honour the cache.
- Integration (gated, manual): a `scripts/bitquery_probe.py` that runs both queries against a
  known token with a real token — verify field names + auth before merging (like the RugCheck /
  DexScreener live probes used during their integrations).
- No live Bitquery call in CI (needs a paid key) — same pattern as the other keyed sources.

## 6. Rollout plan (phased, one PR each)

1. **PR A — `BitqueryClient` + config + live probe.** Client, config knobs, mocked unit tests,
   and `scripts/bitquery_probe.py` to confirm the schema/auth against a real key. No wiring yet.
2. **PR B — holder distribution → security gate.** Primary-with-RPC-fallback in
   `SecurityFilter`; tests for the fallback contract.
3. **PR C — early-buyer analytics → smart-money/scanner AI context.** Enrich candidates +
   surface to the AI; tests.
4. **PR D (optional) — top-holder drift → AI exit.** Background position check.

## 7. Open questions

- **Budget:** what monthly Bitquery point budget is acceptable? Determines cache TTLs + per-cycle
  caps. (Holder queries are the expensive ones.)
- **Known-sniper list:** do we maintain a curated "known early sniper" wallet set beyond the
  user's tracked wallets, for the early-buyer intersection? (Higher signal, more maintenance.)
- **Birdeye overlap:** Birdeye (already wired) also exposes holder data on paid tiers — if the
  Birdeye plan already covers holder distribution, wiring #1 could use Birdeye instead of adding
  Bitquery. Confirm before implementing PR B.
