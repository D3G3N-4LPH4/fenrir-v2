"""
social_signal_adapter.py — FENRIR v2 Twitter/X Social Signal Integration
=========================================================================
Uses `twitter-cli` (jackwener/twitter-cli) as a subprocess data source.
No API key required — runs off your browser cookies.

Install dep:
    pip install twitter-cli
    # Verify auth works:
    twitter feed --max 3 --json

Architecture:
    SocialSignalAdapter
        ├── search_token()         → per-token mention scan
        ├── search_ecosystem()     → pump.fun / solana memecoin feed
        ├── get_claude_context()   → formatted string for AI prompt
        └── drain_events()        → EventBus-ready dicts

Usage:
    adapter = SocialSignalAdapter()
    await adapter.scan_token("BONK", "BonkMintAddressXYZ")
    context = adapter.get_claude_context("BonkMintAddressXYZ")
    events  = adapter.drain_events()
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("fenrir.social_signal")


# ---------------------------------------------------------------------------
# Config — tune these for your risk tolerance
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # How many tweets to pull per token search
    "token_search_max": 30,
    # How many tweets to pull for ecosystem feed
    "ecosystem_search_max": 50,
    # Minimum engagement score to count as a signal
    # score = likes + retweets*2 + replies*1.5 + log10(views+1)*0.5
    "min_engagement_score": 5,
    # Seconds before cached results expire
    "cache_ttl_seconds": 120,
    # CLI timeout per command (seconds)
    "cli_timeout": 20,
    # Proxy (optional) — set to None to disable
    # "proxy": "http://127.0.0.1:7890",
    "proxy": None,
}

# Ecosystem-level search queries — catches general pump.fun chatter
ECOSYSTEM_QUERIES = [
    "pump.fun new token",
    "pump.fun launch",
    "solana memecoin",
]

# Bearish sentiment keywords — presence in a tweet raises bear score
BEARISH_KEYWORDS = [
    "rug", "rugpull", "scam", "dump", "dumping", "exit", "exit liquidity",
    "honeypot", "dev sold", "dev dumped", "bundle", "bundled", "sniper",
    "avoid", "warning", "careful", "suspicious", "fake", "bot",
]

# Bullish sentiment keywords
BULLISH_KEYWORDS = [
    "gem", "moon", "mooning", "buy", "buying", "launch", "just launched",
    "early", "airdrop", "kol", "influencer", "trending", "viral",
    "100x", "1000x", "massive", "huge",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Tweet:
    tweet_id: str
    text: str
    author: str
    author_followers: int
    likes: int
    retweets: int
    replies: int
    views: int
    created_at: str
    engagement_score: float = 0.0
    sentiment: str = "neutral"   # "bullish" | "bearish" | "neutral"
    sentiment_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict) -> "Tweet | None":
        """Parse a raw tweet dict from twitter-cli --json output."""
        try:
            text = raw.get("text", "") or raw.get("full_text", "") or ""
            author_data = raw.get("user") or raw.get("author") or {}
            metrics = raw.get("public_metrics") or {}

            likes     = int(raw.get("favorite_count") or metrics.get("like_count") or 0)
            retweets  = int(raw.get("retweet_count") or metrics.get("retweet_count") or 0)
            replies   = int(raw.get("reply_count") or metrics.get("reply_count") or 0)
            views     = int(raw.get("views") or metrics.get("impression_count") or 0)
            followers = int(author_data.get("followers_count") or 0)
            author    = author_data.get("screen_name") or author_data.get("username") or "unknown"
            tweet_id  = str(raw.get("id_str") or raw.get("id") or "")
            created   = raw.get("created_at") or raw.get("created_at_str") or ""

            import math
            score = (
                likes * 1.0
                + retweets * 2.0
                + replies * 1.5
                + math.log10(max(views, 1)) * 0.5
                + (followers / 10_000)  # KOL bonus
            )

            text_lower = text.lower()
            bull_hits = [k for k in BULLISH_KEYWORDS if k in text_lower]
            bear_hits = [k for k in BEARISH_KEYWORDS if k in text_lower]

            if bear_hits and len(bear_hits) >= len(bull_hits):
                sentiment = "bearish"
                kw = bear_hits
            elif bull_hits:
                sentiment = "bullish"
                kw = bull_hits
            else:
                sentiment = "neutral"
                kw = []

            return cls(
                tweet_id=tweet_id,
                text=text[:280],
                author=author,
                author_followers=followers,
                likes=likes,
                retweets=retweets,
                replies=replies,
                views=views,
                created_at=created,
                engagement_score=round(score, 2),
                sentiment=sentiment,
                sentiment_keywords=kw,
            )
        except Exception as e:
            logger.debug(f"Failed to parse tweet: {e} | raw={raw}")
            return None


@dataclass
class SocialSnapshot:
    """Aggregated social picture for a single token at a point in time."""
    token_mint: str
    ticker: str
    tweet_count: int
    bull_count: int
    bear_count: int
    neutral_count: int
    top_tweets: list[Tweet]           # highest engagement, up to 5
    bear_warnings: list[str]          # deduplicated warning keywords found
    bull_signals: list[str]           # deduplicated bullish keywords found
    kol_mentions: list[str]           # authors with >10k followers
    sentiment_score: float            # -1.0 (max bear) to +1.0 (max bull)
    scanned_at: str = ""

    @property
    def event_type(self) -> str:
        return "SOCIAL_SNAPSHOT"

    def to_dict(self) -> dict:
        return {
            "event": self.event_type,
            "token_mint": self.token_mint,
            "ticker": self.ticker,
            "tweet_count": self.tweet_count,
            "bull_count": self.bull_count,
            "bear_count": self.bear_count,
            "neutral_count": self.neutral_count,
            "bear_warnings": self.bear_warnings,
            "bull_signals": self.bull_signals,
            "kol_mentions": self.kol_mentions,
            "sentiment_score": self.sentiment_score,
            "scanned_at": self.scanned_at,
        }


# ---------------------------------------------------------------------------
# CLI runner (subprocess wrapper)
# ---------------------------------------------------------------------------

class TwitterCLIRunner:
    """
    Thin wrapper around the `twitter` CLI subprocess.
    All commands use --json for structured output.
    """

    def __init__(self, proxy: str | None = None, timeout: int = 20):
        self.proxy = proxy
        self.timeout = timeout
        self._env = self._build_env()

    def _build_env(self) -> dict:
        import os
        env = os.environ.copy()
        if self.proxy:
            env["TWITTER_PROXY"] = self.proxy
        return env

    def run(self, args: list[str]) -> list[dict] | None:
        """
        Run a twitter-cli command and return parsed JSON list.
        Returns None on failure (timeout, auth error, parse error).

        Example:
            runner.run(["search", "pump.fun", "--max", "20", "--json"])
        """
        cmd = ["twitter"] + args + ["--json"]
        logger.debug(f"twitter-cli: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=self._env,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"twitter-cli timeout: {' '.join(cmd)}")
            return None
        except FileNotFoundError:
            logger.error("twitter-cli not found. Run: pip install twitter-cli")
            return None

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "401" in stderr or "403" in stderr or "No Twitter cookies" in stderr:
                logger.error(f"twitter-cli auth error: {stderr}")
            else:
                logger.warning(f"twitter-cli non-zero exit: {stderr}")
            return None

        stdout = result.stdout.strip()
        if not stdout:
            return []

        try:
            data = json.loads(stdout)
            # twitter-cli may return a list or {"data": [...]}
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("data", "tweets", "results"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"twitter-cli JSON parse error: {e} | stdout={stdout[:200]}")
            return None

    async def run_async(self, args: list[str]) -> list[dict] | None:
        """Non-blocking async version — runs CLI in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, args)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class SocialSignalAdapter:
    """
    Polls twitter-cli for social signals relevant to active FENRIR positions.

    Per-token cache prevents hammering the CLI on every candle.
    Cache TTL is configurable (default 2 minutes).
    """

    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.runner = TwitterCLIRunner(
            proxy=self.config["proxy"],
            timeout=self.config["cli_timeout"],
        )

        # Cache: token_mint → (SocialSnapshot, timestamp)
        self._cache: dict[str, tuple[SocialSnapshot, float]] = {}

        # Ecosystem snapshot cache
        self._ecosystem_tweets: list[Tweet] = []
        self._ecosystem_scanned_at: float = 0.0

        # Pending events
        self._event_queue: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan_token(
        self,
        ticker: str,
        token_mint: str,
        extra_terms: list[str] | None = None,
    ) -> SocialSnapshot | None:
        """
        Scan Twitter for mentions of this token.
        Uses cache — won't re-scan within TTL window.

        Parameters
        ----------
        ticker : str
            Token ticker e.g. "BONK"
        token_mint : str
            Solana mint address (used as cache key + fallback search term)
        extra_terms : list[str]
            Additional search terms (e.g. token name, KOL handle)
        """
        # Check cache
        cached = self._get_cached(token_mint)
        if cached:
            return cached

        # Build search query
        # Mint address first (most specific), then ticker, then extras
        terms = [f'"{token_mint[:8]}"']  # partial mint is often enough
        if ticker and ticker != token_mint:
            terms.append(f'${ticker}')
            terms.append(ticker)
        if extra_terms:
            terms.extend(extra_terms)

        query = " OR ".join(terms[:4])  # keep query manageable

        raw_tweets = await self.runner.run_async([
            "search", query,
            "-t", "Latest",
            "--max", str(self.config["token_search_max"]),
        ])

        if raw_tweets is None:
            logger.warning(f"Social scan failed for {ticker}")
            return None

        tweets = [t for raw in raw_tweets if (t := Tweet.from_raw(raw)) is not None]
        snapshot = self._build_snapshot(ticker, token_mint, tweets)

        # Cache it
        self._cache[token_mint] = (snapshot, time.time())

        # Emit to EventBus if bearish signals found
        if snapshot.sentiment_score < -0.3 or snapshot.bear_warnings:
            event = snapshot.to_dict()
            event["alert"] = True
            event["alert_reason"] = f"Negative social sentiment: {snapshot.bear_warnings[:3]}"
            self._event_queue.append(event)
            logger.info(
                f"SOCIAL ALERT {ticker}: score={snapshot.sentiment_score:.2f} "
                f"warnings={snapshot.bear_warnings}"
            )

        return snapshot

    async def scan_ecosystem(self) -> list[Tweet]:
        """
        Pull general pump.fun / Solana memecoin chatter.
        Used for FENRIR's pre-entry scouting and ambient threat detection.
        Cached separately from per-token scans.
        """
        now = time.time()
        if now - self._ecosystem_scanned_at < self.config["cache_ttl_seconds"]:
            return self._ecosystem_tweets

        all_tweets: list[Tweet] = []
        for query in ECOSYSTEM_QUERIES:
            raw = await self.runner.run_async([
                "search", query,
                "-t", "Latest",
                "--max", str(self.config["ecosystem_search_max"] // len(ECOSYSTEM_QUERIES)),
            ])
            if raw:
                tweets = [t for r in raw if (t := Tweet.from_raw(r)) is not None]
                all_tweets.extend(tweets)

        # Deduplicate by tweet_id, sort by engagement
        seen: set[str] = set()
        deduped = []
        for t in all_tweets:
            if t.tweet_id not in seen:
                seen.add(t.tweet_id)
                deduped.append(t)

        deduped.sort(key=lambda t: t.engagement_score, reverse=True)
        self._ecosystem_tweets = deduped
        self._ecosystem_scanned_at = now

        logger.debug(f"Ecosystem scan: {len(deduped)} tweets")
        return deduped

    def get_claude_context(self, token_mint: str, include_ecosystem: bool = False) -> str:
        """
        Returns a compact, structured string for Claude's decision prompt.
        """
        cached = self._get_cached(token_mint)
        if not cached:
            return "[SOCIAL] No data — scan not yet run for this token."

        snap = cached
        lines = [
            f"[SOCIAL SIGNALS — ${snap.ticker} | {snap.tweet_count} tweets scanned]"
        ]

        # Sentiment summary
        score_bar = self._score_to_bar(snap.sentiment_score)
        lines.append(
            f"Sentiment: {score_bar} ({snap.sentiment_score:+.2f}) "
            f"| Bull:{snap.bull_count} Bear:{snap.bear_count} Neutral:{snap.neutral_count}"
        )

        # Warnings first — these are the most critical for FENRIR
        if snap.bear_warnings:
            lines.append(f"⚠️  RUG/SCAM SIGNALS: {', '.join(snap.bear_warnings)}")

        # KOL mentions
        if snap.kol_mentions:
            lines.append(f"KOL mentions (>10k followers): {', '.join(snap.kol_mentions[:5])}")

        # Bull signals
        if snap.bull_signals:
            lines.append(f"Bullish signals: {', '.join(snap.bull_signals[:5])}")

        # Top tweet snippets (max 3, only if engagement score is meaningful)
        top = [t for t in snap.top_tweets if t.engagement_score >= self.config["min_engagement_score"]][:3]
        if top:
            lines.append("\nTop tweets:")
            for t in top:
                flag = "🔴" if t.sentiment == "bearish" else "🟢" if t.sentiment == "bullish" else "⚪"
                lines.append(
                    f"  {flag} @{t.author} ({t.author_followers:,} followers) "
                    f"[❤{t.likes} 🔁{t.retweets}]: {t.text[:120].strip()}..."
                )

        # Ecosystem context
        if include_ecosystem and self._ecosystem_tweets:
            eco_bear = sum(1 for t in self._ecosystem_tweets if t.sentiment == "bearish")
            eco_bull = sum(1 for t in self._ecosystem_tweets if t.sentiment == "bullish")
            eco_total = len(self._ecosystem_tweets)
            lines.append(
                f"\nEcosystem mood (pump.fun): "
                f"Bull:{eco_bull} Bear:{eco_bear} of {eco_total} recent tweets"
            )

        lines.append(f"(Scanned: {snap.scanned_at})")
        return "\n".join(lines)

    def drain_events(self) -> list[dict]:
        """Returns and clears the pending event queue for EventBus."""
        events = self._event_queue.copy()
        self._event_queue.clear()
        return events

    def invalidate(self, token_mint: str) -> None:
        """Force re-scan on next call (e.g. after a large price move)."""
        self._cache.pop(token_mint, None)

    def cleanup(self, token_mint: str) -> None:
        """Release cache entry when exiting a position."""
        self._cache.pop(token_mint, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cached(self, token_mint: str) -> SocialSnapshot | None:
        entry = self._cache.get(token_mint)
        if not entry:
            return None
        snapshot, ts = entry
        if time.time() - ts > self.config["cache_ttl_seconds"]:
            return None
        return snapshot

    def _build_snapshot(
        self, ticker: str, token_mint: str, tweets: list[Tweet]
    ) -> SocialSnapshot:
        bull = [t for t in tweets if t.sentiment == "bullish"]
        bear = [t for t in tweets if t.sentiment == "bearish"]
        neutral = [t for t in tweets if t.sentiment == "neutral"]

        # Sentiment score: weighted by engagement
        total_score = sum(t.engagement_score for t in tweets) or 1
        bull_weight = sum(t.engagement_score for t in bull)
        bear_weight = sum(t.engagement_score for t in bear)
        sentiment_score = (bull_weight - bear_weight) / total_score

        # Deduplicated warning/signal keywords
        bear_warnings = list({kw for t in bear for kw in t.sentiment_keywords})
        bull_signals = list({kw for t in bull for kw in t.sentiment_keywords})

        # KOLs
        kol_threshold = 10_000
        kols = list({
            f"@{t.author}" for t in tweets
            if t.author_followers >= kol_threshold
        })

        # Top tweets by engagement
        top = sorted(tweets, key=lambda t: t.engagement_score, reverse=True)[:5]

        return SocialSnapshot(
            token_mint=token_mint,
            ticker=ticker,
            tweet_count=len(tweets),
            bull_count=len(bull),
            bear_count=len(bear),
            neutral_count=len(neutral),
            top_tweets=top,
            bear_warnings=bear_warnings,
            bull_signals=bull_signals,
            kol_mentions=kols,
            sentiment_score=round(sentiment_score, 3),
            scanned_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _score_to_bar(score: float) -> str:
        """Convert -1..+1 to a visual bar."""
        idx = int((score + 1) / 2 * 8)
        idx = max(0, min(8, idx))
        bars = ["████▓▓▒▒░", "███▓▓▒▒░░", "██▓▓▒▒░░░", "█▓▓▒▒░░░░",
                "▓▒▒░░░░░░", "▒▒░░░░░░░", "▒░░░░░░░░", "░░░░░░░░░", "░░░░░░░░░"]
        label = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
        return f"{bars[idx]} {label}"


# ---------------------------------------------------------------------------
# FENRIR strategy mixin
# ---------------------------------------------------------------------------

class FENRIRSocialMixin:
    """
    Mixin for SniperStrategy / GraduationStrategy.
    Provides async social scanning and Claude context injection.

    Usage:
        class SniperStrategy(FENRIRSocialMixin, BaseStrategy):
            async def on_token_detected(self, token):
                snap = await self.social_scan(token.ticker, token.mint)
                if snap and snap.sentiment_score < -0.5:
                    logger.warning(f"Skipping {token.ticker} — negative social")
                    return
                # proceed with entry logic
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._social = SocialSignalAdapter()

    async def social_scan(
        self,
        ticker: str,
        token_mint: str,
        extra_terms: list[str] | None = None,
    ) -> "SocialSnapshot | None":
        """Scan social + drain events to EventBus."""
        snap = await self._social.scan_token(ticker, token_mint, extra_terms)
        if snap:
            for event in self._social.drain_events():
                if hasattr(self, "event_bus"):
                    self.event_bus.publish(event["event"], event)
        return snap

    def get_social_context(self, token_mint: str, include_ecosystem: bool = False) -> str:
        return self._social.get_claude_context(token_mint, include_ecosystem)

    def invalidate_social(self, token_mint: str) -> None:
        self._social.invalidate(token_mint)

    def cleanup_social(self, token_mint: str) -> None:
        self._social.cleanup(token_mint)

    async def get_ecosystem_mood(self) -> str:
        """Returns a one-liner ecosystem summary for Claude context."""
        tweets = await self._social.scan_ecosystem()
        if not tweets:
            return "[SOCIAL ECOSYSTEM] No data."
        bear = sum(1 for t in tweets if t.sentiment == "bearish")
        bull = sum(1 for t in tweets if t.sentiment == "bullish")
        total = len(tweets)
        rug_hits = [t for t in tweets if any(k in t.text.lower() for k in ["rug", "scam", "honeypot"])]
        out = f"[ECOSYSTEM] pump.fun sentiment: Bull={bull} Bear={bear} / {total} tweets"
        if rug_hits:
            out += f" | ⚠️ {len(rug_hits)} rug/scam warnings in feed"
        return out


# ---------------------------------------------------------------------------
# Claude prompt template additions
# ---------------------------------------------------------------------------

SOCIAL_PROMPT_ADDON = """
--- SOCIAL SIGNALS (Twitter/X) ---
{social_context}

{ecosystem_mood}

Social signal guidance:
- Bear keywords (rug/scam/dump/dev sold) = strong EXIT or NO-ENTRY signal
- KOL mentions with high engagement = potential pump catalyst  
- Ecosystem-wide rug warnings = reduce position sizes globally
- Sentiment score < -0.3 = override HOLD to EXIT
"""


# ---------------------------------------------------------------------------
# Setup verification script
# ---------------------------------------------------------------------------

async def verify_setup() -> bool:
    """
    Run this once to confirm twitter-cli is installed and authenticated.
    Call from a __main__ block or a setup script.
    """
    print("=" * 50)
    print("FENRIR Social Signal — Setup Verification")
    print("=" * 50)

    runner = TwitterCLIRunner()

    print("\n[1/3] Checking twitter-cli installation...")
    result = runner.run(["--help"])
    # --help exits non-zero but we just want to confirm the binary exists
    import shutil
    if shutil.which("twitter"):
        print("  ✓ twitter-cli found")
    else:
        print("  ✗ twitter-cli NOT found. Run: pip install twitter-cli")
        return False

    print("\n[2/3] Testing authentication (fetching 1 tweet)...")
    tweets = await TwitterCLIRunner().run_async(["feed", "--max", "1"])
    if tweets is None:
        print("  ✗ Auth failed. Log in at x.com and retry.")
        print("    Supported browsers: Arc, Chrome, Edge, Firefox, Brave")
        return False
    print(f"  ✓ Auth OK — got {len(tweets)} tweet(s)")

    print("\n[3/3] Testing social scan...")
    adapter = SocialSignalAdapter()
    snap = await adapter.scan_token("BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263")
    if snap is None:
        print("  ✗ Token scan failed (auth may have succeeded but search failed)")
        return False
    print(f"  ✓ Scan OK — {snap.tweet_count} tweets, sentiment={snap.sentiment_score:+.2f}")
    print("\n" + adapter.get_claude_context("DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"))

    print("\n✓ Setup complete. SocialSignalAdapter is ready for FENRIR.")
    return True


if __name__ == "__main__":
    asyncio.run(verify_setup())
