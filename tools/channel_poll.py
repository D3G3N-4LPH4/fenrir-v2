#!/usr/bin/env python3
"""Track Telegram alert channels and run new calls through FENRIR's pipeline.

Two source types:
  - Bot API ("tg"): the Fenrir Scout bot must be admin in the channel; posts
    arrive as channel_post updates via getUpdates.
  - Web preview ("web"): public channels readable at https://t.me/s/<name>
    without any login. NOTE: channels that restrict content ("Please open
    Telegram to view this post") expose no text here and cannot be tracked
    this way.

State file (JSON): {"tg_offset": int, "web": {channel: last_post_id}}.
First run drains the Bot API backlog silently; web channels process the ~20
visible posts once, then only newer ones.

Stdout: {"ts": ..., "scanned": N, "candidates": [...]} — same shape as
tools/scout.py, each candidate carrying "source": "tg:<title>" or
"web:<channel>".

Usage:
  python tools/channel_poll.py --state <path> [--web-channels rhutilmate ...] [--min-score 60]
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.evaluate import enrich_safety  # noqa: E402
from tools.scout import hard_fail, safety_unknown  # noqa: E402

from fenrir.discovery.playbooks import PlaybookTagger  # noqa: E402

from fenrir.discovery.filters import FilterEngine, FilterName  # noqa: E402
from fenrir.discovery.models import Chain  # noqa: E402
from fenrir.discovery.providers.dexscreener import DexScreenerProvider  # noqa: E402
from fenrir.discovery.providers.goplus import GoPlusProvider  # noqa: E402
from fenrir.discovery.providers.perceptor import (  # noqa: E402
    ROBINHOOD_CHAIN_ID,
    PerceptorProvider,
    enrich_robinhood_safety,
)
from fenrir.discovery.scoring import ScoringEngine  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVM_RE = re.compile(r"0x[a-fA-F0-9]{40}")
SOL_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")
UA = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def load_env(path: str) -> dict:
    env: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def extract_addresses(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pat in (EVM_RE, SOL_RE):
        for m in pat.findall(text):
            if m not in seen:
                seen.add(m)
                found.append(m)
    return found


def clean_html(raw: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


async def fetch_web_posts(session: aiohttp.ClientSession, channel: str) -> list[tuple[int, str]]:
    """Return [(post_id, text)] newest-last from the channel's public preview."""
    url = f"https://t.me/s/{channel}"
    async with session.get(
        url, headers={"User-Agent": UA}, timeout=aiohttp.ClientTimeout(total=30)
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"t.me/s/{channel} -> HTTP {resp.status}")
        page = await resp.text()
    # Split the page into per-post chunks on data-post="channel/<id>"
    marker = re.compile(r'data-post="' + re.escape(channel) + r"/(\d+)\"")
    parts = marker.split(page)
    posts: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        pid = int(parts[i])
        chunk = parts[i + 1]
        # stop at the next post's marker (already split) — chunk is post-scoped
        texts = re.findall(r"tgme_widget_message_text[^>]*>(.*?)</div>", chunk, re.S)
        texts += re.findall(r"tgme_widget_message_caption[^>]*>(.*?)</div>", chunk, re.S)
        text = "\n".join(clean_html(t) for t in texts).strip()
        if "open Telegram to view this post" in text:
            text = ""  # restricted preview: no usable content
        posts.append((pid, text))
    posts.sort(key=lambda p: p[0])
    return posts


async def poll_tg_api(session: aiohttp.ClientSession, token: str, offset: int):
    # Request every update type we might see and advance the offset past ALL
    # of them. (If we filtered to channel_post only, the offset would creep
    # past join/message updates and orphan them — the chat ID would be lost.)
    params = {
        "offset": offset,
        "timeout": 0,
        "allowed_updates": json.dumps(["message", "channel_post", "my_chat_member"]),
    }
    async with session.post(
        f"https://api.telegram.org/bot{token}/getUpdates",
        data=params,
        timeout=aiohttp.ClientTimeout(total=35),
    ) as resp:
        data = await resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"getUpdates API error: {data}")
    return data.get("result", [])


async def evaluate(addr: str, ds, gp, engine, scorer, tagger, min_score,
                   perceptor: PerceptorProvider | None = None):
    """Run one address through the pipeline; return candidate dict or None."""
    try:
        snap = await ds.fetch_snapshot(addr)  # chain=None: DexScreener resolves
    except Exception:
        return None
    if snap is None:
        return None
    try:
        await enrich_safety(snap, gp)
    except Exception:
        pass
    if hard_fail(snap):
        return None
    results = {fn.value: engine.evaluate(snap, fn) for fn in FilterName}
    passed = [k for k, r in results.items() if r.passed]
    score = scorer.score(snap)
    # Robinhood safety net: Perceptor forensics when GoPlus has nothing.
    # A landed verdict re-runs the score (safety_unknown cap may lift).
    perceptor_info: dict | None = None
    if perceptor is not None and snap.chain is Chain.ROBINHOOD and safety_unknown(snap):
        try:
            report = await enrich_robinhood_safety(snap, perceptor)
        except Exception:  # noqa: BLE001 - fail-open
            report = None
        if report is not None:
            if hard_fail(snap):
                return None
            score = scorer.score(snap)
            perceptor_info = {
                "status": "complete",
                "band": report.band,
                "band_label": report.band_label,
                "headline": report.headline,
                "investigation_id": report.investigation_id,
            }
        else:
            inv_id = await perceptor.ensure_investigation(ROBINHOOD_CHAIN_ID, snap.token_address)
            if inv_id:
                perceptor_info = {"status": "pending", "investigation_id": inv_id}
    if not passed or score.overall < min_score:
        return None
    return {
        "address": snap.token_address,
        "chain": snap.chain.value,
        "symbol": snap.symbol,
        "name": snap.name,
        "price_usd": snap.price_usd,
        "market_cap_usd": round(snap.market_cap_usd, 2),
        "liquidity_usd": round(snap.liquidity_usd, 2),
        "volume_24h_usd": round(snap.volume_24h_usd, 2),
        "age_minutes": round(snap.age_minutes or 0),
        "buys_1h": snap.txns_1h_buys,
        "sells_1h": snap.txns_1h_sells,
        "buys_24h": snap.txns_24h_buys,
        "sells_24h": snap.txns_24h_sells,
        "holder_count": snap.holder_count,
        "passed_filters": passed,
        "playbooks": tagger.tag(snap).as_dict(),
        "score": score.as_dict(),
        "safety_unknown": safety_unknown(snap),
        "perceptor": perceptor_info,
        "dexscreener": f"https://dexscreener.com/{snap.chain.value}/{snap.token_address}",
    }


async def amain() -> int:
    ap = argparse.ArgumentParser(description="FENRIR telegram channel poll")
    ap.add_argument("--state", required=True)
    ap.add_argument("--web-channels", nargs="*", default=[])
    ap.add_argument("--min-score", type=float, default=60.0)
    args = ap.parse_args()

    env = load_env(os.path.join(REPO_ROOT, ".env"))
    token = env.get("TELEGRAM_BOT_TOKEN", "")

    state: dict = {"tg_offset": 0, "web": {}}
    first_run = not os.path.exists(args.state)
    if not first_run:
        try:
            with open(args.state) as f:
                loaded = json.load(f)
            # tolerate the old {"offset": n} format
            if "tg_offset" in loaded:
                state = loaded
            else:
                state = {"tg_offset": int(loaded.get("offset", 0)), "web": {}}
        except Exception:
            pass
    if "web" not in state:
        state["web"] = {}

    errors: list[str] = []
    found: list[tuple[str, str]] = []  # (source, address)

    async with aiohttp.ClientSession(
        trust_env=True, timeout=aiohttp.ClientTimeout(total=40)
    ) as session:
        # --- Bot API source ---
        if token:
            try:
                updates = await poll_tg_api(session, token, state["tg_offset"])
                max_id = state["tg_offset"]
                if not first_run:
                    for u in updates:
                        max_id = max(max_id, int(u.get("update_id", 0)))
                        cp = u.get("channel_post") or {}
                        text = cp.get("text") or cp.get("caption") or ""
                        title = (cp.get("chat") or {}).get("title", "unknown")
                        for a in extract_addresses(text):
                            found.append((f"tg:{title}", a))
                else:
                    for u in updates:
                        max_id = max(max_id, int(u.get("update_id", 0)))
                state["tg_offset"] = max_id + 1
            except Exception as e:  # noqa: BLE001 - keep the other source alive
                errors.append(f"tg api: {e}")
        else:
            errors.append("tg api: TELEGRAM_BOT_TOKEN missing")

        # --- Web preview sources ---
        for ch in args.web_channels:
            try:
                posts = await fetch_web_posts(session, ch)
            except Exception as e:  # noqa: BLE001
                errors.append(f"web:{ch}: {e}")
                continue
            last = state["web"].get(ch, 0)
            new_posts = [(pid, t) for pid, t in posts if pid > last]
            for pid, text in new_posts:
                for a in extract_addresses(text):
                    found.append((f"web:{ch}", a))
            if posts:
                state["web"][ch] = max(pid for pid, _ in posts)

    with open(args.state, "w") as f:
        json.dump(state, f)

    # de-dupe addresses, keep first source
    uniq: dict[str, str] = {}
    for src, a in found:
        uniq.setdefault(a, src)

    ds = DexScreenerProvider(timeout_seconds=15)
    gp = GoPlusProvider(timeout_seconds=10)
    perceptor = PerceptorProvider()
    engine = FilterEngine()
    scorer = ScoringEngine()
    tagger = PlaybookTagger()
    candidates: list[dict] = []
    scanned = 0
    try:
        for addr, src in uniq.items():
            cand = await evaluate(addr, ds, gp, engine, scorer, tagger, args.min_score,
                                  perceptor)
            scanned += 1
            if cand:
                cand["source"] = src
                candidates.append(cand)
            await asyncio.sleep(0.4)
    finally:
        await ds.close()
        await gp.close()
        await perceptor.close()

    candidates.sort(key=lambda c: -c["score"]["overall"])
    out = {"ts": time.time(), "scanned": scanned, "candidates": candidates}
    if errors:
        out["errors"] = errors
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
