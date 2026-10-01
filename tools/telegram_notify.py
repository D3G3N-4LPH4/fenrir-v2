#!/usr/bin/env python3
"""Send a Telegram message via the Fenrir Scout bot.

Reads TELEGRAM_BOT_TOKEN and the destination chat(s) from the repo .env:
  TELEGRAM_CHAT_IDS="-100111,-100222"  (comma-separated, preferred)
  TELEGRAM_CHAT_ID="-100111"           (legacy single-chat fallback)
Usage:
  python tools/telegram_notify.py "message text"
  python tools/telegram_notify.py --parse-mode Markdown "formatted *text*"
  echo "message text" | python tools/telegram_notify.py
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path: str) -> dict:
    env: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def send_message(token: str, chat_id: str, text: str, parse_mode: str = "") -> dict:
    """POST to sendMessage with a few quick retries on transport-level failures.

    The sandbox proxy occasionally drops connections mid-request ("Remote end closed
    connection without response"); a bounded retry with backoff rides through those
    blips. API-level errors (ok:false) are NOT retried — only connection/timeout
    failures (a successful POST returns immediately, ok:false and all).
    """
    import http.client
    import json
    import time

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    fields = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        fields["parse_mode"] = parse_mode
    payload = urllib.parse.urlencode(fields).encode()
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=payload, method="POST")  # noqa: S310 - fixed https Telegram API URL
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                result: dict = json.loads(resp.read().decode())
                return result
        except (
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as e:
            last_exc = e
            time.sleep(1.5 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("send_message: no attempt was made")  # unreachable


def main() -> int:
    env = load_env(os.path.join(REPO_ROOT, ".env"))
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    raw_ids = env.get("TELEGRAM_CHAT_IDS", "") or env.get("TELEGRAM_CHAT_ID", "")
    chat_ids = [c.strip() for c in raw_ids.split(",") if c.strip()]
    if not token or not chat_ids:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_IDS missing from .env", file=sys.stderr)
        return 1
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        parse_mode = ""
        if "--parse-mode" in args:
            i = args.index("--parse-mode")
            if i + 1 < len(args):
                parse_mode = args[i + 1]
            del args[i : i + 2]
        text = " ".join(args)
    else:
        parse_mode = ""
        text = sys.stdin.read().strip()
    if not text:
        print("no message text", file=sys.stderr)
        return 1
    failures = 0
    for chat_id in chat_ids:
        try:
            result = send_message(token, chat_id, text, parse_mode=parse_mode)
        except Exception as e:  # noqa: BLE001 - report API/transport errors plainly
            print(f"telegram send failed for {chat_id}: {e}", file=sys.stderr)
            failures += 1
            continue
        if not result.get("ok"):
            print(f"telegram API error for {chat_id}: {result}", file=sys.stderr)
            failures += 1
            continue
        print(f"sent message_id={result['result']['message_id']} to {chat_id}")
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
