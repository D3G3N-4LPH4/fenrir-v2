#!/usr/bin/env python3
"""Send a Telegram message via the Fenrir Scout bot.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the repo .env.
Usage:
  python tools/telegram_notify.py "message text"
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


def send_message(token: str, chat_id: str, text: str) -> dict:
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
    payload = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    ).encode()
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
    chat_id = env.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing from .env", file=sys.stderr)
        return 1
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read().strip()
    if not text:
        print("no message text", file=sys.stderr)
        return 1
    try:
        result = send_message(token, chat_id, text)
    except Exception as e:  # noqa: BLE001 - report API/transport errors plainly
        print(f"telegram send failed: {e}", file=sys.stderr)
        return 2
    if not result.get("ok"):
        print(f"telegram API error: {result}", file=sys.stderr)
        return 3
    print(f"sent message_id={result['result']['message_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
