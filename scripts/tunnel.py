#!/usr/bin/env python3
"""
FENRIR — HTTPS tunnel helper for a remote (e.g. Replit-hosted) dashboard.

A browser page served over HTTPS cannot call an http://localhost bot API
(mixed-content), so expose the local FENRIR API over HTTPS/WSS with a tunnel.
This script:

  1. Starts a tunnel (cloudflared or ngrok) to http://localhost:<port>.
  2. Grabs the public https://… URL.
  3. Registers that URL in .env's FENRIR_CORS_ORIGINS (so the API allows the
     browser origin — remember the *dashboard's* origin must also be listed;
     see notes printed at the end).
  4. Prints the API-base / WSS / API-key / restart steps, then streams tunnel
     logs until Ctrl+C.

Usage:
    python scripts/tunnel.py                 # auto-detect provider, port 8000
    python scripts/tunnel.py --provider ngrok --port 8001

Order of operations: run this FIRST (it writes .env), THEN (re)start the API
(`python -m api.server`) so it picks up the new allowed origin.

Install a provider if needed:
    cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
    ngrok:       https://ngrok.com/download   (run `ngrok config add-authtoken <token>` once)
"""

# Intentional for this local dev helper: we spawn a known tunnel binary
# (S603/S607), poll ngrok's fixed localhost API (S310), and retry that poll
# quietly until it's ready (S110).
# ruff: noqa: S603, S607, S310, S110

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
CF_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
NGROK_LOCAL_API = "http://127.0.0.1:4040/api/tunnels"


def _fail(msg: str) -> None:
    print(f"\n[tunnel] {msg}", file=sys.stderr)
    sys.exit(1)


def detect_provider(pref: str) -> str:
    """Resolve which tunnel binary to use, or exit with install guidance."""
    if pref != "auto":
        if not shutil.which(pref):
            _fail(f"'{pref}' not found on PATH. See install links in this script's header.")
        return pref
    for name in ("cloudflared", "ngrok"):
        if shutil.which(name):
            return name
    _fail(
        "No tunnel provider found. Install cloudflared (no signup) or ngrok:\n"
        "  cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/"
        "connect-networks/downloads/\n"
        "  ngrok:       https://ngrok.com/download"
    )
    return ""  # unreachable


def start_cloudflared(port: int) -> tuple[subprocess.Popen, str]:
    """Launch a cloudflared quick tunnel and scrape the trycloudflare.com URL."""
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    url_holder: dict[str, str] = {}

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if "url" not in url_holder:
                m = CF_URL_RE.search(line)
                if m:
                    url_holder["url"] = m.group(0)
            print(f"[cloudflared] {line.rstrip()}")

    threading.Thread(target=_pump, daemon=True).start()

    for _ in range(60):  # up to ~30s
        if "url" in url_holder:
            return proc, url_holder["url"]
        if proc.poll() is not None:
            _fail("cloudflared exited before a URL appeared. Check its output above.")
        time.sleep(0.5)
    proc.terminate()
    _fail("Timed out waiting for the cloudflared URL.")
    raise SystemExit  # for type-checkers


def start_ngrok(port: int) -> tuple[subprocess.Popen, str]:
    """Launch ngrok and read the public URL from its local API (:4040)."""
    proc = subprocess.Popen(
        ["ngrok", "http", str(port), "--log", "stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[ngrok] {line.rstrip()}")

    threading.Thread(target=_pump, daemon=True).start()

    for _ in range(60):
        if proc.poll() is not None:
            _fail("ngrok exited early. Did you run `ngrok config add-authtoken <token>`?")
        try:
            with urllib.request.urlopen(NGROK_LOCAL_API, timeout=1) as resp:
                data = json.load(resp)
            for t in data.get("tunnels", []):
                pub = t.get("public_url", "")
                if pub.startswith("https://"):
                    return proc, pub
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    _fail("Timed out waiting for the ngrok URL.")
    raise SystemExit


def register_cors_origin(url: str) -> None:
    """Add `url` to FENRIR_CORS_ORIGINS in .env (creating the line/file if needed).

    Only touches that one key; other lines (secrets) are preserved untouched and
    never printed.
    """
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    key = "FENRIR_CORS_ORIGINS="
    out: list[str] = []
    found = False
    for line in lines:
        if line.startswith(key):
            found = True
            existing = [o.strip() for o in line[len(key) :].split(",") if o.strip()]
            if url not in existing:
                existing.append(url)
            out.append(key + ",".join(existing))
        else:
            out.append(line)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append("# Remote dashboard origin(s) allowed by CORS (added by scripts/tunnel.py)")
        out.append(key + url)
    ENV_FILE.write_text("\n".join(out) + "\n")


def print_notes(url: str) -> None:
    host = url.split("://", 1)[1]
    wss = f"wss://{host}/ws/updates"
    print("\n" + "=" * 68)
    print("  FENRIR API is now reachable over HTTPS:")
    print(f"    {url}")
    print("-" * 68)
    print("  Next steps:")
    print("    1. (Re)start the API so it reads the updated .env:")
    print("         python -m api.server")
    print("    2. Point your remote dashboard at this API base:")
    print(f"         REST : {url}")
    print(f"         WS   : {wss}")
    print("    3. Auth: send your FENRIR_API_KEY as the 'X-API-Key' header")
    print("       (or run the API with FENRIR_DEV_MODE=true while testing).")
    print("    4. Make sure the DASHBOARD's own origin is in FENRIR_CORS_ORIGINS")
    print("       too (this script added the *tunnel* URL; your Replit page URL")
    print("       must also be listed).")
    print("=" * 68)
    print("  Leave this running. Ctrl+C to close the tunnel.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Expose the local FENRIR API over HTTPS.")
    ap.add_argument("--provider", choices=["auto", "cloudflared", "ngrok"], default="auto")
    ap.add_argument("--port", type=int, default=8000, help="Local API port (default 8000)")
    args = ap.parse_args()

    provider = detect_provider(args.provider)
    print(f"[tunnel] using {provider} -> http://localhost:{args.port}")

    if provider == "cloudflared":
        proc, url = start_cloudflared(args.port)
    else:
        proc, url = start_ngrok(args.port)

    register_cors_origin(url)
    print(f"[tunnel] added {url} to FENRIR_CORS_ORIGINS in .env")
    print_notes(url)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[tunnel] shutting down…")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
