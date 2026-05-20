"""
VPS-side heartbeat pinger for healthchecks.io.

Runs every minute on VPS (via Task Scheduler). Checks:
  1. MT4 heartbeat.json is fresh (mtime < 90 sec)
  2. gold_runner.py is still running

If all healthy: POST to PING_URL (success ping)
If any issue: POST to PING_URL/fail (immediate failure ping)
On any unexpected exception: no ping (healthchecks will alert on missed ping)

This complements the local-PC watchdog: even if local PC is off,
healthchecks.io will trigger Telegram alert on missed pings.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Config ─────────────────────────────────────────────
PING_URL = 'https://hc-ping.com/9fccce5a-bab2-41ad-b53a-890c7f9234b6'

HB_FILE = Path(
    r'C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal'
    r'\35EEC3EFDB656AF6FC775F21FEAD053B\MQL4\Files\DWX\heartbeat.json'
)
HB_FRESH_SEC = 90  # MT4 EA writes every ~5s, allow 90s buffer

LOG_FILE = Path(__file__).parent / 'vps_heartbeat_ping.log'


def log(msg: str) -> None:
    """Append timestamped line to local log + stdout."""
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {msg}'
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def is_mt4_alive() -> tuple[bool, str]:
    """Check MT4 heartbeat.json mtime (file modification time on disk)."""
    if not HB_FILE.exists():
        return False, 'heartbeat.json missing'
    try:
        mtime = HB_FILE.stat().st_mtime
        age = time.time() - mtime
        if age > HB_FRESH_SEC:
            return False, f'heartbeat file stale ({int(age)}s)'
        return True, f'mt4 ok (mtime age {int(age)}s)'
    except Exception as e:
        return False, f'heartbeat stat error: {e}'


def is_runner_alive() -> tuple[bool, str]:
    """Check python.exe gold_runner.py is running via PowerShell."""
    try:
        result = subprocess.run(
            ['powershell', '-Command',
             "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
             "| ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=15,
        )
        if 'gold_runner.py' in result.stdout:
            return True, 'gold_runner ok'
        return False, 'gold_runner.py not running'
    except Exception as e:
        return False, f'process check error: {e}'


def send_ping(success: bool, detail: str) -> None:
    """POST to healthchecks.io; /fail suffix for failures."""
    url = PING_URL if success else (PING_URL + '/fail')
    body = detail.encode('utf-8')
    req = urllib.request.Request(url, data=body, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            log(f"  ping {'OK' if success else 'FAIL'} -> HTTP {r.status}")
    except Exception as e:
        log(f'  ping HTTP error: {e}')


def main() -> int:
    mt4_ok, mt4_msg = is_mt4_alive()
    runner_ok, runner_msg = is_runner_alive()
    healthy = mt4_ok and runner_ok

    detail = f'mt4={mt4_ok} ({mt4_msg}); runner={runner_ok} ({runner_msg})'
    log(detail)

    send_ping(healthy, detail)
    return 0 if healthy else 1


if __name__ == '__main__':
    sys.exit(main())
