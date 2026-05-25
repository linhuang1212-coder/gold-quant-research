"""
VPS External Watchdog — runs on local PC, monitors VPS over SSH, alerts via Telegram.

Checks every run:
  1. SSH connectivity to VPS (with 60s banner_timeout + 1 in-run retry)
  2. python.exe (gold_runner) running on VPS
  3. MT4 heartbeat.json fresh (<2min mtime)

Alert policy:
  - Single failure does NOT trigger alert (the in-run retry handles transient
    sshd banner-read delays caused by VPS memory pressure / Defender scans).
  - First alert fires only on the SECOND consecutive failure across runs.
  - Repeats every 30 min while still down.
  - Sends recovery message when all checks pass again.

Credentials are loaded from .env at project root (see .env.example).
NEVER hardcode passwords or Telegram tokens in this file.

Setup:
  Run via Windows Task Scheduler every 5 minutes (see vps_watchdog_install.md).
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import paramiko
import requests

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Project paths ──────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJ_ROOT / '.env'
STATE_FILE = Path(__file__).parent / '.vps_watchdog_state.json'
LOG_FILE = Path(__file__).parent / 'vps_watchdog.log'


# ── Credentials from .env (never hardcode) ─────────────
def load_env() -> dict:
    """Read .env into a dict, fail loudly if required keys missing."""
    if not ENV_FILE.exists():
        raise SystemExit(
            f'Missing {ENV_FILE}.\n'
            f'Copy `.env.example` to `.env` at the project root and fill in '
            f'MT4_VPS_* + TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID.'
        )
    env = {}
    for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip()
    required = [
        'MT4_VPS_HOST', 'MT4_VPS_USER', 'MT4_VPS_PASSWORD',
        'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID',
    ]
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise SystemExit(f'Missing keys in {ENV_FILE}: {missing}')
    return env


ENV = load_env()
VPS_IP = ENV['MT4_VPS_HOST']
VPS_PORT = int(ENV.get('MT4_VPS_PORT', '22'))
VPS_USER = ENV['MT4_VPS_USER']
VPS_PASS = ENV['MT4_VPS_PASSWORD']
TELEGRAM_TOKEN = ENV['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT = ENV['TELEGRAM_CHAT_ID']

# Path on VPS to MT4 heartbeat file (forward slashes work with SFTP)
HB_PATH = (
    '/C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/'
    '35EEC3EFDB656AF6FC775F21FEAD053B/MQL4/Files/DWX/heartbeat.json'
)
ACCT_PATH = (
    '/C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/'
    '35EEC3EFDB656AF6FC775F21FEAD053B/MQL4/Files/DWX/account.json'
)

# ── Thresholds ────────────────────────────────────────
MT4_HB_FRESH_SEC = 120
SSH_CONNECT_TIMEOUT = 30   # TCP connect timeout
SSH_BANNER_TIMEOUT = 60    # was 20 — VPS memory pressure (2GB) can stall sshd
SSH_AUTH_TIMEOUT = 30
SSH_RETRIES = 1            # extra retries within a single watchdog run (was 0)
SSH_RETRY_SLEEP = 15       # seconds between retries
ALERT_REPEAT_MIN = 30
CONSEC_FAILS_TO_ALERT = 2  # was 1 — need two back-to-back failures to alert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()],
)
log = logging.getLogger('vps_watchdog')


def send_telegram(msg: str) -> bool:
    """Send markdown HTML message via Telegram bot."""
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    try:
        r = requests.post(
            url,
            json={'chat_id': TELEGRAM_CHAT, 'text': msg, 'parse_mode': 'HTML', 'disable_web_page_preview': True},
            timeout=10,
        )
        if r.status_code == 200:
            log.info('Telegram sent OK')
            return True
        log.warning(f'Telegram failed {r.status_code}: {r.text[:200]}')
    except Exception as e:
        log.warning(f'Telegram error: {e}')
    return False


def _ssh_connect_once() -> paramiko.SSHClient:
    """Single connect attempt with the longer timeouts. Raises on failure."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        VPS_IP, port=VPS_PORT, username=VPS_USER, password=VPS_PASS,
        timeout=SSH_CONNECT_TIMEOUT,
        banner_timeout=SSH_BANNER_TIMEOUT,
        auth_timeout=SSH_AUTH_TIMEOUT,
        allow_agent=False, look_for_keys=False,
    )
    return ssh


def ssh_connect_with_retry() -> tuple[paramiko.SSHClient | None, str]:
    """Connect to VPS with up to (1 + SSH_RETRIES) attempts.

    Returns (ssh_client, error_msg). On success error_msg is empty.
    """
    last_err = ''
    for attempt in range(SSH_RETRIES + 1):
        try:
            return _ssh_connect_once(), ''
        except (paramiko.SSHException, socket.error, EOFError) as e:
            last_err = f'{type(e).__name__}: {e}'
            if attempt < SSH_RETRIES:
                log.info(f'  SSH attempt {attempt+1} failed ({last_err}); '
                         f'retrying in {SSH_RETRY_SLEEP}s')
                time.sleep(SSH_RETRY_SLEEP)
        except Exception as e:
            last_err = f'{type(e).__name__}: {e}'
            break
    return None, f'SSH connect failed: {last_err}'


def check_vps() -> dict:
    """Run all health checks. Returns dict with each check result."""
    result = {
        'ssh_ok': False,
        'runner_alive': False,
        'mt4_hb_fresh': False,
        'hb_age_sec': None,
        'account_balance': None,
        'errors': [],
    }
    ssh, err = ssh_connect_with_retry()
    if ssh is None:
        result['errors'].append(err)
        return result

    try:
        result['ssh_ok'] = True

        # Check python.exe (gold_runner)
        try:
            stdin, stdout, stderr = ssh.exec_command(
                'tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH', timeout=15,
            )
            out = stdout.read().decode('utf-8', errors='replace')
            result['runner_alive'] = 'python.exe' in out.lower()
            if not result['runner_alive']:
                result['errors'].append('gold_runner.py process not found')
        except Exception as e:
            result['errors'].append(f'process check failed: {e}')

        # Check MT4 heartbeat file freshness via SFTP
        try:
            sftp = ssh.open_sftp()
            try:
                stat = sftp.stat(HB_PATH)
                age = time.time() - stat.st_mtime
                result['hb_age_sec'] = int(age)
                result['mt4_hb_fresh'] = age < MT4_HB_FRESH_SEC
                if not result['mt4_hb_fresh']:
                    result['errors'].append(f'MT4 heartbeat stale ({int(age)}s ago)')
            except IOError as e:
                result['errors'].append(f'heartbeat.json missing: {e}')

            # Bonus: read account balance for status report
            try:
                with sftp.open(ACCT_PATH, 'r') as f:
                    acct = json.loads(f.read().decode('utf-8', errors='replace'))
                result['account_balance'] = acct.get('equity', acct.get('balance'))
                result['bid'] = acct.get('bid')
                result['ask'] = acct.get('ask')
            except Exception:
                pass
            sftp.close()
        except Exception as e:
            result['errors'].append(f'sftp failed: {e}')
    finally:
        try:
            ssh.close()
        except Exception:
            pass

    return result


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except Exception:
            state = {}
    else:
        state = {}
    state.setdefault('status', 'unknown')
    state.setdefault('down_since', None)
    state.setdefault('last_alert', None)
    state.setdefault('consec_fails', 0)
    state.setdefault('last_errors', [])
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')


def fmt_now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main() -> int:
    log.info('=== VPS watchdog check ===')
    r = check_vps()
    healthy = r['ssh_ok'] and r['runner_alive'] and r['mt4_hb_fresh']
    log.info(
        f"ssh={r['ssh_ok']} runner={r['runner_alive']} mt4_hb={r['mt4_hb_fresh']} "
        f"hb_age={r['hb_age_sec']}s balance={r.get('account_balance')}"
    )
    if r['errors']:
        for e in r['errors']:
            log.warning(f'  ! {e}')

    state = load_state()
    now = fmt_now()

    if healthy:
        # Recovery path — only notify if we were previously in 'down' state.
        # Pending (consec_fails > 0 but never alerted) just resets quietly.
        if state['status'] == 'down':
            outage_start = state.get('down_since') or 'unknown'
            try:
                dur = datetime.now() - datetime.strptime(outage_start, '%Y-%m-%d %H:%M:%S')
                dur_str = f'{int(dur.total_seconds()//60)} min'
            except Exception:
                dur_str = 'unknown'
            msg = (
                f'<b>[VPS RECOVERED]</b>\n\n'
                f'Time: {now}\n'
                f'Outage duration: {dur_str}\n'
                f'Outage started: {outage_start}\n\n'
                f'Balance: ${r.get("account_balance", "?")}\n'
                f'XAUUSD: bid={r.get("bid", "?")} ask={r.get("ask", "?")}'
            )
            send_telegram(msg)
            log.info(f'RECOVERY notification sent. Outage was {dur_str}.')
        elif state.get('consec_fails', 0) > 0:
            log.info(f'recovered after {state["consec_fails"]} pending fail(s) — no alert sent')
        state = {
            'status': 'ok', 'down_since': None, 'last_alert': None,
            'consec_fails': 0, 'last_errors': [],
        }

    else:
        # Unhealthy this run. Increment counter; alert only if we crossed the threshold.
        state['consec_fails'] = int(state.get('consec_fails', 0)) + 1
        state['last_errors'] = r['errors']

        if state['consec_fails'] < CONSEC_FAILS_TO_ALERT:
            log.info(
                f'fail #{state["consec_fails"]}/{CONSEC_FAILS_TO_ALERT} — not alerting yet '
                f'(needs {CONSEC_FAILS_TO_ALERT} consecutive)'
            )
        else:
            if state['status'] != 'down':
                # First alert (now that we've accumulated enough failures)
                state['down_since'] = now
                state['status'] = 'down'
                errs = '\n'.join(f'  - {e}' for e in r['errors']) or '  - unknown'
                msg = (
                    f'<b>[VPS ALERT]</b>\n\n'
                    f'Time: {now}\n'
                    f'Consecutive failures: {state["consec_fails"]}\n'
                    f'Issues:\n{errs}\n\n'
                    f'Will repeat every {ALERT_REPEAT_MIN} min while down.'
                )
                send_telegram(msg)
                state['last_alert'] = now
                log.warning('OUTAGE detected, first alert sent')
            else:
                # Repeat alerts every ALERT_REPEAT_MIN
                try:
                    last = datetime.strptime(
                        state.get('last_alert') or state['down_since'],
                        '%Y-%m-%d %H:%M:%S',
                    )
                    if datetime.now() - last >= timedelta(minutes=ALERT_REPEAT_MIN):
                        errs = '\n'.join(f'  - {e}' for e in r['errors']) or '  - unknown'
                        dur = datetime.now() - datetime.strptime(state['down_since'], '%Y-%m-%d %H:%M:%S')
                        dur_str = f'{int(dur.total_seconds()//60)} min'
                        msg = (
                            f'<b>[VPS STILL DOWN]</b>\n\n'
                            f'Down since: {state["down_since"]} ({dur_str})\n'
                            f'Issues:\n{errs}'
                        )
                        send_telegram(msg)
                        state['last_alert'] = now
                        log.warning(f'Repeat alert sent (down for {dur_str})')
                except Exception as e:
                    log.warning(f'Repeat alert decision failed: {e}')

    save_state(state)
    log.info(f'State: {state}')
    return 0 if healthy else 1


if __name__ == '__main__':
    sys.exit(main())
