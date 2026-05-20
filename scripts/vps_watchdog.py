"""
VPS External Watchdog — runs on local PC, monitors VPS over SSH, alerts via Telegram.

Checks every run:
  1. SSH connectivity to VPS
  2. python.exe (gold_runner) running on VPS
  3. MT4 heartbeat.json fresh (<2min mtime)

Alerts on Telegram once per outage start, repeat every 30min if still down.
Sends recovery message when all checks pass again.

Setup:
  Run via Windows Task Scheduler every 5 minutes (see vps_watchdog_install.md).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import paramiko
import requests

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── VPS connection ─────────────────────────────────────
VPS_IP = '38.180.149.40'
VPS_PORT = 22
VPS_USER = 'administrator'
VPS_PASS = 'h6A7qLuaPo'

# Path on VPS to MT4 heartbeat file (forward slashes work with SFTP)
HB_PATH = (
    '/C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/'
    '35EEC3EFDB656AF6FC775F21FEAD053B/MQL4/Files/DWX/heartbeat.json'
)
ACCT_PATH = (
    '/C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/'
    '35EEC3EFDB656AF6FC775F21FEAD053B/MQL4/Files/DWX/account.json'
)

# ── Telegram ───────────────────────────────────────────
TELEGRAM_TOKEN = '8646871612:AAHNt4TFp6T1pJdJarhnW8svFkhd-45Z7Fs'
TELEGRAM_CHAT = '8531960227'

# ── Thresholds ────────────────────────────────────────
MT4_HB_FRESH_SEC = 120
SSH_TIMEOUT = 20
ALERT_REPEAT_MIN = 30

STATE_FILE = Path(__file__).parent / '.vps_watchdog_state.json'
LOG_FILE = Path(__file__).parent / 'vps_watchdog.log'

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
    ssh = None
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            VPS_IP, port=VPS_PORT, username=VPS_USER, password=VPS_PASS,
            timeout=SSH_TIMEOUT, banner_timeout=SSH_TIMEOUT, auth_timeout=SSH_TIMEOUT,
        )
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

    except Exception as e:
        result['errors'].append(f'SSH connect failed: {e}')
    finally:
        if ssh:
            try:
                ssh.close()
            except Exception:
                pass

    return result


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'status': 'unknown', 'down_since': None, 'last_alert': None}


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
        if state['status'] == 'down':
            # Recovery
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
        state = {'status': 'ok', 'down_since': None, 'last_alert': None}

    else:
        if state['status'] != 'down':
            state['down_since'] = now
            state['status'] = 'down'
            # First alert
            errs = '\n'.join(f'  - {e}' for e in r['errors']) or '  - unknown'
            msg = (
                f'<b>[VPS ALERT]</b>\n\n'
                f'Time: {now}\n'
                f'Issues:\n{errs}\n\n'
                f'Will repeat every {ALERT_REPEAT_MIN} min while down.'
            )
            send_telegram(msg)
            state['last_alert'] = now
            log.warning('OUTAGE detected, first alert sent')
        else:
            # Repeat alerts every ALERT_REPEAT_MIN
            try:
                last = datetime.strptime(state.get('last_alert', state['down_since']), '%Y-%m-%d %H:%M:%S')
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
