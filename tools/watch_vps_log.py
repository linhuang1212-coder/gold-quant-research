"""
实时查看 MT4 VPS 上的 gold_runner.log
====================================

凭据从项目根目录的 `.env` 读取（参考 `.env.example`），不要把密码硬编码到这个文件里。

用法:
    python tools\watch_vps_log.py          # 实时 follow (Ctrl+C 退出)
    python tools\watch_vps_log.py 100      # 显示最后 100 行后实时 follow
    python tools\watch_vps_log.py --once   # 只显示最后 50 行, 不持续

筛选:
    python tools\watch_vps_log.py --filter "TSMOM|tsmom"   # 只显示包含关键字的行
    python tools\watch_vps_log.py --error                  # 只显示 WARNING/ERROR

按行过滤后输出本地终端, 不影响 VPS。
"""
import sys
import time
import re
import argparse
from pathlib import Path

import paramiko

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJ_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJ_ROOT / '.env'

DEFAULT_LOG = r'C:\Users\administrator\gold-quant-trading\logs\gold_runner.log'


def load_env():
    """Read .env into a dict. Fail loudly if the file is missing."""
    if not ENV_FILE.exists():
        raise SystemExit(
            f'Missing {ENV_FILE}.\n'
            f'Copy `.env.example` to `.env` at the project root and fill in '
            f'MT4_VPS_HOST / MT4_VPS_USER / MT4_VPS_PASSWORD.'
        )
    env = {}
    for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip()
    required = ['MT4_VPS_HOST', 'MT4_VPS_USER', 'MT4_VPS_PASSWORD']
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise SystemExit(f'Missing keys in {ENV_FILE}: {missing}')
    return env


ENV = load_env()
HOST = ENV['MT4_VPS_HOST']
PORT = int(ENV.get('MT4_VPS_PORT', '22'))
USER = ENV['MT4_VPS_USER']
PASS = ENV['MT4_VPS_PASSWORD']
LOG = ENV.get('MT4_VPS_LOG', DEFAULT_LOG)


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        HOST, PORT, USER, PASS,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    return ssh


def get_file_size(ssh):
    cmd = (f'powershell -NoProfile -Command "(Get-Item \'{LOG}\').Length"')
    _, out, _ = ssh.exec_command(cmd, timeout=15)
    s = out.read().decode('utf-8', errors='replace').strip()
    try:
        return int(s)
    except ValueError:
        return 0


def tail_lines(ssh, n):
    """Get last N lines via PowerShell Get-Content -Tail."""
    cmd = (f'powershell -NoProfile -Command "[Console]::OutputEncoding = '
           f'[System.Text.Encoding]::UTF8; '
           f'Get-Content \'{LOG}\' -Tail {n} -Encoding UTF8"')
    _, out, _ = ssh.exec_command(cmd, timeout=30)
    return out.read().decode('utf-8', errors='replace')


def read_from(ssh, offset):
    """Read bytes from offset to end of file via SFTP."""
    sftp = ssh.open_sftp()
    with sftp.open(LOG, 'r') as f:
        f.seek(offset)
        data = f.read()
    sftp.close()
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='replace')
    return data


def match(line, pattern, error_only):
    if error_only:
        if 'WARNING' in line or 'ERROR' in line or 'CRITICAL' in line or 'Traceback' in line:
            return True
        return False
    if pattern is not None:
        return bool(pattern.search(line))
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('initial_lines', nargs='?', type=int, default=30,
                    help='initial lines to show (default 30)')
    ap.add_argument('--once', action='store_true',
                    help='show only initial lines and exit (no follow)')
    ap.add_argument('--filter', '-f', type=str, default=None,
                    help='regex filter (e.g. "TSMOM|tsmom")')
    ap.add_argument('--error', action='store_true',
                    help='only WARNING/ERROR/CRITICAL/Traceback lines')
    ap.add_argument('--interval', type=float, default=2.0,
                    help='polling interval seconds (default 2)')
    args = ap.parse_args()

    pat = re.compile(args.filter, re.IGNORECASE) if args.filter else None

    ssh = connect()
    print(f'■ Connected to {HOST}. Tailing {LOG}', flush=True)

    initial = tail_lines(ssh, args.initial_lines)
    for line in initial.splitlines():
        if match(line, pat, args.error):
            print(line, flush=True)

    if args.once:
        ssh.close()
        return

    print(f'\n──────── tailing live (Ctrl+C to exit) ────────', flush=True)

    last_size = get_file_size(ssh)
    last_partial = ''
    try:
        while True:
            time.sleep(args.interval)
            try:
                cur_size = get_file_size(ssh)
            except Exception as e:
                print(f'[!] size probe failed: {e}; reconnecting...', flush=True)
                try:
                    ssh.close()
                except Exception:
                    pass
                ssh = connect()
                last_size = get_file_size(ssh)
                continue

            if cur_size < last_size:
                print(f'[!] log rotated (size shrunk {last_size} -> {cur_size})', flush=True)
                last_size = 0
                last_partial = ''

            if cur_size > last_size:
                try:
                    chunk = read_from(ssh, last_size)
                except Exception as e:
                    print(f'[!] read failed: {e}; will retry', flush=True)
                    continue
                last_size = cur_size
                text = last_partial + chunk
                lines = text.split('\n')
                last_partial = lines[-1]
                for line in lines[:-1]:
                    if match(line, pat, args.error):
                        print(line, flush=True)
    except KeyboardInterrupt:
        print('\n[exit] interrupted by user', flush=True)
    finally:
        try:
            ssh.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
