"""探测：这条 SSH 通道能承受多大的 exec 命令行 / 连续执行多少次（用完即删）。"""
import base64
import os
import sys
import time

import paramiko

HOST = os.environ.get('RC_HOST', '192.168.9.226')
USER = os.environ.get('RC_USER', 'root')
PW = os.environ['RC_PW']


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PW, timeout=20, banner_timeout=30, auth_timeout=30,
              look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, timeout=60):
    chan = c.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = b''
    deadline = time.time() + timeout
    while True:
        if chan.recv_ready():
            out += chan.recv(65536)
        elif chan.recv_stderr_ready():
            out += chan.recv_stderr(65536)
        elif chan.exit_status_ready():
            return out.decode(errors='replace').strip(), chan.recv_exit_status()
        elif time.time() > deadline:
            chan.close()
            return '<timeout>', -1
        time.sleep(0.05)


# 1) 同一条连接上连续执行 5 个小命令
c = connect()
for i in range(5):
    try:
        out, code = run(c, f'echo small{i}')
        print(f'  同连接第 {i + 1} 次: {out} (exit={code})')
    except Exception as exc:
        print(f'  同连接第 {i + 1} 次: 异常 {type(exc).__name__}: {exc}')
        break
c.close()

# 2) 不同大小的命令（每次新连接），写入 /tmp 并校验字节数
payload = os.urandom(200000)
for size in (1000, 2000, 4000, 8000, 16000, 32000, 60000):
    data = payload[:size]
    b64 = base64.b64encode(data).decode()
    cmd = f"printf '%s' '{b64}' | base64 -d > /tmp/rc_probe.bin; wc -c < /tmp/rc_probe.bin"
    try:
        c = connect()
        out, code = run(c, cmd)
        c.close()
        ok = out.strip() == str(size)
        print(f'  命令行 {len(cmd):6d} 字节(数据 {size:6d}): exit={code} out={out!r} {"OK" if ok else "MISMATCH"}')
    except Exception as exc:
        print(f'  命令行 {len(cmd):6d} 字节(数据 {size:6d}): 异常 {type(exc).__name__}: {exc}')
