"""探测 2：单条 exec 命令字符串多大就会被 RST（用完即删）。

对照：同一条命令里只改长度，内容形态一致（printf '<b64>' >> 文件），避免"命令内容"干扰。
"""
import base64
import os
import time

import paramiko

HOST = os.environ.get('RC_HOST', '192.168.9.226')
PW = os.environ['RC_PW']


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username='root', password=PW, timeout=20, banner_timeout=30, auth_timeout=30,
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
        time.sleep(0.03)


def send_sized(cmd_len, tag):
    """构造长度约 cmd_len 的命令：printf '<b64>' >> 文件（b64 长度占大头）"""
    fixed = len("printf '%s' '' >> /tmp/rc_probe2.bin")
    b64_len = max(4, cmd_len - fixed)
    b64_len -= b64_len % 4
    payload = os.urandom(b64_len * 3 // 4)
    b64 = base64.b64encode(payload).decode()[:b64_len]
    cmd = f"printf '%s' '{b64}' >> /tmp/rc_probe2.bin"
    try:
        c = connect()
        out, code = run(c, cmd)
        c.close()
        return f'len={len(cmd):5d} {tag}: OK exit={code} {out[:40]!r}'
    except Exception as exc:
        return f'len={len(cmd):5d} {tag}: 异常 {type(exc).__name__}'


print('=== 单条命令长度扫描（每条都新建连接，追加到同一文件） ===')
for n in (100, 300, 500, 700, 900, 1100, 1300, 1600, 2000, 3000):
    print(' ', send_sized(n, 'first'))

print('=== 同一条连接上连续发 20 条 ~600 字节命令 ===')
c = connect()
ok = 0
for i in range(20):
    fixed = len("printf '%s' '' >> /tmp/rc_probe2.bin")
    b64 = base64.b64encode(os.urandom(400)).decode()
    cmd = f"printf '%s' '{b64}' >> /tmp/rc_probe2.bin"
    try:
        out, code = run(c, cmd)
        ok += 1
    except Exception as exc:
        print(f'  第 {i + 1} 条失败: {type(exc).__name__} （已成功 {ok} 条）')
        break
else:
    print(f'  20 条全部成功')
c.close()

print('=== 校验远端文件大小 ===')
c = connect()
out, code = run(c, 'wc -c < /tmp/rc_probe2.bin')
print('  wc -c:', out, 'exit=', code)
c.close()
