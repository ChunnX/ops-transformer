#!/usr/bin/env python3
"""远程执行小工具（paramiko，密码从环境变量 RC_PW 读，不落盘）。

环境限制（实测）：单条 exec 命令行 >= ~1.5KB 会被对端 RST，且会连带断开整条连接；
总传输量不受限，所以文件传输用「小分块 base64 追加 + 断线重连 + 最终大小校验」。
SFTP 也不可用（同样被 RST）。

用法:
  rc.py "cmd1; cmd2"                    # 在宿主机执行
  rc.py -c wangchao "cmd"               # 在 docker 容器内执行
  rc.py --run-script 本地.sh             # 上传脚本到 <工作目录>/.rc/ 再 bash 执行（推荐）
  rc.py --put 本地文件 远端路径           # 上传文件
  rc.py --put-tree 本地目录 远端目录      # 递归上传目录（小目录用）
  rc.py --get 远端文件 本地路径           # 下载文件

环境变量: RC_PW(必填), RC_HOST(默认 192.168.9.226), RC_USER(默认 root), RC_PORT(默认 22),
          RC_WORKDIR(默认 /home/wchao00893980/deepseek1009，宿主机与容器都能看到)
"""
import argparse
import base64
import os
import shlex
import sys
import time

import paramiko

CHUNK = 800          # 每个 exec 携带的 base64 字符数（保持命令行 <1KB）
TRANSPORT_ERRORS = (paramiko.SSHException, EOFError, OSError)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--container', help='在指定 docker 容器内执行命令')
    p.add_argument('--put', nargs=2, metavar=('LOCAL', 'REMOTE'), help='上传单个文件')
    p.add_argument('--put-tree', nargs=2, metavar=('LOCAL_DIR', 'REMOTE_DIR'), help='递归上传目录')
    p.add_argument('--get', nargs=2, metavar=('REMOTE', 'LOCAL'), help='下载单个文件')
    p.add_argument('--run-script', metavar='LOCAL_SH', help='上传脚本到 <RC_WORKDIR>/.rc/ 后执行')
    p.add_argument('--timeout', type=int, default=3600, help='命令超时秒数（默认 3600）')
    p.add_argument('command', nargs='*', help='要执行的命令')
    return p.parse_args()


def connect():
    password = os.environ.get('RC_PW')
    if not password:
        raise SystemExit('环境变量 RC_PW 未设置（登录密码）')
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        os.environ.get('RC_HOST', '192.168.9.226'),
        port=int(os.environ.get('RC_PORT', '22')),
        username=os.environ.get('RC_USER', 'root'),
        password=password,
        timeout=20, banner_timeout=30, auth_timeout=30,
        look_for_keys=False, allow_agent=False,
    )
    return client


class Remote:
    def __init__(self):
        self.client = connect()

    def reconnect(self):
        try:
            self.client.close()
        except Exception:
            pass
        self.client = connect()

    def exec_once(self, cmd, timeout=900):
        """执行一次；连接类异常直接抛出（调用方决定是否重连重试）。"""
        chan = self.client.get_transport().open_session()
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
                while chan.recv_ready():
                    out += chan.recv(65536)
                return out.decode('utf-8', errors='replace'), chan.recv_exit_status()
            elif time.time() > deadline:
                chan.close()
                return '<timeout>', -1
            else:
                time.sleep(0.03)

    def exec(self, cmd, timeout=900, retries=4):
        """适合幂等命令（mkdir/wc/普通命令）：断线自动重连重试。"""
        last = None
        for _ in range(retries):
            try:
                return self.exec_once(cmd, timeout)
            except TRANSPORT_ERRORS as exc:
                last = exc
                time.sleep(0.4)
                self.reconnect()
        raise RuntimeError(f'连接重试 {retries} 次仍失败: {last}')

    def stream(self, cmd, timeout):
        """流式执行（给构建这类长命令用）。"""
        chan = self.client.get_transport().open_session()
        chan.settimeout(timeout)
        chan.exec_command(cmd)
        deadline = time.time() + timeout
        while True:
            if chan.recv_ready():
                sys.stdout.write(chan.recv(65536).decode('utf-8', errors='replace'))
                sys.stdout.flush()
            elif chan.recv_stderr_ready():
                sys.stderr.write(chan.recv_stderr(65536).decode('utf-8', errors='replace'))
                sys.stderr.flush()
            elif chan.exit_status_ready():
                while chan.recv_ready():
                    sys.stdout.write(chan.recv(65536).decode('utf-8', errors='replace'))
                return chan.recv_exit_status()
            elif time.time() > deadline:
                chan.close()
                print(f'\n[rc.py] 超时 {timeout}s，已中断', file=sys.stderr)
                return -1
            else:
                time.sleep(0.05)

    def size_of(self, remote):
        out, code = self.exec(f'wc -c < {shlex.quote(remote)} 2>/dev/null || echo -1')
        try:
            return int(out.split()[0])
        except (ValueError, IndexError):
            return -1


def put_bytes(r, data, remote, mode=None):
    """小分块写入并校验大小；某块写完后连接才断的情况用大小比对跳过重写。"""
    remote_q = shlex.quote(remote)
    r.exec(f'mkdir -p $(dirname {remote_q})')
    r.exec(f': > {remote_q}')
    b64 = base64.b64encode(data).decode()
    written = 0
    for i in range(0, len(b64), CHUNK):
        part = b64[i:i + CHUNK]
        piece = base64.b64decode(part + '=' * (-len(part) % 4))
        target = written + len(piece)
        for attempt in range(4):
            try:
                out, code = r.exec_once(f"printf '%s' '{part}' | base64 -d >> {remote_q}")
                if code != 0:
                    raise RuntimeError(f'分块写入返回 {code}: {out[:120]}')
                break
            except TRANSPORT_ERRORS as exc:
                time.sleep(0.4)
                r.reconnect()
                if r.size_of(remote) >= target:      # 其实已经写进去了
                    break
                if attempt == 3:
                    raise RuntimeError(f'分块传输失败: {exc}')
        written = target
    actual = r.size_of(remote)
    if actual != len(data):
        raise RuntimeError(f'传输大小不符: 期望 {len(data)}，实际 {actual} -> {remote}')
    if mode:
        r.exec(f'chmod {mode} {remote_q}')
    return len(data)


def iter_files(local_dir):
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            full = os.path.join(root, name)
            yield full, os.path.relpath(full, local_dir)


def main():
    a = parse_args()
    r = Remote()
    workdir = os.environ.get('RC_WORKDIR', '/home/wchao00893980/deepseek1009')

    if a.put:
        n = put_bytes(r, open(a.put[0], 'rb').read(), a.put[1])
        print(f'[rc.py] uploaded {n} bytes -> {a.put[1]}')
        return 0

    if a.put_tree:
        count = 0
        for full, rel in iter_files(a.put_tree[0]):
            put_bytes(r, open(full, 'rb').read(), a.put_tree[1].rstrip('/') + '/' + rel.replace(os.sep, '/'))
            count += 1
        print(f'[rc.py] uploaded {count} files -> {a.put_tree[1]}')
        return 0

    if a.get:
        out, code = r.exec(f'base64 -w0 {shlex.quote(a.get[0])}')
        if code != 0:
            raise SystemExit(f'读取远端文件失败: {out[:200]}')
        with open(a.get[1], 'wb') as f:
            f.write(base64.b64decode(out.strip()))
        print(f'[rc.py] downloaded {a.get[0]} -> {a.get[1]}')
        return 0

    if a.run_script:
        r.exec(f'mkdir -p {shlex.quote(workdir)}/.rc')
        remote = f'{workdir}/.rc/{os.path.basename(a.run_script)}'
        put_bytes(r, open(a.run_script, 'rb').read(), remote, mode='u+x')
        print(f'[rc.py] script -> {remote}', flush=True)
        cmd = f'bash {shlex.quote(remote)}'
    elif a.command:
        cmd = ' '.join(a.command)
    else:
        raise SystemExit('需要给出要执行的命令，或用 --run-script 指定脚本')

    if a.container:
        cmd = f'docker exec {shlex.quote(a.container)} bash -lc {shlex.quote(cmd)}'
    print(f'[rc.py] $ {cmd}', flush=True)
    code = r.stream(cmd, a.timeout)
    print(f'[rc.py] exit={code}')
    r.client.close()
    return 0 if code == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
