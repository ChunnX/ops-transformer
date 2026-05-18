#!/usr/bin/env python3
"""
轻量级定时调度器 — 替代 cron/systemd
- 每天 9:15   发送每日一题 PDF
- 每天 20:00  学习督促跟进
"""

import time
import subprocess
import os
import sys
import random
from datetime import datetime, date, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_SCRIPT = os.path.join(SCRIPT_DIR, "daily_leetcode.py")
LOG_FILE = os.path.join(SCRIPT_DIR, "scheduler.log")
DONE_FILE = os.path.join(SCRIPT_DIR, ".done_today")
CHECKIN_FILE = os.path.join(SCRIPT_DIR, ".checkin_today")

PYTHON_BIN = "/opt/buildtools/Python-3.11.4/bin/python3.11"
LARK_CLI_BIN = "/home/developer/.nvm/versions/node/v18.20.8/bin/lark-cli"
GROUP_CHAT_ID = "oc_2a4c0042cd63810cf518fe7bcc430e70"

EVENING_MESSAGES = [
    "🌙 晚上好！今天的算法题学会了吗？不会的话现在正是攻克的好时机，题就在上面，随时问我！",
    "💡 睡觉前花15分钟复习一下今天的算法题，效果比白天走马观花好十倍~",
    "📖 还没看今天的每日一题？别让它过夜呀，打开 PDF 花几分钟搞定它！",
    "⏰ 一天即将结束，今天题目掌握了吗？有问题随时在群里问。",
    "🎯 坚持就是胜利！今天的每日一题还在等你，抽空看一下吧！",
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_daily():
    log("推送每日一题...")
    try:
        result = subprocess.run(
            [PYTHON_BIN, DAILY_SCRIPT],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "PATH": "/home/developer/.nvm/versions/node/v18.20.8/bin:" + os.environ.get("PATH", "")}
        )
        if result.returncode == 0:
            log(f"推送成功: {result.stdout.strip()}")
        else:
            log(f"推送失败: {result.stderr.strip()}")
    except Exception as e:
        log(f"推送异常: {e}")

    with open(DONE_FILE, "w") as f:
        f.write(date.today().isoformat())


def already_done_today():
    try:
        with open(DONE_FILE) as f:
            return f.read().strip() == date.today().isoformat()
    except FileNotFoundError:
        return False


def checkin_done_today():
    try:
        with open(CHECKIN_FILE) as f:
            return f.read().strip() == date.today().isoformat()
    except FileNotFoundError:
        return False


def mark_checkin_done():
    with open(CHECKIN_FILE, "w") as f:
        f.write(date.today().isoformat())


def send_checkin():
    msg = random.choice(EVENING_MESSAGES)
    log(f"发送学习督促: {msg[:20]}...")
    try:
        result = subprocess.run(
            [LARK_CLI_BIN, "im", "+messages-send",
             "--chat-id", GROUP_CHAT_ID,
             "--text", msg,
             "--as", "bot"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log("督促消息发送成功")
            mark_checkin_done()
        else:
            log(f"督促消息发送失败: {result.stderr.strip()}")
    except Exception as e:
        log(f"督促消息发送异常: {e}")


def seconds_until(hour, minute):
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max(1, int((target - now).total_seconds()))


log("⚡ 调度器启动 — 每日 9:15 推送 + 20:00 督促")
log(f"每日脚本: {DAILY_SCRIPT}")

while True:
    # 计算两个事件分别还需等多久
    wait_push = seconds_until(9, 15)
    wait_checkin = seconds_until(20, 0)

    # 挑最近的下一个事件
    if already_done_today():
        wait_push += 86400  # 今天已推送，跳到明天
    if checkin_done_today():
        wait_checkin += 86400

    if wait_push < wait_checkin:
        wait = wait_push
        desc = "每日一题推送"
    else:
        wait = wait_checkin
        desc = "晚间学习督促"

    log(f"下次事件: {desc}，等待 {wait} 秒 (~{wait//60} 分钟)")

    time.sleep(wait)

    if desc == "每日一题推送" and not already_done_today():
        run_daily()
    elif desc == "晚间学习督促" and not checkin_done_today():
        send_checkin()
