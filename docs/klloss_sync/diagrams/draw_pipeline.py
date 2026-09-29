"""Deterministic, editable redraws of the user's V/C pipeline reference.

This creates editable vector diagrams; user_reference.png remains unchanged.
Run: python3 docs/klloss_sync/diagrams/draw_pipeline.py
"""

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
font_path = Path(os.environ.get("SLIG_DIAGRAM_FONT", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"))
FONT = FontProperties(fname=str(font_path)) if font_path.is_file() else FontProperties(family="sans-serif")
GREEN, BLUE = "#329952", "#307fde"
RED, PURPLE, GRAY = "#cc3944", "#8054ad", "#8391a0"
INK = "#23374d"
VY, CY = 7.65, 4.95


def label(ax, x, y, text, size=13, color=INK, ha="center", background=False):
    ax.text(x, y, text, fontsize=size, fontproperties=FONT, color=color,
            ha=ha, va="center", linespacing=1.35, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=.94)
            if background else None)


def arrow(ax, start, end, color=GRAY, dashed=False, curve=0, width=1.7):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                 connectionstyle=f"arc3,rad={curve}", color=color,
                 linewidth=width, linestyle="--" if dashed else "-", zorder=2))


def node(ax, x, y, text, parity=0, width=1.68, height=.96, state="normal", size=14):
    color = GREEN if parity == 0 else BLUE
    fill = "#ffffff"
    dash = "-"
    if state == "stop":
        color, fill = RED, "#fff0f1"
    elif state == "future":
        color, fill, dash = "#b0b8c4", "#f6f7f9", "--"
    elif state == "suspect":
        color, fill, dash = "#c98e31", "#fff7e8", "--"
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                 boxstyle="round,pad=0.045,rounding_size=.2", edgecolor=color,
                 facecolor=fill, linewidth=1.7, linestyle=dash, zorder=4))
    label(ax, x, y, text, size, color)


def frame(title, subtitle):
    fig, ax = plt.subplots(figsize=(26, 10))
    fig.patch.set_facecolor("#ffffff")
    ax.set(xlim=(0, 24), ylim=(0, 10))
    ax.axis("off")
    label(ax, .25, 9.57, title, 23, ha="left")
    label(ax, .25, 9.08, subtitle, 13, ha="left")
    arrow(ax, (.9, 8.65), (23.5, 8.65), "#afbbc7")
    label(ax, 22.0, 8.9, "执行进度 →（非周期比例）", 12)
    label(ax, .23, 8.35, "AIV / Vector", 13, ha="left")
    label(ax, .23, 4.13, "AIC / Cube", 13, ha="left")
    for y in (VY, CY):
        ax.plot([.7, 23.5], [y, y], color="#e2e7ee", linewidth=1.5, zorder=0)
    fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
    return fig, ax


def dependencies(ax, positions, tasks, omit=()):
    colors = {0: "#db6259", 1: BLUE, 2: "#db6259"}
    for task in tasks:
        chain = [f"v0{task}", f"c1{task}", f"v1{task}", f"c2{task}", f"v2{task}"]
        for source, target in zip(chain, chain[1:]):
            if (source, target) in omit:
                continue
            sx, sy = positions[source]
            tx, ty = positions[target]
            sign = 1 if ty > sy else -1
            arrow(ax, (sx+.35, sy+sign*.52), (tx-.28, ty-sign*.52),
                  colors[task], curve=.04 if abs(tx-sx) > 4 else 0)


def footer(ax, conflict=False):
    label(ax, .25, .62,
          "绿框 = Ping（task 0、2）   蓝框 = Pong（task 1）   实线 = 正常任务依赖   "
          + ("红虚线 = 错配猜想   灰虚框 = 后续未执行阶段" if conflict else
             "紫色说明 = V2 内全 AIV 的 mode 0 会合"), 12, ha="left")
    label(ax, .25, .25,
          "只展开 task 0 / 1 / 2；后续新任务的 V0、V1 等阶段省略。Ping/Pong 是任务相位，不是 flagId 的数值。",
          11, color="#63768a", ha="left")


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    fig.savefig(OUT / f"{name}.png", facecolor="white", dpi=135)
    plt.close(fig)


def normal():
    fig, ax = frame("正常时序：V0 → C1 → V1 → C2 → V2 按任务正确配对",
                    "沿用原图的上下双泳道；补充任务号。下方单独展开一次安全的 flag 0 复用顺序，这不是源码已经保证的顺序。")
    pos = {
        "v00": (1.4, VY), "v01": (3.6, VY), "v10": (5.8, VY),
        "v02": (8.0, VY), "v11": (10.4, VY), "v20": (14.6, VY),
        "v12": (18.0, VY), "v21": (20.4, VY), "v22": (22.6, VY),
        "c10": (3.6, CY), "c20": (8.0, CY), "c11": (9.6, CY),
        "c21": (11.8, CY), "c12": (13.6, CY), "c22": (20.4, CY),
    }
    dependencies(ax, pos, (0, 1, 2))
    for name, (x, y) in pos.items():
        stage, task = name[:2].upper(), int(name[-1])
        node(ax, x, y, f"{stage}_{'Ping' if task % 2 == 0 else 'Pong'}\ntask {task}", task % 2,
             width=1.46 if name.startswith("c") else 1.68)
    label(ax, 2.15, 6.25, "mode 2\nflag 0", 11, background=True)
    label(ax, 4.68, 6.16, "P / SY\n4 / 6", 11, background=True)
    label(ax, 7.03, 6.18, "DW = 8", 11, background=True)
    label(ax, 11.4, 6.07, "SA = 10", 11, background=True)
    label(ax, 3.6, 7.0, "Pong 的 V0 也发 flag 0", 11, BLUE)
    label(ax, 14.6, 7.0, "Wait SA(10) → mode 0 / flag 0", 11, PURPLE)
    label(ax, 20.4, 7.0, "mode 0 / flag 1", 11, PURPLE)
    label(ax, 22.6, 7.0, "mode 0 / flag 0", 11, PURPLE)
    label(ax, 16.25, VY, "…", 22, GRAY)
    label(ax, 16.25, 7.0, "省略 V0(3)", 10, GRAY)

    ax.add_patch(FancyBboxPatch((.3, 1.15), 23.35, 2.5,
                 boxstyle="round,pad=.05,rounding_size=.16", facecolor="#f4faf6",
                 edgecolor="#c5dfce", zorder=0))
    label(ax, .55, 3.25, "放大 flag 0：正常完成的一种安全时序（所有相关参与核的 mode 2 先收尾）", 15, GREEN, ha="left")
    steps = [(3.0, "V0_Ping(task 2)\nSet<2, MTE3>(0)"),
             (8.8, "C1_Ping(task 2)\nWait(0) 消费 V0 通知"),
             (14.6, "V2_Ping(task 0)\nSet<0, MTE2>(0)"),
             (20.4, "全 AIV 本轮会合完成\nV2 Wait(0) 放行")]
    for x, content in steps:
        node(ax, x, 2.22, content, width=4.15, height=.94, size=13)
    for left, right in zip(steps, steps[1:]):
        arrow(ax, (left[0]+2.18, 2.22), (right[0]-2.18, 2.22), GREEN,
              dashed=left[0] == 8.8)
    label(ax, 12, 1.37, "绿色虚线表示安全收尾条件，不是新增代码：原始流水缺少强制这条顺序的确认边。", 12, GREEN)
    footer(ax)
    save(fig, "normal_timing")


def conflict():
    fig, ax = frame("可能冲突时序：旧任务 V2_Ping 与新任务 C1_Ping 同时使用 flag 0",
                    "按共享底层状态与可交叉消费的前提构造；task 0 / 2 为示意。实测只定位了最终 Wait，尚无逐拍计数器轨迹。")
    pos = {
        "v00": (1.4, VY), "v01": (3.6, VY), "v10": (5.8, VY),
        "v02": (8.0, VY), "v11": (10.4, VY), "v20": (14.1, VY),
        "v12": (18.0, VY), "v21": (20.4, VY), "v22": (22.6, VY),
        "c10": (3.6, CY), "c20": (8.0, CY), "c11": (9.6, CY),
        "c21": (11.6, CY), "c12": (14.2, CY), "c22": (18.5, CY),
    }
    dependencies(ax, pos, (0, 1), omit=(("c21", "v21"),))
    for name in ("v00", "v01", "v10", "v02", "v11", "c10", "c20", "c11", "c21"):
        x, y = pos[name]
        task = int(name[-1])
        node(ax, x, y, f"{name[:2].upper()}_{'Ping' if task % 2 == 0 else 'Pong'}\ntask {task}",
             task % 2, width=1.46 if name.startswith("c") else 1.68)
    node(ax, 14.1, VY, "V2_Ping · task 0\nSet<0>(0) → Wait(0) 阻塞", width=3.5, state="stop", size=14)
    node(ax, 14.2, CY, "C1_Ping · task 2\nWait(0) 错配放行？", width=3.05, state="suspect", size=14)
    node(ax, 18.5, CY, "C2_Ping · task 2\nWait DW(8) 阻塞", width=3.2, state="stop", size=14)
    node(ax, 18.0, VY, "V1_Ping\ntask 2 · 未执行", state="future", width=2.5)
    node(ax, 20.5, VY, "V2_Pong\ntask 1 · 未执行", 1, state="future", width=2.05, size=12)
    node(ax, 22.8, VY, "V2_Ping\ntask 2 · 未执行", state="future", width=2.05, size=12)
    arrow(ax, (8.7, 7.13), (13.0, 5.48), "#c18d35", dashed=True)
    label(ax, 10.5, 6.9, "V0 的 mode 2 / flag 0\n未确认全部收尾", 11, "#a7741b", background=True)
    arrow(ax, (14.05, 7.1), (14.1, 5.5), RED, dashed=True, width=2.5)
    label(ax, 15.25, 6.25, "mode 0 通知\n被 mode 2 消费？", 12, RED, background=True)
    arrow(ax, (15.83, CY), (16.83, CY), RED)
    label(ax, 16.25, 4.24, "C1 发 P / SY 后\n继续到 C2 等 DW", 11, RED)
    arrow(ax, (18.1, 7.12), (18.4, 5.51), GRAY, dashed=True)
    label(ax, 19.3, 6.23, "DW(8) 不再发出", 12, RED, background=True)
    label(ax, 14.1, 8.34, "极简流水现场：V2 首道 mode 0 Wait", 11, RED, background=True)
    label(ax, 18.5, 4.02, "极简流水现场：C2 Wait DW", 12, RED)
    label(ax, 22.0, 5.2, "全局会合未完成\n后续流水停顿", 14, RED)

    ax.add_patch(FancyBboxPatch((.3, 1.15), 23.35, 2.5,
                 boxstyle="round,pad=.05,rounding_size=.16", facecolor="#fff6f5",
                 edgecolor="#edc5c6", zorder=0))
    label(ax, .55, 3.25, "卡死链：冲突位置不等于最终超时位置", 15, RED, ha="left")
    contents = ["同 flag 0 跨 mode 重叠\nV2 通知可能错配 / 被消费",
                "V2 本轮通知缺失\n且在途通知不能补齐",
                "AIV 无法进入 V1(task 2)\nDW(task 2) 无法发送",
                "AIC 停在 C2 Wait DW\n最终报 507014 超时"]
    for x, content in zip((3.0, 8.8, 14.6, 20.4), contents):
        node(ax, x, 2.22, content, width=4.15, height=.94, state="stop", size=13)
    for x in (3.0, 8.8, 14.6):
        arrow(ax, (x+2.18, 2.22), (x+3.62, 2.22), RED, dashed=x == 3.0)
    label(ax, 12, 1.37, "现场未采集任务号 / flag 寄存器值。task 0/2、DW=8 是便于解释的示例，不是逐拍日志读数。", 11, RED)
    footer(ax, True)
    save(fig, "conflict_timing")


if __name__ == "__main__":
    normal()
    conflict()
    print(f"Wrote normal_timing and conflict_timing (SVG + PNG) to {OUT}")
