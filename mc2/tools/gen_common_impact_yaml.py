#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: ./gen_common_impact_yaml.py [ops-transformer仓路径] [输出目录]
      两个参数均可省略: 默认仓=本脚本所在仓, 默认输出=<仓>/mc2/tools/impact_map
功能: 对指定的 ops-transformer 检出(任意时期/任意版本), 扫描 mc2 内全部 #include,
      按"裸文件名"判定 mc2/common 与 mc2/3rd 每个文件的反向影响算子范围
      (含 common/3rd 内部传递闭包), 按目录(common/3rd)×芯片(A2/A3/A5)输出6个yaml,
      算子列表全部显式展开(不使用缩写标记)。
输出: <输出目录>/common/{A2,A3,A5}.yaml 与 <输出目录>/3rd/{A2,A3,A5}.yaml
      yaml头部记录commit, 便于跨时期对比
说明: 重名头文件的映射取并集(宁多勿漏), 相关文件带 note 标注; 重新运行会覆盖输出。
"""

import os
import re
import sys
import subprocess
import datetime

REPO = (
    os.path.abspath(sys.argv[1])
    if len(sys.argv) > 1
    else os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
)
OUTDIR = (
    os.path.abspath(sys.argv[2])
    if len(sys.argv) > 2
    else os.path.join(REPO, "mc2", "tools", "impact_map")
)
MC2 = os.path.join(REPO, "mc2")
SRC_EXT = (".h", ".hpp", ".cuh", ".cpp", ".cc", ".c")
CHIP_MAP = {"A2": "ascend910b", "A3": "ascend910_93", "A5": "ascend950"}
INC_RE = re.compile(r'#include\s+"([^"]+)"')
SOC_RE = re.compile(r'AddConfig\("([a-z0-9_]+)"')
EXCLUDE_OPS = ("common", "3rd", "tools")

if not os.path.isdir(MC2):
    sys.exit(f"错误: 不是 ops-transformer 仓(找不到 {MC2})")

# ---------- 1. 索引 mc2 下所有源文件 ----------
idx = {}  # mc2相对路径 -> 绝对路径
for dp, _, fs in os.walk(MC2):
    top = os.path.relpath(dp, MC2).split("/")[0]
    if top == "tools":
        continue
    for f in fs:
        if f.endswith(SRC_EXT):
            idx[os.path.relpath(os.path.join(dp, f), MC2)] = os.path.join(dp, f)

targets = {r for r in idx if r.startswith(("common/", "3rd/"))}  # 影响源(相对mc2路径)
by_base = {}  # 裸文件名 -> [mc2相对路径]
for r in targets:
    by_base.setdefault(os.path.basename(r), []).append(r)
dup_groups = {k: v for k, v in by_base.items() if len(v) > 1}

# ---------- 2. 建 裸文件名 -> 引用者 边(所有include形态统一取basename) ----------
referers = {r: set() for r in targets}
for rel, path in idx.items():
    for m in INC_RE.finditer(open(path, errors="ignore").read()):
        for t in by_base.get(os.path.basename(m.group(1)), ()):
            if t != rel:  # 跳过自引用
                referers[t].add(rel)

# ---------- 3. common/3rd 内部传递闭包 ----------
memo = {}


def ops_of(rel, stack=()):
    if rel in memo:
        return memo[rel]
    if rel in stack:
        return set()
    out = set()
    for rf in referers[rel]:
        top = rf.split("/")[0]
        if top not in EXCLUDE_OPS:
            out.add(top)
        elif rf in targets:
            out |= ops_of(rf, stack + (rel,))
    memo[rel] = out
    return out


impact = {r: ops_of(r) for r in targets}

# ---------- 4. 算子芯片注册(来自 *_def.cpp) ----------
socs = {}
for op in sorted(os.listdir(MC2)):
    if op in EXCLUDE_OPS or not os.path.isdir(os.path.join(MC2, op)):
        continue
    s = set()
    for dp, _, fs in os.walk(os.path.join(MC2, op)):
        for f in fs:
            if f.endswith("_def.cpp"):
                s |= set(
                    SOC_RE.findall(open(os.path.join(dp, f), errors="ignore").read())
                )
    socs[op] = s

# ---------- 5. 零引用实现文件继承同名头 ----------
inh = 0
for r in targets:
    if not impact[r] and r.endswith((".cpp", ".cc", ".c")):
        h = r.rsplit(".", 1)[0] + ".h"
        if h in impact and impact[h]:
            impact[r] = impact[h]
            inh += 1


# ---------- 6. git信息(yaml头, 跨时期对比用) ----------
def git(*a):
    try:
        return (
            subprocess.check_output(
                ["git", "-C", REPO] + list(a), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "?"


commit = git("rev-parse", "--short", "HEAD")
branch = git("rev-parse", "--abbrev-ref", "HEAD")

# ---------- 7. 输出 yaml: 目录(common/3rd) x 芯片(A2/A3/A5) = 6个文件 ----------
os.makedirs(OUTDIR, exist_ok=True)
summary = []
for grp in ("common", "3rd"):
    grp_files = {r for r in targets if r.startswith(grp + "/")}
    for chip_cn, soc in CHIP_MAP.items():
        lines = [
            f"# mc2/{grp} 文件影响范围映射表 - {chip_cn} ({soc})",
            f"# 生成: {datetime.date.today()}  基线commit: {commit} ({branch})",
            "# 生成器: mc2/tools/gen_common_impact_yaml.py (可对任意时期检出重新生成; 本yaml为生成物, 勿手改)",
            f"# 键路径: 相对mc2目录({grp}/开头), 与pr_files.sh输出中mc2/后的段落一致",
            "#",
            "# 条目三态语义:",
            "#   1) ops 列出算子 = 该文件被这些算子(直接或间接)include, 改动它需在本芯片测试这些算子",
            f"#   2) ops: []      = 该文件有引用, 但引用它的算子均未在{chip_cn}注册, 本芯片无需测试",
            "#   3) unref.yaml   = 完全无mc2源文件include的零引用文件(单独成表), 改动需人工确认",
            "#",
            "# 判定规则:",
            "#   - mc2内所有#include按裸文件名匹配, 重名文件取并集(note:字段标注), 宁多勿漏",
            "#   - 间接include沿common/3rd内部引用链传递闭包, 爬到算子目录为止",
            f"#   - 算子范围 = 其*_def.cpp的AICore().AddConfig()注册 与 {chip_cn}({soc}) 的交集",
            "",
        ]
        n_map = n_empty = 0
        for r in sorted(grp_files):
            ops = {o for o in impact[r] if soc in socs.get(o, set())}
            if not impact[r]:
                continue
            lines.append(f"{r}:")
            if ops:
                lines.append("  ops:")
                for o in sorted(ops):
                    lines.append(f"    - {o}")
                n_map += 1
            else:
                lines.append("  ops: []")
                n_empty += 1
            if os.path.basename(r) in dup_groups:
                lines.append(
                    f"  note: 同名文件{len(dup_groups[os.path.basename(r)])}份, 映射取并集"
                )
        grp_out = os.path.join(OUTDIR, grp)
        os.makedirs(grp_out, exist_ok=True)
        out = os.path.join(grp_out, f"{chip_cn}.yaml")
        open(out, "w").write("\n".join(lines) + "\n")
        summary.append((f"{grp}/{chip_cn}", n_map, n_empty, out))

# ---------- 7.5 零引用文件单独输出 unref.yaml(防漏查) ----------
for grp in ("common", "3rd"):
    grp_files = {r for r in targets if r.startswith(grp + "/")}
    unref = sorted(r for r in grp_files if not impact[r])
    lines = [
        f"# mc2/{grp} 零引用文件清单 (与芯片无关, 全芯片共用)",
        f"# 生成: {datetime.date.today()}  基线commit: {commit} ({branch})",
        "#",
        "# 语义: 下列文件当前无任何mc2源文件include(含间接), 改动不影响mc2算子编译;",
        "#       注意与芯片映射表中 ops: [] 的区别:",
        "#         ops: []  = 有引用, 但引用它的算子不支持该芯片 (精确的空, 无需测试)",
        "#         本表     = 完全没有引用, 属异常状态, 修改前需人工确认:",
        "#                    - 是否被仓外代码/构建脚本引用",
        "#                    - 是否include关系尚未建立(如新增文件)",
        "",
    ]
    for r in unref:
        lines.append(f"{r}:")
    os.makedirs(os.path.join(OUTDIR, grp), exist_ok=True)
    open(os.path.join(OUTDIR, grp, "unref.yaml"), "w").write("\n".join(lines) + "\n")
    print(f"零引用 {grp}: {len(unref)} 个 -> impact_map/{grp}/unref.yaml")

# ---------- 8. 摘要 ----------
print(f"仓: {REPO}  commit: {commit} ({branch})")
print(
    f"common/3rd 源文件: {len(targets)}, 重名组: {len(dup_groups)}, 零引用.cpp继承同名.h: {inh}"
)
top = sorted(targets, key=lambda r: -len(impact[r]))[:5]
print(
    "引用算子数 Top5: "
    + "; ".join(f"{os.path.basename(r)}({len(impact[r])})" for r in top)
)
for c, m, e, out in summary:
    print(f"输出 {out}: 有映射 {m} / 空 {e}")
