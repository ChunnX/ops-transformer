#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: ./gen_common_impact_yaml.py [ops-transformer仓路径] [输出目录]
      两个参数均可省略: 默认仓=本脚本所在仓, 默认输出=<仓>/mc2/tools/impact_map
功能: 对指定的 ops-transformer 检出(任意时期/任意版本), 扫描 mc2 内全量 #include,
      计算 mc2/common 与 mc2/3rd 头文件的反向影响算子范围(含传递闭包),
      按芯片 A2/A3/A5 输出映射表 + 一份全量回归算子清单。

影响面模型(只对 .h 建模, 最简):
  - 头文件: 被include才生效, 影响面 = 直接+间接include它的算子(不动点迭代, 含环)
  - .cpp / .py / .json / .ini / 其他一切文件: 不建模, 改动一律按全量回归处理
输出:
  <输出目录>/common/{A2,A3,A5}.yaml  头文件 -> 算子列表(显式)
  <输出目录>/3rd/{A2,A3,A5}.yaml
  <输出目录>/full_regression.yaml    全量回归算子清单(按芯片)
"""
import os, re, sys, subprocess, datetime

REPO = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUTDIR = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(REPO, 'mc2', 'tools', 'impact_map')
MC2 = os.path.join(REPO, 'mc2')
SRC_EXT = ('.h', '.hpp', '.cuh')
INC_RE = re.compile(r'#include\s+"([^"]+)"')
SOC_RE = re.compile(r'AddConfig\("([a-z0-9_]+)"')
CHIP_MAP = {'A2': 'ascend910b', 'A3': 'ascend910_93', 'A5': 'ascend950'}
EXCLUDE_OPS = ('common', '3rd', 'tools')

if not os.path.isdir(MC2):
    sys.exit(f'错误: 不是 ops-transformer 仓(找不到 {MC2})')

# ---------- 1. 索引: 建模对象=common/3rd头文件; include扫描源=mc2全部C/C++源文件 ----------
SCAN_EXT = ('.h', '.hpp', '.cuh', '.cpp', '.cc', '.c')
scan_files = {}
for dp, _, fs in os.walk(MC2):
    top = os.path.relpath(dp, MC2).split('/')[0]
    if top == 'tools':
        continue
    for f in fs:
        if f.endswith(SCAN_EXT):
            scan_files[os.path.relpath(os.path.join(dp, f), MC2)] = os.path.join(dp, f)
idx = {r: p for r, p in scan_files.items() if r.endswith(('.h', '.hpp', '.cuh'))}
targets = {r for r in idx if r.startswith(('common/', '3rd/'))}   # 影响源: common/3rd 头文件
by_base = {}
for r in targets:
    by_base.setdefault(os.path.basename(r), []).append(r)
dup_groups = {k: v for k, v in by_base.items() if len(v) > 1}

# ---------- 2. 裸文件名 include 边(重名取并集, 宁多勿漏) ----------
by_base = {}
for r in targets:
    by_base.setdefault(os.path.basename(r), []).append(r)
referers = {r: set() for r in targets}   # 头 -> 直接引用者(任何mc2源文件, 含cpp)
for rel, path in scan_files.items():
    for m in INC_RE.finditer(open(path, errors='ignore').read()):
        for t in by_base.get(os.path.basename(m.group(1)), ()):
            if t != rel:
                referers[t].add(rel)

# ---------- 3. 算子芯片注册(来自 *_def.cpp) ----------
socs = {}
for op in sorted(os.listdir(MC2)):
    if op in EXCLUDE_OPS or not os.path.isdir(os.path.join(MC2, op)):
        continue
    s = set()
    for dp, _, fs in os.walk(os.path.join(MC2, op)):
        for f in fs:
            if f.endswith('_def.cpp'):
                s |= set(SOC_RE.findall(open(os.path.join(dp, f), errors='ignore').read()))
    socs[op] = s
ALL_OPS = set(socs)

# ---------- 4. 影响面: 单一不动点(单调收敛, 与遍历顺序无关) ----------
# 头的影响 = 直接引用它的算子 ∪ 引用者的影响(传递闭包, 天然处理环)
base_ops = {r: {rf.split('/')[0] for rf in referers[r]
                if rf.split('/')[0] not in EXCLUDE_OPS} for r in targets}
impact = {r: set(base_ops[r]) for r in targets}
changed = True
while changed:
    changed = False
    for r in targets:
        new = set(impact[r])
        for rf in referers[r]:
            if rf in targets and impact[rf] - new:
                new |= impact[rf]
                changed = True
        if new != impact[r]:
            impact[r] = new
            changed = True

grp_files_map = {g: sorted(r for r in targets if r.startswith(g + '/'))
                 for g in ('common', '3rd')}

# ---------- 5. git信息 ----------
def git(*a):
    try:
        return subprocess.check_output(['git', '-C', REPO] + list(a),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return '?'
commit = git('rev-parse', '--short', 'HEAD')
branch = git('rev-parse', '--abbrev-ref', 'HEAD')

# ---------- 6. 输出 ----------
os.makedirs(OUTDIR, exist_ok=True)
summary = []
for grp in ('common', '3rd'):
    grp_files = {r for r in targets if r.startswith(grp + '/')}
    for chip_cn, soc in CHIP_MAP.items():
        lines = [
            f'# mc2/{grp} 头文件影响范围映射表 - {chip_cn} ({soc})',
            f'# 生成: {datetime.date.today()}  基线commit: {commit} ({branch})',
            '# 生成器: mc2/tools/gen_common_impact_yaml.py (可对任意时期检出重新生成; 本yaml为生成物, 勿手改)',
            f'# 键路径: 相对mc2目录({grp}/开头), 与PR修改文件列表中mc2/后的段落一致',
            '#',
            '# 影响面: 被include才生效, ops = 直接+间接include该头的算子(传递闭包, 显式全列表)',
            f'# ops: [] = 该头有引用, 但引用算子均未在{chip_cn}注册, 本芯片无需测试',
            '# 不在本表中的头 = 无明确include到算子关系(冗余/仅测试引用), 改动按全量回归处理',
            f'# 算子范围 = 其*_def.cpp的AICore().AddConfig()注册 与 {chip_cn}({soc}) 的交集',
            '# 重名头文件映射取并集(note:字段标注), 宁多勿漏',
            '',
        ]
        n_map = n_empty = 0
        for r in sorted(grp_files):
            ops = {o for o in impact[r] if soc in socs.get(o, set())}
            if not impact[r]:
                continue          # 零算子可达头: 不入表 -> 消费侧按全量处理
            lines.append(f'{r}:')
            if ops:
                lines.append('  ops:')
                for o in sorted(ops):
                    lines.append(f'    - {o}')
                n_map += 1
            else:
                lines.append('  ops: []')
                n_empty += 1
            if os.path.basename(r) in dup_groups:
                lines.append(
                    f'  note: 同名文件{len(dup_groups[os.path.basename(r)])}份, 映射取并集'
                )
        grp_out = os.path.join(OUTDIR, grp)
        os.makedirs(grp_out, exist_ok=True)
        out = os.path.join(grp_out, f'{chip_cn}.yaml')
        open(out, 'w').write('\n'.join(lines) + '\n')
        summary.append((f'{grp}/{chip_cn}', n_map, n_empty, out))

# file_lists.yaml: 全量匹配表(消费侧第一步先查本表, 命中即全量; 未命中再查对应soc的芯片yaml)
unrel_all = {g: sorted(r for r in targets if r.startswith(g + '/') and not impact[r])
             for g in ('common', '3rd')}
lines = [
    '# mc2 全量匹配表(消费侧第一步先查本表)',
    f'# 生成: {datetime.date.today()}  基线commit: {commit} ({branch})',
    '# 用途: 对每个修改文件, 先在本表匹配:',
    '#   1) 扩展名命中 full_regression.non_header_extensions -> 全量',
    '#   2) 路径命中 full_regression.headers_no_operator_relation -> 全量',
    '# 均未命中 -> 改到对应soc的 {A2,A3,A5}.yaml 查映射(有算子关系的头全在其中)取ops',
    '# 最终算子集需叠加CMake depends反向闭包后去重',
    '',
    'full_regression:',
    '  non_header_extensions:',
]
for ext in ('.cpp', '.cc', '.c', '.py', '.json', '.ini', '.sh', '.csv', '.txt', '.md'):
    lines.append(f'    - "{ext}"')
lines.append('  headers_no_operator_relation:')
for g in ('common', '3rd'):
    lines.append(f'    {g}:')
    for r in unrel_all[g]:
        lines.append(f'      - {r}')
os.makedirs(os.path.join(OUTDIR, '3rd'), exist_ok=True)
open(os.path.join(OUTDIR, '3rd', 'file_lists.yaml'), 'w').write('\n'.join(lines) + '\n')
print(f'输出 {os.path.join(OUTDIR, "3rd", "file_lists.yaml")}: '
      f'无关系头 common {len(unrel_all["common"])} / 3rd {len(unrel_all["3rd"])}')

# ---------- 7. 摘要 ----------
print(f'仓: {REPO}  commit: {commit} ({branch})')
print(f'common/3rd 头文件: {len(targets)}, 重名组: {len(dup_groups)}')
top = sorted(targets, key=lambda r: -len(impact[r]))[:5]
print('引用算子数 Top5: ' + '; '.join(f'{os.path.basename(r)}({len(impact[r])})' for r in top))
for c, m, e, out in summary:
    print(f'输出 {out}: 有映射 {m} / 该芯片无算子 {e}')
