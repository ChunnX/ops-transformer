#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: ./gen_common_impact_yaml.py [ops-transformer仓路径] [输出目录]
      两个参数均可省略: 默认仓=本脚本所在仓, 默认输出=<仓>/mc2/tools/impact_map
功能: 对指定的 ops-transformer 检出(任意时期/任意版本), 扫描 mc2 内全量 #include,
      将 mc2/common 与 mc2/3rd 的每个文件归入两类之一:
        A) 测部分算子: 有include到算子关系、且非全算子使用的 .h
           -> 按芯片 A2/A3/A5 输出映射表(三张表键集合一致, 按芯片取算子子集)
        B) 测全量: 以下三种情况, 统一放入 full_regression.yaml
           1) .h 被全部算子使用
           2) .h 无任何文件引用
           3) 非 .h 文件(.cpp/.py/.json/.ini/.sh/.csv/.txt/.md 等, 按扩展名模式)
输出:
  <输出目录>/common/{A2,A3,A5}.yaml  部分算子映射(键集合三芯片一致)
  <输出目录>/3rd/{A2,A3,A5}.yaml
  <输出目录>/full_regression.yaml    全量清单(按芯片的全量算子列表 + 三种全量文件分类)
覆盖性保证: 每个common/3rd文件恰好归入A类或B类之一, A2/A3/A5三张表键集合一致。
"""
import os, re, sys, subprocess, datetime

REPO = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUTDIR = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(REPO, 'mc2', 'tools', 'impact_map')
MC2 = os.path.join(REPO, 'mc2')
SCAN_EXT = ('.h', '.hpp', '.cuh', '.cpp', '.cc', '.c')
HDR_EXT = ('.h', '.hpp', '.cuh')
INC_RE = re.compile(r'#include\s+"([^"]+)"')
SOC_RE = re.compile(r'AddConfig\("([a-z0-9_]+)"')
CHIP_MAP = {'A2': 'ascend910b', 'A3': 'ascend910_93', 'A5': 'ascend950'}
EXCLUDE_OPS = ('common', '3rd', 'tools')
NON_HDR_EXT = ('.py', '.json', '.ini', '.sh', '.csv', '.txt', '.md')

if not os.path.isdir(MC2):
    sys.exit(f'错误: 不是 ops-transformer 仓(找不到 {MC2})')

# ---------- 1. 索引: 扫描源=mc2全部C/C++源文件; 建模对象=common/3rd头文件 ----------
scan_files = {}
hdrs = {}
for dp, _, fs in os.walk(MC2):
    top = os.path.relpath(dp, MC2).split('/')[0]
    if top == 'tools':
        continue
    for f in fs:
        rel = os.path.relpath(os.path.join(dp, f), MC2)
        if f.endswith(SCAN_EXT):
            scan_files[rel] = os.path.join(dp, f)
        if f.endswith(HDR_EXT) and rel.startswith(('common/', '3rd/')):
            hdrs[rel] = os.path.join(dp, f)

# ---------- 2. 裸文件名 include 边(重名取并集, 宁多勿漏) ----------
by_base = {}
for h in hdrs:
    by_base.setdefault(os.path.basename(h), []).append(h)
DUP_GROUPS = {}
_by = {}
for h in hdrs:
    _by.setdefault(os.path.basename(h), []).append(h)
DUP_GROUPS = {k: v for k, v in _by.items() if len(v) > 1}
referers = {h: set() for h in hdrs}   # 头 -> 直接引用者(任何mc2源文件)
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
base_ops = {h: {rf.split('/')[0] for rf in referers[h]
                if rf.split('/')[0] not in EXCLUDE_OPS} for h in hdrs}
impact = {h: set(base_ops[h]) for h in hdrs}
changed = True
while changed:
    changed = False
    for h in hdrs:
        new = set(impact[h])
        for rf in referers[h]:
            if rf in hdrs and impact[rf] - new:
                new |= impact[rf]
                changed = True
        if new != impact[h]:
            impact[h] = new
            changed = True

# ---------- 5.5 CMake *_depends 反向闭包: 算子级依赖烙进最终影响面 ----------
DEP_RE = re.compile(r'set\(\w+_depends\s+([^)]*)\)')
dep_dependents = {}
for op in sorted(os.listdir(MC2)):
    if op in EXCLUDE_OPS or not os.path.isdir(os.path.join(MC2, op)):
        continue
    for dp, _, fs in os.walk(os.path.join(MC2, op)):
        for f in fs:
            if f != 'CMakeLists.txt':
                continue
            for m in DEP_RE.finditer(open(os.path.join(dp, f), errors='ignore').read()):
                for t in m.group(1).split():
                    if t.startswith('mc2/') and t.split('/')[1] in socs and t.split('/')[1] != op:
                        dep_dependents.setdefault(t.split('/')[1], set()).add(op)

def dep_closure(seeds):
    out = set(seeds)
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        for dp_ in dep_dependents.get(cur, ()):
            if dp_ not in out:
                out.add(dp_)
                stack.append(dp_)
    return out


# ---------- 5. 分类: 全量(三种) / 部分算子 ----------
used_by_all = set()    # ① 被全部算子使用
no_reference = set()   # ② 无任何文件引用
partial = set()        # 其余: 影响部分算子
for h in hdrs:
    if not referers[h]:
        no_reference.add(h)
    elif dep_closure(impact[h]) >= ALL_OPS:
        used_by_all.add(h)
    else:
        partial.add(h)

# ---------- 6. git信息 ----------
def git(*a):
    try:
        return subprocess.check_output(['git', '-C', REPO] + list(a),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return '?'
commit = git('rev-parse', '--short', 'HEAD')
branch = git('rev-parse', '--abbrev-ref', 'HEAD')

# ---------- 7. 输出: 芯片映射表(键集合三芯片一致) + full_regression.yaml ----------
os.makedirs(OUTDIR, exist_ok=True)
summary = []
for grp in ('common', '3rd'):
    partial_g = sorted(h for h in partial if h.startswith(grp + '/'))
    for chip_cn, soc in CHIP_MAP.items():
        lines = [
            f'# mc2/{grp} 头文件影响范围映射表 - {chip_cn} ({soc})',
            f'# 生成: {datetime.date.today()}  基线commit: {commit} ({branch})',
            '# 生成器: mc2/tools/gen_common_impact_yaml.py (可对任意时期检出重新生成; 本yaml为生成物, 勿手改)',
            f'# 键路径: 相对mc2目录({grp}/开头), 与PR修改文件列表中mc2/后的段落一致',
            '#',
            '# 本表 = 影响部分算子的头文件(键集合与A2/A3/A5三张表完全一致)。',
            '#   ops = 直接+间接include该头的算子中, 注册于本芯片的子集(显式全列表)',
            f'#   ops: [] = 该头影响的算子均未在{chip_cn}注册(该芯片无匹配算子, 后续策略待定)',
            '# 不在本表中的文件 -> 全量(清单见本目录file_lists.yaml)',
            f'# 算子范围 = 其*_def.cpp的AICore().AddConfig()注册 与 {chip_cn}({soc}) 的交集',
            '',
        ]
        for h in partial_g:
            ops = sorted(o for o in dep_closure(impact[h]) if soc in socs.get(o, set()))
            lines.append(f'{h}:')
            if ops:
                lines.append('  ops:')
                for o in ops:
                    lines.append(f'    - {o}')
            else:
                lines.append('  ops: []')
            if os.path.basename(h) in DUP_GROUPS:
                lines.append(
                    f'  note: 同名文件{len(DUP_GROUPS[os.path.basename(h)])}份, 映射取并集'
                )
        grp_out = os.path.join(OUTDIR, grp)
        os.makedirs(grp_out, exist_ok=True)
        out = os.path.join(grp_out, f'{chip_cn}.yaml')
        open(out, 'w').write('\n'.join(lines) + '\n')
        summary.append((f'{grp}/{chip_cn}', len(partial_g), out))

# file_lists.yaml: 全量匹配清单(按grp分放, 消费侧先查本表; 未命中再查对应soc的芯片yaml)
cpp_no_same = {}
for grp in ('common', '3rd'):
    cpp_no_same[grp] = sorted(
        r for r in scan_files
        if r.startswith(grp + '/') and r.endswith(('.cpp', '.cc', '.c'))
        and (r.rsplit('.', 1)[0] + '.h') not in hdrs
    )
for grp in ('common', '3rd'):
    used_g = sorted(h for h in used_by_all if h.startswith(grp + '/'))
    noref_g = sorted(h for h in no_reference if h.startswith(grp + '/'))
    lines = [
        f'# mc2/{grp} 全量匹配清单(消费侧第一步先查本表)',
        f'# 生成: {datetime.date.today()}  基线commit: {commit} ({branch})',
        '# 用途: 对每个修改文件, 先在本表匹配:',
        '#   1) 扩展名命中 full_regression.non_header_extensions -> 全量',
        '#   2) cpp路径命中 full_regression.cpp_no_same_header(无同名.h) -> 全量',
        '#   3) 头路径命中 full_regression.headers_used_by_all_operators -> 全量',
        '#   4) 头路径命中 full_regression.headers_no_reference -> 全量',
        f'# 其余.h属"测部分算子"类, 改查 {{A2,A3,A5}}.yaml 取ops; cpp先找同目录同名.h, 按.h规则走',
        '# 最终算子集需叠加CMake depends反向闭包后去重',
        '',
        'full_regression:',
        '  non_header_extensions:',
    ]
    for ext in NON_HDR_EXT:
        lines.append(f'    - "{ext}"')
    lines.append('  cpp_no_same_header:')
    for r in cpp_no_same[grp]:
        lines.append(f'      - {r}')
    lines.append('  headers_used_by_all_operators:')
    for h in used_g:
        lines.append(f'      - {h}')
    lines.append('  headers_no_reference:')
    for h in noref_g:
        lines.append(f'      - {h}')
    os.makedirs(os.path.join(OUTDIR, grp), exist_ok=True)
    open(os.path.join(OUTDIR, grp, 'file_lists.yaml'), 'w').write('\n'.join(lines) + '\n')
    print(f'输出 {os.path.join(OUTDIR, grp, "file_lists.yaml")}: '
          f'全算子使用 {len(used_g)} / 无引用 {len(noref_g)}')

# ---------- 8. 覆盖性自检 ----------
n_hdr = len(hdrs)
covered = len(used_by_all) + len(no_reference) + len(partial)
print(f'仓: {REPO}  commit: {commit} ({branch})')
print(f'common/3rd 头文件: {n_hdr} = 全算子使用 {len(used_by_all)} + 无引用 {len(no_reference)} '
      f'+ 部分算子 {len(partial)}  (覆盖: {covered}/{n_hdr})')
print(f'非.h文件: 按扩展名模式 {NON_HDR_EXT} 归入全量, 不逐个枚举')
grp_check = {}
for grp in ('common', '3rd'):
    keysets = []
    for chip_cn in CHIP_MAP:
        d = open(os.path.join(OUTDIR, grp, f'{chip_cn}.yaml')).read()
        keysets.append({l[:-1] for l in d.splitlines() if l and not l.startswith((' ', '#'))})
    same = keysets[0] == keysets[1] == keysets[2]
    grp_check[grp] = (len(keysets[0]), same)
    print(f'{grp}: 三芯片键集合一致={same}, 条目数={len(keysets[0])}')
