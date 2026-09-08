#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法: ./gen_common_impact_yaml.py [ops-transformer仓路径] [输出目录]
      两个参数均可省略: 默认仓=本脚本所在仓, 默认输出=<仓>/mc2/tools/impact_map
功能: 对指定的 ops-transformer 检出(任意时期/任意版本), 统计 mc2/common 与 mc2/3rd
      每个文件的反向影响算子范围, 按目录(common/3rd)×芯片(A2/A3/A5)输出6个yaml。

影响面模型:
  1) 头文件(.h): 被 include 才生效。影响面 = 直接+间接 include 它的算子
     (沿 common/3rd 内部引用链传递闭包, 不动点迭代, 天然处理环)。
  2) 编译级引用(.cpp): 由 CMake file(GLOB)/add_modules_sources(DIR) 按目录整包
     编译进全局目标, 无需被 include。影响面 = 同名头的接口使用方算子(接口反推),
     即"改了实现, 只有调用该接口的算子受影响"。
     - 无同名头的 .cpp: 接口不明, 保守回退全算子, note 标注待人工确认
     - 同名头无任何算子引用的 .cpp: 疑似死代码, ops:[] + note 待人工确认
输出: <输出目录>/common/{A2,A3,A5,zero_reference}.yaml 与 <输出目录>/3rd/{...}
      零引用文件(无include且非编译单元)单独写入 zero_reference.yaml。
说明: 重名头文件的映射取并集(宁多勿漏), 相关文件带 note 标注; 重新运行覆盖输出。
"""
import os, re, sys, subprocess, datetime

REPO = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUTDIR = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(REPO, 'mc2', 'tools', 'impact_map')
MC2 = os.path.join(REPO, 'mc2')
SRC_EXT = ('.h', '.hpp', '.cuh', '.cpp', '.cc', '.c')
CHIP_MAP = {'A2': 'ascend910b', 'A3': 'ascend910_93', 'A5': 'ascend950'}
INC_RE = re.compile(r'#include\s+"([^"]+)"')
SOC_RE = re.compile(r'AddConfig\("([a-z0-9_]+)"')
EXCLUDE_OPS = ('common', '3rd', 'tools')

if not os.path.isdir(MC2):
    sys.exit(f'错误: 不是 ops-transformer 仓(找不到 {MC2})')

# ---------- 1. 索引 mc2 下所有源文件 ----------
idx = {}
for dp, _, fs in os.walk(MC2):
    top = os.path.relpath(dp, MC2).split('/')[0]
    if top == 'tools':
        continue
    for f in fs:
        if f.endswith(SRC_EXT):
            idx[os.path.relpath(os.path.join(dp, f), MC2)] = os.path.join(dp, f)

targets = {r for r in idx if r.startswith(('common/', '3rd/'))}
by_base = {}
for r in targets:
    by_base.setdefault(os.path.basename(r), []).append(r)
dup_groups = {k: v for k, v in by_base.items() if len(v) > 1}

# ---------- 2. 裸文件名 include 边 ----------
referers = {r: set() for r in targets}      # 目标文件 -> 直接引用者
for rel, path in idx.items():
    for m in INC_RE.finditer(open(path, errors='ignore').read()):
        for t in by_base.get(os.path.basename(m.group(1)), ()):
            if t != rel:
                referers[t].add(rel)

def is_test(p):
    return any(s in ('tests', 'test') for s in p.split('/'))

compiled = {r for r in targets
            if r.endswith(('.cpp', '.cc', '.c')) and not is_test(r)}

# ---------- 3. 不动点求值(单调收敛, 与遍历顺序无关) ----------
base_ops = {r: {rf.split('/')[0] for rf in referers[r]
                if rf.split('/')[0] not in EXCLUDE_OPS}
            for r in targets}

def fixpoint(seeds):
    """seeds: 编译级cpp的固定影响集; 其余节点按引用链聚合, 迭代到稳定"""
    imp = {r: (set(seeds[r]) if r in seeds else set(base_ops[r]))
           for r in targets}
    changed = True
    while changed:
        changed = False
        for r in targets:
            if r in seeds:
                continue
            new = set(imp[r])
            for rf in referers[r]:
                if rf in targets:
                    if imp[rf] - new:
                        new |= imp[rf]
                        changed = True
            if new != imp[r]:
                imp[r] = new
                changed = True
    return imp

# 3.1 纯头文件世界: 编译级cpp不传播(其对同名头的自引用不算使用方)
imp_pure = fixpoint(seeds={})

# 3.2 编译级cpp影响面 = 同名头接口使用方; 无同名头回退全算子
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

cpp_seed, cpp_note = {}, {}
for c in sorted(compiled):
    h = c.rsplit('.', 1)[0] + '.h'
    if h in targets:
        cpp_seed[c] = imp_pure[h]
        cpp_note[c] = f'编译级引用, 影响面按同名头{h}接口反推'
    else:
        cpp_seed[c] = set(ALL_OPS)
        cpp_note[c] = '编译级引用, 无同名头, 按全算子处理'

# 3.3 头文件最终影响面: cpp作为使用方以其seed贡献回去
impact = fixpoint(seeds=cpp_seed)
for c in compiled:
    impact[c] = cpp_seed[c]

# ---------- 4. 零引用.cpp继承同名头(编译级模型下不再需要, 保留兜底) ----------
# ---------- 5. git信息 ----------
def git(*a):
    try:
        return subprocess.check_output(['git', '-C', REPO] + list(a),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return '?'
commit = git('rev-parse', '--short', 'HEAD')
branch = git('rev-parse', '--abbrev-ref', 'HEAD')

# ---------- 6. 输出 yaml: 目录(common/3rd) x 芯片(A2/A3/A5) = 6个文件 + unref ----------
os.makedirs(OUTDIR, exist_ok=True)
summary = []
for grp in ('common', '3rd'):
    grp_files = {r for r in targets if r.startswith(grp + '/')}
    for chip_cn, soc in CHIP_MAP.items():
        lines = [
            f'# mc2/{grp} 头文件影响范围映射表 - {chip_cn} ({soc})',
            f'# 生成: {datetime.date.today()}  基线commit: {commit} ({branch})',
            '# 生成器: mc2/tools/gen_common_impact_yaml.py (可对任意时期检出重新生成; 本yaml为生成物, 勿手改)',
            f'# 键路径: 相对mc2目录({grp}/开头), 与pr_files.sh输出中mc2/后的段落一致',
            '#',
            '# 本表仅保存头文件(.h/.hpp/.cuh)条目, 影响面=直接+间接include它的算子(传递闭包)。',
            '# 其他文件类型由消费脚本按以下规则处理(与本表配合):',
            '#   .cpp: 同目录同名.h存在 -> 查本表同名头条目; 同名.h在zero_reference.yaml -> 疑似死代码不触发;',
            '#         同目录无同名.h -> 全量',
            '#   其余一切(.py/json/ini/sh/csv/txt/md等): 一律全量',
            f'#   算子范围 = 其*_def.cpp的AICore().AddConfig()注册 与 {chip_cn}({soc}) 的交集',
            '#   注意: 本表结果为文件直接传递的算子, 消费侧还需叠加CMake depends反向闭包',
            '',
        ]
        n_map = n_empty = 0
        for r in sorted(grp_files):
            if not r.endswith(('.h', '.hpp', '.cuh')):
                continue
            ops = {o for o in impact[r] if soc in socs.get(o, set())}
            if not impact[r]:
                continue
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
    # unref
    unref = sorted(r for r in grp_files if r.endswith(('.h', '.hpp', '.cuh'))
                   and not impact[r])
    lines = [
        f'# mc2/{grp} 零引用文件清单 (与芯片无关, 全芯片共用)',
        f'# 生成: {datetime.date.today()}  基线commit: {commit} ({branch})',
        '#',
        '# 语义: 下列头文件无任何算子可达的引用(无人直接/间接include),',
        '#       改动不影响mc2算子编译; .cpp的判定见映射表头部规则(同名.h在zero_reference=死代码);',
        '#       注意与芯片映射表中 ops: [] 的区别:',
        '#         ops: []  = 头文件无算子引用 / .cpp同名头无算子引用(疑似死代码, 看note)',
        '#         本表     = 完全没有引用, 修改前需人工确认:',
        '#                    - 是否被仓外代码/构建脚本引用',
        '#                    - 是否include关系尚未建立(如新增文件)',
        '',
    ]
    for r in unref:
        lines.append(f'{r}:')
    open(os.path.join(OUTDIR, grp, 'zero_reference.yaml'), 'w').write('\n'.join(lines) + '\n')
    print(f'零引用 {grp}: {len(unref)} 个 -> impact_map/{grp}/zero_reference.yaml')

# ---------- 7. 摘要 ----------
print(f'仓: {REPO}  commit: {commit} ({branch})')
print(
    f'common/3rd 源文件: {len(targets)}, 重名组: {len(dup_groups)}, '
    f'编译级.cpp: {len(compiled)} (yaml不保存, 消费侧按同名头规则处理)'
)
top = sorted(targets, key=lambda r: -len(impact[r]))[:5]
print(
    '引用算子数 Top5: '
    + '; '.join(f'{os.path.basename(r)}({len(impact[r])})' for r in top)
)
for c, m, e, out in summary:
    print(f'输出 {out}: 头文件条目 {m} / 该芯片无算子 {e}')
n_cpp = len(compiled)
n_noh = sum(1 for c in compiled if c.rsplit(".", 1)[0] + ".h" not in targets)
print(f'编译级.cpp {n_cpp} 个不在yaml中(消费规则): 有同名头 {n_cpp - n_noh} / 无同名头走全量 {n_noh}')
