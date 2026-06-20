#!/usr/bin/env python3
# 用法: python inject_anchors.py fa_arch35_design_v2.md
import re, sys

def header_id(text):
    t = text.strip()
    m = re.match(r'^第([零一二三四五])篇', t)
    if m: return f'sec-part{"零一二三四五".index(m.group(1))}'
    m = re.match(r'^附录\s*([A-Za-z])', t)
    if m: return f'sec-appendix-{m.group(1).lower()}'
    m = re.match(r'^([A-Za-z])\.(\d+(?:\.\d+)*)\s', t)         # A.1 / B.1.1
    if m: return 'sec-' + m.group(1).lower() + '-' + m.group(2).replace('.', '-')
    m = re.match(r'^(\d+(?:\.\d+)*)\.?\s', t)                   # 1. / 2.5 / 6.4.1
    if m: return 'sec-' + m.group(1).replace('.', '-')
    for k, v in {'文档信息与基线常量':'sec-front', '目录':'sec-toc',
                 '版本历史':'sec-front-version', '术语表':'sec-front-glossary',
                 '记号表':'sec-front-notation', '基线常量表':'sec-front-baseline',
                 '运行示例':'sec-front-shape'}.items():
        if t.startswith(k): return v
    return None

path = sys.argv[1]
lines = open(path, encoding='utf-8').read().split('\n')
out, in_fence = [], False
hdr = re.compile(r'^(#{2,4})\s+(.*)$')
for line in lines:
    if line.lstrip().startswith('```'):
        in_fence = not in_fence
    if not in_fence:
        m = hdr.match(line)
        if m:
            hid = header_id(m.group(2))
            if hid and not (out and out[-1].strip() == f'<a id="{hid}"></a>'):
                out.append(f'<a id="{hid}"></a>')
    out.append(line)
open(path, 'w', encoding='utf-8').write('\n'.join(out))
print('anchors injected')