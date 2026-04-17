#!/usr/bin/env python3
import re
from pathlib import Path

def find_if_else_chains(lines):
    """找到所有的 if/else if 链，并返回它们的信息"""
    chains = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 检查是否是 if (TILING_KEY_IS(...))
        if_match = re.match(r'^(\s*)if\s*\(\s*TILING_KEY_IS\s*\(\s*([^)]+)\s*\)\s*\)\s*\{?\s*$', line)

        if if_match:
            indent = if_match.group(1)
            chain = {
                'start': i,
                'indent': indent,
                'conditions': [{'line': i, 'value': if_match.group(2).strip()}],
                'has_or': False
            }

            # 检查是否有 ||
            if '||' in line:
                chain['has_or'] = True

            # 继续查找 else if
            i += 1
            while i < len(lines):
                next_line = lines[i]

                # 检查是否是 } else if (TILING_KEY_IS(...))
                elif_match = re.match(r'^(\s*)}\s*else\s+if\s*\(\s*TILING_KEY_IS\s*\(\s*([^)]+)\s*\)\s*\)\s*\{?\s*$',next_line)

                if elif_match and elif_match.group(1) == indent:
                    chain['conditions'].append({'line': i, 'value': elif_match.group(2).strip()})
                    if '||' in next_line:
                        chain['has_or'] = True
                    i += 1
                else:
                    break

            chain['end'] = i - 1
            chains.append(chain)
        else:
            i += 1

    return chains

def process_file(file_path):
    """处理单个文件，转换 TILING_KEY_IS 为预处理指令"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    # 找到所有的 if/else if 链
    chains = find_if_else_chains(lines)

    # 只转换没有 || 的链
    lines_to_replace = []
    for chain in chains:
        if not chain['has_or']:
            for cond in chain['conditions']:
                lines_to_replace.append(cond['line'])

    # 创建新的行列表
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if i in lines_to_replace:
            # 检查是否是 if (TILING_KEY_IS(...))
            if_match = re.match(r'^(\s*)if\s*\(\s*TILING_KEY_IS\s*\(\s*([^)]+)\s*\)\s*\)\s*\{?\s*$', line)
            # 检查是否是 } else if (TILING_KEY_IS(...))
            elif_match = re.match(r'^(\s*)}\s*else\s+if\s*\(\s*TILING_KEY_IS\s*\(\s*([^)]+)\s*\)\s*\)\s*\{?\s*$', line)

            if if_match:
                indent = if_match.group(1)
                key_value = if_match.group(2).strip()
                new_lines.append(f'{indent}#if TILING_KEY_VAR == {key_value}')
            elif elif_match:
                indent = elif_match.group(1)
                key_value = elif_match.group(2).strip()
                new_lines.append(f'{indent}#elif TILING_KEY_VAR == {key_value}')
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

        i += 1

    # 现在需要添加 #endif
    final_lines = []
    i = 0
    if_stack = []

    while i < len(new_lines):
        new_line = new_lines[i]

        # 检查 #if
        if re.match(r'^(\s*)#if\s+TILING_KEY_VAR\s*==', new_line):
            if_stack.append(i)
            final_lines.append(new_line)
            i += 1
        # 检查 #elif
        elif re.match(r'^(\s*)#elif\s+TILING_KEY_VAR\s*==', new_line):
            final_lines.append(new_line)
            i += 1
        # 检查 } (可能是 if-else 链的结束)
        elif re.match(r'^(\s*)}\s*$', new_line) and if_stack:
            # 找到对应的 #if 的缩进
            if_idx = if_stack[-1]
            if_line = new_lines[if_idx]
            if_indent = re.match(r'^(\s*)', if_line).group(1)

            # 检查当前 } 的缩进是否匹配
            current_indent = re.match(r'^(\s*)', new_line).group(1)

            if current_indent == if_indent:
                # 这是一个 if-else 链的结束，添加 #endif
                final_lines.append(new_line)
                final_lines.append(f'{if_indent}#endif')
                if_stack.pop()
            else:
                final_lines.append(new_line)
            i += 1
        else:
            final_lines.append(new_line)
            i += 1

    # 写回文件
    new_content = '\n'.join(final_lines)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return len([line for line in new_lines if re.match(r'^\s*#if\s+TILING_KEY_VAR\s*==', line)])

def main():
    # 测试单个文件
    test_file = Path('matmul_all_reduce_add_rms_norm/op_kernel/matmul_all_reduce_add_rms_norm.cpp')

    if test_file.exists():
        if_count = process_file(test_file)
        print(f'处理: {test_file} (转换了 {if_count} 个 TILING)')
    else:
        print(f'文件不存在: {test_file}')

if __name__ == '__main__':
    main()
