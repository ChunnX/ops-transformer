#!/usr/bin/env python3
import re
from pathlib import Path

def process_file(file_path):
    """处理单个文件，转换 TILING_KEY_IS 为预处理指令"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 检查是否是 if (TILING_KEY_IS(...))
        if_match = re.match(r'^(\s*)if\s*\(\s*TILING_KEY_IS\s*\(\s*([^)]+)\s*\)\s*\)\s*\{?\s*$', line)
        # 检查是否是 } else if (TILING_KEY_IS(...))
        elif_match = re.match(r'^(\s*)}\s*else\s+if\s*\(\s*TILING_KEY_IS\s*\(\s*([^)]+)\s*\)\s*\)\s*\{?\s*$', line)

        if if_match:
            indent = if_match.group(1)
            key_value = if_match.group(2).strip()
            new_lines.append(f'{indent}#if TILING_KEY_VAR == {key_value}')
            i += 1
        elif elif_match:
            indent = elif_match.group(1)
            key_value = elif_match.group(2).strip()
            new_lines.append(f'{indent}#elif TILING_KEY_VAR == {key_value}')
            i += 1
        else:
            new_lines.append(line)
            i += 1

    # 现在需要添加 #endif
    # 我们需要找到 #if/#elif 链，并在它们对应的 } 后添加 #endif
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
    # 查找所有 op_kernel 目录下的 cpp 文件
    base_dir = Path('.')
    cpp_files = []

    for op_kernel_dir in base_dir.rglob('op_kernel'):
        if op_kernel_dir.is_dir():
            for cpp_file in op_kernel_dir.glob('*.cpp'):
                cpp_files.append(cpp_file)

    print(f'找到 {len(cpp_files)} 个 op_kernel 目录下的 cpp 文件')

    total_if_count = 0
    for cpp_file in cpp_files:
        try:
            if_count = process_file(cpp_file)
            if if_count > 0:
                print(f'处理: {cpp_file} (转换了 {if_count} 个 TILING_KEY_IS)')
                total_if_count += if_count
        except Exception as e:
            print(f'处理 {cpp_file} 时出错: {e}')

    print(f'\n总共转换了 {total_if_count} 个 TILING_KEY_IS')

if __name__ == '__main__':
    main()
