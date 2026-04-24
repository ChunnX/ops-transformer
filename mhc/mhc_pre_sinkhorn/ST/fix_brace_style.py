#!/usr/bin/env python3
import re
import json
import os

def fix_brace_style(file_path, issues):
    """修复大括号风格问题"""
    base_path = "/data/hey/git/ops-transformer"
    full_path = os.path.join(base_path, file_path)
    
    if not os.path.exists(full_path):
        print(f"文件不存在: {full_path}")
        return 0
    
    with open(full_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 收集需要修复的行号
    lines_to_fix = {}
    for issue in issues:
        line_num = int(issue['line']) - 1  # 转为 0-based 索引
        if line_num < len(lines):
            if line_num not in lines_to_fix:
                lines_to_fix[line_num] = []
            lines_to_fix[line_num].append(issue['issue'])
    
    # 修复每一行
    fixed_count = 0
    for line_num, issue_list in sorted(lines_to_fix.items(), reverse=True):  # 从后往前修复，避免行号变化
        original_line = lines[line_num]
        
        for issue in issue_list:
            # 函数定义：左大括号应该在新行
            if 'function should start a new line' in issue:
                # 检查是否是函数定义（以 { 结尾）
                if original_line.rstrip().endswith('{'):
                    # 将 { 移到下一行
                    lines[line_num] = original_line.rstrip()[:-1].rstrip() + '\n'
                    # 在下一行插入 {
                    indent = len(original_line) - len(original_line.lstrip())
                    lines.insert(line_num + 1, ' ' * indent + '{\n')
                    fixed_count += 1
            
            # TEST_F 宏：左大括号应该在新行
            elif 'left brace should follow the statement' in issue:
                if 'TEST_F' in original_line and original_line.rstrip().endswith('{'):
                    # 将 { 移到下一行
                    lines[line_num] = original_line.rstrip()[:-1].rstrip() + '\n'
                    # 在下一行插入 {
                    indent = len(original_line) - len(original_line.lstrip())
                    lines.insert(line_num + 1, ' ' * indent + '{\n')
                    fixed_count += 1
    
    # 写回文件
    with open(full_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✓ {file_path}: 修复了 {fixed_count} 处大括号风格问题")
    return fixed_count

def main():
    # 读取问题列表
    with open('/data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/docs/brace_style_issues.json', 'r', encoding='utf-8') as f:
        issues = json.load(f)
    
    # 按文件分组
    from collections import defaultdict
    issues_by_file = defaultdict(list)
    for issue in issues:
        issues_by_file[issue['path']].append(issue)
    
    total_fixed = 0
    for file_path, file_issues in issues_by_file.items():
        fixed = fix_brace_style(file_path, file_issues)
        if fixed:
            total_fixed += fixed
    
    print(f"\n总计修复: {total_fixed} 处大括号风格问题")

if __name__ == '__main__':
    main()
