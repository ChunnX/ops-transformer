#!/usr/bin/env python3
import re
import json
import os

def split_long_line(line, max_length=120):
    """智能分割长行"""
    if len(line.rstrip()) <= max_length:
        return [line]
    
    # 获取缩进
    indent = len(line) - len(line.lstrip())
    indent_str = ' ' * indent
    
    # 去除首尾空格
    content = line.strip()
    
    # 策略1: 在逗号后分割（适用于函数调用、列表等）
    if ',' in content:
        # 找到合适的分割点
        parts = []
        current = ""
        depth = 0  # 括号深度
        
        for i, char in enumerate(content):
            current += char
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif char == ',' and depth == 0:
                # 在逗号后分割
                if len(indent_str + current) > max_length and parts:
                    # 当前行太长，需要分割
                    parts.append(current.rstrip())
                    current = ""
                elif len(indent_str + current) > max_length * 0.8:  # 超过80%就分割
                    parts.append(current.rstrip())
                    current = ""
        
        if current:
            parts.append(current.rstrip())
        
        if len(parts) > 1:
            # 重新格式化
            result = []
            for i, part in enumerate(parts):
                if i == 0:
                    result.append(indent_str + part + '\n')
                else:
                    # 增加额外的缩进
                    result.append(indent_str + '    ' + part + '\n')
            return result
    
    # 策略2: 在操作符处分割
    operators = [' + ', ' - ', ' * ', ' / ', ' = ', ' += ', ' -= ', ' && ', ' || ', ' < ', ' > ', ' <= ', ' >= ']
    for op in operators:
        if op in content:
            parts = content.split(op, 1)  # 只在第一个操作符处分割
            if len(parts) == 2:
                return [
                    indent_str + parts[0] + op.rstrip() + '\n',
                    indent_str + '    ' + parts[1].lstrip() + '\n'
                ]
    
    # 策略3: 在空格处分割
    if ' ' in content:
        words = content.split()
        result = []
        current_line = indent_str
        
        for word in words:
            if len(current_line + ' ' + word) > max_length:
                if current_line != indent_str:
                    result.append(current_line + '\n')
                current_line = indent_str + '    ' + word
            else:
                if current_line == indent_str:
                    current_line += word
                else:
                    current_line += ' ' + word
        
        if current_line != indent_str:
            result.append(current_line + '\n')
        
        if len(result) > 1:
            return result
    
    # 无法分割，返回原行
    return [line]

def fix_line_length(file_path, issues):
    """修复行长度问题"""
    base_path = "/data/hey/git/ops-transformer"
    full_path = os.path.join(base_path, file_path)
    
    if not os.path.exists(full_path):
        print(f"文件不存在: {full_path}")
        return 0
    
    with open(full_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 收集需要修复的行号
    lines_to_fix = set()
    for issue in issues:
        line_num = int(issue['line']) - 1  # 转为 0-based 索引
        if line_num < len(lines):
            lines_to_fix.add(line_num)
    
    # 修复每一行
    fixed_count = 0
    new_lines = []
    
    for i, line in enumerate(lines):
        if i in lines_to_fix and len(line.rstrip()) > 120:
            split_lines = split_long_line(line)
            if len(split_lines) > 1:
                new_lines.extend(split_lines)
                fixed_count += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # 写回文件
    with open(full_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✓ {file_path}: 修复了 {fixed_count} 处行长度问题")
    return fixed_count

def main():
    # 读取问题列表
    with open('/data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/docs/line_length_issues.json', 'r', encoding='utf-8') as f:
        issues = json.load(f)
    
    # 按文件分组
    from collections import defaultdict
    issues_by_file = defaultdict(list)
    for issue in issues:
        issues_by_file[issue['path']].append(issue)
    
    total_fixed = 0
    for file_path, file_issues in sorted(issues_by_file.items()):
        fixed = fix_line_length(file_path, file_issues)
        if fixed:
            total_fixed += fixed
    
    print(f"\n总计修复: {total_fixed} 处行长度问题")

if __name__ == '__main__':
    main()
