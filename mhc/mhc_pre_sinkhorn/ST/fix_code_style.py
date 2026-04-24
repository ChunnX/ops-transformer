#!/usr/bin/env python3
import re
import json
import os

def fix_file(file_path, issues):
    """修复单个文件的问题"""
    base_path = "/data/hey/git/ops-transformer"
    full_path = os.path.join(base_path, file_path)
    
    if not os.path.exists(full_path):
        print(f"文件不存在: {full_path}")
        return
    
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
    for line_num, issue_list in sorted(lines_to_fix.items()):
        original_line = lines[line_num]
        fixed_line = original_line
        
        for issue in issue_list:
            # 1. 移除行尾空格
            if 'Do not add spaces at the end of each line' in issue:
                fixed_line = fixed_line.rstrip() + '\n'
            
            # 2. 二元操作符前后添加空格
            elif 'Need add a space before and after binary operator' in issue:
                # 修复 = 操作符
                fixed_line = re.sub(r'(\S)=(\S)', r'\1 = \2', fixed_line)
            
            # 3. 移除右括号前的空格
            elif "Do not add spaces before the right parenthesis" in issue:
                fixed_line = re.sub(r'\s+\)', ')', fixed_line)
            
            # 4. 大括号前添加空格
            elif 'There is no space before curly bracket' in issue:
                fixed_line = re.sub(r'(\S)\{', r'\1 {', fixed_line)
            
            # 5. 注释格式修复
            elif 'There must be 1 space between the comment character' in issue:
                # 修复 // 注释
                fixed_line = re.sub(r'//(\S)', r'// \1', fixed_line)
                fixed_line = re.sub(r'//\s{2,}', r'// ', fixed_line)
            
            # 6. 代码后注释前添加空格
            elif 'Must have at least 1 space between the right comment' in issue:
                # 在代码后的注释前添加空格
                fixed_line = re.sub(r'(\S)//', r'\1 //', fixed_line)
            
            # 7. Tab 替换为空格
            elif "Avoid using 'Tab' for indentation" in issue or 'Use spaces to indent' in issue:
                fixed_line = fixed_line.replace('\t', '    ')
            
            # 8. 移除多余空行（在代码块末尾）
            elif 'Do not add blank lines at the end of a code block' in issue:
                # 检查下一行是否是 }
                if line_num + 1 < len(lines) and lines[line_num + 1].strip() == '}':
                    fixed_line = ''  # 移除空行
            
            # 9. 移除函数内的连续空行
            elif 'Do not put two or more continuous blank lines' in issue:
                # 检查是否是连续空行
                if fixed_line.strip() == '':
                    # 向前查找非空行
                    prev_non_empty = line_num - 1
                    while prev_non_empty >= 0 and lines[prev_non_empty].strip() == '':
                        prev_non_empty -= 1
                    # 如果前面有空行，移除当前空行
                    if prev_non_empty < line_num - 1:
                        fixed_line = ''
            
            # 10. 移除代码块开始的空行
            elif 'Do not add blank lines on the start of a code block' in issue:
                # 检查上一行是否是 {
                if line_num > 0 and lines[line_num - 1].strip() == '{':
                    fixed_line = ''  # 移除空行
            
            # 11. 移除类型定义中的连续空行
            elif 'Do not put two or more continuous blank lines inside type definition' in issue:
                if fixed_line.strip() == '':
                    # 向前查找非空行
                    prev_non_empty = line_num - 1
                    while prev_non_empty >= 0 and lines[prev_non_empty].strip() == '':
                        prev_non_empty -= 1
                    if prev_non_empty < line_num - 1:
                        fixed_line = ''
            
            # 12. 移除多余连续空行（超过2个）
            elif 'Do not use more than 2 continuous blank lines' in issue:
                if fixed_line.strip() == '':
                    # 检查前面有多少个连续空行
                    blank_count = 0
                    for i in range(line_num - 1, -1, -1):
                        if lines[i].strip() == '':
                            blank_count += 1
                        else:
                            break
                    if blank_count >= 2:
                        fixed_line = ''
        
        if fixed_line != original_line:
            lines[line_num] = fixed_line
            fixed_count += 1
    
    # 写回文件
    with open(full_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✓ {file_path}: 修复了 {fixed_count} 处问题")
    return fixed_count

def main():
    # 读取问题列表
    with open('/data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/docs/fixable_issues.json', 'r', encoding='utf-8') as f:
        issues_by_file = json.load(f)
    
    total_fixed = 0
    for file_path, issues in issues_by_file.items():
        fixed = fix_file(file_path, issues)
        if fixed:
            total_fixed += fixed
    
    print(f"\n总计修复: {total_fixed} 处问题")

if __name__ == '__main__':
    main()
