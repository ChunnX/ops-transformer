#!/usr/bin/env python3
"""
Excel Sheet 信息读取工具
用于获取指定 Sheet 的行数信息，并验证行范围的合理性
"""
import sys
import argparse
import pandas as pd
import json

def get_sheet_info(excel_file, sheet_name=None, start_row=None, end_row=None):
    """
    获取 Excel 文件的 Sheet 信息，并验证行范围
    
    Args:
        excel_file: Excel 文件路径
        sheet_name: 指定 Sheet 名称
        start_row: 用户指定的开始行（可选）
        end_row: 用户指定的结束行（可选）
    
    Returns:
        dict: Sheet 信息，包含验证结果
    """
    try:
        # 检查文件是否存在
        import os
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel 文件不存在: {excel_file}")
        
        if sheet_name:
            # 读取指定 Sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            total_rows = len(df)
            min_row = 2  # 第1行是表头
            max_row = total_rows + 1  # 实际结束行
            
            result = {
                'sheet_name': sheet_name,
                'total_rows': total_rows,
                'min_row': min_row,
                'max_row': max_row,
                'default_start_row': min_row,
                'default_end_row': max_row,
                'warnings': [],
                'errors': []
            }
            
            # 验证用户指定的行范围
            if start_row is not None:
                result['user_start_row'] = start_row
                if start_row < min_row:
                    result['warnings'].append(f"开始行 {start_row} 小于最小行 {min_row}，已自动修正为 {min_row}")
                    start_row = min_row
                elif start_row > max_row:
                    result['errors'].append(f"开始行 {start_row} 超出最大行范围 {max_row}")
                    result['valid'] = False
                    return result
            
            if end_row is not None:
                result['user_end_row'] = end_row
                if end_row > max_row:
                    result['warnings'].append(f"结束行 {end_row} 超出实际数据范围（最大 {max_row}），已自动修正为 {max_row}")
                    end_row = max_row
                elif end_row < min_row:
                    result['errors'].append(f"结束行 {end_row} 小于最小行 {min_row}")
                    result['valid'] = False
                    return result
            
            # 应用用户指定或使用默认值
            final_start_row = start_row if start_row is not None else min_row
            final_end_row = end_row if end_row is not None else max_row
            
            # 验证最终行范围的合理性
            if final_start_row > final_end_row:
                result['errors'].append(f"开始行 ({final_start_row}) 大于结束行 ({final_end_row})")
                result['valid'] = False
                return result
            
            result['start_row'] = final_start_row
            result['end_row'] = final_end_row
            result['valid'] = True
            
            return result
        else:
            # 获取所有 Sheet 信息
            xl_file = pd.ExcelFile(excel_file)
            sheets_info = []
            for sheet in xl_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                total_rows = len(df)
                sheets_info.append({
                    'sheet_name': sheet,
                    'total_rows': total_rows,
                    'min_row': 2,
                    'max_row': total_rows + 1,
                    'default_start_row': 2,
                    'default_end_row': total_rows + 1
                })
            return sheets_info
            
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        if "No sheet named" in str(e):
            # Sheet 不存在，尝试列出可用的 Sheets
            try:
                xl_file = pd.ExcelFile(excel_file)
                available_sheets = xl_file.sheet_names
                print(f"错误: Sheet '{sheet_name}' 不存在", file=sys.stderr)
                print(f"可用的 Sheets: {', '.join(available_sheets)}", file=sys.stderr)
            except:
                print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"错误: 读取 Excel 文件失败 - {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='获取 Excel Sheet 信息并验证行范围')
    parser.add_argument('excel_file', help='Excel 文件路径')
    parser.add_argument('-s', '--sheet', help='指定 Sheet 名称')
    parser.add_argument('--start-row', type=int, help='用户指定的开始行')
    parser.add_argument('--end-row', type=int, help='用户指定的结束行')
    parser.add_argument('--list-sheets', action='store_true', help='列出所有 Sheet 名称')
    parser.add_argument('--json', action='store_true', help='以 JSON 格式输出')
    parser.add_argument('--validate-only', action='store_true', help='仅验证行范围，不执行')
    
    args = parser.parse_args()
    
    if args.list_sheets:
        # 只列出 Sheet 名称
        try:
            xl_file = pd.ExcelFile(args.excel_file)
            for sheet in xl_file.sheet_names:
                print(sheet)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # 获取详细信息并验证
        if args.sheet:
            info = get_sheet_info(
                args.excel_file, 
                args.sheet,
                args.start_row,
                args.end_row
            )
        else:
            info = get_sheet_info(args.excel_file)
        
        if args.json:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            if isinstance(info, list):
                # 显示所有 Sheet 信息
                print(f"{'Sheet名称':<20} {'总行数':<8} {'最小行':<8} {'最大行':<8}")
                print("-" * 50)
                for item in info:
                    print(f"{item['sheet_name']:<20} {item['total_rows']:<8} {item['min_row']:<8} {item['max_row']:<8}")
            else:
                # 显示单个 Sheet 详细信息
                print(f"Sheet: {info['sheet_name']}")
                print(f"总数据行数: {info['total_rows']}")
                print(f"有效行范围: {info['min_row']} - {info['max_row']}")
                
                if 'user_start_row' in info:
                    print(f"用户指定开始行: {info['user_start_row']}")
                if 'user_end_row' in info:
                    print(f"用户指定结束行: {info['user_end_row']}")
                
                # 显示警告
                if info.get('warnings'):
                    print("\n⚠️  警告:")
                    for warning in info['warnings']:
                        print(f"  - {warning}")
                
                # 显示错误
                if info.get('errors'):
                    print("\n❌ 错误:")
                    for error in info['errors']:
                        print(f"  - {error}")
                
                if info.get('valid', True):
                    print(f"\n✅ 最终使用行范围: {info['start_row']} - {info['end_row']}")
                else:
                    print("\n❌ 行范围验证失败")
                    sys.exit(1)

if __name__ == '__main__':
    main()