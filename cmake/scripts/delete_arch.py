#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
在编译tiling的时候，用来删除不需要的arch文件夹下面的文件
"""

import os
import sys
import shutil
from pathlib import Path

def should_skip_directory(dir_name, skip_patterns=None):
    """检查目录是否应该被跳过"""
    if skip_patterns is None:
        skip_patterns = [
            "moe_distribute_dispatch_v2",
            "moe_distribute_combine_v2"
        ]
    
    return dir_name in skip_patterns

def main():
    if len(sys.argv) != 3:
        print(f"用法: {sys.argv[0]} <基础目录> <相对路径>", file=sys.stderr)
        print(f"示例: {sys.argv[0]} /path/to/ops-transformer/mc2 op_host/op_tiling/arch35", file=sys.stderr)
        sys.exit(1)
    
    base_dir, rel_path = sys.argv[1], sys.argv[2]
    base_path = Path(base_dir)
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"错误: 无效的基础目录", file=sys.stderr)
        sys.exit(1)
    
    deleted = 0
    failed = 0
    skipped = 0
    
    print(f"正在处理目录: {base_dir}")
    
    # 遍历基础目录下的所有子目录
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
            
        dir_name = item.name
        
        # 检查是否应该跳过
        if should_skip_directory(dir_name) and rel_path.endswith('/arch35'):
            print(f"跳过目录: {dir_name}")
            skipped += 1
            continue
        
        # 构建目标路径
        target = item / rel_path
        if target.exists() and target.is_dir():
            try:
                shutil.rmtree(target)
                print(f"✓ 已删除: {target}")
                deleted += 1
            except PermissionError:
                print(f"✗ 权限不足: {target}", file=sys.stderr)
                failed += 1
            except Exception as e:
                print(f"✗ 删除失败: {target} ({e})", file=sys.stderr)
                failed += 1
        else:
            # 输出找不到的路径（可选）
            # print(f"未找到: {target}")
            pass
    
    # 输出结果
    print(f"\n处理完成:")
    print(f"  成功删除: {deleted} 个")
    print(f"  删除失败: {failed} 个")
    print(f"  跳过目录: {skipped} 个")
    
    if deleted == 0 and failed == 0 and skipped == 0:
        print(f"在 '{base_dir}' 下未找到任何 '{rel_path}' 路径")
    
    # 如果有删除失败，返回1
    sys.exit(1 if failed > 0 else 0)

if __name__ == "__main__":
    main()