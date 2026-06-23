#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
在编译kernel的时候，用来删除不需要的arch文件夹下面的文件
"""

import os
import sys
import shutil
from pathlib import Path

def should_skip_directory(dir_name, skip_patterns=None):
    if skip_patterns is None:
        skip_patterns = [
            "all_gather_matmul_v2",
            "quant_grouped_mat_mul_allto_allv"
        ]
    
    return dir_name in skip_patterns

def main():
    if len(sys.argv) != 3:
        print(f"method: {sys.argv[0]} <based folder> <deleted folder>", file=sys.stderr)
        print(f"example: {sys.argv[0]} /path/to/ops-transformer/mc2 op_host/op_tiling/arch35", file=sys.stderr)
        sys.exit(1)
    
    base_dir, rel_path = sys.argv[1], sys.argv[2]
    base_path = Path(base_dir)
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"error: invalid directory", file=sys.stderr)
        sys.exit(1)
    
    deleted = 0
    failed = 0
    skipped = 0
    
    print(f"processing directory: {base_dir}")
    
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
            
        dir_name = item.name
        
        if should_skip_directory(dir_name) and rel_path.endswith('/arch35'):
            print(f"skip directory: {dir_name}")
            skipped += 1
            continue
        
        target = item / rel_path
        if target.exists() and target.is_dir():
            try:
                shutil.rmtree(target)
                print(f"✓ Deleted: {target}")
                deleted += 1
            except PermissionError:
                print(f"✗ PermissionError: {target}", file=sys.stderr)
                failed += 1
            except Exception as e:
                print(f"✗ Failed to delete: {target} ({e})", file=sys.stderr)
                failed += 1
        else:
            pass
    
    print(f"\nfinish:")
    print(f"  deleted num: {deleted}")
    print(f"  failed num: {failed}")
    print(f"  skipped num: {skipped}")
    
    if deleted == 0 and failed == 0 and skipped == 0:
        print(f"in '{base_dir}', '{rel_path}' is not founded")
    
    sys.exit(1 if failed > 0 else 0)

if __name__ == "__main__":
    main()