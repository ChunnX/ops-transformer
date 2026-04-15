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
在编译的时候，用来删除不需要的arch文件夹下面的文件
"""

import os
import sys
import shutil
from pathlib import Path

if len(sys.argv) != 3:
    print(f"用法: {sys.argv[0]} <基础目录> <相对路径>", file=sys.stderr)
    sys.exit(1)

base_dir, relative_path = sys.argv[1], sys.argv[2]
target_path = Path(base_dir) / relative_path

if not target_path.exists():
    print(f"路径不存在: {target_path}", file=sys.stderr)
    sys.exit(1)

if not target_path.is_dir():
    print(f"不是文件夹: {target_path}", file=sys.stderr)
    sys.exit(1)

try:
    shutil.rmtree(target_path)
    print(f"已删除: {target_path}")
    sys.exit(0)
except PermissionError:
    print(f"权限不足: {target_path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"删除失败: {e}", file=sys.stderr)
    sys.exit(1)