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

import os, sys, shutil
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(1)

target_dir, folder_name = sys.argv[1], sys.argv[2]

if not Path(target_dir).exists():
    sys.exit(1)

deleted = 0
for root, dirs, _ in os.walk(target_dir):
    if folder_name in dirs:
        try:
            shutil.rmtree(Path(root) / folder_name)
            deleted += 1
        except:
            pass  # 静默失败

# 返回删除的数量（0表示未找到或失败）
sys.exit(0 if deleted > 0 else 1)