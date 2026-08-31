#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 对指定范围内的提交变更文件运行 pre-commit 检查
# 用法:
#   bash scripts/pc.sh HEAD~2 HEAD   # 检查最近2笔提交
#   bash scripts/pc.sh HEAD~1        # 检查最近1笔提交（末尾HEAD可省略）
#   bash scripts/pc.sh origin/master...HEAD  # 检查PR全部提交
#
# 可选：配置 git 别名后支持 git pc <范围> 短命令（每个环境执行一次，全局生效）:
#   git config --global alias.pc '!bash scripts/pc.sh "$@"'
# 配置后用法: git pc HEAD~2 HEAD
exec pre-commit run --files $(git diff --name-only "${@:-HEAD}")
