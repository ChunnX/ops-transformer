#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""L2 cache flush helper（B9⑤-5 性能采集，每轮 msprof 前执行）.

口径来源：
  - ascendc-mc2-best-practice references/shared/profiling_mc2.md：每轮主 kernel
    前刷 L2（heavy kernel 扫 > L2 容量的 buffer），禁止只分配不 flush；
  - ops-profiling references/msprof-guide.md「MC² 多 rank 算子采集」；
  - 任务拍板（B9⑤-5）：每轮实际执行 L2 cache flush。

实现（框架侧轮前 flush，example 不可改）：
  - buffer 256 MB bf16/卡 > Ascend950PR PCIE L2 112 MB（npu-arch skill
    唯一知识源：950PR PCIE L2=112MB，2.3× 余量；对齐参考工程 256MB 示例）；
  - 每卡 2 遍 in-place add（读改写全量扫过，挤出上一轮驻留行）；
  - torch_npu 实现（向量算子自动铺满 AIV 核），flush 进程独立于 msprof
    采集进程，flush kernel 不污染 msprof task 记录；
  - 结束打印 FLUSH_OK 证据行（collect_perf.sh 校验该行判定本轮 flush 生效）。

方法学注记（报告登记）：flush 位于轮前（进程外），无法消除同轮 example 内
H2D 写入的 L2 写分配效应——该效应 v1/APACE 两分支同构，主判据（双分支同
条件对比）公平；绝对带宽解读时注意此口径。

Usage: AGM_V2_DEV_NUM=2 python3 l2_flush.py
"""

import os
import sys
import time

import torch
import torch_npu

# Ascend950PR PCIE L2 = 112 MB（npu-arch skill）；256 MB buffer = 2.3x 余量
FLUSH_MB = 256
PASSES = 2


def main():
    dev_num = int(os.environ.get("AGM_V2_DEV_NUM", "2"))
    devices = list(range(dev_num))
    t0 = time.perf_counter()
    for d in devices:
        torch.npu.set_device(d)
        # bf16 2 bytes/elem
        numel = FLUSH_MB * 1024 * 1024 // 2
        buf = torch.empty(numel, dtype=torch.bfloat16, device=f"npu:{d}")
        for p in range(PASSES):
            buf.add_(1)  # in-place 读改写：全量扫过 buffer（非仅分配）
        torch.npu.synchronize()
        del buf
        torch.npu.empty_cache()
    elapsed_us = (time.perf_counter() - t0) * 1e6
    print(
        f"[l2_flush] FLUSH_OK devices={devices} buf_mb={FLUSH_MB} "
        f"dtype=bf16 passes={PASSES} elapsed_us={elapsed_us:.0f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
