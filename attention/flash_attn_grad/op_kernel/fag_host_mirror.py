
# --- generated imports (fag_fiximports.py) ---
from fag_common import CUBE_BASEM, CUBE_BASEN
# --- end generated imports ---
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""host 判据的 Python 镜像 —— **不参与 kernel trace**。

生产路径上这些判断全部由 host 承担（``op_host/plan/*``）。这里逐行照抄一份，
只为了让 JIT 脚本（test / prof / bench）能自己拼出 tilingkey —— JIT 不走
host，拿不到 host 算好的值。

**改判据必须同时改两处，且以 host 的 C++ 实现为准。** 两边算得不一样不会报
任何错，只会让 JIT 验证的二进制与 host 实际派发的那份不是同一个。

把它们单独拎成一个模块，就是为了让「这些函数不会被 trace」这件事在目录结构
上是显然的：kernel 侧任何模块都不 import 本文件。
"""

BN2_MAX_S = 128
BN2_MAX_D = 128
BN2_MULTIBLK_SEQ = 640
BN2_MULTIBLK_BN_128 = 128
BN2_MULTIBLK_BN_256 = 256



def decide_bn2_multiblk(b, n1, n2, s1, s2, d, dv=None, is_tnd=False, mask_mode=0):
    """老 FAG SetSplitAxis 的 MultiBlk 门。不满足则不要硬走 BN2。"""
    if is_tnd:
        return 0
    if n1 != n2 or d > BN2_MAX_D:
        return 0
    if dv is not None and dv != d:
        return 0
    if not ((s1 > BN2_MAX_S or s2 > BN2_MAX_S) and
            s1 <= BN2_MULTIBLK_SEQ and s2 <= BN2_MULTIBLK_SEQ):
        return 0
    bn = b * n1
    if bn >= BN2_MULTIBLK_BN_256:
        return 1
    if bn >= BN2_MULTIBLK_BN_128 and (s1 % 128 == 0) and (s2 % 128 == 0):
        return 1
    return 0



def decide_template(n1, n2, s1, s2, d, is_tnd=False, b=None, dv=None, mask_mode=0):
    """host 侧选分核模板。BN2 本轮不接 GQA、不接 D=192。

    b 缺省时 S>128 不升 MultiBlk，避免旧调用方在不知道 B 时静默改模板。
    """
    if is_tnd:
        return 0
    if n1 != n2 or d > BN2_MAX_D:
        return 0
    if s1 <= BN2_MAX_S and s2 <= BN2_MAX_S:
        return 1
    if b is None:
        return 0
    return 1 if decide_bn2_multiblk(b, n1, n2, s1, s2, d, dv=dv,
                                    is_tnd=is_tnd, mask_mode=mask_mode) else 0



def decide_d_align(d):
    """对齐档：(0,64]->64，(64,128]->128，(128,192]->192。"""
    if d <= 64:
        return 64
    if d <= 128:
        return 128
    return 192



def decide_dv_align(dv, d_align):
    if d_align == 64:
        return 128
    return 128 if dv <= 128 else 192



SPARSE_DENSE = 0
SPARSE_CASUAL = 1
SPARSE_BAND = 2
INT32_MAX_TOKEN = 2147483647



def correct_sparse_tokens(mask_mode, win_left, win_right, s1, s2):
    """对齐 ASC ProcessTokensInfo 非 TND：causal 填 token，再做右下 pad 校正。"""
    if mask_mode == 3:
        s1_token = INT32_MAX_TOKEN
        s2_token = 0
    elif mask_mode == 4:
        s1_token = win_left
        s2_token = win_right
    else:
        return INT32_MAX_TOKEN, INT32_MAX_TOKEN
    return s1_token + s1 - s2, s2_token - s1 + s2



def decide_sparse_type(mask_mode, s1, s2, win_left=-1, win_right=-1):
    """对齐 GetSparseType 非 TND。1=CASUAL，2=BAND，0=DENSE。"""
    if mask_mode != 3 and mask_mode != 4:
        return SPARSE_DENSE
    s1_token, s2_token = correct_sparse_tokens(mask_mode, win_left, win_right, s1, s2)
    if (mask_mode == 3 and s1 <= s2) or (mask_mode == 4 and s1_token >= s1 and s2_token == 0):
        return SPARSE_CASUAL
    return SPARSE_BAND



def total_per_batch_num(s1, s2, mask_mode=0, win_left=-1, win_right=-1):
    """对齐 ASC GetTotalPerBatchNum：CASUAL 三角闭式，BAND 按列走 p/q。"""
    s1_outer = (s1 + CUBE_BASEM - 1) // CUBE_BASEM
    s2_outer = (s2 + CUBE_BASEN - 1) // CUBE_BASEN
    sparse = decide_sparse_type(mask_mode, s1, s2, win_left, win_right)
    if sparse == SPARSE_DENSE:
        return s1_outer * s2_outer
    if sparse == SPARSE_CASUAL:
        if s1 < s2:
            if mask_mode == 3:
                return ((((s2_outer << 1) - s1_outer + 1) * s1_outer) >> 1) + (s1_outer - 1)
            return (((s1_outer << 1) - s1_outer + 1) * s1_outer) >> 1
        return (((s1_outer << 1) - s2_outer + 1) * s2_outer) >> 1
    s1_token, s2_token = correct_sparse_tokens(mask_mode, win_left, win_right, s1, s2)
    p = int((s1_token + CUBE_BASEM - 1) / CUBE_BASEM)
    q = int((s2_token + CUBE_BASEN - 1) / CUBE_BASEN)
    total = 0
    for s2o in range(s2_outer):
        x_min = s2o - q if (s2o - q) > 0 else 0
        x_max = s2o + p if (s1_outer - 1) > (s2o + p) else (s1_outer - 1)
        length = (x_max - x_min + 1) if x_max >= x_min else 0
        if length > 0:
            total += length
    return total



# ==================================================================
#  IsValid + GetNextValidIdx —— 按 tilingkey mask_mode 静态分支
#  对应 kernel_base.h::IsValid / CheckIsValidBlock
# ==================================================================
# ==================================================================
#  Swizzle 分核 —— 让同时在跑的核落在同一 head 内，命中 L2
# ==================================================================
# 线性切分(核 c 领 metadata 给的连续区间)在大 N 下会让 36 个核同时
# 落在 32~36 个不同 head 上。实测 S=8192：
#   N=1  36核跨 1 head，K+V 工作集   4MB -> L2(128MB) 内，0.94x ASC
#   N=32 36核跨 32 head，          128MB -> 刚撑满，    1.46x
#   N=64 36核跨 36 head，          144MB -> 溢出，      1.54x
# 归一化后 mte2 从 93.9 涨到 227.5us，成为新瓶颈(占用率 0.72)，mac 反而
# 从 0.938 掉到 0.733 —— 即 L2 未命中退化成 HBM 带宽受限。
#
# 对标 ASC 的 swizzle(M/N_SWIZZLE_SIZE=32768, 触发条件 isExceedL2Cache,
# common.h:67)：把块网格按 tile 分组，同时在跑的核锁在同一 tile 内。
#
# 这里的做法：以「s2 列」为分配单位(一列 = 一个 (b,n2,s2o) 组 = s1Outer
# 个连续块)，按核跨步发放 —— 核 c 领第 c, c+coreNum, ... 列。
#   * 同核在一列内连续处理 s1Outer 个块 -> s2 不变
#     => K/V 只载一次、dK/dV 在 L0C 连续累加，这两项优化都保住
#   * 第 k 轮时 36 核处理列 k*coreNum+c，几乎总在同一 head 内
#     => K+V 工作集回落到 4MB
#
# 代价：分配粒度变粗(一列 s1Outer 块)。N=1 时总列数仅 64 < 2*coreNum，
# 核 0 拿 2 列而核 35 只拿 1 列，负载失衡 2 倍 —— 实测 N=1 由 193.8
# 慢到 220.8us。故与 ASC 一样按「工作集是否超 L2」开关。
#
# 实测(S=8192, 36 核, ASC 为基准):
#   N     线性切分         swizzle          选择
#   1     193.8 (0.94x)   220.8 (1.07x)    线性
#   32   10017 (1.46x)    6650 (0.97x)     swizzle
#   64   21993 (1.54x)   13434 (0.94x)     swizzle
# 标定机型 Ascend950PR_9599 的 L2 容量，仅作 JIT 脚本的缺省值。
# host 侧不再硬编码这个数，而是从平台取（见 op_host 的 FagParsedInfo::l2CacheSize）：
# ascend950 各变体差别很大 —— 950PR_950z 16MB、950DT_950x 32MB、957d 96MB、
# 957b/957c/958b 等 112MB、9599/9581/9589 等 128MB。
# 在非 128MB 的型号上跑 JIT，必须给 decide_swizzle 显式传 l2_cache_bytes，
# 否则算出来的 tilingkey 与 host 实际派发的不是同一个。
L2_CACHE_BYTES = 128 * 1024 * 1024



def decide_swizzle(b, n, s1, s2, d, core_num, dtype_bytes=2, g_size=1,
                   l2_cache_bytes=L2_CACHE_BYTES, mask_mode=0, win_left=-1, win_right=-1):
    """**host 判据的 Python 镜像**，仅供 JIT 脚本使用。

    生产路径上这个判据由 host 承担，权威实现是
    `op_host/plan/flash_attn_grad_tiling_swizzle.cpp` 的 `DecideSwizzle()`，
    全部标定依据（29 个实测点）也记在那里。本函数只是把那段整数运算逐行
    照抄成 Python，让 JIT 脚本(test/prof/bench)能自己拼出 tilingkey ——
    JIT 不走 host，拿不到 host 算好的值。

    **改判据必须同时改两处，且以 DecideSwizzle() 为准。** 两边算得不一样不会
    报任何错，只会让 JIT 验证的二进制与 host 实际派发的那份不是同一个
    (GQA 下曾因此有 19 个 shape 静默不一致)。
    `test_..._pypto.py` 里有 `test_gate_matches_host()` 守护这件事。

    入参口径与 host 一一对应：
      n              <- n1Size (Q 侧 head 数)，KV 侧 N2 = n // g_size
      g_size         <- gSize
      core_num       <- aicNum (物理 cube 核数，**不是** blockDim)
      l2_cache_bytes <- FagParsedInfo::l2CacheSize (平台上报值，不是常量)。
                        缺省是标定机型 9599 的 128MB；换型号必须显式传，
                        否则与 host 分叉。0 表示平台没报，host 此时不开
                        swizzle，这里同样返回 0。
      mask_mode/win  <- 属性；per_head_blocks 走 totalPerBatchNum。
    """
    if l2_cache_bytes == 0:
        return 0
    s1_outer = (s1 + CUBE_BASEM - 1) // CUBE_BASEM
    s2_outer = (s2 + CUBE_BASEN - 1) // CUBE_BASEN
    per_head_blocks = total_per_batch_num(s1, s2, mask_mode, win_left, win_right)
    if per_head_blocks <= 0:
        per_head_blocks = s1_outer * s2_outer
    total = b * n * per_head_blocks
    per_core = (total + core_num - 1) // core_num
    live_heads = len({(c * per_core) // per_head_blocks
                      for c in range(core_num)})

    fp32 = 4
    # q + dy + y 读 (fp16) + dq 写 (fp32) —— 这些是本 g 私有的
    per_head = 3 * s1 * d * dtype_bytes + s1 * d * fp32
    # k + v 读 (fp16) + dk + dv 写 (fp32) —— GQA 下同一个 KV head 被 G 个
    # Q head 共享，故 KV 侧字节按 G 摊薄，只记一份(与 host 的 `/ gSize` 一致)
    per_head += (2 * s2 * d * dtype_bytes + 2 * s2 * d * fp32) // g_size
    # 判据依赖 S1/S2 的**方向**，不只看工作集大小：S2 更长时恒不开、
    # S1 明显更长时拐点更早(1.25x L2)、对称时保持 2x L2。
    # 三档阈值的实测标定数据(29 个点)全部记录在 DecideSwizzle() 的同一段里，
    # 此处不复制 —— 复制两份必然漂移，要看依据请看那边。
    if s2_outer > s1_outer:
        return 0
    # 与 host 相同的整除：S1 重用 5*L2/4（1.25x），对称用 2*L2。
    threshold = ((5 * l2_cache_bytes) // 4 if s1_outer >= 2 * s2_outer
                 else 2 * l2_cache_bytes)
    if live_heads * per_head <= threshold:
        return 0
    # swizzle 的分配粒度是「一整列 s1Outer 块」，列数太少会失衡到不划算
    # (N=1 时仅 64 列 < 2*36 核，核0 拿 2 列而核35 拿 1 列，实测反慢 14%)
    if b * n * s2_outer < 2 * core_num:
        return 0
    return 1
