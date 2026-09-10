# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer 输出比对工具(集合比对策略)。

比对策略:
  1. indices(第一输出 sparseIndices) —— 唯一精度门禁:
     逐行提取有效索引(值 >= 0, 无效位为 -1), 排序后做集合相等比较;
     要求 100% 行通过(严格标准, 不允许任何行差异)。
     另校验无效位取值必须精确为 -1(接口约定的 INT32 填充值)。
     说明: 逐行有效索引的有序集合要求完全一致(允许 TopK 输出顺序不同,
     但选中的索引集合本身不得有差)。

  2. values(第二输出 sparseValues) —— 仅参考, 不做通过/失败判定:
     逐行提取有效值(有限值, 无效位为 -inf), 排序后按 rtol/atol 统计
     数值接近程度, 结果打印日志供人工参考。
     直接对算子原始输出比对, 不引入任何归一化/重排/重建等后处理,
     避免后处理掩盖或放大真实差异导致精度误判。
"""

import torch

RTOL = 1e-3
ATOL = 1e-3


def _to_3d(tensor):
    """(B, S, L) / (T, L) -> (B, S, L), 并转 CPU。"""
    t = tensor.cpu()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def compare_indices(
    golden: torch.Tensor, npu: torch.Tensor, pct_thd: float = 1.0
) -> bool:
    """indices 集合比对(精度门禁, 严格 100% 标准)。

    逐行提取有效索引(>= 0)升序排序后, 要求 golden 与 npu 的索引集合完全相等;
    行通过率必须达到 100%(默认 pct_thd=1.0, 不允许任何行差异)。
    同时校验 npu 无效位(<0)取值必须精确为 -1, 不允许其他负值。

    Args:
        golden: 标杆 indices, shape (B, S1, L) 或 (T1, L), 无效位置 -1。
        npu: 算子输出 indices, shape 同 golden。
        pct_thd: 行通过率阈值, 默认 1.0(严格 100%); 仅诊断场景可显式调低。

    Returns:
        bool: 是否通过。
    """
    g = _to_3d(golden)
    n = _to_3d(npu)
    if g.shape != n.shape:
        print(f"[FAIL] indices shape mismatch: golden {g.shape} vs npu {n.shape}")
        return False

    # 无效位取值校验: 接口约定无效位置必须精确为 -1(INT32 填充值),
    # 不允许 -2/-999 等其他负值。仅过滤 >=0 的集合比对无法暴露该类错误。
    bad_neg_rows = []
    for b in range(g.shape[0]):
        for s in range(g.shape[1]):
            nv = n[b, s]
            n_neg = nv[nv < 0]
            if n_neg.numel() > 0 and not bool((n_neg == -1).all()):
                bad_neg_rows.append((b, s))
    if bad_neg_rows:
        print(
            f"[FAIL] invalid slots must be exactly -1, but {len(bad_neg_rows)} rows "
            f"contain other negative values, e.g. row {bad_neg_rows[0]}: "
            f"{n[bad_neg_rows[0][0], bad_neg_rows[0][1]][n[bad_neg_rows[0][0], bad_neg_rows[0][1]] < 0][:8].tolist()}"
        )
        return False

    total_rows = 0
    pass_rows = 0
    for b in range(g.shape[0]):
        for s in range(g.shape[1]):
            gv = g[b, s]
            nv = n[b, s]
            g_valid = gv[gv >= 0].sort().values
            n_valid = nv[nv >= 0].sort().values
            total_rows += 1
            if g_valid.numel() == n_valid.numel() and torch.equal(g_valid, n_valid):
                pass_rows += 1
            elif total_rows - pass_rows <= 10:  # 最多打印 10 行失败明细
                if g_valid.numel() != n_valid.numel():
                    print(
                        f"[FAIL] row ({b},{s}): valid count golden={g_valid.numel()} "
                        f"npu={n_valid.numel()}"
                    )
                else:
                    diffs = (g_valid != n_valid).nonzero(as_tuple=True)[0]
                    idx = diffs[0].item() if len(diffs) > 0 else 0
                    only_g = [int(v) for v in g_valid if v not in n_valid][:8]
                    only_n = [int(v) for v in n_valid if v not in g_valid][:8]
                    print(
                        f"[FAIL] row ({b},{s}): {len(diffs)} index diffs, "
                        f"first@{idx} golden={int(g_valid[idx])} npu={int(n_valid[idx])}, "
                        f"golden-only{only_g} npu-only{only_n}"
                    )

    ratio = pass_rows / total_rows if total_rows > 0 else 0.0
    if ratio >= pct_thd:
        print(f"[PASS] indices: {pass_rows}/{total_rows} rows matched ({ratio:.2%})")
        return True
    print(
        f"[FAIL] indices: only {pass_rows}/{total_rows} rows matched ({ratio:.2%}), "
        f"threshold={pct_thd:.2%}"
    )
    return False


def compare_values(
    golden: torch.Tensor,
    npu: torch.Tensor,
    dt: torch.dtype = None,
    rtol: float = RTOL,
    atol: float = ATOL,
    label: str = "values",
) -> dict:
    """values 参考性比对(仅打印日志, 不做通过/失败判定)。

    逐行提取有效值(有限值, 无效位 -inf)升序排序后, 按 rtol/atol 统计
    golden 与 npu 的数值接近程度, 打印汇总日志并返回统计信息。
    TopK 并列分数取舍不同时选中池可能不同, 但选中分数的有序集合仍应一致,
    因此该统计可反映分数通道的整体数值质量。

    Args:
        golden: 标杆 values, shape (B, S1, K) 或 (T1, K), 无效位置 -inf。
        npu: 算子输出 values, shape 同 golden。
        dt: 输入 dtype(仅用于日志展示, 可为 None)。
        rtol/atol: allclose 容差。
        label: 日志标签。

    Returns:
        dict: 统计信息 {total_rows, pass_rows, ratio, max_abs_err, mismatch_rows}。
    """
    g = _to_3d(golden).to(torch.float32)
    n = _to_3d(npu).to(torch.float32)
    stats = {
        "total_rows": 0,
        "pass_rows": 0,
        "ratio": 0.0,
        "max_abs_err": 0.0,
        "mismatch_rows": [],
    }
    if g.shape != n.shape:
        print(f"[REF] {label}: shape mismatch golden {g.shape} vs npu {n.shape} (参考)")
        return stats

    for b in range(g.shape[0]):
        for s in range(g.shape[1]):
            gv = g[b, s]
            nv = n[b, s]
            g_valid = gv[torch.isfinite(gv)].sort().values
            n_valid = nv[torch.isfinite(nv)].sort().values
            stats["total_rows"] += 1
            row_err = 0.0
            if g_valid.numel() == n_valid.numel():
                if g_valid.numel() > 0:
                    # 通过行也记录真实误差, 避免 all-pass 时 max_abs_err 恒为 0
                    row_err = (g_valid - n_valid).abs().max().item()
                stats["max_abs_err"] = max(stats["max_abs_err"], row_err)
                if g_valid.numel() == 0 or torch.allclose(
                    g_valid, n_valid, rtol=rtol, atol=atol
                ):
                    stats["pass_rows"] += 1
                    continue
            if len(stats["mismatch_rows"]) < 5:
                stats["mismatch_rows"].append(
                    {
                        "row": (b, s),
                        "golden_cnt": int(g_valid.numel()),
                        "npu_cnt": int(n_valid.numel()),
                        "max_err": row_err,
                    }
                )

    total = stats["total_rows"]
    stats["ratio"] = stats["pass_rows"] / total if total > 0 else 0.0

    # ---- 参考日志输出(不判定) ----
    dt_name = str(dt).replace("torch.", "") if dt is not None else "n/a"
    print(
        f"[REF] {label}: {stats['pass_rows']}/{total} rows matched "
        f"({stats['ratio']:.2%}), max_abs_err={stats['max_abs_err']:.6f}, "
        f"rtol={rtol}, atol={atol}, dtype={dt_name} (参考信息, 不作为判定)"
    )
    for m in stats["mismatch_rows"]:
        print(
            f"[REF]   row {m['row']}: valid count golden={m['golden_cnt']} "
            f"npu={m['npu_cnt']}, max_err={m['max_err']:.6f}"
        )
    return stats
