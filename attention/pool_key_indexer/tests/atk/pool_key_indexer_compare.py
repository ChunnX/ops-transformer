#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer ATK 双标杆精度比较插件 (pki_double_benchmark)。

注册名: "pki_double_benchmark" (对应 YAML standard.acc, 名字包含 "benchmark"
以触发 ATK 框架的真值任务: 标杆升精度(fp16/bf16->fp32)在 CPU 上重跑,
结果落盘到 {output}/cpu_benchmark/{save_name}/{id}/)。

三方数据:
  - operator  : pyaclnn 算子输出(pyaclnn_operator/), 原始精度输入;
  - 标杆      : NPU 组合标杆输出(npu_npu_golden/), 原始精度输入;
  - 真值      : CPU 金标升精度输出(cpu_benchmark/), ATK benchmark 任务生成。

比对策略(与 pytest tests/pytest/result_compare.py 口径一致):
  1. indices(第一输出 sparseIndices, INT32) —— 集合对比法(唯一索引门禁):
     算子 vs 真值, 逐行提取有效索引(>=0, 无效位 -1)升序排序后做集合相等
     比较, 要求 100% 行通过(允许 TopK 输出顺序不同, 选中索引集合不得有差异);
     同时校验算子无效位取值必须精确为 -1。
  2. values(第二输出 sparseValues, FLOAT32) —— 按索引排序后双标杆对比:
     每行提取 (pool 索引, value) 对并按索引排序(TopK 顺序差异被吸收),
     算子与标杆分别与真值配对计算相对误差:
       Actual    = 算子 vs 真值   -> max_re / avg_re / rmse
       Benchmark = 标杆 vs 真值   -> max_re / avg_re / rmse
     门禁(双标杆: 标杆和算子都跟真值对比, 以标杆固有误差标定容差):
       Actual有效值 / Benchmark有效值 <= 对应 ratio;
       误差低于 abs_floor 视为 0(双方均达到数值噪声 floor 时通过);
       标杆有效误差为 0 而算子有效误差 > 0 时判定失败。

YAML 配置(standard.acc.pki_double_benchmark):
  max_re_ratio: 2        # max_re  的 Actual/Benchmark 比例上限
  avg_re_ratio: 1.2      # avg_re  的 Actual/Benchmark 比例上限
  rmse_ratio: 1.2        # rmse    的 Actual/Benchmark 比例上限
  row_pass_ratio: 1.0    # indices 集合比对行通过率阈值(1.0 = 严格 100%)
  abs_floor: 1.0e-5      # 误差噪声 floor(低于该值视为 0)
  denom_floor: 1.0e-3    # 相对误差分母下限(小值域用绝对误差口径)
  skip_values_compare: false  # values 精度比对开关: true=跳过 values 比对
                              # (rv=True 用例 values 直接判定通过, 仅保留
                              #  indices 门禁); 默认 false=正常比对
"""

import math
import os

import torch

from atk.common.log import Logger
from atk.common.utils import get_output_path_by_case, torch_load_safe
from atk.configs.nodes_config import BENCHMARK_PATH_PATTERN
from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare

logging = Logger().get_logger()


def _to_3d(tensor):
    """(B, S, L) / (T, L) -> (B, S, L), 并转 CPU。"""
    t = tensor.cpu()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def _case_input_value(case_config, name, default=None):
    """从用例配置读取输入/属性取值(range_values[0])。"""
    for inp in case_config.inputs:
        if getattr(inp, "name", None) == name:
            rv = getattr(inp, "range_values", None)
            if rv:
                return rv[0]
            return default
    return default


def _row_pairs(idx_row, val_row, pool_size, sparse):
    """从 indices/values 单行提取 (pool_idx, value) 对, 按索引排序。

    values 第 k 个有效值对应 indices 前 topk 列中第 k 个被选中的池
    (其 token 展开块首元素 = indices[k * pool_size], pool = token // pool_size)。
    """
    pairs = []
    for k in range(sparse):
        tok = int(idx_row[k * pool_size])
        val = float(val_row[k])
        if tok >= 0 and math.isfinite(val):
            pairs.append((tok // pool_size, val))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _pair_rel_errors(pairs_a, pairs_truth, denom_floor):
    """按 pool 索引交集配对, 计算相对误差列表(小值域用绝对误差口径)。"""
    truth_map = dict(pairs_truth)
    errors = []
    for pool_idx, value in pairs_a:
        truth_value = truth_map.get(pool_idx)
        if truth_value is None:
            continue
        errors.append(abs(value - truth_value) / max(abs(truth_value), denom_floor))
    return errors


def _metrics_of(errors):
    """(max_re, avg_re, rmse) 聚合。"""
    if not errors:
        return 0.0, 0.0, 0.0
    tensor = torch.tensor(errors, dtype=torch.float64)
    max_re = tensor.max().item()
    avg_re = tensor.mean().item()
    rmse = tensor.pow(2).mean().sqrt().item()
    return max_re, avg_re, rmse


def _indices_set_compare(op_indices, truth_indices, row_pass_ratio, tie_ctx=None):
    """indices 集合比对(算子 vs 真值), 与 pytest compare_indices 一致,
    可选并列翻转容忍(tie_ctx)。

    并列翻转容忍(为什么需要): TopK 边界处两池分数差可低至 1e-7,
    远小于算子 cube 侧与 CPU 真值的固有数值偏差(~1e-4), 排名相邻的
    近并列分数取舍会随计算路径翻转 —— 属数值噪声而非选池错误。

    仲裁策略(两级):
      1) 分数仲裁(优先, rv=True 有 values 时): 对差异行提取
         truth-only 池 A(取真值分数) 与 op-only 池 B(取算子分数),
         按分数排序配对, 全部 |sA - sB| <= tie_rtol*max(|sA|,denom_floor)
         则该行判为并列翻转(严格可信);
      2) 结构仲裁(rv=False 无 values 时): 差集等量且 <= max_flip_pools
         判为翻转(启发式), 受 flip_row_ratio 总量比例约束兜底
         (大量翻转仍失败, 防选池错误蒙混)。
      差集不等量/分数差过大/结构翻转超比例 -> 该行失败。

    tie_ctx: None(严格集合比对) 或 dict(
        op_values, truth_values, pool_size, sparse,
        tie_rtol, denom_floor, max_flip_pools, flip_row_ratio)

    返回 (ok, error_info, diag): ok=是否通过; diag=诊断明细。
    """
    op = _to_3d(op_indices)
    truth = _to_3d(truth_indices)
    if op.shape != truth.shape:
        return (
            False,
            (
                f"indices shape mismatch: operator {tuple(op.shape)} vs truth {tuple(truth.shape)}"
            ),
            "",
        )

    # 无效位取值校验: 接口约定无效位置必须精确为 -1
    neg = op[op < 0]
    if neg.numel() > 0 and not bool((neg == -1).all()):
        bad = neg[neg != -1][:8].tolist()
        return False, f"invalid slots must be exactly -1, got {bad}", ""

    # ---- 快速路径: 整批向量化预筛相等行 ----
    # 逐行 Python 循环(排序+set 构造)在大 rows×topk 下每行毫秒级, 10 万行
    # ×topk 8192 总耗时可到 10+ 分钟, 撞 celery compare 队列软时限
    # (SoftTimeLimitExceeded, 与 atk -to 挂钩; 默认 60s 必超)。行内升序排序
    # 后逐元素全等 <=> 两行多重集合相等(有效 token 唯一, 与集合比较等价;
    # 无效位已校验全为 -1, 参与排序不引入差异)。相等行直接判 pass, 仅
    # 差异行走下方逐行差集+仲裁慢路径(差异行通常为 0 或并列翻转少量行)。
    op_sorted, _ = op.sort(dim=-1)
    tr_sorted, _ = truth.sort(dim=-1)
    eq_rows = (op_sorted == tr_sorted).all(dim=-1)  # (B, S1m)

    # values 预处理(分数仲裁用): 空输出(rv=False)时仅剩结构仲裁
    tie_ctx = tie_ctx or {}
    op_vals = tie_ctx.get("op_values")
    truth_vals = tie_ctx.get("truth_values")
    has_values = (
        op_vals is not None
        and truth_vals is not None
        and getattr(op_vals, "numel", lambda: 0)() > 0
        and getattr(truth_vals, "numel", lambda: 0)() > 0
    )
    if has_values:
        op_vals = _to_3d(op_vals).to(torch.float32)
        truth_vals = _to_3d(truth_vals).to(torch.float32)
        if op_vals.shape[:2] != op.shape[:2] or truth_vals.shape[:2] != truth.shape[:2]:
            has_values = False

    pool_size = int(tie_ctx.get("pool_size", 1))
    sparse = int(tie_ctx.get("sparse", 1))
    tie_rtol = float(tie_ctx.get("tie_rtol", 0.0))
    denom_floor = float(tie_ctx.get("denom_floor", 1e-3))
    max_flip_pools = int(tie_ctx.get("max_flip_pools", 2))
    flip_row_ratio = float(tie_ctx.get("flip_row_ratio", 0.0))

    def _row_pool_val_map(idx_row, val_row):
        return dict(_row_pairs(idx_row, val_row, pool_size, sparse))

    total_rows = op.shape[0] * op.shape[1]
    pass_rows = 0
    tie_rows = 0  # 分数仲裁通过的翻转行(严格可信)
    struct_rows = 0  # 结构仲裁通过的翻转行(启发式, 受比例约束)
    first_diffs = []
    eq_mask = eq_rows.reshape(-1).tolist()  # 展平行序与下方双循环一致
    row_iter = 0
    for b in range(op.shape[0]):
        for s in range(op.shape[1]):
            if eq_mask[row_iter]:
                pass_rows += 1
                row_iter += 1
                continue
            row_iter += 1
            g_row = truth[b, s]
            n_row = op[b, s]
            g_valid = g_row[g_row >= 0].sort().values
            n_valid = n_row[n_row >= 0].sort().values
            if g_valid.numel() == n_valid.numel() and torch.equal(g_valid, n_valid):
                pass_rows += 1
                continue
            # ---- 差异行: 翻转仲裁 ----
            # 差集换算到 pool 粒度(ps>1 时一个池的翻转 = ps 个连续 token)
            g_set = set(int(v) // pool_size for v in g_valid.tolist())
            n_set = set(int(v) // pool_size for v in n_valid.tolist())
            only_g = sorted(g_set - n_set)  # truth-only pools A
            only_n = sorted(n_set - g_set)  # op-only pools B
            # 仲裁容量: 有分数仲裁(has_values)时不限翻转池数 —— 逐对校验分数差,
            # 全部并列才容忍, 数学上与大翻转数不矛盾; 随机数据下 relu 产生大量
            # 精确 0 分, topk 边界落入 0 分并列区时单行翻转可达数十池(测例实测
            # 中位 17), 按 max_flip_pools=2 封顶会把合法并列翻转误判为失败。
            # 无分数(rv=False)仅剩结构仲裁, 维持小容量启发式(防选池错误蒙混)。
            if has_values and tie_rtol > 0:
                flip_cap = max(len(only_g), len(only_n))
            else:
                flip_cap = max_flip_pools
            tolerated = False
            if len(only_g) == len(only_n) and 0 < len(only_g) <= flip_cap:
                if has_values and tie_rtol > 0:
                    # 分数仲裁: A 取真值分数, B 取算子分数, 排序配对比分数差
                    t_map = _row_pool_val_map(g_row, truth_vals[b, s])
                    o_map = _row_pool_val_map(n_row, op_vals[b, s])
                    scores_a = sorted(t_map.get(p, float("-inf")) for p in only_g)
                    scores_b = sorted(o_map.get(p, float("-inf")) for p in only_n)
                    if all(
                        math.isfinite(a) and math.isfinite(bb)
                        for a, bb in zip(scores_a, scores_b)
                    ):
                        tolerated = all(
                            abs(a - bb) <= tie_rtol * max(abs(a), denom_floor)
                            for a, bb in zip(scores_a, scores_b)
                        )
                elif not has_values and max_flip_pools > 0:
                    # 结构仲裁: 等量小差集视为翻转(无分数可验证)
                    tolerated = True
            if tolerated:
                if has_values:
                    tie_rows += 1
                else:
                    struct_rows += 1
            elif len(first_diffs) < 3:
                first_diffs.append(
                    f"row({b},{s}): golden-only{only_g[:6]} npu-only{only_n[:6]}"
                )
    fail_rows = total_rows - pass_rows - tie_rows - struct_rows
    ratio = (pass_rows + tie_rows + struct_rows) / total_rows if total_rows > 0 else 0.0
    struct_ratio = struct_rows / total_rows if total_rows > 0 else 0.0
    ok = ratio >= row_pass_ratio and struct_ratio <= flip_row_ratio
    info = (
        f"indices set compare: {pass_rows}/{total_rows} matched"
        f" (+{tie_rows} tie-flip, +{struct_rows} struct-flip)"
        f" ({ratio:.2%})"
    )
    if not ok:
        if struct_ratio > flip_row_ratio:
            info += (
                f"; struct-flip ratio {struct_ratio:.2%} > {flip_row_ratio:.2%}"
                f" (mass flips without value arbitration)"
            )
        info += "; first diffs: " + "; ".join(first_diffs)
    return (
        ok,
        info,
        f"rows={total_rows}, matched={pass_rows}, tie={tie_rows}, struct={struct_rows}",
    )


def _values_double_benchmark(
    op_indices,
    op_values,
    bm_indices,
    bm_values,
    truth_indices,
    truth_values,
    pool_size,
    topk,
    denom_floor,
    abs_floor,
    ratio_thresholds,
):
    """values 按索引排序后的双标杆对比。

    返回 (ok, error_info)。
    """
    sparse = topk // pool_size
    op_idx = _to_3d(op_indices)
    bm_idx = _to_3d(bm_indices)
    truth_idx = _to_3d(truth_indices)

    op_vals = _to_3d(op_values).to(torch.float32)
    bm_vals = _to_3d(bm_values).to(torch.float32)
    truth_vals = _to_3d(truth_values).to(torch.float32)

    if op_vals.shape != truth_vals.shape:
        return False, (
            f"values shape mismatch: operator {tuple(op_vals.shape)} vs truth {tuple(truth_vals.shape)}"
        )

    actual_errors = []
    benchmark_errors = []
    for b in range(truth_idx.shape[0]):
        for s in range(truth_idx.shape[1]):
            truth_pairs = _row_pairs(
                truth_idx[b, s], truth_vals[b, s], pool_size, sparse
            )
            op_pairs = _row_pairs(op_idx[b, s], op_vals[b, s], pool_size, sparse)
            bm_pairs = _row_pairs(bm_idx[b, s], bm_vals[b, s], pool_size, sparse)
            actual_errors.extend(_pair_rel_errors(op_pairs, truth_pairs, denom_floor))
            benchmark_errors.extend(
                _pair_rel_errors(bm_pairs, truth_pairs, denom_floor)
            )

    a_max, a_avg, a_rmse = _metrics_of(actual_errors)
    b_max, b_avg, b_rmse = _metrics_of(benchmark_errors)

    def _effective(value):
        return value if value > abs_floor else 0.0

    checks = []
    ok = True
    for name, a_val, b_val, ratio in (
        ("max_re", a_max, b_max, ratio_thresholds["max_re_ratio"]),
        ("avg_re", a_avg, b_avg, ratio_thresholds["avg_re_ratio"]),
        ("rmse", a_rmse, b_rmse, ratio_thresholds["rmse_ratio"]),
    ):
        eff_a = _effective(a_val)
        eff_b = _effective(b_val)
        if eff_a == 0.0:
            checks.append(f"{name}: actual={a_val:.3e} <= floor(pass)")
        elif eff_b == 0.0:
            ok = False
            checks.append(
                f"{name}: actual={a_val:.3e} > floor but benchmark={b_val:.3e} exact(FAIL)"
            )
        else:
            real_ratio = eff_a / eff_b
            passed = real_ratio <= ratio
            ok = ok and passed
            checks.append(
                f"{name}: actual/benchmark={real_ratio:.3f}"
                f"(a={a_val:.3e}, b={b_val:.3e}, thd={ratio})"
                f"{'pass' if passed else 'FAIL'}"
            )

    info = (
        f"values double benchmark (pairs: actual={len(actual_errors)}, "
        f"benchmark={len(benchmark_errors)}): " + "; ".join(checks)
    )
    return ok, info


@ACCURACY_REGISTRY.register("pki_double_benchmark")
class PKIDoubleBenchmarkCompare(BaseAccuracyCompare):
    """PoolKeyIndexer 双标杆精度比较(indices 集合比对 + values 双标杆)。"""

    def __init__(self, case_config, **kwargs):
        super().__init__(case_config, **kwargs)
        self.max_re_ratio = float(kwargs.get("max_re_ratio", 2.0))
        self.avg_re_ratio = float(kwargs.get("avg_re_ratio", 1.2))
        self.rmse_ratio = float(kwargs.get("rmse_ratio", 1.2))
        self.row_pass_ratio = float(kwargs.get("row_pass_ratio", 1.0))
        self.abs_floor = float(kwargs.get("abs_floor", 1e-5))
        self.denom_floor = float(kwargs.get("denom_floor", 1e-3))
        # 索引并列翻转容忍参数(见 _indices_set_compare docstring)
        self.tie_rtol = float(kwargs.get("tie_rtol", 1e-2))
        self.max_flip_pools = int(kwargs.get("max_flip_pools", 2))
        self.flip_row_ratio = float(kwargs.get("flip_row_ratio", 1e-2))
        # values 精度比对开关: true=跳过 values 比对(仅保留 indices 门禁),
        # 默认 false=正常比对(旧用例 JSON 无该字段时同样回退到 False)
        self.skip_values_compare = bool(kwargs.get("skip_values_compare", False))

    def _load_output(self, output_dir, filename):
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return None
        return torch_load_safe(path)

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        """抽象方法实体化(本比较器整体接管 precision_compare_for_case,
        该方法不会被调用, 仅满足 BaseAccuracyCompare 实例化约束)。"""
        return AccuracyConfig(filename=data_file, result=True)

    def precision_compare_for_case(
        self, main_node, remote_node, is_npu=False, device_id=0
    ):
        """三方(算子/标杆/真值)对比, 产出 output_0/output_1 两条精度结果。"""
        operator_dir = get_output_path_by_case(
            main_node.get_output_path(), main_node.get_backend_name(), self.case_config
        )
        benchmark_dir = get_output_path_by_case(
            remote_node.get_output_path(),
            remote_node.get_backend_name(),
            self.case_config,
        )
        benchmark_device = getattr(
            getattr(self.executor, "task_result", None), "benchmark_device", "cpu"
        )
        truth_dir = get_output_path_by_case(
            main_node.get_output_path(),
            BENCHMARK_PATH_PATTERN.format(benchmark_device),
            self.case_config,
        )

        op_indices = self._load_output(operator_dir, "output_0.pt")
        op_values = self._load_output(operator_dir, "output_1.pt")
        bm_indices = self._load_output(benchmark_dir, "output_0.pt")
        bm_values = self._load_output(benchmark_dir, "output_1.pt")
        truth_indices = self._load_output(truth_dir, "output_0.pt")
        truth_values = self._load_output(truth_dir, "output_1.pt")

        if op_indices is None or op_values is None:
            return [
                AccuracyConfig(
                    filename="output_0.pt",
                    result=False,
                    error_info=f"operator output missing in {operator_dir}",
                )
            ]
        if truth_indices is None or truth_values is None:
            return [
                AccuracyConfig(
                    filename="output_0.pt",
                    result=False,
                    error_info=(
                        f"truth(cpu_benchmark) output missing in {truth_dir}; "
                        f"check standard name contains 'benchmark' and bm task ran"
                    ),
                )
            ]
        if bm_indices is None or bm_values is None:
            return [
                AccuracyConfig(
                    filename="output_0.pt",
                    result=False,
                    error_info=f"benchmark node output missing in {benchmark_dir}",
                )
            ]

        # ---- 1. indices 集合比对(算子 vs 真值, 唯一索引门禁, 并列翻转容忍) ----
        topk = int(_case_input_value(self.case_config, "topk", 2048))
        pool_size = int(_case_input_value(self.case_config, "pool_size", 1))
        tie_ctx = dict(
            op_values=op_values,
            truth_values=truth_values,
            pool_size=pool_size,
            sparse=max(1, topk // pool_size),
            tie_rtol=self.tie_rtol,
            denom_floor=self.denom_floor,
            max_flip_pools=self.max_flip_pools,
            flip_row_ratio=self.flip_row_ratio,
        )
        idx_ok, idx_info, _ = _indices_set_compare(
            op_indices, truth_indices, self.row_pass_ratio, tie_ctx=tie_ctx
        )

        # 标杆 vs 真值集合匹配率(仅诊断, 不判定)
        bm_ok, bm_diag, _ = _indices_set_compare(
            bm_indices, truth_indices, self.row_pass_ratio
        )

        idx_result = AccuracyConfig(
            filename="output_0.pt",
            result=idx_ok,
            error_info=idx_info + f"; [ref] bm-vs-truth: {bm_diag} (reference only)",
        )

        # ---- 2. values 按索引排序后双标杆对比 ----

        if self.skip_values_compare:
            val_result = AccuracyConfig(
                filename="output_1.pt",
                result=True,
                error_info="values compare disabled by config (skip_values_compare=true)",
            )
        elif truth_values.numel() == 0 and op_values.numel() == 0:
            val_result = AccuracyConfig(
                filename="output_1.pt",
                result=True,
                error_info="values empty (return_value=False), skip compare",
            )
        else:
            val_ok, val_info = _values_double_benchmark(
                op_indices,
                op_values,
                bm_indices,
                bm_values,
                truth_indices,
                truth_values,
                pool_size,
                topk,
                self.denom_floor,
                self.abs_floor,
                {
                    "max_re_ratio": self.max_re_ratio,
                    "avg_re_ratio": self.avg_re_ratio,
                    "rmse_ratio": self.rmse_ratio,
                },
            )
            val_result = AccuracyConfig(
                filename="output_1.pt",
                result=val_ok,
                error_info=val_info,
            )

        logging.info(
            f"[case {self.case_config.id}] pki_double_benchmark: "
            f"indices={'PASS' if idx_ok else 'FAIL'} | {idx_result.error_info}"
        )
        logging.info(
            f"[case {self.case_config.id}] pki_double_benchmark: "
            f"values={'PASS' if val_result.result else 'FAIL'} | {val_result.error_info}"
        )
        return [idx_result, val_result]
