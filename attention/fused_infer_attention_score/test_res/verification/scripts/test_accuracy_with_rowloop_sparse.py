#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLASST FIA 精度验证套件 - 精度对比版（增强版）
=================================================================
功能说明：
1. 支持从 .pt dump 文件读取 Query, Key, Value 及其原始输出
2. 对比四种精度：
   - NPU BLASST vs Baseline NPU (无BLASST)
   - Python BLASST vs Python Dense (BLASST跳块误差)
   - NPU BLASST vs Python BLASST (NPU与Python仿真差异)
   - Baseline NPU vs Python Dense (NPU与Dense基准差异)
3. 支持 sparse_mode=0 (dense) 和 sparse_mode=3 (rightDownCausal)
4. 支持批量测试多个 THRESHOLD_LOG 配置
5. 支持拼接算子 (Concat) 精度分析
6. 支持可选的详细误差分析
7. 【新增】NPU 算子一致性检查：多次调用 NPU 算子，验证结果确定性
8. 【新增】Top‑K 误差分析：找出 NPU BLASST 与 Python BLASST 差异最大的 (token, head) 位置

用法示例：
----------
基本用法（使用默认阈值列表，保存在 GlobalConfig.THRESHOLD_LOGS 中）：
    python blasst_precision_suite.py

自定义阈值：
    python blasst_precision_suite.py --thresholds "-7,-5,-3,-1,0"

保存 CSV 结果：
    python blasst_precision_suite.py --save-csv

启用详细误差分析（分析误差超过 1% 的 token）：
    python blasst_precision_suite.py --detailed --error-threshold 0.01

NPU 算子一致性检查（重复运行 5 次）：
    python blasst_precision_suite.py --check-consistency

自定义一致性检查次数（例如 10 次）：
    python blasst_precision_suite.py --check-consistency --consistency-runs 10

Top‑K 误差分析（使用第一个阈值，显示前 10 个最大误差位置）：
    python blasst_precision_suite.py --topk

使用第二个阈值（索引 1）进行 Top‑K 分析，并显示前 20 个最大误差：
    python blasst_precision_suite.py --topk --topk-threshold-idx 1 --topk-k 20

同时启用一致性检查和 Top‑K 分析：
    python blasst_precision_suite.py --check-consistency --topk

NPU head 模式（逐个 head 计算）：
    python blasst_precision_suite.py --npu-head-mode

阈值扫描模式（分析不同阈值对稀疏度的影响）：
    python blasst_precision_suite.py --threshold-scan --threshold-start -10 --threshold-end 0 --threshold-steps 20

完整参数列表：
    --sparse-mode         稀疏模式：0=dense, 3=rightDownCausal (默认 3)
    --thresholds          自定义阈值列表，逗号分隔，如 "-7,-5,-3,-1,0"
    --save-csv            保存结果到 CSV 文件
    --detailed            启用详细误差分析
    --error-threshold     误差阈值，超过此值的 token 会被详细分析 (默认 0.01 = 1%)
    --npu-head-mode       启用 NPU 的 head 处理策略（逐个 head 计算）
    --threshold-scan      启用阈值扫描模式
    --threshold-start     扫描起始值（log scale，默认 -10.0）
    --threshold-end       扫描结束值（log scale，默认 0.0）
    --threshold-steps     扫描步数（默认 20）
    --check-consistency   启用 NPU 算子一致性检查（多次运行）
    --consistency-runs    一致性检查重复次数（默认 5）
    --topk                启用 Top‑K 误差位置分析
    --topk-threshold-idx  用于 Top‑K 分析的阈值索引（默认 0）
    --topk-k              显示前 K 个最大误差位置（默认 10）

注意：
    - 需要提前配置 GlobalConfig 中的 DUMP_FILES、OUTPUT_DIR、BASELINE_NPU_DIR 等路径。
    - 依赖 row_loop_python_blasst.py 中的 optimized_blasst_sim 和 GlobalConfig。
    - 运行前请确保 NPU 环境已正确安装 torch_npu。
=================================================================
"""

import math
import os
import re
import torch
import torch_npu
import numpy as np
from collections import defaultdict
from datetime import datetime
from row_loop_python_blasst import optimized_blasst_sim, GlobalConfig as OrigGlobalConfig

# ============================================================
# 0. 扩展 GlobalConfig（添加新功能所需配置）
# ============================================================
class GlobalConfig(OrigGlobalConfig):
    # 继承原有配置，添加新参数
    NPU_CONSISTENCY_RUNS = 1          # NPU 一致性检查重复次数（1 表示不检查）
    ENABLE_TOPK_ANALYSIS = False      # 是否启用 Top‑K 误差分析
    TOPK_THRESHOLD_IDX = 0            # 使用第几个阈值进行 Top‑K 分析（0 为第一个）
    TOPK_K = 10                       # 显示前 K 个最大误差位置
    # 基线根目录指向 npu_blasst.py 生成的 baseline 目录
    BASELINE_NPU_DIR = os.path.join(
        OrigGlobalConfig.BASE_PATH,
        "attention/fused_infer_attention_score/tests/tests/baseline"
    )


# ============================================================
# 1. 原有辅助函数（保持不变）
# ============================================================
def compute_error_metrics(out_test, out_ref):
    """计算两个输出之间的误差指标"""
    out_test_f32 = out_test.float()
    out_ref_f32 = out_ref.float()

    rel_error = torch.norm(out_test_f32 - out_ref_f32) / (
        torch.norm(out_ref_f32) + 1e-9
    )
    abs_error = torch.norm(out_test_f32 - out_ref_f32)
    max_element_error = torch.max(torch.abs(out_test_f32 - out_ref_f32)).item()
    mean_abs_error = torch.mean(torch.abs(out_test_f32 - out_ref_f32)).item()

    return {
        "rel_error": rel_error.item(),
        "abs_error": abs_error.item(),
        "max_element_error": max_element_error,
        "mean_abs_error": mean_abs_error,
    }


def compute_tokenwise_errors(out_test, out_ref, S_q):
    """计算逐 token 的误差"""
    out_test_f32 = out_test[:S_q].float()
    out_ref_f32 = out_ref[:S_q].float()
    token_errors = (
        torch.norm(out_test_f32 - out_ref_f32, dim=-1).mean(dim=-1).cpu().numpy()
    )
    return token_errors


def print_error_comparison(name1, name2, metrics, prefix=""):
    """打印两个结果的精度对比"""
    if metrics is None:
        print(f"{prefix}  {name1} vs {name2}: Baseline NPU not available")
        return
    print(f"{prefix}  {name1} vs {name2}:")
    print(
        f"{prefix}    Relative Error: {metrics['rel_error']:.6e} ({metrics['rel_error'] * 100:.4f}%)"
    )
    print(f"{prefix}    Absolute Error: {metrics['abs_error']:.6e}")
    print(f"{prefix}    Max Element Error: {metrics['max_element_error']:.6e}")
    print(f"{prefix}    Mean Abs Error: {metrics['mean_abs_error']:.6e}")


def detailed_error_analysis(
    out_dense,
    out_blasst,
    out_npu_blasst,
    skip_info,
    block_contributions,
    S_q,
    threshold_log,
    error_threshold=0.01,
):
    out_dense_f32 = out_dense[:S_q].float()
    out_blasst_f32 = out_blasst[:S_q].float()
    out_npu_f32 = out_npu_blasst[:S_q].float()

    token_error_blasst_vs_dense = (
        torch.norm(out_blasst_f32 - out_dense_f32, dim=-1).mean(dim=-1).cpu().numpy()
    )
    token_error_blasst_vs_npu = (
        torch.norm(out_blasst_f32 - out_npu_f32, dim=-1).mean(dim=-1).cpu().numpy()
    )
    token_error_dense_vs_npu = (
        torch.norm(out_dense_f32 - out_npu_f32, dim=-1).mean(dim=-1).cpu().numpy()
    )

    high_error_positions = np.where(token_error_blasst_vs_npu > error_threshold)[0]

    if len(high_error_positions) == 0:
        print(f"    没有发现超过 {error_threshold * 100}% 误差的位置")
        return

    print(f"\n  【高误差位置详细分析】(误差 > {error_threshold * 100}%)")
    print(f"  发现 {len(high_error_positions)} 个高误差位置:")
    print(f"  {'-' * 60}")

    max_display = min(10, len(high_error_positions))
    for pos in high_error_positions[:max_display]:
        print(f"\n  Position: {pos}")
        print(
            f"    Error (BLASST vs Dense): {token_error_blasst_vs_dense[pos]:.6f} ({token_error_blasst_vs_dense[pos] * 100:.4f}%)"
        )
        print(f"    Error (BLASST vs NPU): {token_error_blasst_vs_npu[pos]:.6f}")
        print(f"    Error (Dense vs NPU): {token_error_dense_vs_npu[pos]:.6f}")

        is_skipped = pos in skip_info.get("skipped_positions", set())
        print(f"    Is Skipped by BLASST: {is_skipped}")

        if block_contributions:
            processed_blocks = 0
            skipped_blocks = 0
            for block_key, block_info in block_contributions.items():
                q_range = block_info.get("q_range")
                if q_range and q_range[0] <= pos < q_range[1]:
                    if block_info.get("skipped", False):
                        skipped_blocks += 1
                    elif block_info.get("processed", False):
                        processed_blocks += 1
            print(f"    Affected KV Blocks: {processed_blocks + skipped_blocks} total")
            print(
                f"      Processed: {processed_blocks}, Skipped by BLASST: {skipped_blocks}"
            )

    if len(high_error_positions) > max_display:
        print(
            f"\n  ... 还有 {len(high_error_positions) - max_display} 个高误差位置未显示"
        )

    skipped_high_error = sum(
        1
        for pos in high_error_positions
        if pos in skip_info.get("skipped_positions", set())
    )
    print(f"\n  高误差位置统计:")
    print(
        f"    被 BLASST 跳过的: {skipped_high_error} ({skipped_high_error / len(high_error_positions) * 100:.1f}%)"
    )
    print(
        f"    未被跳过的: {len(high_error_positions) - skipped_high_error} ({(len(high_error_positions) - skipped_high_error) / len(high_error_positions) * 100:.1f}%)"
    )


def analyze_concat_operator(
    out_blasst_list, out_dense_list, out_npu_blasst, threshold_logs, S_q,
    out_dense_from_file=None
):
    print(f"\n{'=' * 60}")
    print("拼接算子 (Concat) 精度分析")
    print(f"{'=' * 60}")

    if out_dense_from_file is not None:
        out_dense_ref = out_dense_from_file
        print("  使用从文件加载的 Dense 输出作为基准")
    else:
        out_dense_ref = out_dense_list[0]
        print("  使用当前计算的 Dense 输出作为基准")

    seq_len = S_q
    concat_result = torch.zeros_like(out_dense_ref)
    best_choice_count = defaultdict(int)

    for pos in range(seq_len):
        best_error = float("inf")
        best_idx = 0
        for idx, out_blasst in enumerate(out_blasst_list):
            error = torch.norm(
                (out_blasst[pos] - out_dense_ref[pos]).float()
            ).item()
            if error < best_error:
                best_error = error
                best_idx = idx
        concat_result[pos] = out_blasst_list[best_idx][pos]
        best_choice_count[threshold_logs[best_idx]] += 1

    metrics_concat_vs_dense = compute_error_metrics(concat_result, out_dense_ref)
    metrics_concat_vs_npu = compute_error_metrics(concat_result, out_npu_blasst)

    print(f"\n  拼接策略: 按位置选择误差最小的 BLASST 结果")
    print(f"  各阈值被选择的次数:")
    for th in threshold_logs:
        print(
            f"    λ={th}: {best_choice_count[th]} 次 ({best_choice_count[th] / seq_len * 100:.1f}%)"
        )

    print(f"\n  拼接结果精度:")
    print_error_comparison("Concat", "Dense", metrics_concat_vs_dense, "    ")
    print_error_comparison("Concat", "NPU", metrics_concat_vs_npu, "    ")
    return concat_result


# ============================================================
# 2. 新增功能函数
# ============================================================
def check_npu_consistency(query, key, value, atten_mask, block_size,
                          q_len, kv_len, num_heads, num_kv_heads,
                          scale, sparse_mode, n_runs=5, device='npu:0',
                          topk=10, save_dir=None, base_name=None):
    """
    多次调用 NPU 算子，检查输出是否完全一致，并将每次结果保存到文件。
    
    参数:
        save_dir: 保存结果的目录（如果为 None 则不保存）
        base_name: 基础文件名（如 dump 文件名），用于生成带标识的文件名
    """
    print("\n  [一致性检查] NPU 融合算子重复运行 {} 次...".format(n_runs))
    outputs = []
    lses = []
    
    for i in range(n_runs):
        out, gLse = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=atten_mask,
            block_table=None,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=q_len,
            actual_seq_lengths_kv=kv_len,
            num_key_value_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            sparse_mode=sparse_mode,
            softmax_lse_flag=True
        )
        torch.npu.synchronize()
        out_cpu = out.cpu()
        lse_cpu = gLse.cpu()
        outputs.append(out_cpu)
        lses.append(lse_cpu)
        
        # 保存每次运行的结果到文件
        if save_dir is not None and base_name is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{base_name}_consistency_run{i}_out.pt")
            lse_path = os.path.join(save_dir, f"{base_name}_consistency_run{i}_lse.pt")
            torch.save(out_cpu, out_path)
            torch.save(lse_cpu, lse_path)
            print(f"      已保存: {out_path}, {lse_path}")
        
        # 打印 gLse 统计信息（可选）
        print("======================================")
        blockNum = 0
        blockSparseNum = 0
        data_1d = gLse.flatten()
        coreNum = min(20, len(data_1d) // 16)
        for core in range(coreNum):
            blockSparseNum += data_1d[core * 16]
            blockNum += data_1d[core * 16 + 1]
        npu_block_sparsity = (blockSparseNum / blockNum).item() if blockNum > 0 else 0.0
        print(f"      NPU block sparse rate: {npu_block_sparsity:.2%}")
        print("======================================")
    
    # 两两比较最大绝对误差，并记录差异最大的一对
    max_diff = 0.0
    worst_pair = (0, 1)
    worst_diff_tensor = None
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            diff_tensor = (outputs[i] - outputs[j]).abs()
            diff = diff_tensor.max().item()
            if diff > max_diff:
                max_diff = diff
                worst_pair = (i, j)
                worst_diff_tensor = diff_tensor
    
    # 保存一致性检查元数据
    if save_dir is not None and base_name is not None:
        meta = {
            "n_runs": n_runs,
            "max_diff": max_diff,
            "worst_pair": worst_pair,
            "sparse_mode": sparse_mode,
            "block_size": block_size,
            "scale": scale,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
        }
        if max_diff >= 1e-6 and worst_diff_tensor is not None:
            meta["worst_diff_tensor"] = worst_diff_tensor.cpu()
        meta_path = os.path.join(save_dir, f"{base_name}_consistency_meta.pt")
        torch.save(meta, meta_path)
        print(f"      一致性元数据已保存: {meta_path}")
    
    if max_diff < 1e-6:
        print(f"    ✅ 一致性通过: 所有运行结果完全一致 (最大差异 {max_diff:.2e})")
    else:
        print(f"    ❌ 一致性失败: 运行结果存在差异 (最大差异 {max_diff:.2e})")
        print(f"    差异最大的一对: run#{worst_pair[0]} vs run#{worst_pair[1]}")
        if worst_diff_tensor is not None:
            S_q, num_heads, head_dim = worst_diff_tensor.shape
            token_head_diff = worst_diff_tensor.mean(dim=-1)
            flat = token_head_diff.flatten()
            topk_vals, topk_indices = torch.topk(flat, min(topk, len(flat)))
            print(f"\n    [Top‑K 不一致位置] 差异最大的 {len(topk_indices)} 个 (token, head):")
            print(f"      {'Rank':<6} {'Token':<8} {'Head':<8} {'MeanDiff':<14} {'MaxDiff':<14} VecFirst4")
            print("      " + "-" * 70)
            for rank, (idx, mean_val) in enumerate(zip(topk_indices, topk_vals)):
                t = idx // num_heads
                h = idx % num_heads
                max_val = worst_diff_tensor[t, h].max().item()
                vec_front0 = outputs[worst_pair[0]][t, h, :4].tolist()
                vec_front1 = outputs[worst_pair[1]][t, h, :4].tolist()
                print(f"      {rank + 1:<6} {t:<8} {h:<8} {mean_val:<14.6e} {max_val:<14.6e}")
                print(f"        run#{worst_pair[0]}: {vec_front0}")
                print(f"        run#{worst_pair[1]}: {vec_front1}")
    return max_diff


def print_topk_errors(out_npu_blasst, out_python_blasst, out_npu_baseline,
                      threshold_val, head_dim, topk=10):
    """
    找出 NPU BLASST 与 Python BLASST 之间向量 L2 误差最大的 topk 个 (token, head) 位置，
    并打印三者之间的误差对比表格。
    """
    if out_npu_baseline is None:
        print("    [Top‑K] 跳过：未加载 NPU Baseline 输出")
        return
    
    # 形状假设: (S_q, num_heads, head_dim)
    S_q, num_heads, _ = out_npu_blasst.shape
    # 计算每个 (t, h) 的 L2 误差
    errors = torch.zeros((S_q, num_heads), device='cpu')
    for t in range(S_q):
        for h in range(num_heads):
            diff = out_npu_blasst[t, h] - out_python_blasst[t, h]
            # errors[t, h] = torch.norm(diff).item()
            errors[t, h] = diff.abs().mean().item()
    
    # 找出 topk 个最大值的位置
    flat_errors = errors.flatten()
    topk_vals, topk_indices = torch.topk(flat_errors, min(topk, len(flat_errors)))
    
    print(f"\n  [Top‑K 误差分析] 阈值 λ = {threshold_val}")
    print(f"  差异最大的 {len(topk_indices)} 个 (token, head) 位置:")
    # 表头
    print(f"    {'#':<4} {'Token':<8} {'Head':<8} "
          f"{'BLASST: NPU vs Python L2':<16} {'NPU: Blasst vs Base L2':<16} {'Python Blasst vs NPU Base L2':<16}")
    print("    " + "-" * 72)
    
    for rank, (idx, err_val) in enumerate(zip(topk_indices, topk_vals)):
        t = idx // num_heads
        h = idx % num_heads
        # 计算 NPU BLASST vs NPU Baseline
        diff_npu_base = out_npu_blasst[t, h] - out_npu_baseline[t, h]
        err_npu_base = torch.norm(diff_npu_base).item()
        # 计算 Python BLASST vs NPU Baseline
        diff_py_base = out_python_blasst[t, h] - out_npu_baseline[t, h]
        err_py_base = torch.norm(diff_py_base).item()
        
        print(f"    {rank+1:<4} {t:<8} {h:<8} "
              f"{err_val:<16.6e} {err_npu_base:<16.6e} {err_py_base:<16.6e}")
    
    # 可选：展示前几个位置的向量前4维（便于调试）
    if head_dim <= 8 and topk > 0:
        print("\n    示例（前4维）:")
        for rank, idx in enumerate(topk_indices[:3]):
            t = idx // num_heads
            h = idx % num_heads
            print(f"      (t={t},h={h}): NPU_Blasst={out_npu_blasst[t,h,:4]}, "
                  f"Py_Blasst={out_python_blasst[t,h,:4]}, Base={out_npu_baseline[t,h,:4]}")


# ============================================================
# 3. 核心对比逻辑（添加新功能调用）
# ============================================================
def process_single_dump(dump_path, sparse_mode=3, npu_head_mode=False):
    device = GlobalConfig.DEVICE
    if not os.path.exists(dump_path):
        print(f"File not found: {dump_path}")
        return None

    print(f"\n{'=' * 80}")
    print(f"Processing: {os.path.basename(dump_path)}")
    print(f"{'=' * 80}")

    data = torch.load(dump_path, map_location=device,weights_only=False)
    params = data["inputs"]

    query = params["query"]
    key = params["key"]
    value = params["value"]
    q_len = params["actual_seq_qlen"]
    kv_len = params["actual_seq_kvlen"]
    num_heads = params["num_heads"].item()
    num_kv_heads = params["num_kv_heads"].item()
    scale = params["scale"].item()
    block_size = params["block_size"].item()
    S_q = int(q_len.item())

    print(f"\n  输入信息:")
    print(f"    Q shape: {query.shape}, K shape: {key.shape}, V shape: {value.shape}")
    print(f"    Q length: {S_q}, KV length: {int(kv_len.item())}")
    print(f"    Num heads: {num_heads}, Num KV heads: {num_kv_heads}")
    print(f"    Scale: {scale}, Block size: {block_size}")

    # 获取 base_name 和时间戳
    base_name = os.path.splitext(os.path.basename(dump_path))[0]
    match = re.search(r'(\d+)$', base_name)
    timestamp = match.group(1) if match else str(int(datetime.now().timestamp() * 1000))
    save_dir = os.path.join(GlobalConfig.OUTPUT_DIR, "fixblasst", f"sparse{sparse_mode}", timestamp, base_name)

    # 加载 Baseline NPU 输出（无BLASST）和 Dense 输出
    baseline_npu_dir = os.path.join(GlobalConfig.BASELINE_NPU_DIR, f"sparse{sparse_mode}")
    baseline_npu_path = os.path.join(baseline_npu_dir, f"{base_name}_out_npu.pt")
    out_npu_baseline = None
    out_dense_from_file = None

    if os.path.exists(baseline_npu_path):
        out_npu_baseline = torch.load(baseline_npu_path, map_location=device)
        print(f"  加载 Baseline NPU (无BLASST) 从: {baseline_npu_path}")
    else:
        print(f"  警告: Baseline NPU 文件不存在: {baseline_npu_path}")

    # 1. NPU 融合算子结果（当前 BLASST 版本）
    print(f"\n  [1/4] 运行 NPU 融合算子 (BLASST 版本)...")
    atten_mask = None
    if sparse_mode == 3:
        atten_mask = torch.triu(
            torch.ones(2048, 2048, device=device, dtype=torch.bool), diagonal=1
        )

    out_npu_blasst, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        atten_mask=atten_mask,
        block_table=None,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=q_len,
        actual_seq_lengths_kv=kv_len,
        num_key_value_heads=num_kv_heads,
        num_heads=num_heads,
        scale=scale,
        sparse_mode=sparse_mode,
    )
    torch.npu.synchronize()
    # return
    
    # 提前构建保存目录（用于一致性检查和后续保存）
    base_name = os.path.splitext(os.path.basename(dump_path))[0]
    match = re.search(r'(\d+)$', base_name)
    timestamp = match.group(1) if match else str(int(datetime.now().timestamp() * 1000))
    save_dir = os.path.join(GlobalConfig.OUTPUT_DIR, "fixblasst", f"sparse{sparse_mode}", timestamp, base_name)

    # ----- NPU 一致性检查（包含结果保存）-----
    if GlobalConfig.NPU_CONSISTENCY_RUNS > 1:
        check_npu_consistency(
            query, key, value, atten_mask, block_size, q_len, kv_len,
            num_heads, num_kv_heads, scale, sparse_mode,
            n_runs=GlobalConfig.NPU_CONSISTENCY_RUNS,
            device=device,
            topk=GlobalConfig.TOPK_K,
            save_dir=save_dir,
            base_name=base_name
        )

    # 2. 仿真小算子 (Dense 模式)
    print(f"\n  [2/4] 运行 Dense 仿真 (threshold=-99)...")
    out_dense, _, _ = optimized_blasst_sim(
        query,
        key,
        value,
        q_len,
        kv_len,
        scale,
        threshold_log=-99.0,
        sparse_mode=sparse_mode,
        return_block_info=False,
        npu_head_mode=npu_head_mode,
        print_flag=False,
        trace_pos=None,
        trace_head=None,
        trace_output_dir=None        
    )

    # 3. 批量测试不同 BLASST 阈值配置
    print(f"\n  [3/4] 运行 BLASST 仿真 (多阈值)...")
    results = []
    out_blasst_list = []
    out_npu_blasst_list = []
    GlobalConfig.VERBOSE_SPARSITY = True

    for th_idx, threshold in enumerate(GlobalConfig.THRESHOLD_LOGS):
        print(f"\n    --- Testing threshold_log = {threshold} ---")

        # 运行 NPU 算子（传入 key_antiquant_mode 作为 sparseLambda 载体）
        out_npu_blasst, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=atten_mask,
            block_table=None,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=q_len,
            actual_seq_lengths_kv=kv_len,
            num_key_value_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            sparse_mode=sparse_mode,
            key_antiquant_mode=int(10 * threshold - 100),
        )
        torch.npu.synchronize()
        out_npu_blasst_list.append(out_npu_blasst)

        if GlobalConfig.DETAILED_ANALYSIS:
            out_blasst, row_sparsity, block_sparsity, skip_info, block_contributions = (
                optimized_blasst_sim(
                    query,
                    key,
                    value,
                    q_len,
                    kv_len,
                    scale,
                    threshold_log=threshold,
                    sparse_mode=sparse_mode,
                    return_block_info=True,
                    npu_head_mode=npu_head_mode,
                    print_flag=True,
                    trace_pos=GlobalConfig.qs_idx_target,
                    trace_head=GlobalConfig.h_target,
                    trace_output_dir=GlobalConfig.TRACE_OUTPUT_DIR
                )
            )
        else:
            out_blasst, row_sparsity, block_sparsity = optimized_blasst_sim(
                query,
                key,
                value,
                q_len,
                kv_len,
                scale,
                threshold_log=threshold,
                sparse_mode=sparse_mode,
                return_block_info=False,
                npu_head_mode=npu_head_mode,
                print_flag=True,
                trace_pos=GlobalConfig.qs_idx_target,
                trace_head=GlobalConfig.h_target,
                trace_output_dir=GlobalConfig.TRACE_OUTPUT_DIR
            )
            skip_info = None
            block_contributions = None

        out_blasst_list.append(out_blasst)

        # 计算四种对比指标
        # 1) NPU BLASST vs Baseline NPU
        if out_npu_baseline is not None:
            metrics_npu_blasst_vs_baseline = compute_error_metrics(out_npu_blasst, out_npu_baseline)
        else:
            metrics_npu_blasst_vs_baseline = None

        # 2) Python BLASST vs Python Dense
        metrics_python_blasst_vs_baseline = compute_error_metrics(out_blasst, out_dense)

        # 3) NPU BLASST vs Python BLASST
        metrics_blasst_npu_vs_python = compute_error_metrics(out_npu_blasst, out_blasst)

        # 4) Baseline NPU vs Python Dense
        if out_npu_baseline is not None:
            metrics_baseline_npu_vs_python = compute_error_metrics(out_npu_baseline, out_dense)
        else:
            metrics_baseline_npu_vs_python = None

        # 计算逐 token 误差（用于统计，可选）
        token_errors = compute_tokenwise_errors(out_blasst, out_npu_blasst, S_q)

        result = {
            "threshold_log": threshold,
            "block_sparsity": block_sparsity,
            "row_sparsity": row_sparsity,
            "sparsity_percent": row_sparsity * 100,
            # 四种对比
            "metrics_npu_blasst_vs_baseline": metrics_npu_blasst_vs_baseline,
            "metrics_python_blasst_vs_baseline": metrics_python_blasst_vs_baseline,
            "metrics_blasst_npu_vs_python": metrics_blasst_npu_vs_python,
            "metrics_baseline_npu_vs_python": metrics_baseline_npu_vs_python,
            # token 误差统计
            "token_error_mean": np.mean(token_errors),
            "token_error_std": np.std(token_errors),
            "token_error_max": np.max(token_errors),
            "token_error_p95": np.percentile(token_errors, 95),
            "token_error_p99": np.percentile(token_errors, 99),
        }

        # 打印四种对比结果
        print(f"      Row Sparsity: {row_sparsity:.2%} (Block: {block_sparsity:.2%})")
        print(f"      Token Error (Python BLASST vs Python Dense): mean={np.mean(token_errors):.6e}, max={np.max(token_errors):.6e}, p95={np.percentile(token_errors, 95):.6e}")
        print_error_comparison("NPU BLASST", "NPU Baseline", metrics_npu_blasst_vs_baseline, "      ")
        print_error_comparison("Python BLASST", "Python Baseline", metrics_python_blasst_vs_baseline, "      ")
        print_error_comparison("NPU BLASST", "Python BLASST", metrics_blasst_npu_vs_python, "      ")
        print_error_comparison("NPU Baseline", "Python Baseline", metrics_baseline_npu_vs_python, "      ")

        # ----- 新增：Top‑K 误差分析（仅对指定阈值）-----
        if (GlobalConfig.ENABLE_TOPK_ANALYSIS and 
            th_idx == GlobalConfig.TOPK_THRESHOLD_IDX and
            out_npu_baseline is not None):
            head_dim = out_npu_blasst.shape[-1]
            print_topk_errors(
                out_npu_blasst, out_blasst, out_npu_baseline,
                threshold_val=threshold,
                head_dim=head_dim,
                topk=GlobalConfig.TOPK_K
            )
        # ------------------------------------------------

        # 详细分析（如果需要）
        if GlobalConfig.DETAILED_ANALYSIS and skip_info:
            detailed_error_analysis(
                out_dense,
                out_blasst,
                out_npu_blasst,
                skip_info,
                block_contributions,
                S_q,
                threshold,
                GlobalConfig.ERROR_THRESHOLD,
            )

        results.append(result)

    # 4. 拼接算子精度分析
    print(f"\n  [4/4] 拼接算子精度分析...")
    concat_result = analyze_concat_operator(
        out_blasst_list, [out_dense], out_npu_blasst, GlobalConfig.THRESHOLD_LOGS, S_q,
        out_dense_from_file=out_dense_from_file
    )

    # 保存张量到文件
    if GlobalConfig.SAVE_TENSORS:
        save_tensors_for_dump(
            dump_path,
            out_npu_blasst_list,
            out_dense,
            out_blasst_list,
            concat_result,
            GlobalConfig.THRESHOLD_LOGS,
            sparse_mode,
            npu_head_mode
        )

    return {
        "layer": data.get("layer_id", "unknown"),
        "dump_file": os.path.basename(dump_path),
        "results": results,
    }


# ============================================================
# 4. 结果汇总与输出（保持不变）
# ============================================================
def print_summary_table(all_results):
    """打印汇总表格，显示稀疏度和四种对比的相对误差"""
    if not all_results:
        return

    print("\n" + "=" * 160)
    print("汇总结果表格")
    print("=" * 160)

    # 表头
    print(
        f"{'Layer':<30} {'λ':<6} {'RowSparsity':<12} {'BlockSparsity':<13} "
        f"{'NPU_Blasst_vs_NPU_Baseline':<22} {'Python_Blasst_vs_Python_Baseline':<22} "
        f"{'NPU_Blasst_vs_Python_Blasst':<22} {'NPU_Baseline_vs_Python_Baseline':<22}"
    )
    print("-" * 160)

    for layer_result in all_results:
        dump_file = layer_result["dump_file"]
        for res in layer_result["results"]:
            # 获取四种相对误差，如果为 None 则显示 "N/A"
            def get_rel(metric):
                if metric is None:
                    return "N/A"
                return f"{metric['rel_error']:.6e}"

            print(
                f"{dump_file:<30} {res['threshold_log']:<6} "
                f"{res['row_sparsity']:<12.2%} "
                f"{res['block_sparsity']:<13.2%} "
                f"{get_rel(res['metrics_npu_blasst_vs_baseline']):<22} "
                f"{get_rel(res['metrics_python_blasst_vs_baseline']):<22} "
                f"{get_rel(res['metrics_blasst_npu_vs_python']):<22} "
                f"{get_rel(res['metrics_baseline_npu_vs_python']):<22}"
            )
        print("-" * 160)


def save_results_to_csv(all_results, filename="blasst_precision_results.csv"):
    """保存结果到 CSV 文件，包含四种对比指标"""
    import csv

    with open(filename, "w", newline="") as csvfile:
        fieldnames = [
            "layer",
            "threshold_log",
            "row_sparsity",
            "block_sparsity",
            "rel_error_npu_blasst_vs_baseline",
            "rel_error_python_blasst_vs_dense",
            "rel_error_npu_blasst_vs_python",
            "rel_error_baseline_npu_vs_dense",
            "token_error_mean",
            "token_error_std",
            "token_error_max",
            "token_error_p95",
            "token_error_p99",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for layer_result in all_results:
            for res in layer_result["results"]:
                def get_rel(metric):
                    return metric["rel_error"] if metric is not None else float('nan')

                writer.writerow(
                    {
                        "layer": layer_result["dump_file"],
                        "threshold_log": res["threshold_log"],
                        "row_sparsity": res["row_sparsity"],
                        "block_sparsity": res["block_sparsity"],
                        "rel_error_npu_blasst_vs_baseline": get_rel(res["metrics_npu_blasst_vs_baseline"]),
                        "rel_error_python_blasst_vs_dense": get_rel(res["metrics_python_blasst_vs_baseline"]),
                        "rel_error_npu_blasst_vs_python": get_rel(res["metrics_blasst_npu_vs_python"]),
                        "rel_error_baseline_npu_vs_dense": get_rel(res["metrics_baseline_npu_vs_python"]),
                        "token_error_mean": res["token_error_mean"],
                        "token_error_std": res["token_error_std"],
                        "token_error_max": res["token_error_max"],
                        "token_error_p95": res["token_error_p95"],
                        "token_error_p99": res["token_error_p99"],
                    }
                )

    print(f"\n结果已保存到: {filename}")


def save_tensors_for_dump(dump_file, out_npu_blasst_list, out_dense, out_blasst_list, concat_result, thresholds, sparse_mode, npu_head_mode):
    """保存张量到文件（修复：每个阈值保存对应的 NPU 结果）"""
    base_name = os.path.splitext(os.path.basename(dump_file))[0]
    match = re.search(r'(\d+)$', base_name)
    timestamp = match.group(1) if match else str(int(datetime.now().timestamp() * 1000))
    save_dir = os.path.join(GlobalConfig.OUTPUT_DIR, "fixblasst", f"sparse{sparse_mode}", timestamp, base_name)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(out_dense.cpu(), os.path.join(save_dir, "out_dense.pt"))

    for i, th in enumerate(thresholds):
        th_str = str(th).replace('.', '_')
        torch.save(out_blasst_list[i].cpu(), os.path.join(save_dir, f"out_blasst_th{th_str}.pt"))
        torch.save(out_npu_blasst_list[i].cpu(), os.path.join(save_dir, f"out_npu_th{th_str}.pt"))

    if concat_result is not None:
        torch.save(concat_result.cpu(), os.path.join(save_dir, "out_concat.pt"))

    meta = {
        "dump_file": dump_file,
        "thresholds": thresholds,
        "sparse_mode": sparse_mode,
        "npu_head_mode": npu_head_mode,
        "saved_files": [f"out_npu_th{str(th).replace('.','_')}.pt" for th in thresholds] +
                      ["out_dense.pt"] +
                      [f"out_blasst_th{str(th).replace('.','_')}.pt" for th in thresholds] +
                      (["out_concat.pt"] if concat_result is not None else [])
    }
    torch.save(meta, os.path.join(save_dir, "metadata.pt"))
    print(f"    张量已保存至: {save_dir}")


# ============================================================
# 5. 主函数（添加新参数）
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="BLASST FIA 精度验证套件（增强版）")
    parser.add_argument(
        "--sparse-mode",
        type=int,
        default=0,
        choices=[0, 3],
        help="稀疏模式: 0=dense, 3=rightDownCausal",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="自定义阈值列表，用逗号分隔，如: -7,-5,-3,-1,0",
    )
    parser.add_argument("--save-csv", action="store_true", help="保存结果到 CSV 文件")
    parser.add_argument(
        "--detailed",
        action="store_true",
        default=False,
        help="是否进行详细误差分析 (默认: False)",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.01,
        help="误差阈值，超过此值的token会被详细分析 (默认: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--npu-head-mode",
        action="store_true",
        help="启用NPU的head处理策略（逐个head计算）",
    )
    parser.add_argument(
        "--threshold-scan",
        action="store_true",
        help="启用阈值扫描模式，分析不同阈值对稀疏度的影响",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=-10.0,
        help="阈值扫描起始值（log scale）",
    )
    parser.add_argument(
        "--threshold-end", type=float, default=0.0, help="阈值扫描结束值（log scale）"
    )
    parser.add_argument("--threshold-steps", type=float, default=20, help="阈值扫描步数")
    
    # ----- 新增参数 -----
    parser.add_argument("--check-consistency", action="store_true",
                        help="启用 NPU 算子一致性检查（重复运行多次）")
    parser.add_argument("--consistency-runs", type=int, default=5,
                        help="NPU 一致性检查的重复次数")
    parser.add_argument("--topk", action="store_true",
                        help="启用 Top‑K 误差位置分析")
    parser.add_argument("--topk-threshold-idx", type=int, default=0,
                        help="用于 Top‑K 分析的阈值索引（0 表示第一个）")
    parser.add_argument("--topk-k", type=int, default=10,
                        help="显示前 K 个最大误差位置")
    # -------------------
    
    args = parser.parse_args()

    if args.thresholds:
        GlobalConfig.THRESHOLD_LOGS = [
            float(x.strip()) for x in args.thresholds.split(",")
        ]
    GlobalConfig.DETAILED_ANALYSIS = args.detailed
    GlobalConfig.ERROR_THRESHOLD = args.error_threshold

    # 设置新增配置
    if args.check_consistency:
        GlobalConfig.NPU_CONSISTENCY_RUNS = args.consistency_runs
    else:
        GlobalConfig.NPU_CONSISTENCY_RUNS = 1
    GlobalConfig.ENABLE_TOPK_ANALYSIS = args.topk
    GlobalConfig.TOPK_THRESHOLD_IDX = args.topk_threshold_idx
    GlobalConfig.TOPK_K = args.topk_k

    print("=" * 80)
    print("BLASST FIA 精度验证套件（增强版）")
    print("=" * 80)
    head_mode_str = "NPU_HeadMode" if args.npu_head_mode else "Standard_HeadMode"
    print(f"  Head Mode: {head_mode_str}")
    print(f"  Sparse Mode: {args.sparse_mode}")
    print(f"  Thresholds: {GlobalConfig.THRESHOLD_LOGS}")
    print(f"  Detailed Analysis: {GlobalConfig.DETAILED_ANALYSIS}")
    print(f"  Error Threshold: {GlobalConfig.ERROR_THRESHOLD * 100}%")
    print(f"  NPU Consistency Runs: {GlobalConfig.NPU_CONSISTENCY_RUNS}")
    print(f"  Top‑K Analysis: {GlobalConfig.ENABLE_TOPK_ANALYSIS}")
    if GlobalConfig.ENABLE_TOPK_ANALYSIS:
        print(f"    Top‑K Threshold Index: {GlobalConfig.TOPK_THRESHOLD_IDX}")
        print(f"    Top‑K K: {GlobalConfig.TOPK_K}")
    print(f"  Device: {GlobalConfig.DEVICE}")
    print("=" * 80)

    all_results = []
    for dump_file in GlobalConfig.DUMP_FILES:
        result = process_single_dump(
            dump_file, sparse_mode=args.sparse_mode, npu_head_mode=args.npu_head_mode
        )
        if result:
            all_results.append(result)

    print_summary_table(all_results)

    if args.save_csv:
        save_results_to_csv(all_results)


if __name__ == "__main__":
    main()