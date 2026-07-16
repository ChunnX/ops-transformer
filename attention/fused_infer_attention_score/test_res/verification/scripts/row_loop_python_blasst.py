import math
import os
import re
import torch
import torch_npu
import numpy as np
from collections import defaultdict

# ============================================================
# 0. 全局配置
# ============================================================


def format_value(value):
    """格式化数值为6位小数"""
    return round(float(value), 6)

def format_list(values):
    """格式化列表为6位小数"""
    return [round(float(v), 6) for v in values]

class GlobalConfig:
    # 仿真参数运行
    THRESHOLD_LOGS = [-7,-3,-2,-1]  # 支持多个阈值批量测试
    QS_BLOCK = 128         # Q 分块大小
    KV_BLOCK = 128         # KV 分块大小 (对齐 NPU s2BaseSize=128)
    SOFTMAX_ROWLOOP_ROWS = 16   # Online Softmax rowloop 内行数 (NPU: rowNumTile=16)
    RESCALEO_ROWLOOP_ROWS = 64  # RescaleO rowloop 内行数 (NPU: rowNumTile=64)woy
    # 设备与精度
    DEVICE = "npu:14"
    COMPUTE_DTYPE = torch.float32

    BASE_PATH = "/home/z00603376/ops-transformer-dev"
    # 输出目录（保存张量）
    OUTPUT_DIR = os.path.join(BASE_PATH, "attention/fused_infer_attention_score/tests/tests/output/blasst_outputs_tensor")
    SAVE_TENSORS = True    # 是否保存张量到文件

    # Baseline NPU 输出（无 BLASST）的根目录
    BASELINE_NPU_DIR = os.path.join(BASE_PATH, "attention/fused_infer_attention_score/tests/tests/baseline")

    # 待处理的 Dump 全量文件列表
    DUMP_FILES = [
        os.path.join(BASE_PATH, "attention/fused_infer_attention_score/tests/tests/input/layer_30_1775227380534.pt"),
    ]

    # Trace 打印 JSON 输出路径
    TRACE_OUTPUT_DIR = os.path.join(BASE_PATH, "attention/fused_infer_attention_score/tests/tests/output/trace_outputs")   # 自定义路径

    # 误差阈值（超过此值的token会被详细分析，仅在 DETAILED_ANALYSIS=True 时生效）
    ERROR_THRESHOLD = 0.01  # 1%
    # 是否进行详细误差分析
    DETAILED_ANALYSIS = True
    VERBOSE_SPARSITY = True
    qs_idx_target = 7135
    h_target = 0


# ============================================================
# 1. BLASST 仿真核心逻辑
# ============================================================
def optimized_blasst_sim(
    Q,
    K,
    V,
    actual_qlen,
    actual_kvlen,
    scale,
    threshold_log=-7.2,
    sparse_mode=0,
    return_block_info=False,
    npu_head_mode=False,
    print_flag=False,
    trace_pos=None,
    trace_head=None,
    trace_output_dir=None
):
    """
    BLASST 仿真核心逻辑
    参数:
        return_block_info: 是否返回详细的块级信息（用于详细分析）
        npu_head_mode: 是否使用NPU的head处理策略（逐个head计算）
        trace_pos: 需要打点追踪的 position（如 202）
        trace_head: 需要打点追踪的 head（如 0）
        trace_output_dir: 打点数据保存目录
    """
    device = Q.device
    # 模拟 NPU fp16 精度：先转 half 再 float，使 S 矩阵计算与 NPU cube 精度一致
    q_f32 = Q.half().float()
    k_f32 = K.half().float()
    v_f32 = V.float()

    S_q = int(actual_qlen.item())
    S_kv = int(actual_kvlen.item())
    H, D = q_f32.shape[1], q_f32.shape[2]

    # 提取有效部分
    q_effective = q_f32[:S_q]
    k_effective = k_f32[:S_kv]
    v_effective = v_f32[:S_kv]

    if npu_head_mode:
        H_kv = k_effective.shape[1]
        group_size = H // H_kv
        k_exp = k_effective.repeat_interleave(group_size, dim=1)
        v_exp = v_effective.repeat_interleave(group_size, dim=1)
    else:
        group_size = H // k_effective.shape[1]
        k_exp = k_effective.repeat_interleave(group_size, dim=1)
        v_exp = v_effective.repeat_interleave(group_size, dim=1)

    gm = torch.full((S_q, H), -1e9, device=device, dtype=torch.float32)
    gl = torch.zeros(S_q, H, device=device, dtype=torch.float32)
    go = torch.zeros(S_q, H, D, device=device, dtype=torch.float32)

    sparse_count, total_count = 0, 0
    sparse_block_count, total_block_count = 0, 0
    mask_factor = -1e9

    # 详细分析信息
    skip_info = None
    block_contributions = None
    if return_block_info:
        skip_info = {
            "blasst_skip_blocks": 0,
            "mask_skip_blocks": 0,
            "total_blocks": 0,
            "skipped_blocks": [],
            "skipped_positions": set(),
        }
        block_contributions = defaultdict(
            lambda: {"processed": False, "skipped": False, "skip_reason": None}
        )

    SUBBLOCK_ROWS = GlobalConfig.QS_BLOCK // 2
    ROWLOOPS_PER_SUBBLOCK = SUBBLOCK_ROWS // GlobalConfig.SOFTMAX_ROWLOOP_ROWS

    qs_block = GlobalConfig.QS_BLOCK
    kv_block = GlobalConfig.KV_BLOCK
    softmax_rowloop_rows = GlobalConfig.SOFTMAX_ROWLOOP_ROWS

    # 打点追踪初始化（按 block 粒度）
    trace_data = []
    enable_trace = (trace_pos is not None) and (trace_head is not None)

    if npu_head_mode:
        kv_s_num = math.ceil(S_kv / kv_block)
        for kv_s_idx in range(kv_s_num):
            k_s, k_e = kv_s_idx * kv_block, min((kv_s_idx + 1) * kv_block, S_kv)
            curr_k = k_exp[k_s:k_e]
            curr_v = v_exp[k_s:k_e]
            is_first_stack = kv_s_idx == 0

            qs_num = math.ceil(S_q / qs_block)
            for qs_idx in range(qs_num):
                q_s, q_e = qs_idx * qs_block, min((qs_idx + 1) * qs_block, S_q)
                curr_q = q_effective[q_s:q_e]

                for h in range(H):
                    # 按 block 粒度检查：Q block 包含目标 position 且 head 匹配
                    is_target_block = enable_trace and (q_s <= trace_pos < q_e) and (h == trace_head)

                    kv_head_idx = h // group_size
                    curr_k_h = k_effective[k_s:k_e, kv_head_idx : kv_head_idx + 1, :]
                    curr_v_h = v_effective[k_s:k_e, kv_head_idx : kv_head_idx + 1, :]
                    curr_q_h = q_effective[q_s:q_e, h : h + 1, :]

                    S_ij = torch.einsum("qhd,khd->qhk", curr_q_h, curr_k_h) * scale
                    if sparse_mode == 3:
                        q_range = torch.arange(q_s, q_e, device=device).view(-1, 1)
                        k_range = torch.arange(k_s, k_e, device=device).view(1, -1)
                        mask = q_range < (k_range - (S_kv - S_q))
                        S_ij.masked_fill_(mask.unsqueeze(1), mask_factor)

                    row_loop_num = math.ceil((q_e - q_s) / softmax_rowloop_rows)
                    block_sparse = True
                    total_block_count += 1

                    if return_block_info:
                        skip_info["total_blocks"] += 1
                        block_key = (kv_s_idx, qs_idx, h)
                        block_contributions[block_key] = {
                            "q_range": (q_s, q_e),
                            "k_range": (k_s, k_e),
                            "kv_idx": kv_s_idx,
                            "qs_idx": qs_idx,
                            "head_idx": h,
                        }

                    block_skipped = False
                    block_skipped_reason = None

                    # 如果是目标 block，初始化 block 级 trace
                    if is_target_block:
                        rel_pos = trace_pos - q_s
                        block_trace = {
                            "kv_block_idx": kv_s_idx,
                            "qs_block_idx": qs_idx,
                            "head_idx": h,
                            "q_range": (q_s, q_e),
                            "k_range": (k_s, k_e),
                            "rel_pos_in_qblock": rel_pos,
                            "is_first_stack": is_first_stack,
                            "rowloops": [],
                        }

                    sp_flag_sb_init = 1 if not is_first_stack else 0
                    sp_flag_sb = [sp_flag_sb_init, sp_flag_sb_init]
                    if (q_e - q_s) <= SUBBLOCK_ROWS:
                        sp_flag_sb[1] = 0

                    for rl_idx in range(row_loop_num):
                        total_count += 1
                        sb_idx = min(rl_idx // ROWLOOPS_PER_SUBBLOCK, 1)
                        sp_flag = sp_flag_sb[sb_idx]
                        r_s, r_e = (
                            rl_idx * softmax_rowloop_rows,
                            min((rl_idx + 1) * softmax_rowloop_rows, q_e - q_s),
                        )
                        abs_s, abs_e = q_s + r_s, q_s + r_e

                        lm = S_ij[r_s:r_e].amax(dim=-1)
                        gm_old = gm[abs_s:abs_e, h : h + 1]

                        diff_max = (lm - gm_old).max().item()
                        is_all_masked = (lm < -1e8).all()
                        is_sparse = (not is_first_stack) and (diff_max < threshold_log)
                        block_sparse = (
                            block_sparse and is_sparse and (not is_all_masked)
                        )

                        is_target_rowloop = is_target_block and (abs_s <= trace_pos < abs_e)
                        if is_target_rowloop:
                            rl_rel_pos = trace_pos - abs_s
                            rl_trace = {
                                "rowloop_idx": rl_idx,
                                "subblock_idx": sb_idx,
                                "q_range": (abs_s, abs_e),
                                "k_range": (k_s, k_e),
                                "rel_pos_in_rowloop": rl_rel_pos,
                                "lm_all": format_list(lm[:, 0].tolist()),
                                "lm_target": format_value(lm[rl_rel_pos, 0].item()),
                                "gm_old_all": format_list(gm_old[:, 0].tolist()),
                                "gm_before": format_value(gm_old[rl_rel_pos, 0].item()),
                                "gl_before": format_value(gl[abs_s:abs_e, h : h + 1][rl_rel_pos, 0].item()),
                                "go_norm_before": format_value(torch.norm(go[abs_s:abs_e, h : h + 1, :][rl_rel_pos, 0, :]).item()),
                                "diff_max": format_value(diff_max),
                                "is_all_masked": is_all_masked.item(),
                                "is_sparse": is_sparse,
                                "sp_flag": sp_flag,
                                "sp_flag_sb": list(sp_flag_sb),
                            }

                        if is_sparse and sp_flag == 1:
                            if not is_all_masked:
                                sparse_count += 1
                                if return_block_info:
                                    block_skipped = True
                                    block_skipped_reason = "blasst"
                                    skip_info["blasst_skip_blocks"] += 1
                                    for pos in range(abs_s, abs_e):
                                        skip_info["skipped_positions"].add(pos)
                                if is_target_rowloop:
                                    rl_trace["action"] = "skipped_blasst"
                                    rl_trace["gm_after"] = rl_trace["gm_before"]
                                    rl_trace["gl_after"] = rl_trace["gl_before"]
                                    rl_trace["go_norm_after"] = rl_trace["go_norm_before"]
                                    rl_trace["sp_flag"] = sp_flag
                                    block_trace["rowloops"].append(rl_trace)
                            else:
                                if return_block_info:
                                    block_skipped = True
                                    block_skipped_reason = "mask"
                                    skip_info["mask_skip_blocks"] += 1
                                if is_target_rowloop:
                                    rl_trace["action"] = "skipped_mask"
                                    rl_trace["gm_after"] = rl_trace["gm_before"]
                                    rl_trace["gl_after"] = rl_trace["gl_before"]
                                    rl_trace["go_norm_after"] = rl_trace["go_norm_before"]
                                    rl_trace["sp_flag"] = sp_flag
                                    block_trace["rowloops"].append(rl_trace)
                            continue

                        sp_flag_sb[sb_idx] = 0

                        hm = torch.maximum(lm, gm_old)
                        dm = torch.exp(gm_old - hm)
                        ls = torch.exp(S_ij[r_s:r_e] - hm.unsqueeze(-1))
                        pv = torch.einsum("qhk,khd->qhd", ls, curr_v_h)

                        if is_target_rowloop:
                            rl_trace["dm_all"] = format_list(dm[:, 0].tolist())
                            rl_trace["dm_target"] = format_value(dm[rl_rel_pos, 0].item())
                            rl_trace["hm_all"] = format_list(hm[:, 0].tolist())
                            rl_trace["hm_target"] = format_value(hm[rl_rel_pos, 0].item())
                            rl_trace["ls_target_row"] = format_list(ls[rl_rel_pos, 0, :].tolist())
                            rl_trace["ls_sum"] = format_value(ls[rl_rel_pos, 0, :].sum().item())
                            rl_trace["pv_norm"] = format_value(torch.norm(pv[rl_rel_pos, 0, :]).item())
                            rl_trace["S_ij_target"] = format_list(S_ij[r_s:r_e, 0 : 1, :][rl_rel_pos, 0, :].tolist())
                            rl_trace["pv_target_row"] = format_list(pv[rl_rel_pos, 0, :].tolist())
                            rl_trace["kv_blk"] = kv_s_idx
                            rl_trace["qs_blk"] = qs_idx
                            rl_trace["head"] = h
                            rl_trace["rowloop_idx"] = rl_idx
                            rl_trace["gm_old_value"] = format_value(gm_old[rl_rel_pos, 0].item())
                            rl_trace["gl_before_value"] = format_value(gl[abs_s:abs_e, h : h + 1][rl_rel_pos, 0].item())
                            rl_trace["go_before_norm"] = format_value(torch.norm(go[abs_s:abs_e, h : h + 1, :][rl_rel_pos, 0, :]).item())

                        go[abs_s:abs_e, h : h + 1, :] = (
                            dm.unsqueeze(-1) * go[abs_s:abs_e, h : h + 1, :] + pv
                        )
                        gl[abs_s:abs_e, h : h + 1] = dm * gl[
                            abs_s:abs_e, h : h + 1
                        ] + ls.sum(dim=-1)
                        gm[abs_s:abs_e, h : h + 1] = hm

                        if is_target_rowloop:
                            rl_trace["action"] = "processed"
                            rl_trace["gm_after_all"] = format_list(gm[abs_s:abs_e, h : h + 1][:, 0].tolist())
                            rl_trace["gm_after"] = format_value(gm[abs_s:abs_e, h : h + 1][rl_rel_pos, 0].item())
                            rl_trace["gl_after"] = format_value(gl[abs_s:abs_e, h : h + 1][rl_rel_pos, 0].item())
                            rl_trace["go_norm_after"] = format_value(torch.norm(go[abs_s:abs_e, h : h + 1, :][rl_rel_pos, 0, :]).item())
                            rl_trace["go_after_row"] = format_list(go[abs_s:abs_e, h : h + 1, :][rl_rel_pos, 0, :].tolist())
                            go_normalized = go[abs_s:abs_e, h : h + 1, :][rl_rel_pos, 0, :] / gl[abs_s:abs_e, h : h + 1][rl_rel_pos, 0]
                            rl_trace["go_normalized_row"] = format_list(go_normalized.tolist())
                            rl_trace["is_last_kv_blk"] = (kv_s_idx == kv_s_num - 1)
                            block_trace["rowloops"].append(rl_trace)

                    if block_sparse:
                        sparse_block_count += 1

                    if return_block_info and block_skipped:
                        skip_info["skipped_blocks"].append(
                            {
                                "kv_block_idx": kv_s_idx,
                                "qs_block_idx": qs_idx,
                                "head_idx": h,
                                "q_range": (q_s, q_e),
                                "k_range": (k_s, k_e),
                                "reason": block_skipped_reason,
                            }
                        )
                        block_contributions[block_key]["skipped"] = True
                        block_contributions[block_key]["skip_reason"] = (
                            block_skipped_reason
                        )
                    elif return_block_info:
                        block_contributions[block_key]["processed"] = True

                    # 目标 block 处理完毕，保存
                    if is_target_block:
                        block_trace["final_gm"] = format_value(gm[trace_pos, h].item())
                        block_trace["final_gl"] = format_value(gl[trace_pos, h].item())
                        block_trace["final_go_norm"] = format_value(torch.norm(go[trace_pos, h, :]).item())
                        trace_data.append(block_trace)
    else:
        kv_s_num = math.ceil(S_kv / kv_block)
        for kv_s_idx in range(kv_s_num):
            k_s, k_e = kv_s_idx * kv_block, min((kv_s_idx + 1) * kv_block, S_kv)
            curr_k = k_exp[k_s:k_e]
            curr_v = v_exp[k_s:k_e]
            is_first_stack = kv_s_idx == 0

            qs_num = math.ceil(S_q / qs_block)
            for qs_idx in range(qs_num):
                q_s, q_e = qs_idx * qs_block, min((qs_idx + 1) * qs_block, S_q)
                curr_q = q_effective[q_s:q_e]

                # 按 block 粒度检查：Q block 包含目标 position
                is_target_block = enable_trace and (q_s <= trace_pos < q_e) and (trace_head < H)

                # S_ij: [q_len_blk, H, k_len_blk]
                S_ij = torch.einsum("qhd,khd->qhk", curr_q, curr_k) * scale

                # sparse_mode=3 掩码应用
                if sparse_mode == 3:
                    q_range = torch.arange(q_s, q_e, device=device).view(-1, 1)
                    k_range = torch.arange(k_s, k_e, device=device).view(1, -1)
                    mask = q_range < (k_range - (S_kv - S_q))
                    S_ij.masked_fill_(mask.unsqueeze(1), mask_factor)

                row_loop_num = math.ceil((q_e - q_s) / softmax_rowloop_rows)
                block_sparse = True
                total_block_count += 1

                if return_block_info:
                    skip_info["total_blocks"] += 1
                    block_key = (kv_s_idx, qs_idx)
                    block_contributions[block_key] = {
                        "q_range": (q_s, q_e),
                        "k_range": (k_s, k_e),
                        "kv_idx": kv_s_idx,
                        "qs_idx": qs_idx,
                    }

                block_skipped = False
                block_skipped_reason = None

                # 如果是目标 block，初始化 block 级 trace
                if is_target_block:
                    rel_pos = trace_pos - q_s
                    block_trace = {
                        "kv_block_idx": kv_s_idx,
                        "qs_block_idx": qs_idx,
                        "head_idx": trace_head,
                        "q_range": (q_s, q_e),
                        "k_range": (k_s, k_e),
                        "rel_pos_in_qblock": rel_pos,
                        "is_first_stack": is_first_stack,
                        "rowloops": [],
                    }

                sp_flag_sb_init = 1 if not is_first_stack else 0
                sp_flag_sb = [sp_flag_sb_init, sp_flag_sb_init]
                if (q_e - q_s) <= SUBBLOCK_ROWS:
                    sp_flag_sb[1] = 0

                for rl_idx in range(row_loop_num):
                    total_count += 1
                    sb_idx = min(rl_idx // ROWLOOPS_PER_SUBBLOCK, 1)
                    sp_flag = sp_flag_sb[sb_idx]
                    r_s, r_e = (
                        rl_idx * softmax_rowloop_rows,
                        min((rl_idx + 1) * softmax_rowloop_rows, q_e - q_s),
                    )
                    abs_s, abs_e = q_s + r_s, q_s + r_e

                    lm = S_ij[r_s:r_e].amax(dim=-1)
                    gm_old = gm[abs_s:abs_e]

                    diff_max = (lm - gm_old).max().item()
                    is_all_masked = (lm < -1e8).all()
                    is_sparse = (not is_first_stack) and (diff_max < threshold_log)
                    block_sparse = block_sparse and is_sparse and (not is_all_masked)

                    is_target_rowloop = is_target_block and (abs_s <= trace_pos < abs_e)
                    if is_target_rowloop:
                        rl_rel_pos = trace_pos - abs_s
                        rl_trace = {
                            "rowloop_idx": rl_idx,
                            "subblock_idx": sb_idx,
                            "q_range": (abs_s, abs_e),
                            "k_range": (k_s, k_e),
                            "rel_pos_in_rowloop": rl_rel_pos,
                            "lm_all": format_list(lm[rl_rel_pos, :].tolist()),
                            "lm_target": format_value(lm[rl_rel_pos, trace_head].item()),
                            "gm_old_all": format_list(gm_old[:, trace_head].tolist()),
                            "gm_before": format_value(gm_old[rl_rel_pos, trace_head].item()),
                            "gl_before": format_value(gl[abs_s:abs_e, trace_head][rl_rel_pos].item()),
                            "go_norm_before": format_value(torch.norm(go[abs_s:abs_e, trace_head, :][rl_rel_pos, :]).item()),
                            "diff_max": format_value(diff_max),
                            "is_all_masked": is_all_masked.item(),
                            "is_sparse": is_sparse,
                            "sp_flag": sp_flag,
                            "sp_flag_sb": list(sp_flag_sb),
                        }

                    if is_sparse and sp_flag == 1:
                        if not is_all_masked:
                            sparse_count += 1
                            if return_block_info:
                                block_skipped = True
                                block_skipped_reason = "blasst"
                                skip_info["blasst_skip_blocks"] += 1
                                for pos in range(abs_s, abs_e):
                                    skip_info["skipped_positions"].add(pos)
                            if is_target_rowloop:
                                rl_trace["action"] = "skipped_blasst"
                                rl_trace["gm_after"] = rl_trace["gm_before"]
                                rl_trace["gl_after"] = rl_trace["gl_before"]
                                rl_trace["go_norm_after"] = rl_trace["go_norm_before"]
                                rl_trace["sp_flag"] = sp_flag
                                block_trace["rowloops"].append(rl_trace)
                        else:
                            if return_block_info:
                                block_skipped = True
                                block_skipped_reason = "mask"
                                skip_info["mask_skip_blocks"] += 1
                            if is_target_rowloop:
                                rl_trace["action"] = "skipped_mask"
                                rl_trace["gm_after"] = rl_trace["gm_before"]
                                rl_trace["gl_after"] = rl_trace["gl_before"]
                                rl_trace["go_norm_after"] = rl_trace["go_norm_before"]
                                rl_trace["sp_flag"] = sp_flag
                                block_trace["rowloops"].append(rl_trace)
                        continue

                    sp_flag_sb[sb_idx] = 0

                    hm = torch.maximum(lm, gm_old)
                    dm = torch.exp(gm_old - hm)
                    ls = torch.exp(S_ij[r_s:r_e] - hm.unsqueeze(-1))
                    pv = torch.einsum("qhk,khd->qhd", ls, curr_v)

                    if is_target_rowloop:
                        rl_trace["dm_all"] = format_list(dm[:, trace_head].tolist())
                        rl_trace["dm_target"] = format_value(dm[rl_rel_pos, trace_head].item())
                        rl_trace["hm_all"] = format_list(hm[:, trace_head].tolist())
                        rl_trace["hm_target"] = format_value(hm[rl_rel_pos, trace_head].item())
                        rl_trace["ls_target_row"] = format_list(ls[rl_rel_pos, trace_head, :].tolist())
                        rl_trace["ls_sum"] = format_value(ls[rl_rel_pos, trace_head, :].sum().item())
                        rl_trace["pv_norm"] = format_value(torch.norm(pv[rl_rel_pos, trace_head, :]).item())
                        rl_trace["S_ij_target"] = format_list(S_ij[r_s:r_e, :, :][rl_rel_pos, trace_head, :].tolist())
                        rl_trace["pv_target_row"] = format_list(pv[rl_rel_pos, trace_head, :].tolist())
                        rl_trace["kv_blk"] = kv_s_idx
                        rl_trace["qs_blk"] = qs_idx
                        rl_trace["rowloop_idx"] = rl_idx
                        rl_trace["gm_old_value"] = format_value(gm_old[rl_rel_pos, trace_head].item())
                        rl_trace["gl_before_value"] = format_value(gl[abs_s:abs_e, trace_head][rl_rel_pos].item())
                        rl_trace["go_before_norm"] = format_value(torch.norm(go[abs_s:abs_e, trace_head, :][rl_rel_pos, :]).item())

                    go[abs_s:abs_e] = dm.unsqueeze(-1) * go[abs_s:abs_e] + pv
                    gl[abs_s:abs_e] = dm * gl[abs_s:abs_e] + ls.sum(dim=-1)
                    gm[abs_s:abs_e] = hm

                    if is_target_rowloop:
                        rl_trace["action"] = "processed"
                        rl_trace["gm_after_all"] = format_list(gm[abs_s:abs_e, trace_head].tolist())
                        rl_trace["gm_after"] = format_value(gm[abs_s:abs_e, trace_head][rl_rel_pos].item())
                        rl_trace["gl_after"] = format_value(gl[abs_s:abs_e, trace_head][rl_rel_pos].item())
                        rl_trace["go_norm_after"] = format_value(torch.norm(go[abs_s:abs_e, trace_head, :][rl_rel_pos, :]).item())
                        rl_trace["go_after_row"] = format_list(go[abs_s:abs_e, trace_head, :][rl_rel_pos, :].tolist())
                        go_normalized = go[abs_s:abs_e, trace_head, :][rl_rel_pos, :] / gl[abs_s:abs_e, trace_head][rl_rel_pos]
                        rl_trace["go_normalized_row"] = format_list(go_normalized.tolist())
                        rl_trace["is_last_kv_blk"] = (kv_s_idx == kv_s_num - 1)
                        block_trace["rowloops"].append(rl_trace)

                if block_sparse:
                    sparse_block_count += 1

                if return_block_info and block_skipped:
                    skip_info["skipped_blocks"].append(
                        {
                            "kv_block_idx": kv_s_idx,
                            "qs_block_idx": qs_idx,
                            "q_range": (q_s, q_e),
                            "k_range": (k_s, k_e),
                            "reason": block_skipped_reason,
                        }
                    )
                    block_contributions[block_key]["skipped"] = True
                    block_contributions[block_key]["skip_reason"] = block_skipped_reason
                elif return_block_info:
                    block_contributions[block_key]["processed"] = True

                # 目标 block 处理完毕，保存
                if is_target_block:
                    block_trace["final_gm"] = format_value(gm[trace_pos, trace_head].item())
                    block_trace["final_gl"] = format_value(gl[trace_pos, trace_head].item())
                    block_trace["final_go_norm"] = format_value(torch.norm(go[trace_pos, trace_head, :]).item())
                    trace_data.append(block_trace)

    output = go / (gl.unsqueeze(-1) + 1e-9)
    full_output = torch.zeros(
        (q_f32.shape[0], H, D), device=device, dtype=torch.float32
    )
    full_output[:S_q] = output

    block_sparsity = sparse_block_count / (total_block_count + 1e-9)
    row_sparsity = sparse_count / (total_count + 1e-9)

    # 保存打点数据
    if enable_trace and trace_data:
        import json
        output_row = format_list(output[trace_pos, trace_head, :].tolist())
        trace_summary = {
            "trace_pos": trace_pos,
            "trace_head": trace_head,
            "threshold_log": threshold_log,
            "sparse_mode": sparse_mode,
            "S_q": S_q,
            "S_kv": S_kv,
            "H": H,
            "D": D,
            "qs_block": qs_block,
            "kv_block": kv_block,
            "softmax_rowloop_rows": softmax_rowloop_rows,
            "subblock_rows": SUBBLOCK_ROWS,
            "rowloops_per_subblock": ROWLOOPS_PER_SUBBLOCK,
            "total_trace_blocks": len(trace_data),
            "total_rowloops": sum(len(bt["rowloops"]) for bt in trace_data),
            "processed_rowloops": sum(
                sum(1 for rl in bt["rowloops"] if rl["action"] == "processed")
                for bt in trace_data
            ),
            "skipped_blasst_rowloops": sum(
                sum(1 for rl in bt["rowloops"] if rl["action"] == "skipped_blasst")
                for bt in trace_data
            ),
            "skipped_mask_rowloops": sum(
                sum(1 for rl in bt["rowloops"] if rl["action"] == "skipped_mask")
                for bt in trace_data
            ),
            "final_output": output_row,
        }

        # 序列化 trace_data
        trace_entries_serializable = []
        for bt in trace_data:
            serializable = {}
            for k, v in bt.items():
                if k == "rowloops":
                    serializable[k] = v
                elif isinstance(v, torch.Tensor):
                    serializable[k] = v.tolist()
                else:
                    serializable[k] = v
            trace_entries_serializable.append(serializable)

        if trace_output_dir:
            os.makedirs(trace_output_dir, exist_ok=True)
            trace_file = os.path.join(trace_output_dir, f"trace_pos{trace_pos}_head{trace_head}_th{threshold_log}.json")
            with open(trace_file, "w") as f:
                json.dump({"summary": trace_summary, "blocks": trace_entries_serializable}, f, indent=2)
            print(f"\n[TRACE] 打点数据已保存至: {trace_file}")
            print(f"[TRACE] 共记录 {len(trace_data)} 个 Q-KV block 对")
            print(f"[TRACE] 总 rowloop 数: {trace_summary['total_rowloops']}")
            print(f"[TRACE]   - 实际处理: {trace_summary['processed_rowloops']} 次")
            print(f"[TRACE]   - BLASST跳过: {trace_summary['skipped_blasst_rowloops']} 次")
            print(f"[TRACE]   - Mask跳过: {trace_summary['skipped_mask_rowloops']} 次")
        else:
            print(f"\n[TRACE] pos={trace_pos}, head={trace_head} 打点数据:")
            print(f"  共记录 {len(trace_data)} 个 Q-KV block 对")
            for bt in trace_entries_serializable:
                print(f"\n  Block: kv_blk={bt['kv_block_idx']}, qs_blk={bt['qs_block_idx']}, head={bt['head_idx']}")
                print(f"    Q range: {bt['q_range']}, K range: {bt['k_range']}")
                for rl in bt['rowloops']:
                    action = rl.get("action", "unknown")
                    print(f"    rl={rl['rowloop_idx']}, q={rl['q_range']}, action={action}")
                    if action == "processed":
                        print(f"      gm: {rl['gm_before']:.4f} -> {rl['gm_after']:.4f}")
                        print(f"      gl: {rl['gl_before']:.4f} -> {rl['gl_after']:.4f}")
                        print(f"      go_norm: {rl['go_norm_before']:.6e} -> {rl['go_norm_after']:.6e}")
                        print(f"      pv_norm: {rl['pv_norm']:.6e}")
                    else:
                        print(f"      (skipped, gm/gl/go unchanged)")

    if return_block_info:
        return full_output, row_sparsity, block_sparsity, skip_info, block_contributions
    else:
        return full_output, row_sparsity, block_sparsity