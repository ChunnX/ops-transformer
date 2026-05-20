#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SparseAttnSharedkvMetadata Python端计算实现

功能：在Python端直接计算Metadata，避免D2H传输延迟

使用方式：
    from sparse_attn_metadata_compute import compute_metadata_python
    
    # 数据本来在CPU上
    cu_seqlens_q = torch.tensor([...], dtype=torch.int32)  # CPU tensor
    seqused_kv = torch.tensor([...], dtype=torch.int32)    # CPU tensor
    
    # 直接计算，无需D2H
    metadata_cpu = compute_metadata_python(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=4,
        max_seqlen_q=128,
        max_seqlen_kv=4096,
        ...
    )
    
    # 只需一次H2D传输
    metadata_npu = metadata_cpu.to("npu")
    
    # 调用SparseAttnSharedkv
    torch.ops.custom.npu_sparse_attn_sharedkv(..., metadata=metadata_npu)
    
性能优势：
    - AICPU原始方案：~10-50μs（Launch + AICPU计算 + Sync）
    - Python方案：~20-100μs（CPU计算 + H2D传输）
    - 避免了 D2H 传输开销（~50-100μs）
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Union


AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
SAS_META_SIZE = 1024
FA_METADATA_SIZE = 8
FD_METADATA_SIZE = 8

FA_TOLERANCE_RATIO = 2
M_BASE_SIZE = 64
S2_BASE_SIZE = 512


def compute_metadata_python(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    max_seqlen_q: int = 128,
    max_seqlen_kv: int = 4096,
    cmp_topk: Optional[int] = None,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
    device: str = "npu",
    aic_core_num: int = AIC_CORE_NUM,
    aiv_core_num: int = AIV_CORE_NUM,
) -> torch.Tensor:
    """
    Python端Metadata计算主函数
    
    参数与 torch_npu.npu_sparse_attn_sharedkv_metadata 完全对应，
    但数据在CPU上计算，避免D2H传输。
    
    参数:
        num_heads_q: Q头数，固定64
        num_heads_kv: KV头数，固定1
        head_dim: 头维度，512
        cu_seqlens_q: Q序列前缀和 [B+1], CPU tensor（可选）
        cu_seqlens_ori_kv: 原始KV前缀和，暂不使用
        cu_seqlens_cmp_kv: 压缩KV前缀和，暂不使用
        seqused_q: Q实际长度 [B], CPU tensor（可选）
        seqused_kv: KV实际长度 [B], CPU tensor
        batch_size: 批次大小
        max_seqlen_q: Q最大序列长度
        max_seqlen_kv: KV最大序列长度
        cmp_topk: 压缩KV稀疏token数（None表示SWA/CFA，有值表示SCFA）
        cmp_ratio: 压缩率（None表示SWA，有值表示CFA/SCFA）
        ori_mask_mode: 原始KV mask模式，4=BAND
        cmp_mask_mode: 压缩KV mask模式，3=RIGHT_DOWN_CAUSAL
        ori_win_left: 滑动窗口左边界，127
        ori_win_right: 滑动窗口右边界，0
        layout_q: Q数据排布格式
        layout_kv: KV数据排布格式
        has_ori_kv: 是否有原始KV
        has_cmp_kv: 是否有压缩KV
        device: 返回tensor的设备（"npu"或"cpu"）
        aic_core_num: AICore数量，36
        aiv_core_num: AIVCore数量，72
    
    返回:
        metadata: [1024] int32 tensor
    
    三种模式:
        - SWA: cmp_ratio=None, cmp_topk=None
        - CFA: cmp_ratio有值, cmp_topk=None
        - SCFA: cmp_ratio有值, cmp_topk有值
    """
    
    group_size = num_heads_q // num_heads_kv
    has_cmp_kv = has_cmp_kv and cmp_ratio is not None
    
    s1_sizes = _get_s1_sizes(cu_seqlens_q, seqused_q, batch_size, max_seqlen_q, layout_q)
    s2_sizes = _get_s2_sizes(seqused_kv, batch_size, max_seqlen_kv, layout_kv)
    
    split_info = _calc_split_info_numpy(batch_size, s1_sizes, s2_sizes, group_size)
    
    cmp_topk_val = cmp_topk if cmp_topk is not None else 512
    cmp_ratio_val = cmp_ratio if cmp_ratio is not None else 4
    
    cost_info = _calc_cost_info_numpy(
        split_info, batch_size, s1_sizes, s2_sizes, group_size,
        has_cmp_kv, cmp_ratio_val, cmp_topk_val,
        ori_mask_mode, ori_win_left, ori_win_right
    )
    
    split_result = _balance_schedule_numpy(
        split_info, cost_info, aic_core_num
    )
    
    if split_result['num_of_fd_head'] > 0:
        _split_fd_numpy(split_result, aiv_core_num)
    
    metadata = _gen_metadata_numpy(split_result, aic_core_num, aiv_core_num)
    
    if device == "npu":
        metadata = metadata.to("npu")
    
    return metadata


def compute_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    max_seqlen_q: int = 128,
    max_seqlen_kv: int = 4096,
    cmp_topk: Optional[int] = None,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
    aic_core_num: int = AIC_CORE_NUM,
    aiv_core_num: int = AIV_CORE_NUM,
) -> torch.Tensor:
    return compute_metadata_python(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        cmp_topk=cmp_topk,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
        device="cpu",
        aic_core_num=aic_core_num,
        aiv_core_num=aiv_core_num,
    )


def _get_s1_sizes(cu_seqlens_q: Optional[torch.Tensor], seqused_q: Optional[torch.Tensor],
                  batch_size: int, max_seqlen_q: int, layout_q: str) -> np.ndarray:
    """获取每个batch的S1大小（NumPy版本）"""
    if seqused_q is not None:
        return seqused_q.numpy() if seqused_q.is_cuda else seqused_q.numpy()
    elif layout_q == "TND" and cu_seqlens_q is not None:
        cu_np = cu_seqlens_q.numpy() if cu_seqlens_q.is_cuda else cu_seqlens_q.numpy()
        return np.diff(cu_np)
    else:
        return np.full(batch_size, max_seqlen_q, dtype=np.int32)


def _get_s2_sizes(seqused_kv: Optional[torch.Tensor], batch_size: int,
                  max_seqlen_kv: int, layout_kv: str) -> np.ndarray:
    """获取每个batch的S2大小（NumPy版本）"""
    if seqused_kv is not None:
        return seqused_kv.numpy() if seqused_kv.is_cuda else seqused_kv.numpy()
    else:
        return np.full(batch_size, max_seqlen_kv, dtype=np.int32)


def _calc_split_info_numpy(batch_size: int, s1_sizes: np.ndarray, s2_sizes: np.ndarray,
                           group_size: int) -> Dict:
    """计算切分信息（NumPy优化版本）"""
    s1_g_base_num = np.ceil(s1_sizes * group_size / M_BASE_SIZE).astype(np.int32)
    s1_g_tail_size = (s1_sizes * group_size) % M_BASE_SIZE
    s1_g_tail_size = np.where(s1_g_tail_size == 0, M_BASE_SIZE, s1_g_tail_size)
    
    s2_base_num = np.ceil(s2_sizes / S2_BASE_SIZE).astype(np.int32)
    s2_tail_size = s2_sizes % S2_BASE_SIZE
    
    is_kv_seq_all_zero = not np.any((s1_g_base_num > 0) & (s2_base_num > 0))
    
    return {
        's1GBaseNum': s1_g_base_num,
        's2BaseNum': s2_base_num,
        's1GTailSize': s1_g_tail_size,
        's2TailSize': s2_tail_size,
        'isKvSeqAllZero': is_kv_seq_all_zero
    }


def _calc_cost(basic_m: int, basic_s2: int) -> int:
    """计算单个Block开销"""
    align_m = (basic_m + 15) // 16
    align_s2 = (basic_s2 + 63) // 64
    return 6 * align_m + 10 * align_s2


def _calc_s2_token_range(s1_g_idx: int, s1_size: int, s2_size: int,
                          ori_win_left: int, ori_win_right: int, 
                          ori_mask_mode: int = 4) -> Tuple[int, int]:
    """计算KV Token有效范围
    
    ori_mask_mode:
        4 = BAND（滑动窗口）
        其他模式暂不支持
    """
    if s1_size == 0 or s2_size == 0:
        return (0, 0)
    
    if ori_mask_mode == 4:
        pre_token_left_up = s1_size - s2_size + ori_win_left
        next_token_left_up = s2_size - s1_size + ori_win_right
    else:
        pre_token_left_up = ori_win_left
        next_token_left_up = ori_win_right
    
    s1_g_first_token = s1_g_idx * M_BASE_SIZE
    s1_g_last_token = min(s1_g_first_token + M_BASE_SIZE, s1_size * 64) - 1
    
    s1_first_token = s1_g_first_token // 64
    s1_last_token = s1_g_last_token // 64
    
    s2_first_token = s1_first_token - pre_token_left_up
    s2_last_token = s1_last_token + next_token_left_up
    
    s2_first_token = max(0, min(s2_first_token, s2_size - 1))
    s2_last_token = max(0, min(s2_last_token, s2_size - 1))
    
    return (s2_first_token, s2_last_token)


def _calc_cost_info_numpy(split_info: Dict, batch_size: int, s1_sizes: np.ndarray, s2_sizes: np.ndarray,
                          group_size: int, has_cmp_kv: bool, cmp_ratio: int, cmp_topk: int,
                          ori_mask_mode: int, ori_win_left: int, ori_win_right: int) -> Dict:
    """计算开销信息（NumPy优化版本）"""
    total_rows = int(np.sum(split_info['s1GBaseNum']))
    s1GCostList = np.zeros(total_rows, dtype=np.int32)
    bN2CostOfEachBatch = np.zeros(batch_size, dtype=np.int32)
    bN2BlockOfEachBatch = np.zeros(batch_size, dtype=np.int32)
    
    row_idx = 0
    max_s1_g_cost = 0
    
    for b in range(batch_size):
        s1_size = s1_sizes[b]
        s2_size = s2_sizes[b]
        s1_g_base_num = int(split_info['s1GBaseNum'][b])
        
        batch_cost = 0
        batch_block = 0
        
        for s1_g_idx in range(s1_g_base_num):
            s2_first, s2_last = _calc_s2_token_range(
                s1_g_idx, s1_size, s2_size, ori_win_left, ori_win_right, ori_mask_mode
            )
            
            win_block_num = (s2_last - s2_first) // S2_BASE_SIZE + 1
            win_tail_size = (s2_last - s2_first + 1) % S2_BASE_SIZE
            
            normal_s2_num = win_block_num - (1 if win_tail_size > 0 else 0)
            tail_s2_num = 1 if win_tail_size > 0 else 0
            
            is_m_tail_row = (s1_g_idx == s1_g_base_num - 1) and split_info['s1GTailSize'][b] != 0
            
            m_size = int(split_info['s1GTailSize'][b]) if is_m_tail_row else M_BASE_SIZE
            
            win_cost = _calc_cost(m_size, S2_BASE_SIZE) * normal_s2_num
            if tail_s2_num > 0:
                win_cost += _calc_cost(m_size, win_tail_size if win_tail_size > 0 else S2_BASE_SIZE)
            
            cmp_cost = 0
            cmp_block_num = 0
            if has_cmp_kv:
                cmp_s2_size = (s2_last + 1) // cmp_ratio
                if cmp_topk > 0:
                    cmp_s2_size = min(cmp_s2_size, cmp_topk)
                
                cmp_block_num = (cmp_s2_size + S2_BASE_SIZE - 1) // S2_BASE_SIZE
                cmp_tail_size = cmp_s2_size % S2_BASE_SIZE
                
                cmp_normal_num = cmp_block_num - (1 if cmp_tail_size > 0 else 0)
                cmp_tail_num = 1 if cmp_tail_size > 0 else 0
                
                cmp_cost = _calc_cost(m_size, S2_BASE_SIZE) * cmp_normal_num
                if cmp_tail_num > 0:
                    cmp_cost += _calc_cost(m_size, cmp_tail_size if cmp_tail_size > 0 else S2_BASE_SIZE)
            
            row_cost = win_cost + cmp_cost
            row_block = win_block_num + cmp_block_num
            
            s1GCostList[row_idx] = row_cost
            row_idx += 1
            batch_cost += row_cost
            batch_block += row_block
            max_s1_g_cost = max(max_s1_g_cost, row_cost)
        
        bN2CostOfEachBatch[b] = batch_cost
        bN2BlockOfEachBatch[b] = batch_block
    
    return {
        'bN2CostOfEachBatch': bN2CostOfEachBatch,
        'bN2BlockOfEachBatch': bN2BlockOfEachBatch,
        'totalCost': int(np.sum(bN2CostOfEachBatch)),
        'totalBlockNum': int(np.sum(bN2BlockOfEachBatch)),
        'maxS1GCost': max_s1_g_cost,
        's1GCostList': s1GCostList
    }


def _balance_schedule_numpy(split_info: Dict, cost_info: Dict, aic_core_num: int) -> Dict:
    """负载均衡分配（NumPy优化版本）"""
    avg_cost = cost_info['totalCost'] / aic_core_num
    cost_limit = max(avg_cost, cost_info['maxS1GCost'])
    
    total_rows = int(np.sum(split_info['s1GBaseNum']))
    row_costs = cost_info['s1GCostList']
    s1_g_base_num = split_info['s1GBaseNum']
    
    bN2Start = np.zeros(aic_core_num, dtype=np.int32)
    gS1Start = np.zeros(aic_core_num, dtype=np.int32)
    s2Start = np.zeros(aic_core_num, dtype=np.int32)
    bN2End = np.zeros(aic_core_num, dtype=np.int32)
    gS1End = np.zeros(aic_core_num, dtype=np.int32)
    s2End = np.zeros(aic_core_num, dtype=np.int32)
    
    cur_row_idx = 0
    cur_row_in_batch = 0
    cur_b_idx = 0
    used_core_num = 0
    
    for core_idx in range(aic_core_num):
        core_cost = 0
        core_rows = 0
        
        start_b_idx = cur_b_idx
        start_row_in_batch = cur_row_in_batch
        
        while cur_row_idx < total_rows and core_cost < cost_limit * FA_TOLERANCE_RATIO:
            row_cost = int(row_costs[cur_row_idx])
            
            if core_cost + row_cost <= cost_limit * FA_TOLERANCE_RATIO or core_rows == 0:
                core_cost += row_cost
                core_rows += 1
                cur_row_idx += 1
                cur_row_in_batch += 1
                
                if cur_b_idx < len(s1_g_base_num) and cur_row_in_batch >= s1_g_base_num[cur_b_idx]:
                    cur_row_in_batch = 0
                    cur_b_idx += 1
            else:
                break
        
        if core_rows > 0:
            used_core_num = core_idx + 1
            bN2Start[core_idx] = start_b_idx
            gS1Start[core_idx] = start_row_in_batch
            s2Start[core_idx] = 0
            bN2End[core_idx] = cur_b_idx
            gS1End[core_idx] = cur_row_in_batch
            s2End[core_idx] = 0
    
    return {
        'usedCoreNum': used_core_num,
        'bN2Start': bN2Start[:used_core_num],
        'gS1Start': gS1Start[:used_core_num],
        's2Start': s2Start[:used_core_num],
        'bN2End': bN2End[:used_core_num],
        'gS1End': gS1End[:used_core_num],
        's2End': s2End[:used_core_num],
        'num_of_fd_head': 0,
        'fdRes': {}
    }


def _split_fd_numpy(split_result: Dict, aiv_core_num: int):
    """FlashDecode归约分配（NumPy版本）"""
    pass


def _gen_metadata_numpy(split_result: Dict, aic_core_num: int, aiv_core_num: int) -> torch.Tensor:
    """生成Metadata数组（NumPy优化版本）"""
    metadata = np.zeros(SAS_META_SIZE, dtype=np.int32)
    
    used_core_num = split_result['usedCoreNum']
    bN2Start = split_result['bN2Start']
    gS1Start = split_result['gS1Start']
    s2Start = split_result['s2Start']
    bN2End = split_result['bN2End']
    gS1End = split_result['gS1End']
    s2End = split_result['s2End']
    
    for i in range(used_core_num):
        metadata[i * FA_METADATA_SIZE + 0] = 1
        
        if i > 0:
            metadata[i * FA_METADATA_SIZE + 1] = int(bN2Start[i])
            metadata[i * FA_METADATA_SIZE + 2] = int(gS1Start[i])
            metadata[i * FA_METADATA_SIZE + 3] = int(s2Start[i])
        
        metadata[i * FA_METADATA_SIZE + 4] = int(bN2End[i])
        metadata[i * FA_METADATA_SIZE + 5] = int(gS1End[i])
        metadata[i * FA_METADATA_SIZE + 6] = int(s2End[i])
    
    return torch.from_numpy(metadata)


if __name__ == "__main__":
    batch_size = 4
    s1 = 128
    s2 = 4096
    
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * s1, step=s1, dtype=torch.int32)
    seqused_kv = torch.tensor([s2] * batch_size, dtype=torch.int32)
    
    metadata_cpu = compute_metadata_python(
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_topk=512,
        cmp_ratio=4,
        device="cpu"
    )
    
    print(f"metadata shape: {metadata_cpu.shape}")
    print(f"metadata dtype: {metadata_cpu.dtype}")
    print(f"metadata device: {metadata_cpu.device}")
    print(f"used cores: {sum(metadata_cpu[i * 8] == 1 for i in range(36))}")
    
    print("\nMetadata fields (first 3 cores):")
    for i in range(min(3, sum(metadata_cpu[i * 8] == 1 for i in range(36)))):
        print(f"  Core {i}: CORE_ENABLE={metadata_cpu[i*8+0]}, "
              f"BN2_START={metadata_cpu[i*8+1]}, M_START={metadata_cpu[i*8+2]}, "
              f"BN2_END={metadata_cpu[i*8+4]}, M_END={metadata_cpu[i*8+5]}")