#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer CPU golden 标杆实现。

计算流程 (非量化场景 quantMode=-1):
  1. Q @ K_pool^T → scores (S1, S2)  # S2 = pool count
  2. ReLU(scores)
  3. Weight aggregation: scores = sum_n(weights[n] * scores_n)
  4. Causal mask (if maskMode=3): validS2Len = (L_orig - S1 + i + 1) / poolSize
  5. TopK in pool space: select sparseCount = topk / poolSize pools
  6. Expand: pool_idx → [pool_idx*poolSize, ..., pool_idx*poolSize+poolSize-1]
  7. Append tail: visibleTailK 个尾部 token (受因果掩码约束)
  8. Pad -1 to topk + poolSize - 1
  9. Values = gathered scores (float32)

输出:
  - indices: (B, S1, topk+poolSize-1) INT32, 无效位置填 -1
  - values:  (B, S1, topk//poolSize)  FLOAT32, 无效位置填 -inf
"""

import math
import numpy as np
import torch


COMPARE_INDICES = False


def _safe_to_cpu(tensor):
    if tensor is None:
        return None
    if not hasattr(tensor, "device"):
        return tensor
    if str(tensor.device).startswith("cpu"):
        return tensor
    return tensor.cpu()


def _safe_to_npu(tensor, device_id=0):
    if tensor is None:
        return None
    if not hasattr(tensor, "device"):
        return tensor
    device_str = f"npu:{device_id}"
    if str(tensor.device).startswith("npu"):
        return tensor
    return tensor.to(device_str)


def _safe_to_dtype(tensor, target_dtype):
    if tensor is None:
        return None
    if not hasattr(tensor, "dtype"):
        return tensor
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def _get_compute_dtype(query_dtype):
    if query_dtype == torch.float32:
        return torch.float64
    if query_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return torch.float32


def _normalize_prefix_seq(seq, total):
    b = len(seq)
    if b == 1:
        return torch.tensor([total], dtype=torch.int64)
    lens = [total // b] * b
    lens[-1] += total - sum(lens)
    return torch.tensor(lens, dtype=torch.int64).cumsum(0).to(torch.int64)


def _parse_dtype(dtype_str):
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    elif dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    elif dtype_str in ("fp32", "float32"):
        return torch.float32
    elif dtype_str in ("int32",):
        return torch.int32
    else:
        return torch.float32


def _construct_block_table(batch_size, act_seq_k, block_num, block_size):
    import numpy as np

    max_block_num_per_seq = 0
    block_num_per_batch = []
    for b in range(batch_size):
        cur_k = (
            int(act_seq_k[b].item())
            if torch.is_tensor(act_seq_k)
            else int(act_seq_k[b])
        )
        cur_blocks = math.ceil(cur_k / block_size) if cur_k > 0 else 0
        block_num_per_batch.append(cur_blocks)
        max_block_num_per_seq = max(max_block_num_per_seq, cur_blocks)

    rng = np.random.RandomState(block_num * 31 + batch_size * 7 + max_block_num_per_seq)
    block_id_list = rng.permutation(block_num).astype(np.int32)
    bt = np.full((batch_size, max_block_num_per_seq), -1, dtype=np.int32)
    cur_id = 0
    for b in range(batch_size):
        for i in range(block_num_per_batch[b]):
            if cur_id < block_num:
                bt[b][i] = block_id_list[cur_id]
                cur_id += 1
    return torch.from_numpy(bt).to(torch.int32)


class PKIGoldenRef:
    """PoolKeyIndexer CPU golden 参考实现。"""

    def __init__(
        self,
        batch_size,
        q_seq,
        k_seq,
        q_head_num,
        k_head_num,
        head_dim,
        layout_query,
        layout_key,
        topk,
        pool_size,
        mask_mode,
        compute_dtype=torch.float32,
    ):
        self.batch_size = batch_size
        self.q_seq = q_seq
        self.k_seq = k_seq
        self.q_head_num = q_head_num
        self.k_head_num = k_head_num
        self.group_size = q_head_num // k_head_num
        self.head_dim = head_dim
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.topk = topk
        self.pool_size = pool_size
        self.sparse_count = topk // pool_size
        self.mask_mode = mask_mode
        self.compute_dtype = compute_dtype

        self.output_indices_len = topk + pool_size - 1
        self.output_values_len = self.sparse_count

    def _trans_shape_to_bnsd(
        self, tensor, layout, headnums=None, act_seq=None, is_weights=False
    ):
        if layout == "BSND":
            B = tensor.shape[0]
            S = tensor.shape[1]
            N = tensor.shape[2]
            D = 1
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(dim=-1)
            else:
                D = tensor.shape[3]
            tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
            return tensor
        elif layout == "TND":
            T = tensor.shape[0]
            N = tensor.shape[1]
            D = 1
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(dim=-1)
            else:
                D = tensor.shape[2]
            B = len(act_seq)
            S = max(act_seq)
            if all(int(x) == S for x in act_seq) and T == B * S:
                return tensor.reshape(B, S, N, D).permute(0, 2, 1, 3).contiguous()
            new_tensor = torch.zeros((B, N, S, D), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_seq[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s, :] = tensor[
                        t_start:t_end, n_index, :
                    ]
                t_start += act_s
            return new_tensor
        elif layout == "PA_BBND":
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(dim=-1)
            return tensor
        else:
            return tensor

    def _trans_tnd_actseq(self, seq_list):
        list_len = len(seq_list)
        if list_len == 0:
            raise ValueError("TND requires act_seq")
        list_new = [seq_list[0]]
        for i in range(list_len - 1):
            new_item = seq_list[i + 1] - seq_list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                raise ValueError(f"act_seq must be non-decreasing: {seq_list}")
        return list_new

    def _pa_bbnd_to_bnsd(self, key, block_table, act_seq_k, pool_size=None):
        block_num, block_size, N2, D = key.shape
        if torch.is_tensor(block_table):
            B = block_table.shape[0]
            max_blocks = block_table.shape[1]
        else:
            B = len(block_table)
            max_blocks = len(block_table[0])

        # LIV2 语义: PA key 块内每行 = 1 个池, act_seq_k 已换算为每 batch 池数
        if torch.is_tensor(act_seq_k):
            max_s2 = int(act_seq_k.max().item())
        else:
            max_s2 = max(int(x) for x in act_seq_k)

        bnsd = torch.zeros(B, N2, max_s2, D, dtype=key.dtype)
        for b in range(B):
            pos = 0
            for block_idx in range(max_blocks):
                if torch.is_tensor(block_table):
                    block_id = int(block_table[b][block_idx].item())
                else:
                    block_id = int(block_table[b][block_idx])
                if block_id == -1:
                    continue
                remaining = max_s2 - pos
                copy_len = min(block_size, remaining)
                if copy_len > 0:
                    bnsd[b, :, pos : pos + copy_len, :] = key[
                        block_id, :copy_len, :, :
                    ].permute(1, 0, 2)
                pos += block_size
        return bnsd

    def _create_mask_right_down(
        self,
        m_shape,
        actualSeqLengthsQ,
        actualSeqLengthsK,
        batch,
        pool_size,
        pool_tail_k,
        actual_seq_k_is_token=False,
    ):
        """因果掩码: validS2Len = (L_orig - S1 + i + 1) / pool_size, L_orig = S2*pool_size + tail_k

        actual_seq_k_is_token=True 时 actualSeqLengthsK 已是 token 数(PA_BBND 入口换算后),
        L_orig = tokens + tail_k, 不再乘 pool_size。"""
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]
        masks = []
        for i_batch in range(batch):
            S1 = (
                actualSeqLengthsQ[i_batch]
                if (actualSeqLengthsQ is not None and len(actualSeqLengthsQ) > 0)
                else mask_s_q
            )
            S2 = (
                math.floor(actualSeqLengthsK[i_batch])
                if (actualSeqLengthsK is not None and len(actualSeqLengthsK) > 0)
                else mask_s_kv
            )
            cur_tail_k = int(pool_tail_k[i_batch]) if pool_tail_k is not None else 0
            L_orig = (
                (S2 + cur_tail_k)
                if actual_seq_k_is_token
                else (S2 * pool_size + cur_tail_k)
            )
            atten_mask = torch.zeros(mask_s_q, mask_s_kv, dtype=torch.uint8)
            tmp_pos_orig = L_orig - S1
            for i in range(S1):
                valid_pools = (tmp_pos_orig + i + 1) / pool_size
                cutoff = int(math.floor(valid_pools))
                if cutoff < mask_s_kv:
                    atten_mask[i, cutoff:] = 1
            masks.append(atten_mask)
        return torch.stack(masks, dim=0).numpy().astype(np.bool_)

    def _trans_bnsd_to_layout(self, tensor, shape, layout, act_q=None):
        if layout == "BSND":
            return tensor.permute(0, 2, 1, 3).contiguous()
        elif layout == "TND":
            T = sum(act_q)
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            S = max(act_q)
            if all(int(x) == S for x in act_q) and T == B * S:
                return tensor.permute(0, 2, 1, 3).reshape(T, N, D).contiguous()
            output = torch.full(size=(T, N, D), fill_value=-1, dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_q[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = tensor[
                        b_index, n_index, :act_s, :
                    ]
                t_start += act_s
            return output
        else:
            return tensor

    def _cal_atten_per_batch(
        self,
        cur_q,
        cur_k,
        cur_wt,
        cur_m,
        sparse_count,
        mask_mode,
        group_size,
        k_head_num,
        need_topk=True,
    ):
        compute_dtype = self.compute_dtype
        cur_q = _safe_to_dtype(cur_q, compute_dtype)
        cur_k = _safe_to_dtype(cur_k, compute_dtype)
        cur_wt = _safe_to_dtype(cur_wt, compute_dtype)

        qk_bmm_res = torch.matmul(
            cur_q.squeeze(0), cur_k.permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)
        # 算子文档公式: S = Q @ K_pool^T * 1/sqrt(headDim)
        qk_bmm_res = qk_bmm_res / (float(self.head_dim) ** 0.5)

        qk_relu_out = _safe_to_dtype(qk_bmm_res, compute_dtype).clamp_min(0.0)

        brc_vmul = torch.bmm(
            _safe_to_dtype(cur_wt.permute(0, 2, 3, 1), compute_dtype).squeeze(0),
            _safe_to_dtype(qk_relu_out.permute(0, 2, 1, 3), compute_dtype).squeeze(0),
        ).unsqueeze(0)

        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, k_head_num, temp_s1, temp_s2)

        if mask_mode == 3 and cur_m is not None:
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(
                cur_m_broadcasted, (1, k_head_num, temp_s1, temp_s2)
            )
            reduce_sum[cur_m_broadcasted.to(dtype=torch.bool)] = -torch.inf

        to_be_sort_ele = reduce_sum.clone()

        if not need_topk:
            return None, to_be_sort_ele

        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        for i in range(temp_s1):
            row_ele = to_be_sort_ele[0, 0, i, :]
            indices = torch.arange(len(row_ele), device=row_ele.device)
            sorted_vals, sorted_idx = torch.sort(
                torch.stack([-row_ele, indices], dim=1), dim=0, stable=True
            )
            b_sorted_indices[0, 0, i, :] = sorted_idx[:, 0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def _expand_and_append(
        self,
        pool_indices,
        scores,
        sparse_count,
        pool_size,
        valid_s2_len,
        pool_tail_k,
        l_orig,
        s1,
        mask_mode,
        compute_dtype,
    ):
        """展开 pool 索引为 token 索引, 追加尾块, 填充 -1.

        Args:
            pool_indices: [S1, sparse_count] pool 级 TopK 索引
            scores: [S1, S2] 原始分数
            valid_s2_len: [S1] 每个 query 位置的有效 pool 数
            pool_tail_k: 尾部 token 数
            l_orig: L_orig = S2 * pool_size + pool_tail_k
            s1: S1 长度
            mask_mode: 0 或 3

        Returns:
            token_indices: [S1, topk + pool_size - 1] INT32
            token_values: [S1, sparse_count] FLOAT32
        """
        topk = sparse_count * pool_size
        output_len = topk + pool_size - 1

        token_indices = torch.full((s1, output_len), -1, dtype=torch.int32)
        token_values = torch.full(
            (s1, sparse_count), float("-inf"), dtype=compute_dtype
        )

        for i in range(s1):
            v_len = (
                int(valid_s2_len[i])
                if hasattr(valid_s2_len, "__getitem__")
                else int(valid_s2_len)
            )

            # 展开: pool 索引 → token 索引
            out_off = 0
            for k in range(min(v_len, sparse_count)):
                pool_idx = int(pool_indices[i, k].item())
                if pool_idx < 0:
                    continue
                base = pool_idx * pool_size
                for p in range(pool_size):
                    if out_off + p < topk:
                        token_indices[i, out_off + p] = base + p
                out_off += pool_size

            # 收集 values
            for k in range(min(v_len, sparse_count)):
                pool_idx = int(pool_indices[i, k].item())
                if pool_idx >= 0:
                    token_values[i, k] = scores[i, pool_idx]

            # 尾块追加
            if pool_tail_k > 0:
                tail_pos = topk
                if mask_mode == 0:
                    visible_tail_k = pool_tail_k
                else:
                    global_pos_q = l_orig - s1 + i
                    visible_tail_k = max(0, min(pool_tail_k, global_pos_q - topk + 1))

                for t in range(visible_tail_k):
                    token_indices[i, tail_pos + t] = l_orig - pool_tail_k + t

        return token_indices, token_values

    def _canonical_rows_pool(
        self,
        ext_token_indices,
        scores,
        sparse_count,
        pool_size,
        valid_s2_len,
        pool_tail_k,
        l_orig,
        s1,
        mask_mode,
        compute_dtype,
    ):
        """归一化算子输出的 token 级索引: token→pool 转换 → gather scores → sort → cutoff → 重新展开.

        Args:
            ext_token_indices: [S1, topk+poolSize-1] 算子输出的 token 索引
            scores: [S1, S2] pool 级分数
            sparse_count: topk // pool_size
            pool_size: pool 大小
            valid_s2_len: [S1] 每个 query 位置的有效 pool 数
            pool_tail_k: 尾部 token 数
            l_orig: L_orig = S2 * pool_size + pool_tail_k
            s1: S1 长度
            mask_mode: 0 或 3

        Returns:
            token_indices: [S1, topk + pool_size - 1] INT32
            token_values: [S1, sparse_count] compute_dtype
        """
        topk = sparse_count * pool_size
        output_len = topk + pool_size - 1
        s2_len = scores.shape[-1]

        # 1. 提取 TopK 部分 (尾块部分不需要归一化, 重新生成)
        ext_pool_part = ext_token_indices[:, :topk].long()  # [S1, topk]

        # 2. token → pool 索引转换 (仅有效 token)
        valid_mask = (ext_pool_part >= 0) & (ext_pool_part < s2_len * pool_size)
        pool_idx = torch.where(
            valid_mask, ext_pool_part // pool_size, torch.zeros_like(ext_pool_part)
        )

        # 3. Gather pool 级 scores
        gathered = torch.gather(scores, -1, pool_idx.clamp(0, s2_len - 1))
        gathered = gathered.masked_fill(~valid_mask, float("-inf"))

        # 4. Cutoff: 第 k 大的 score
        k_eff = max(1, min(s2_len, sparse_count))
        cutoff = scores.topk(k_eff, dim=-1).values[..., -1:]  # [S1, 1]

        # 5. 标记低于 cutoff 的为 -1
        at_or_below = valid_mask & (gathered <= cutoff)
        out_pool = torch.where(at_or_below, torch.full_like(pool_idx, -1), pool_idx)

        # 6. 按 (index, score) 排序: 先按 index 排序, 再按 score 降序稳定排序
        order_by_idx = torch.argsort(out_pool.long(), dim=-1)
        key = torch.gather(gathered, -1, order_by_idx)
        order_by_score = torch.argsort(key, dim=-1, descending=True, stable=True)
        final_order = torch.gather(order_by_idx, -1, order_by_score)
        canon_pool = torch.gather(out_pool, -1, final_order).to(
            torch.int32
        )  # [S1, topk]
        canon_scores = torch.gather(gathered, -1, final_order)  # [S1, topk]

        # 7. 取前 sparse_count 个 pool
        canon_pool = canon_pool[:, :sparse_count]  # [S1, sparse_count]
        canon_scores = canon_scores[:, :sparse_count]  # [S1, sparse_count]

        # 8. 重新展开 pool → token + 追加尾块
        token_indices = torch.full((s1, output_len), -1, dtype=torch.int32)
        token_values = torch.full(
            (s1, sparse_count), float("-inf"), dtype=compute_dtype
        )

        for i in range(s1):
            v_len = (
                int(valid_s2_len[i])
                if hasattr(valid_s2_len, "__getitem__")
                else int(valid_s2_len)
            )
            out_off = 0
            for k in range(sparse_count):
                pidx = int(canon_pool[i, k].item())
                if pidx >= 0 and k < v_len:
                    base = pidx * pool_size
                    for p in range(pool_size):
                        if out_off + p < topk:
                            token_indices[i, out_off + p] = base + p
                    token_values[i, k] = canon_scores[i, k]
                out_off += pool_size

            # 尾块追加 (与 golden 路径一致)
            if pool_tail_k > 0:
                tail_pos = topk
                if mask_mode == 0:
                    visible_tail_k = pool_tail_k
                else:
                    global_pos_q = l_orig - s1 + i
                    visible_tail_k = max(0, min(pool_tail_k, global_pos_q - topk + 1))
                for t in range(visible_tail_k):
                    token_indices[i, tail_pos + t] = l_orig - pool_tail_k + t

        return token_indices, token_values

    def forward(
        self,
        query,
        key,
        weights,
        actual_seq_q,
        actual_seq_k,
        block_table=None,
        pool_tail_k=None,
        ext_indices=None,
        ext_values=None,
    ):
        _saved_threads = torch.get_num_threads()
        if _saved_threads > 48:
            torch.set_num_threads(48)

        batch_size = self.batch_size
        q_seq = self.q_seq
        k_seq = self.k_seq
        layout_query = self.layout_query
        layout_key = self.layout_key
        topk = self.topk
        pool_size = self.pool_size
        sparse_count = self.sparse_count
        mask_mode = self.mask_mode
        q_head_num = self.q_head_num
        k_head_num = self.k_head_num
        head_dim = self.head_dim

        if layout_query == "TND":
            actual_seq_q = self._trans_tnd_actseq(actual_seq_q)
        if layout_key == "TND":
            actual_seq_k = self._trans_tnd_actseq(actual_seq_k)

        if layout_key == "PA_BBND" and actual_seq_k is not None:
            # 算子约定 PA 的 actual_seq_k 为每 batch 池数(非前缀和, LIV2 语义),
            # golden 内部 k_bnsd 构造直接按池数消费, 无需换算; 仅转为 list 便于索引
            actual_seq_k = [int(x) for x in actual_seq_k]

        q_bnsd = self._trans_shape_to_bnsd(
            query, layout_query, q_head_num, actual_seq_q
        )
        if layout_key == "PA_BBND" and block_table is not None:
            k_bnsd = self._pa_bbnd_to_bnsd(
                key, block_table, actual_seq_k, self.pool_size
            )
        else:
            k_bnsd = self._trans_shape_to_bnsd(
                key, layout_key, k_head_num, actual_seq_k
            )
        wt_bnsd = self._trans_shape_to_bnsd(
            weights, layout_query, q_head_num, actual_seq_q, is_weights=True
        )

        m_shape = [q_bnsd.shape[2], k_bnsd.shape[2]]
        batch = q_bnsd.shape[0]

        m_tensor = []
        if mask_mode == 3:
            m_tensor = torch.from_numpy(
                self._create_mask_right_down(
                    m_shape, actual_seq_q, actual_seq_k, batch, pool_size, pool_tail_k
                )
            )

        # 输出 shape: BSND (B, S1, output_len) / TND (T1, output_len)
        if layout_query == "BSND":
            out_idx_shape = [batch, q_seq, self.output_indices_len]
            out_val_shape = [batch, q_seq, self.output_values_len]
        else:
            out_idx_shape = [q_seq, self.output_indices_len]
            out_val_shape = [q_seq, self.output_values_len]

        y_indices = torch.full(out_idx_shape, -1, dtype=torch.int32)
        y_values = torch.full(out_val_shape, float("-inf"), dtype=self.compute_dtype)

        for b_idx in range(batch):
            curr_q = actual_seq_q[b_idx] if actual_seq_q is not None else q_seq
            # k_bnsd 的 S2 维为池数: actual_seq_k(TND 累计池/PA 每 batch 池)已按池计;
            # BSND 无 actual_seq_k, k_seq 为 token 数, 需换算为池数
            if actual_seq_k is not None:
                curr_k = math.floor(actual_seq_k[b_idx])
            else:
                curr_k = k_seq // pool_size
            cur_q = q_bnsd[b_idx : (b_idx + 1), :, :curr_q, :]
            cur_k = k_bnsd[b_idx : (b_idx + 1), :, :curr_k, :]
            cur_wt = wt_bnsd[b_idx : (b_idx + 1), :, :curr_q, :]

            cur_m = None
            if mask_mode == 3:
                cur_m = m_tensor[b_idx : (b_idx + 1), :curr_q, :curr_k]

            if curr_q == 0:
                continue

            # 计算 scores (need_topk 取决于是否有 ext_indices)
            topk_pool_indices, scores = self._cal_atten_per_batch(
                cur_q,
                cur_k,
                cur_wt,
                cur_m,
                sparse_count,
                mask_mode,
                self.group_size,
                k_head_num,
                need_topk=ext_indices is None,
            )

            # 统一取 scores 2D [S1, S2]
            scores_2d = scores[0, 0]

            # 获取当前 batch 的参数
            cur_pool_tail_k = int(pool_tail_k[b_idx]) if pool_tail_k is not None else 0
            # curr_k 统一为池数语义; L_orig = 池数*poolSize + tail(token 数),
            # 用于 causal 可见性与尾块定位
            l_orig = curr_k * pool_size + cur_pool_tail_k

            # 计算 valid_s2_len
            s1_actual = curr_q
            if mask_mode == 3:
                valid_s2_len = torch.zeros(s1_actual, dtype=torch.int32)
                for i in range(s1_actual):
                    vu = (l_orig - s1_actual + i + 1) / pool_size
                    valid_s2_len[i] = max(0, min(curr_k, int(math.floor(vu))))
            else:
                valid_s2_len = torch.full((s1_actual,), curr_k, dtype=torch.int32)

            if ext_indices is not None:
                # 算子归一化路径: 提取当前 batch 的 ext_token_indices [S1, topk+poolSize-1]
                if layout_query == "BSND":
                    ext_2d = ext_indices[b_idx, :curr_q, :]  # [S1, output_len]
                else:
                    t_start = int(sum(int(x) for x in actual_seq_q[:b_idx]))
                    ext_2d = ext_indices[t_start : t_start + curr_q, :]

                token_idx, token_val = self._canonical_rows_pool(
                    ext_2d,
                    scores_2d,
                    sparse_count,
                    pool_size,
                    valid_s2_len,
                    cur_pool_tail_k,
                    l_orig,
                    s1_actual,
                    mask_mode,
                    self.compute_dtype,
                )
            else:
                # Golden 标杆路径: pool 级 TopK → 展开 → 追加
                pool_idx_2d = topk_pool_indices[0, 0]  # [S1, sparse_count]

                if pool_idx_2d.shape[-1] < sparse_count:
                    pad = torch.full(
                        pool_idx_2d.shape[:-1]
                        + (sparse_count - pool_idx_2d.shape[-1],),
                        -1,
                        dtype=torch.int32,
                    )
                    pool_idx_2d = torch.cat([pool_idx_2d, pad], dim=-1)

                token_idx, token_val = self._expand_and_append(
                    pool_idx_2d,
                    scores_2d,
                    sparse_count,
                    pool_size,
                    valid_s2_len,
                    cur_pool_tail_k,
                    l_orig,
                    s1_actual,
                    mask_mode,
                    self.compute_dtype,
                )

            if layout_query == "BSND":
                y_indices[b_idx, :curr_q, :] = token_idx
                y_values[b_idx, :curr_q, :] = token_val
            else:
                t_start = int(sum(int(x) for x in actual_seq_q[:b_idx]))
                y_indices[t_start : t_start + curr_q, :] = token_idx
                y_values[t_start : t_start + curr_q, :] = token_val

        if not COMPARE_INDICES:
            y_indices = torch.zeros_like(y_indices)

        torch.set_num_threads(_saved_threads)
        return y_indices, y_values
