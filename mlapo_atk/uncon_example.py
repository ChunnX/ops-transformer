import torch
import torch_npu

import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
from atk.tasks.backends.lib_interface.acl_wrapper import VoidPtr
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, nnopbase, AclFormat
import logging
import numpy as np
from array import array
import random
from ml_dtypes import bfloat16
from dataclasses import dataclass
import flash_decoding_tiling

# np.random.seed(44)

# IS_SINK_FLAG = True

def inspect_kwargs(kwargs):
    """
    遍历打印 kwargs 中的 key 和 value。
    如果是 Tensor/Array，只打印 shape 和 dtype。
    如果是标量或普通列表，打印具体值。
    """
    print(f"\n{'='*20} DEBUG: Inspect Kwargs {'='*20}")
    
    if not isinstance(kwargs, dict):
        print(f"Error: Expected dict, got {type(kwargs)}")
        return

    for key, value in kwargs.items():
        # 尝试获取类型名称
        type_name = type(value).__name__
        
        # 1. 处理 PyTorch Tensor
        if isinstance(value, torch.Tensor):
            print(f"[Tensor]  Key: {key:<25} | Shape: {list(value.shape)} | Dtype: {value.dtype} | Device: {value.device}")
            
        # 2. 处理 Numpy Array
        elif isinstance(value, np.ndarray):
            print(f"[Numpy]   Key: {key:<25} | Shape: {value.shape} | Dtype: {value.dtype}")
            
        # 3. 处理包含 Tensor 的列表 (常见于 List[Tensor])
        elif isinstance(value, (list, tuple)) and len(value) > 0 and (isinstance(value[0], torch.Tensor) or isinstance(value[0], np.ndarray)):
            shape_info = list(value[0].shape) if hasattr(value[0], 'shape') else "N/A"
            print(f"[List_T]  Key: {key:<25} | Len: {len(value)} | Item0_Shape: {shape_info}")
            
        # 4. 处理自定义对象 (如果有 shape 属性)
        elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
             print(f"[Custom]  Key: {key:<25} | Shape: {value.shape} | Dtype: {value.dtype}")

        # 5. 其他标量或普通对象，直接打印值
        else:
            # 如果值太长，截断一下
            str_val = str(value)
            if len(str_val) > 100:
                str_val = str_val[:100] + "..."
            print(f"[Scalar]  Key: {key:<25} | Value: {str_val}")

    print(f"{'='*62}\n")

dtypeMap = {
    torch.float16: np.float16,
    torch.bfloat16: bfloat16,
    torch.float32: np.float32
}

maskTypeMap = {
    ## sparseMode : golden maskType
    0: 0,
    3: 1
}

class TestFIAV4SplitFuse():
    @dataclass
    class AuxAttrs:
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int
        layout_dtype: int
        max_q_seqlen: int
        max_kv_seqlen: int
        inner_prec: int
        scale: float
        lse_flag: int = 0
        ## FD切分的时候需要指定平台的数 enable_flash_decoding
        core_num: int = 24

    
    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        learnable_sink: any
        auxAttrs: any

    @classmethod
    def group_matmul(cls, head, kv_head, left, right, high_prec = 1, is_benchmark_task=True):
        group_num = head // kv_head
        score = None
        dtype = np.float32
        if is_benchmark_task:
            dtype = np.float64
        for i in range(kv_head):
            group_score = np.matmul(
                left[i * group_num:(i + 1) * group_num, :, :].astype(dtype),
                right[i:(i + 1), :, :].astype(dtype)
            )
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim, sink_matrix):
        row_max = np.max(sim, axis=-1, keepdims=True)

        # add sink rowmax
        if sink_matrix is not None:
            assert sink_matrix.shape == row_max.shape, \
                f"sink_matrix 形状 {sink_matrix.shape} 与 row_max 形状 {row_max.shape} 不一致！"
            # 更新含sink的rowmax
            # print("lwg-sink_come in\n")
            row_max = np.maximum(row_max, sink_matrix)

        sim_sub = sim - row_max
        sim_sub_high = sim.astype(np.float64) - row_max.astype(np.float64)

        sim_sub = np.exp(sim_sub)
        sim_sub_high = np.exp(sim_sub_high)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        row_sum_high = np.sum(sim_sub_high, axis=-1, keepdims=True)

        if sink_matrix is not None:
            # 更新含sink的rowmax
            sink_exp = np.exp(sink_matrix - row_max)
            sink_exp_high = np.exp(sink_matrix.astype(np.float64) - row_max.astype(np.float64))
            row_sum = row_sum + sink_exp
            row_sum_high = row_sum_high + sink_exp_high

        soft_res = sim_sub / row_sum
        lse = np.squeeze((np.log(row_sum_high) + row_max.astype(np.float64)), axis=-1)
        # lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)

        return soft_res, lse, row_max

    def softmax1(
        self,
        qk_result,
        is_first,
        gm,
        is_kvs_last_loop,
        sink_matrix,
        interm_dtype = np.float16
    ):
        sim = qk_result.astype(interm_dtype)
        # sim = qk_result
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        
        if sink_matrix is not None and is_kvs_last_loop:
            assert sink_matrix.shape == hm.shape, \
            f"sink_matrix 形状 {sink_matrix.shape} 与 hm 形状 {hm.shape} 不一致！"
            hm = np.maximum(hm, sink_matrix)
            dm = gm - hm if not is_first  else 0

        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub.astype(interm_dtype))
        # sim_sub = sim_sub.astype(np.float16)

        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)

        if sink_matrix is not None and is_kvs_last_loop:
            sink_exp = np.exp(sink_matrix - hm)
            row_sum = row_sum + sink_exp

        return sim_sub, row_sum, dm, gm


    def qkMM1(
        self,
        query,
        key,
        is_benchmark_task
    ):
        result = None
        qk_k = key.shape[1]
        qk_k_split = 128
        qk_k_loop = (qk_k + 127) // 128
        for qk_k_loop_idx in range(qk_k_loop):
            sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
            partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
            partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
            result_split = self.group_matmul(
                partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0, is_benchmark_task
            )
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result
    
    def pvMM2(
        self,
        p,
        value,
        is_benchmark_task
    ):
        result = None
        pv_k = value.shape[1]
        pv_k_split = 128
        pv_k_loop = (pv_k + 127) // 128
        for pv_k_loop_idx in range(pv_k_loop):
            sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
            partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
            partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :]
            result_split = self.group_matmul(
                partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0, is_benchmark_task
            )
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def ref_flash_attention(
        self,
        query,
        key,
        value,
        scale,
        mask,
        attention_inputs: AttentionInputs,
        sink_matrix,
        is_benchmark_task,
        force_interm_dtype=None
    ):
        data_type = attention_inputs.auxAttrs.dtype
        inner_prec = attention_inputs.auxAttrs.inner_prec
        if force_interm_dtype is not None:
            interm_dtype = force_interm_dtype
        else:
            interm_dtype = np.float16 if inner_prec == 1 else np.float32
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        value = np.transpose(value, (1, 0, 2))
        scale = np.float16(scale) if inner_prec == 1 else np.float32(scale)
        context_len = key.shape[2]
        context_size = 512
        group_num = query.shape[0] // key.shape[0]
        gl = None
        gl_high = None
        go = None
        go_high = None

        is_kvs_last_loop = False

        for kv_start in range(0, context_len, context_size):
            sub_len = context_size
            if kv_start + context_size > context_len:
                sub_len = context_len - kv_start
            
            is_kvs_last_loop = (kv_start + context_size >= context_len)

            sub_key = key[:, :, kv_start: kv_start + sub_len]
            sub_mask = None
            if mask is not None:
                sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len].astype(interm_dtype)
            sub_value = value[:, kv_start: kv_start + sub_len, :]
            qk_result = self.qkMM1(query, sub_key, is_benchmark_task).astype(interm_dtype)
            qk_result = qk_result * scale

            if mask is not None:
                qk_result += sub_mask
            if kv_start == 0:
                gm = None
            p_result, row_sum, dm, gm = self.softmax1(qk_result, kv_start == 0, gm, is_kvs_last_loop, sink_matrix, interm_dtype)
            p_result = p_result.astype(data_type)
            if kv_start == 0:
                gm_high = None
            lo = self.pvMM2(p_result, sub_value, is_benchmark_task).astype(interm_dtype)
            
            if kv_start == 0:
                gl = row_sum
                go = lo
            else:
                dm = np.exp(dm)
                gl = gl * dm
                gl = gl + row_sum
                go = go * dm
                go = go + lo
        go = go / gl
        go = np.transpose(go, (1, 0, 2))
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32) # lse仅支持fp32输出，无论采取何种精度运算，最终结果都是fp32
        return go.astype(data_type), lse

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask,    # (q_seqlen, k_seqlen)
            sink_matrix,
            is_benchmark_task
    ):
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key, 1, is_benchmark_task)  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float32)
        p_high, lse_high, gm = self.softmax_numpy(sim_high, sink_matrix)
        lse_high = lse_high.astype(np.float64)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))
        
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value, 1, is_benchmark_task)
        out_high = np.transpose(out_high, (1, 0, 2))
        return out_high, lse_high

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output, is_benchmark_task) -> None:
        num_heads = attention_inputs.auxAttrs.num_heads
        kv_heads = attention_inputs.auxAttrs.kv_heads
        head_size_qk = attention_inputs.auxAttrs.head_size
        head_size_vo = attention_inputs.auxAttrs.head_size
        block_size = attention_inputs.auxAttrs.block_size
        max_q_seqlen = attention_inputs.auxAttrs.max_q_seqlen
        inner_prec = attention_inputs.auxAttrs.inner_prec
        scale = attention_inputs.auxAttrs.scale
        learnable_sink = attention_inputs.learnable_sink
        if learnable_sink is not None:
            learnable_sink = learnable_sink.astype(np.float32)
            # print("sink_lwg")

        batch = len(attention_inputs.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = None
            if attention_inputs.auxAttrs.layout_dtype == 1:
                q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            else:
                q = attention_inputs.query[i * max_q_seqlen:(i * max_q_seqlen + q_seqlen), :, :]
            keys = None
            values = None
            if attention_inputs.auxAttrs.kv_dtype == 1:
                keys = []
                values = []
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen):
                    if block_size == 0:
                        pass
                        # print(f"lwg-block_size: {block_size} \n")
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size

                    k = attention_inputs.key_cache[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)

                    v = attention_inputs.value_cache[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
                keys = np.stack(keys, axis=0)
                values = np.stack(values, axis=0)
            elif attention_inputs.auxAttrs.kv_dtype == 0:
                if attention_inputs.auxAttrs.layout_dtype == 1:
                    keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                    values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                else:
                    keys = attention_inputs.key_cache[i, :, :, :]
                    values = attention_inputs.value_cache[i, :, :, :]
            
            if attention_inputs.auxAttrs.mask_type == 1:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            elif attention_inputs.auxAttrs.mask_type == 0:
                mask = None

            sink_matrix = None
            if learnable_sink is not None:
                # [num_heads → [num_heads, 1, 1]
                sink_expanded = np.expand_dims(learnable_sink, axis=1)
                sink_expanded = np.expand_dims(sink_expanded, axis=2)
                # [num_heads, 1, 1] → [num_heads, q_seqlen, 1]
                sink_matrix = np.broadcast_to(sink_expanded, shape=(learnable_sink.shape[0], q_seqlen, 1))
            
            out_normal, lse = self.ref_masked_attention(q, keys, values, scale, mask, sink_matrix, is_benchmark_task)
            out_gpu, lse_gpu = self.ref_flash_attention(q, keys, values, scale, mask, attention_inputs, sink_matrix, is_benchmark_task)

            out = out_normal.reshape(-1, num_heads, head_size_vo)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_gpu = out_gpu.reshape(-1, num_heads, head_size_vo)

            if attention_inputs.auxAttrs.layout_dtype == 1:
                output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse_gpu
            else:
                output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse_gpu
            
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def get_batch_inputs(self, attention_inputs, batch_idx, cu_seqlen, kv_seqlen_now):
        num_heads = attention_inputs.auxAttrs.num_heads
        kv_heads = attention_inputs.auxAttrs.kv_heads
        head_size_qk = attention_inputs.auxAttrs.head_size
        head_size_vo = attention_inputs.auxAttrs.head_size
        block_size = attention_inputs.auxAttrs.block_size
        max_q_seqlen = attention_inputs.auxAttrs.max_q_seqlen
        
        q_seqlen = int(attention_inputs.q_seqlen_list[batch_idx])
        k_seqlen = int(attention_inputs.k_seqlen_list[batch_idx])
        
        q = None
        if attention_inputs.auxAttrs.layout_dtype == 1:
            q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
        else:
            q = attention_inputs.query[batch_idx * max_q_seqlen:(batch_idx * max_q_seqlen + q_seqlen), :, :]
            
        keys = None
        values = None
        if attention_inputs.auxAttrs.kv_dtype == 1:
            keys = []
            values = []
            block_table = attention_inputs.block_tables[batch_idx]
            for j in range(k_seqlen):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = attention_inputs.key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)

                v = attention_inputs.value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)
            keys = np.stack(keys, axis=0)
            values = np.stack(values, axis=0)
        elif attention_inputs.auxAttrs.kv_dtype == 0:
            if attention_inputs.auxAttrs.layout_dtype == 1:
                keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
            else:
                keys = attention_inputs.key_cache[batch_idx, :, :, :]
                values = attention_inputs.value_cache[batch_idx, :, :, :]
                
        scale = attention_inputs.auxAttrs.scale
        
        mask = None
        if attention_inputs.auxAttrs.mask_type == 1:
            mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
        elif attention_inputs.auxAttrs.mask_type == 2:
            mask = attention_inputs.global_mask
            
        return q, keys, values, mask, scale

    def ref_flash_decoding(self, attention_inputs: AttentionInputs, golden_gpu_output, is_benchmark_task, core_num = 24):
        # print("Running ref_flash_decoding to generate golden...")
        # 1. Setup Tiling Context
        faInfo = flash_decoding_tiling.FAInferContext()
        faInfo.numHeads = attention_inputs.auxAttrs.num_heads
        faInfo.kvHeads = attention_inputs.auxAttrs.kv_heads
        faInfo.batch = len(attention_inputs.q_seqlen_list)
        faInfo.qSeqlenList = attention_inputs.q_seqlen_list
        faInfo.kvSeqlenList = attention_inputs.k_seqlen_list
        faInfo.pagedCacheFlag = (attention_inputs.auxAttrs.kv_dtype == 1)
        faInfo.layout = "TND" if attention_inputs.auxAttrs.layout_dtype == 1 else "BSND"
        faInfo.embeddingSizeV = attention_inputs.auxAttrs.head_size

        # 2. Run Tiling
        faTilingData = flash_decoding_tiling.FAInferTilingData(core_num)
        flash_decoding_tiling.splitBN2S1GS2(faInfo, core_num, faTilingData)
        
        # 3. Pre-calculate batch offsets
        batch_count = len(attention_inputs.q_seqlen_list)
        cu_seqlen_list = [0] * (batch_count + 1)
        kv_seqlen_now_list = [0] * (batch_count + 1)
        for i in range(batch_count):
            cu_seqlen_list[i+1] = cu_seqlen_list[i] + attention_inputs.q_seqlen_list[i]
            kv_seqlen_now_list[i+1] = kv_seqlen_now_list[i] + attention_inputs.k_seqlen_list[i]
            
        # 4. Iterate over splits
        head_size_vo = attention_inputs.auxAttrs.head_size
        max_q_seqlen = attention_inputs.auxAttrs.max_q_seqlen

        for splitIdx in range(faTilingData.totalSplitNodeNum):
            batchIdx = faTilingData.splitInfo.batchIdx[splitIdx]
            headStartIdx = faTilingData.splitInfo.headStartIdx[splitIdx]
            headEndIdx = faTilingData.splitInfo.headEndIdx[splitIdx]
            qStartIdx = faTilingData.splitInfo.qStartIdx[splitIdx]
            qEndIdx = faTilingData.splitInfo.qEndIdx[splitIdx]
            kvRanges = faTilingData.splitInfo.kvRanges[splitIdx]
            
            if not kvRanges:
                continue

            # Get full batch inputs
            q_full, k_full, v_full, mask_full, scale = self.get_batch_inputs(
                attention_inputs, 
                batchIdx, 
                cu_seqlen_list[batchIdx], 
                kv_seqlen_now_list[batchIdx]
            )
            
            # Slice Q to current Q chunk
            q_chunk = q_full[qStartIdx:qEndIdx, headStartIdx:headEndIdx, :]
            
            # Calculate KV heads range
            num_heads = attention_inputs.auxAttrs.num_heads
            kv_heads = attention_inputs.auxAttrs.kv_heads
            group_size = num_heads // kv_heads
            
            kvHeadStart = headStartIdx // group_size
            kvHeadEnd = (headEndIdx + group_size - 1) // group_size
            
            # Slice K, V to corresponding KV heads
            k_chunk_heads = k_full[:, kvHeadStart:kvHeadEnd, :]
            v_chunk_heads = v_full[:, kvHeadStart:kvHeadEnd, :]
            
            # Slice Q mask
            mask_chunk = None
            if mask_full is not None:
                mask_chunk = mask_full[qStartIdx:qEndIdx, :]

            # Prepare sink_matrix for this chunk if needed
            sink_matrix_chunk = None
            if attention_inputs.learnable_sink is not None:
                learnable_sink = attention_inputs.learnable_sink.astype(np.float32)
                # Get the heads for this chunk
                sink_heads = learnable_sink[headStartIdx:headEndIdx]
                # Expand to [num_heads_chunk, q_chunk_len, 1]
                sink_expanded = np.expand_dims(sink_heads, axis=1)
                sink_expanded = np.expand_dims(sink_expanded, axis=2)
                q_chunk_len = qEndIdx - qStartIdx
                sink_matrix_chunk = np.broadcast_to(sink_expanded, shape=(sink_heads.shape[0], q_chunk_len, 1))
                
            partial_outputs = [] # List of (O, LSE)
            
            for (kvStart, kvEnd) in kvRanges:
                # Slice K, V along seqlen
                k_tile = k_chunk_heads[kvStart:kvEnd, :, :]
                v_tile = v_chunk_heads[kvStart:kvEnd, :, :]
                
                # Slice Mask along kv seqlen
                mask_tile = None
                if mask_chunk is not None:
                    mask_tile = mask_chunk[:, kvStart:kvEnd]
                
                # Call ref_flash_attention
                # Note: ref_flash_attention returns (go, lse) in FIA_splitFuse_golden.py
                out_low, lse_low = self.ref_flash_attention(
                    q_chunk, k_tile, v_tile, scale, mask_tile, attention_inputs, sink_matrix_chunk, False,
                    force_interm_dtype=np.float16
                )
                
                partial_outputs.append((out_low.astype(np.float32), lse_low))
                
            # Combine Logic (Match CombineScale.hpp)
            if not partial_outputs:
                continue
                
            # Stack results
            outs = [p[0] for p in partial_outputs] # List of (Seq, Head, Dim)
            lses = [p[1] for p in partial_outputs] # List of (Head, Seq)
            
            # Stack to (NumSplits, Head, Seq)
            lse_stack = np.stack(lses, axis=0) 
            
            # 1. Global Max LSE
            global_max_lse = np.max(lse_stack, axis=0) # (Head, Seq)
            
            # 2. Exp(LSE - Max)
            exp_diff = np.exp(lse_stack - global_max_lse)
            
            # 3. Sum Exp
            sum_exp = np.sum(exp_diff, axis=0) # (Head, Seq)
            
            # 4. Global LSE
            global_lse = np.log(sum_exp) + global_max_lse
            
            # 5. Weights
            # weights = np.exp(lse_stack - global_lse)
            weights = np.exp(lse_stack - global_lse).astype(np.float16).astype(np.float32)
            
            # 6. Final O
            # CombineScale.hpp uses serial accumulation
            final_out = np.zeros_like(outs[0], dtype=np.float32)
            for i in range(len(outs)):
                # weights[i]: (Head, Seq) -> Transpose to (Seq, Head) -> (Seq, Head, 1)
                weight_i = np.transpose(weights[i], (1, 0))[:, :, np.newaxis]
                
                # outs[i]: (Seq, Head, Dim)
                term = outs[i] * weight_i
                final_out = final_out + term
            
            # Write back to golden_gpu_output
            # Cast
            final_out = final_out.astype(attention_inputs.auxAttrs.dtype)

            if attention_inputs.auxAttrs.layout_dtype == 1:
                # TND
                global_q_start = cu_seqlen_list[batchIdx] + qStartIdx
                global_q_end = cu_seqlen_list[batchIdx] + qEndIdx
                
                # golden_gpu_output is (TotalSeq, NumHeads, HeadSize)
                # print(f"[DEBUG] Layout=TND Batch={batchIdx} Qrange=[{global_q_start}:{global_q_end}] HeadRange=[{headStartIdx}:{headEndIdx}]")
                old_val = golden_gpu_output[global_q_start:global_q_end, headStartIdx:headEndIdx, :]
                # print(f"    Before Replace (Old O): Shape={old_val.shape}, Mean={np.mean(old_val):.6f}, Max={np.max(old_val):.6f}, Min={np.min(old_val):.6f}")
                # print(f"    New Value (Final O)   : Shape={final_out.shape}, Mean={float(np.mean(final_out)):.6f}, Max={float(np.max(final_out)):.6f}, Min={float(np.min(final_out)):.6f}")
                
                golden_gpu_output[global_q_start:global_q_end, headStartIdx:headEndIdx, :] = final_out
                
            else:
                # BSND
                # golden_gpu_output is (Batch * MaxQ, NumHeads, HeadSize)
                global_q_start = batchIdx * max_q_seqlen + qStartIdx
                global_q_end = batchIdx * max_q_seqlen + qEndIdx
                
                # print(f"[DEBUG] Layout=BSND Batch={batchIdx} Qrange=[{global_q_start}:{global_q_end}] HeadRange=[{headStartIdx}:{headEndIdx}]")
                old_val = golden_gpu_output[global_q_start:global_q_end, headStartIdx:headEndIdx, :]
                # print(f"    Before Replace (Old O): Shape={old_val.shape}, Mean={np.mean(old_val):.6f}, Max={np.max(old_val):.6f}, Min={np.min(old_val):.6f}")
                # print(f"    New Value (Final O)   : Shape={final_out.shape}, Mean={float(np.mean(final_out)):.6f}, Max={float(np.max(final_out)):.6f}, Min={float(np.min(final_out)):.6f}")

                golden_gpu_output[global_q_start:global_q_end, headStartIdx:headEndIdx, :] = final_out
    
    def calc_data(self, attention_inputs:AttentionInputs, is_benchmark_task):
        num_tokens = attention_inputs.query.shape[0]
        shape_out = (num_tokens, attention_inputs.auxAttrs.num_heads, attention_inputs.auxAttrs.head_size)
        golden_output = np.zeros(shape_out, dtype=np.float32)
        golden_gpu_output = np.zeros(shape_out, dtype=np.float32)

        lse_shape_out = (attention_inputs.auxAttrs.num_heads, num_tokens)
        golden_lse_output = np.zeros(lse_shape_out, dtype=np.float32)
        golden_gpu_lse_output = np.zeros(lse_shape_out, dtype=np.float32)

        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            golden_output,
            golden_gpu_output,
            golden_lse_output,
            golden_gpu_lse_output,
            is_benchmark_task
        )

        # check flash decoding condition
        # if ((!lseFlag) && (faInfo.pagedCacheFlag) && (faInfo.batch <= 8) && ((maxQSeqlen * faInfo.numHeads <= 32)) && (maxQSeqlen <= 6) && (minKVSeqlen >= 1024) && (minQSeqlen > 0))
        max_q_seqlen_actual = max(attention_inputs.q_seqlen_list)
        min_q_seqlen_actual = min(attention_inputs.q_seqlen_list)
        min_kv_seqlen_actual = min(attention_inputs.k_seqlen_list)
        batch_size = len(attention_inputs.q_seqlen_list)
        is_paged_attention = (attention_inputs.auxAttrs.kv_dtype == 1)

        num_tasks = batch_size * attention_inputs.auxAttrs.kv_heads
        core_num = attention_inputs.auxAttrs.core_num  ## 默认30
        is_long_seq = (num_tasks <= 0.8 * core_num) and (min_kv_seqlen_actual >= core_num * 512)
        is_short_seq = (num_tasks <= 0.4 * core_num) and (min_kv_seqlen_actual >= 1024)

        enable_flash_decoding = (
            (attention_inputs.auxAttrs.mask_type !=3) and
            (attention_inputs.auxAttrs.inner_prec != 1) and
            (attention_inputs.auxAttrs.lse_flag == 0) and
            is_paged_attention and
            (attention_inputs.auxAttrs.head_size <= 128) and
            (max_q_seqlen_actual * (attention_inputs.auxAttrs.num_heads // attention_inputs.auxAttrs.kv_heads) <= 64) and
            (max_q_seqlen_actual <= 16) and
            (min_q_seqlen_actual > 0) and 
            (is_long_seq or is_short_seq)
        )
        # 拆解所有条件
        conditions = {
            "mask_type != 3": attention_inputs.auxAttrs.mask_type != 3,
            "inner_prec != 1": attention_inputs.auxAttrs.inner_prec != 1,
            "lse_flag == 0": attention_inputs.auxAttrs.lse_flag == 0,
            "is_paged_attention": is_paged_attention,
            "head_size <= 128": attention_inputs.auxAttrs.head_size <= 128,
            "q_heads/kv_heads ratio <= 128": (max_q_seqlen_actual * (attention_inputs.auxAttrs.num_heads // attention_inputs.auxAttrs.kv_heads) <= 128),
            "max_q_seqlen <= 16": max_q_seqlen_actual <= 16,
            "min_q_seqlen > 0": min_q_seqlen_actual > 0,
            "is_long_or_short_seq": (is_long_seq or is_short_seq)
        }
        # 打印结果
        for desc, result in conditions.items():
            print(f"{'[PASS]' if result else '[FAIL]'} : {desc}")
        

        if enable_flash_decoding:
            print(f"xxxxxxxxxxxx:enbale fd")
            self.ref_flash_decoding(attention_inputs, golden_gpu_output, is_benchmark_task)

        golden_lse_output = np.transpose(golden_lse_output, (1, 0))
        golden_lse_output = np.expand_dims(golden_lse_output, axis=2)
        golden_gpu_lse_output = np.transpose(golden_gpu_lse_output, (1, 0))
        golden_gpu_lse_output = np.expand_dims(golden_gpu_lse_output, axis=2)

        return golden_output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output
        

def gen_list_from_cumSum(seqlenArray):
    seqlenList = []
    preSeqSum = 0
    for i in range(len(seqlenArray)):
        seqlenList.append(seqlenArray[i] - preSeqSum)
        preSeqSum = seqlenArray[i]
    return seqlenList

def gen_actual_seqlen_list_golden(actualseqlengths, actualseqlengthskv, inputLayout, pagedAttentionFlag):
    qSeqlenList = []
    kvSeqlenList = []
    if inputLayout == 'TND':
        qSeqlenList = gen_list_from_cumSum(actualseqlengths)
        if pagedAttentionFlag:
            kvSeqlenList = list(actualseqlengthskv)
        else:
            kvSeqlenList = gen_list_from_cumSum(actualseqlengthskv)
    else:
        qSeqlenList = list(actualseqlengths)
        kvSeqlenList = list(actualseqlengthskv)
    return qSeqlenList, kvSeqlenList


def aclnn_op_func_fia_split_fuse_golden(input_data : InputDataset, is_benchmark_task):
    input_data_dtype = input_data.kwargs["query"].dtype

    if input_data_dtype == torch.float16:
         query = input_data.kwargs["query"].numpy()
    elif input_data_dtype == torch.bfloat16:
        query = input_data.kwargs["query"].to(torch.float32).numpy().astype(bfloat16)
    else:
        query = input_data.kwargs["query"].numpy()

    if input_data_dtype == torch.float16:
        key = input_data.kwargs["key"][0].numpy()
    elif input_data_dtype == torch.bfloat16:
        key = input_data.kwargs["key"][0].to(torch.float32).numpy().astype(bfloat16)
    else:
        key = input_data.kwargs["key"][0].numpy()

    if input_data_dtype == torch.float16:
        value = input_data.kwargs["value"][0].numpy()
    elif input_data_dtype == torch.bfloat16:
        value = input_data.kwargs["value"][0].to(torch.float32).numpy().astype(bfloat16)
    else:
        value = input_data.kwargs["value"][0].numpy()

    # query = input_data.kwargs["query"].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["query"].to(torch.float32).numpy().astype(bfloat16)
    # key = input_data.kwargs["key"][0].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["key"][0].to(torch.float32).numpy().astype(bfloat16)
    # value = input_data.kwargs["value"][0].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["value"][0].to(torch.float32).numpy().astype(bfloat16)
    blockTable = None
    pagedAttentionFlag = False
    if input_data.kwargs["blockTableOptional"] != None:
        blockTable = input_data.kwargs["blockTableOptional"].numpy()
        pagedAttentionFlag = True
    ## gen actual seqlen
    inputLayout = input_data.kwargs["inputLayout"]
    batch = len(input_data.kwargs["actualSeqLengthsOptional"])
    actualseqlengths = [0] * batch
    actualseqlengthsKv = [0] * batch
    for i in range(batch):
        actualseqlengths[i] = input_data.kwargs["actualSeqLengthsOptional"][i]
        actualseqlengthsKv[i] = input_data.kwargs["actualSeqLengthsKvOptional"][i]
    qSeqlenList, kvSeqlenList = gen_actual_seqlen_list_golden(actualseqlengths, actualseqlengthsKv, inputLayout, pagedAttentionFlag)
    maxKvSeqlen = max(kvSeqlenList)
    maxQSeqlen = max(qSeqlenList)
    totalQTokens = sum(qSeqlenList)
    ## gen mask
    fullMask = None
    pre_mask_factor = -3e38 if input_data_dtype == torch.bfloat16 or input_data_dtype == torch.float32 else -6e4
    if input_data.kwargs["attenMaskOptional"] != None and input_data.kwargs["sparseMode"] == 3:
        maskDtype = dtypeMap[input_data_dtype]
        fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(maskDtype)
        prevQseqlen = 0
        for i in range(len(qSeqlenList)):
            qSeqlen = qSeqlenList[i]
            kSeqlen = kvSeqlenList[i]
            tri = np.ones((qSeqlen, qSeqlen))
            tri = np.triu(tri, 1)
            tri *= pre_mask_factor
            fullMask[prevQseqlen : (prevQseqlen + qSeqlen), kSeqlen - qSeqlen : kSeqlen] = tri
            prevQseqlen += qSeqlen

    learnable_sink = None 
    if input_data.kwargs.get("learnableSinkOptional") is not None:
        # print("golden, sink不为空 \n")
        # 格式转换：torch张量 → numpy数组（和query保持一致的dtype规则）
        sink_torch_dtype = input_data.kwargs["learnableSinkOptional"].dtype
        if sink_torch_dtype == torch.float16:
            # 可以报错
            logging.error(f"[ERROR] learnableSinkOptional dtype is {sink_torch_dtype} (float16), not allowed!")
            # sys.exit(-99)
        elif sink_torch_dtype == torch.bfloat16:
            learnable_sink = input_data.kwargs["learnableSinkOptional"].to(torch.float32).numpy()
        else:
            # bf16场景：先转float32再转bfloat16
            # learnable_sink = input_data.kwargs["learnableSinkOptional"].to(torch.float32).numpy()
            # # bf16场景：先转float32再转bfloat16
            # print("sink_bf16 \n")
            learnable_sink = input_data.kwargs["learnableSinkOptional"].numpy()
    else:
        print("golden, sink为空 \n")
    
    numHeads = input_data.kwargs["numHeads"]
    # if numHeads == 0:
        # exit(-100)
    kvHeads = input_data.kwargs["numKeyValueHeads"]
    headSize = query.shape[2] if inputLayout == 'TND' else 0
    numBlocks = key.shape[0] if pagedAttentionFlag == True else 0
    blockSize = input_data.kwargs["blockSize"]
    # print(f"lwg-blockSize-coming:{blockSize} \n")
    maskType = maskTypeMap[input_data.kwargs["sparseMode"]]
    dtype = dtypeMap[input_data_dtype]
    kvOrgMode = 1 if pagedAttentionFlag == True else 0
    layoutMode = 1 if inputLayout == 'TND' else 0
    goldenGpuPrecision = input_data.kwargs["innerPrecise"]
    softmaxLseFlag = input_data.kwargs["softmaxLseFlag"]
    scale = float(input_data.kwargs["scaleValue"])

    if pagedAttentionFlag == True:
        key = key.reshape(key.shape[:-1] + (kvHeads, headSize))
        value = value.reshape(value.shape[:-1] + (kvHeads, headSize))
    # print(key.shape)
    testObj = TestFIAV4SplitFuse()
    # lse_flag=0 means LSE output is not required (can enable flash decoding)
    # lse_flag=1 means LSE output is required (cannot enable flash decoding)
    # softmaxLseFlag=True means LSE output is required
    lse_flag = 1 if softmaxLseFlag else 0
    auxAttrs = testObj.AuxAttrs(numHeads, kvHeads, headSize, numBlocks, blockSize, maskType, dtype, kvOrgMode, layoutMode, maxQSeqlen, maxKvSeqlen, goldenGpuPrecision, scale, lse_flag)
    attentionInputs = testObj.AttentionInputs(query, key, value, blockTable, qSeqlenList, kvSeqlenList, fullMask, learnable_sink, auxAttrs)

    golden_output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output = testObj.calc_data(attentionInputs, is_benchmark_task)
    if golden_output.dtype == "bfloat16":
        # print("=================================走入bf16分支",golden_output.dtype)
        golden_output = torch.from_numpy(golden_output.astype(np.float32))
        # print(f"golden_output_dtype:{golden_output.dtype} \n")
        golden_gpu_output = torch.from_numpy(golden_gpu_output.astype(np.float32))
        golden_lse_output = torch.from_numpy(golden_lse_output.astype(np.float32))
        golden_gpu_lse_output = torch.from_numpy(golden_gpu_lse_output.astype(np.float32))
    else:
        golden_output = torch.from_numpy(golden_output)
        golden_gpu_output = torch.from_numpy(golden_gpu_output)
        golden_lse_output = torch.from_numpy(golden_lse_output)
        golden_gpu_lse_output = torch.from_numpy(golden_gpu_lse_output)
    # golden_output = torch.from_numpy(golden_output)
    # golden_gpu_output = torch.from_numpy(golden_gpu_output)
    # golden_lse_output = torch.from_numpy(golden_lse_output)
    # golden_gpu_lse_output = torch.from_numpy(golden_gpu_lse_output)
    if not softmaxLseFlag:
        golden_lse_output = torch.tensor([])
        golden_gpu_lse_output = torch.tensor([])

    if not is_benchmark_task:
        # 标杆返回这个
        return golden_gpu_output, golden_gpu_lse_output
    else:
        # 真值返回下面的
        return golden_output, golden_lse_output

@register("executor_fused_infer_attention_score_v4_fd")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.name == "perf" or "abnormal" in self.task_result.case_config.name:
            if input_data.kwargs["softmaxLseFlag"] == True:
                return torch.Tensor([1]), torch.Tensor([1])
            else:
                return torch.Tensor([1])
        # input_data.kwargs["learnableSinkOptional"] = None
        output, output_lse = aclnn_op_func_fia_split_fuse_golden(input_data, self.task_result.is_benchmark_task)
        if input_data.kwargs["softmaxLseFlag"] == True:
            return output, output_lse
        else:
            return output

@register("executor_aclnn_fused_infer_attention_score_v4_fd")
class aclnnFusedInferAttentionScoreApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnFusedInferAttentionScoreApi, self).__init__(task_result, backend)
    
    def gen_compressed_triU_mask(self, dim_num, mask_dtype):
        mask_shape_four_dims = (1, 1, 2048, 2048)
        mask_four_dims = torch.zeros(mask_shape_four_dims, dtype = mask_dtype)
        mask_triU = torch.triu(torch.ones(2048, 2048), diagonal=1)
        mask_four_dims[:] = mask_triU
        if dim_num == 2:
            return mask_four_dims[0][0]
        elif dim_num == 3:
            return mask_four_dims[0]
        elif dim_num == 4:
            return mask_four_dims
        else:
            print("invalid dim num, will provide a four dim mask anyway")
            return mask_four_dims
        return mask_four_dims[0][0]
    
    def init_by_input_data(self, input_data: InputDataset):
        np.random.seed(self.task_result.case_config.id)
        # np.random.seed(44)
        if input_data.kwargs["attenMaskOptional"] != None:
            mask_dim_num = len(input_data.kwargs["attenMaskOptional"].shape)
            mask_dtype = input_data.kwargs["attenMaskOptional"].dtype
            input_data.kwargs["attenMaskOptional"] = self.gen_compressed_triU_mask(mask_dim_num, mask_dtype).npu()
        inspect_kwargs(input_data.kwargs)
        input_args = []  # 算子的入参列表

        if True and (input_data.kwargs.get("blockTableOptional") != None):
            # 保留这部分的代码，之前BaseApi的重复代码可以直接删除
            print("====================== 不连续输入 ===================================")
            key_tensor_list = input_data.kwargs["key"]
            # print("key_tensor_list[0]", key_tensor_list[0].shape)
            num_blocks = key_tensor_list[0].shape[0]
            block_size = key_tensor_list[0].shape[1]
            kv_heads = input_data.kwargs["numKeyValueHeads"]
            print("key_tensor_list[0].shape")
            # print(key_tensor_list[0].shape)
            # print(kv_heads)
            # print(input_data.kwargs["blockTableOptional"])
            print("blockTableOptional" in input_data.kwargs)
            print(input_data.kwargs.get("blockTableOptional") == None)
            # print(input_data.kwargs["blockTableOptional"].shape)
            embedding_size = key_tensor_list[0].shape[2] // kv_heads

            kv_shape = [2, num_blocks, block_size, kv_heads * embedding_size]
            import math
            cache_size = math.prod(kv_shape)

            input_dtype = key_tensor_list[0].dtype

            if input_dtype == torch.bfloat16:
                # kv_cache_tensor = torch.zeros(cache_size, device="npu", dtype=torch.bfloat16).npu()  #?需要转npu吗，不知道框架是如何设定的
                # kv_cache_tensor = torch.zeros(cache_size, device="npu", dtype=torch.bfloat16)
                kv_cache_tensor = torch.zeros(cache_size, dtype=torch.bfloat16)
            else:
                # kv_cache_tensor = torch.zeros(cache_size, device="npu", dtype=torch.float16).npu()
                # kv_cache_tensor = torch.zeros(cache_size, device="npu", dtype=torch.bfloat16)
                kv_cache_tensor = torch.zeros(cache_size, dtype=torch.float16)
            kv_cache = kv_cache_tensor.view(kv_shape)

            print(f"{kv_cache.shape=}")
            print("kv_cache_tensor.is_contiguous =", kv_cache_tensor.is_contiguous())   # 一定输出 False

            hidden_size = kv_cache.shape[2:].numel()
            kv_cache.as_strided_(
                size=kv_cache.shape,
                stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
            )

            # 你的读取格式：(num_blocks, 128, kv_n*D) → 对应 unbind 后 shape
            k_np = input_data.kwargs["key"]
            v_np = input_data.kwargs["value"]

            # 转为 NPU tensor
            k_data = [t.npu() for t in k_np]
            v_data = [t.npu() for t in v_np]

            print(f"{kv_cache.shape=}")
            k, v = kv_cache.unbind(0)  # k, v 都是 shape [num_blocks, 128, kv_n*D]

            print("======", len(key_tensor_list))

            # # 按 block index 赋值：key_cache[0] = 0, key_cache[1] = 1, ...
            # for i in range(num_blocks):
            #     key_cache[i] = i

            # # 按 block index 赋值：value_cache[0] = 100, value_cache[1] = 101, ...
            # for i in range(num_blocks):
            #     value_cache[i] = 100 + i


            # 把 list 数据复制到 cache
            k.copy_(torch.stack(k_data).squeeze(0))  # squeeze(0) 删掉第0维的1
            v.copy_(torch.stack(v_data).squeeze(0))


            # 按 block index 赋值：key_cache[0] = 0, key_cache[1] = 1, ...
            # for i in range(num_blocks):
            #     k[i] = i

            # 按 block index 赋值：value_cache[0] = 100, value_cache[1] = 101, ...
            # for i in range(num_blocks):
            #     v[i] = 100 + i
            # ==========================================================================

            # ======================
            # ✅ 你要的：输出长度=1 的 list
            # ======================
            # print("k的device是：", k.device)
            # print("v的device是：", v.device)
            # k = k.npu()
            # v = v.npu()
            # print("k的device是：", k.device)
            # print("v的device是：", v.device)

            input_data.kwargs["key"] = [k]       # 直接包进 list，长度=1
            # print("===== k ", k[0].flatten()[100])
            # print("===== k ", k[1].flatten()[100])
            input_data.kwargs["value"] = [v]
            # print("===== v ", v[0].flatten()[100])
            # print("===== v ", v[1].flatten()[100])

            # print("input_data.kwargs["key"]的device是：", input_data.kwargs["key"].device)


            # 下面打印不变
            print("k.is_contiguous =", k.is_contiguous())
            # print("tensor.is_contiguous =", tensor.is_contiguous())
            print("v.is_contiguous =", v.is_contiguous())
            print(f"k stride: {k.stride()}")

            # 检查连续性
            if True:
                # ... 在你原有的代码之后 ...

                print("\n" + "=" * 60)
                print("通过检查内存地址来验证 key_cache 的连续性")
                print("=" * 60)

                # 1. 获取 k[0] 和 k[1] 的内存起始地址
                if k.shape[0] > 1:
                    addr_0 = k[0].data_ptr()
                    addr_1 = k[1].data_ptr()

                    # 2. 计算单个 block 的字节数
                    element_size = k.element_size()
                    numel_per_block = k[0].numel()
                    expected_stride_bytes = element_size * numel_per_block

                    # 3. 计算实际地址差并给出结论
                    actual_stride_bytes = addr_1 - addr_0
                    is_continuous = actual_stride_bytes == expected_stride_bytes

                    print(f"k[0] 的内存首地址: {addr_0}")
                    print(f"k[1] 的内存首地址: {addr_1}")
                    print(f"如果连续，地址差应为: {expected_stride_bytes} 字节")
                    print(f"实际的地址差是:       {actual_stride_bytes} 字节")
                    print(f"结论: k 在内存中是{'连续的' if is_continuous else '不连续的！地址之间存在跳跃'}。")

                    print("=" * 60)

        input_args, output_packages = super().init_by_input_data(input_data)

        # 将所有type是tensor values是None的输入 改为AclTensor类型的空指针
        for i, (name, kwarg) in enumerate(input_data.kwargs.items()):
            if kwarg is None and self.task_result.case_config.inputs[i].type == "tensor":
                from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
                input_args[i] = TensorPtr()
        
        # input_data.kwargs["learnableSinkOptional"] = None
            
        output_packages = []  # 算子的出参数据包列表
        input_args.pop()
        if input_data.kwargs["softmaxLseFlag"] == True:
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            output_packages.append(input_args[-2])
        return input_args, output_packages
    
    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(dtype=torch.float)
            output.append(temp_output_pack)
            # print(f"out_npu_dtype: {temp_output_pack.dtype}")
        # print("aclnn_output is: ", output)

        return output