import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
import ctypes
from ml_dtypes import bfloat16
from dataclasses import dataclass
random.seed(1)      # Python 内置 random
np.random.seed(1)   # NumPy 随机数
# torch.manual_seed(1) # PyTorch 随机数（如果其他地方用到）
# torch.set_printoptions(precision=6, sci_mode=False)
# torch.set_printoptions(threshold=float('inf'))

# 全局开关
DEBUG = True
# 重新定义print
if not DEBUG:
    print = lambda *args, **kwargs: None

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

def gen_block_sparse_pattern(q_seqlen_list: any,
        kv_seqlen_list: any,
        block_shape: any,
        batch: int,
        num_heads: int,
        sparsity_ratio: float,
        is_varied_sparsity: int
):
    q_s_aligned = max(q_seqlen_list)
    kv_s_aligned = max(kv_seqlen_list)
    s_block_x = block_shape[0]
    s_block_y = block_shape[1]
    q_s_block_num_aligned = (q_s_aligned + s_block_x - 1) // s_block_x
    kv_s_block_num_aligned = (kv_s_aligned + s_block_y - 1) // s_block_y

    block_sparse_mask = np.zeros((batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned), dtype=np.int8)
    block_sparse_idx = np.ones((batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned), dtype=np.int32)
    # sparse idx 无效位置为-1，便于和idx=0做区分
    block_sparse_idx = -block_sparse_idx
    block_sparse_count = np.zeros((batch, num_heads, q_s_block_num_aligned), dtype=np.int32)

    for b in range(batch):
        q_s_block_num_aval = (q_seqlen_list[b] + s_block_x - 1) // s_block_x
        kv_s_block_num_aval = (kv_seqlen_list[b] + s_block_y - 1) // s_block_y
        max_rsvd_kv_s_block_num = max(1, int(kv_s_block_num_aval * sparsity_ratio))
        for n1 in range(num_heads):
            for q_blk_id in range(q_s_block_num_aval):
                cur_row_sparse_mask = np.zeros(kv_s_block_num_aval, dtype=np.int8)
                # 当前rainfusion稀疏算法的原理：每个BN组合的所有X方向分块都采取同一个topK的方式选择Y方向的块
                # is_varied_sparsity是依据上述算法发散的进一步泛化场景：每个BN组合的每个X方向分块的topK的值也不相同
                rsvd_kv_s_block_num = max_rsvd_kv_s_block_num
                if is_varied_sparsity:
                    random.seed(b * num_heads * q_s_block_num_aval + n1 * q_s_block_num_aval + q_blk_id)
                    rsvd_kv_s_block_num = random.randint(1, max_rsvd_kv_s_block_num)
                # 依据随机出的保留块计数生成随机的稀疏pattern以及保留块下标
                cur_row_sparse_idx = sorted(np.random.choice(kv_s_block_num_aval, rsvd_kv_s_block_num, replace=False))
                for idx in cur_row_sparse_idx:
                    cur_row_sparse_mask[idx] = 1
                # 将生成的mask，idx，count填入对应位置
                block_sparse_mask[b, n1, q_blk_id, 0:kv_s_block_num_aval] = cur_row_sparse_mask
                block_sparse_idx[b, n1, q_blk_id, 0:rsvd_kv_s_block_num] = cur_row_sparse_idx
                block_sparse_count[b, n1, q_blk_id] = rsvd_kv_s_block_num
    return block_sparse_mask, block_sparse_idx, block_sparse_count

class TestBlockSparseAttentionGenBmGt():
    @dataclass
    class AuxAttrs:
        num_heads: int
        kv_heads: int
        embedding_size: int
        dtype: any
        inner_prec: int
        scale: float
        q_layout: str
        kv_layout: str
        attention_out_dtype: any # tkd

    @dataclass
    class AttentionInputs:
        query: any
        key: any
        value: any
        block_sparse_idx: any
        block_sparse_count: any
        block_sparse_shape:any
        q_seqlen_list: any
        kv_seqlen_list: any
        q_dequant_scale: any # tkd
        k_dequant_scale: any # tkd
        v_dequant_scale: any # tkd
        aux_attrs: any

    def fp32_to_fp8_RINT(self, tensor: np.ndarray, round_mode: str = "rint") -> np.ndarray:
        """将 FP32 numpy 数组转换为 FP8 浮点值（向量化版本）"""
        original_shape = tensor.shape
        flat = tensor.flatten().astype(np.float32)
        
        # 获取编码后的 bits（向量化）
        bits = self._float32_to_fp8_e4m3_vec(flat, round_mode)
        
        # 解码回 float32（向量化）
        result = self._fp8_e4m3_to_float32_vec(bits)
        
        return result.reshape(original_shape)
    
    def _float32_to_fp8_e4m3_vec(self, arr: np.ndarray, round_mode: str = "rint") -> np.ndarray:
        """向量化版本的 FP32 到 FP8 编码（返回编码后的整数）"""
        MAX_FP8_E4M3 = 448.0
        MIN_NORMAL = 2 ** (-6)
        
        # 初始化结果
        result_bits = np.zeros_like(arr, dtype=np.uint8)
        
        # 符号和绝对值
        sign = (arr < 0).astype(np.uint8)
        abs_val = np.abs(arr)
        
        # 1. 处理溢出 (>= MAX)
        overflow_mask = abs_val >= MAX_FP8_E4M3
        if np.any(overflow_mask):
            result_bits[overflow_mask] = np.where(sign[overflow_mask] == 0, 0x7E, 0xFE)
        
        # 2. 处理零值
        zero_mask = (arr == 0.0)
        result_bits[zero_mask] = 0x00
        
        # 3. 处理正常范围的值
        valid_mask = ~(overflow_mask | zero_mask)
        
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            valid_abs = abs_val[valid_indices]
            valid_sign = sign[valid_indices]
            
            # 4. 处理下溢 (次正规数)
            underflow_mask = valid_abs < MIN_NORMAL
            
            if np.any(underflow_mask):
                under_indices = valid_indices[underflow_mask]
                under_abs = valid_abs[underflow_mask]
                under_sign = valid_sign[underflow_mask]
                
                mantissa_scaled = under_abs / MIN_NORMAL * 8.0
                mantissa = np.clip(np.rint(mantissa_scaled), 0, 7).astype(np.uint8)
                result_bits[under_indices] = (under_sign << 7) | mantissa
            
            # 5. 处理正常范围的值 (>= MIN_NORMAL)
            normal_mask = ~underflow_mask
            if np.any(normal_mask):
                normal_indices = valid_indices[normal_mask]
                normal_abs = valid_abs[normal_mask]
                normal_sign = valid_sign[normal_mask]
                
                # 计算指数：使用 frexp
                _, exponent_int = np.frexp(normal_abs)
                exponent = exponent_int - 1
                exponent = np.clip(exponent, -6, 8)
                
                # 计算尾数 (1.mmm)
                mantissa_float = normal_abs / (2.0 ** exponent.astype(np.float32)) - 1.0
                mantissa_scaled = mantissa_float * 8.0
                
                # 舍入
                if round_mode == "rint":
                    mantissa = np.rint(mantissa_scaled).astype(np.int32)
                else:
                    mantissa = np.round(mantissa_scaled).astype(np.int32)
                
                # 处理进位 (mantissa == 8)
                carry_mask = (mantissa == 8)
                mantissa[carry_mask] = 0
                exponent[carry_mask] += 1
                
                # 检查进位后是否上溢
                overflow_carry_mask = exponent > 8
                if np.any(overflow_carry_mask):
                    carry_overflow_indices = normal_indices[overflow_carry_mask]
                    carry_overflow_sign = normal_sign[overflow_carry_mask]
                    result_bits[carry_overflow_indices] = np.where(
                        carry_overflow_sign == 0, 0x7E, 0xFE
                    )
                    
                    # 移除这些值
                    keep_mask = ~overflow_carry_mask
                    if np.any(keep_mask):
                        normal_indices = normal_indices[keep_mask]
                        normal_sign = normal_sign[keep_mask]
                        mantissa = mantissa[keep_mask]
                        exponent = exponent[keep_mask]
                    else:
                        normal_indices = np.array([], dtype=np.int64)
                        mantissa = np.array([], dtype=np.int32)
                        exponent = np.array([], dtype=np.int32)
                
                # 处理剩余的
                if len(normal_indices) > 0:
                    mantissa = np.clip(mantissa, 0, 7).astype(np.uint8)
                    stored_exp = (exponent + 7).astype(np.uint8)
                    stored_exp = np.clip(stored_exp, 1, 15)
                    
                    result_bits[normal_indices] = (
                        (normal_sign << 7) | (stored_exp << 3) | mantissa
                    )
        
        return result_bits
    
    def _fp8_e4m3_to_float32_vec(self, bits: np.ndarray) -> np.ndarray:
        """向量化版本的 FP8 解码（返回 float32）"""
        sign = (bits >> 7) & 0x1
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        
        result = np.zeros_like(bits, dtype=np.float32)
        
        # 正常值 (exponent != 0)
        normal_mask = (exponent != 0)
        if np.any(normal_mask):
            exp_normal = exponent[normal_mask].astype(np.float32)
            mant_normal = mantissa[normal_mask].astype(np.float32)
            sign_normal = sign[normal_mask]
            
            value = (2.0 ** (exp_normal - 7)) * (1.0 + mant_normal / 8.0)
            value = np.where(sign_normal == 1, -value, value)
            result[normal_mask] = value
        
        # 次正规数 (exponent == 0, mantissa != 0)
        subnormal_mask = (exponent == 0) & (mantissa != 0)
        if np.any(subnormal_mask):
            mant_sub = mantissa[subnormal_mask].astype(np.float32)
            sign_sub = sign[subnormal_mask]
            
            value = (2.0 ** (-6)) * (mant_sub / 8.0)
            value = np.where(sign_sub == 1, -value, value)
            result[subnormal_mask] = value
        
        # exponent == 0 and mantissa == 0 已经是 0
        
        return result

    def base_tile_mm(self, left, right, mm_k_tile):
        res = None
        mm_k_loop = (left.shape[1] + mm_k_tile - 1) // mm_k_tile
        for mm1_k_loop_idx in range(mm_k_loop):
            sub_k = (left.shape[1] - mm1_k_loop_idx * mm_k_tile) if (mm1_k_loop_idx == mm_k_loop - 1) else mm_k_tile
            left_slice = left[:, mm1_k_loop_idx * mm_k_tile: mm1_k_loop_idx * mm_k_tile + sub_k]
            right_slice = right[mm1_k_loop_idx * mm_k_tile: mm1_k_loop_idx * mm_k_tile + sub_k, :]
            res_slice = np.matmul(left_slice, right_slice).astype(np.float32)
            res = res_slice if res is None else (res + res_slice)
        return res

    def base_tile_mm_particular(self, left, right, mm_k_tile, dtype):
        res = None
        mm_k_loop = (left.shape[1] + mm_k_tile - 1) // mm_k_tile
        for mm1_k_loop_idx in range(mm_k_loop):
            sub_k = (left.shape[1] - mm1_k_loop_idx * mm_k_tile) if (mm1_k_loop_idx == mm_k_loop - 1) else mm_k_tile
            left_slice = left[:, mm1_k_loop_idx * mm_k_tile: mm1_k_loop_idx * mm_k_tile + sub_k].astype(np.float32)
            right_slice = right[mm1_k_loop_idx * mm_k_tile: mm1_k_loop_idx * mm_k_tile + sub_k, :].astype(np.float32)
            res_slice = np.matmul(left_slice, right_slice)
            res = res_slice if res is None else (res + res_slice)
        return res

    def online_softmax(self, qk_tile_res, gm, is_first_tile, interm_dtype_sm=np.float32):
        sim = qk_tile_res.astype(interm_dtype_sm)
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first_tile:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        gm = hm
        sim_sub = sim - hm
        p = np.exp(sim_sub.astype(interm_dtype_sm))
        ll = np.sum(p, axis=-1, keepdims=True)
        return p, ll, dm, gm, lm

    def rescale_o(self, lo, ll, dm, go, gl, is_first_tile, interm_dtype_re=np.float32):
        if is_first_tile:
            gl = ll
            go = lo
        else:
            dm = np.exp(dm)
            gl = gl * dm
            gl = gl + ll
            go = go * dm.astype(interm_dtype_re)
            go = go + lo
        return go, gl.astype(interm_dtype_re)

    def ref_flash_block_sparse_attention(self, query, key, value, softmax_scale, inner_prec,
                                         q_deq_scale, # tkd
                                         k_deq_scale_gathered, # tkd
                                         v_deq_scale_gathered, # tkd
                                         block_shape_y, # tkd
                                         attention_out_dtype, # tkd
                                         attn_mask=None):
        cur_kv_s_gathered = key.shape[1]
        kv_s_base_tile = 256 if inner_prec == 4 else 128
        interm_dtype_sm = attention_out_dtype # tkd
        interm_dtype_re = query.dtype if inner_prec == 1 else np.float32
        softmax_scale = np.asarray(softmax_scale, dtype=query.dtype) # tkd
        gl = None
        go = None

        # loop_count = block_shape_y // kv_s_base_tile # tkd
        deq_scale_index = 0 # tkd
        already_deq_scale_time = 0 # tkd

        for kv_s_start in range(0, cur_kv_s_gathered, kv_s_base_tile):
            cur_kv_s_tile = (cur_kv_s_gathered - kv_s_start) if (kv_s_start + kv_s_base_tile > cur_kv_s_gathered) else kv_s_base_tile
            key_tile = key[:, kv_s_start:kv_s_start + cur_kv_s_tile]
            value_tile = value[kv_s_start:kv_s_start + cur_kv_s_tile, :]
            mm1_k_tile = 128

            qk_tile_res = self.base_tile_mm_particular(query, key_tile, mm1_k_tile, query.dtype)

            y_blk_idx = kv_s_start // block_shape_y # tkd
            combined_deq_scale = q_deq_scale * k_deq_scale_gathered[y_blk_idx] # tkd
            qk_tile_res = qk_tile_res * combined_deq_scale # tkd
            qk_tile_res = qk_tile_res.astype(interm_dtype_sm) # tkd
            qk_tile_res = qk_tile_res * softmax_scale

            if kv_s_start == 0:
                gm = None
            p, ll, dm, gm, lm = self.online_softmax(qk_tile_res, gm, kv_s_start == 0, interm_dtype_sm)
            p = p * 448 # tkd
            p = self.fp32_to_fp8_RINT(p) # tkd
            p = p.astype(query.dtype)
            mm2_k_tile = 128
            lo = self.base_tile_mm_particular(p, value_tile, mm2_k_tile, np.float32)
            lo = lo * v_deq_scale_gathered[y_blk_idx]
            go, gl = self.rescale_o(lo, ll, dm, go, gl, kv_s_start == 0, interm_dtype_re)
        go = go / gl
        go = go * (1.0 / 448.0)
        go = go.astype(query.dtype)
        lse = np.squeeze(np.log(gl) + gm, axis=-1).astype(np.float32)
        return go, lse

    def ref_attention(self, query, key, value, softmax_scale, inner_prec, attn_mask=None):
        s = np.matmul(query.astype(np.float32), key.astype(np.float32)).astype(np.float32)
        s = s * np.float32(softmax_scale)
        row_max = np.max(s, axis=-1, keepdims=True)
        s_sub = s - row_max
        s_sub = np.exp(s_sub)
        row_sum = np.sum(s_sub, axis=-1, keepdims=True)
        p = s_sub / row_sum
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)
        o = np.matmul(p.astype(np.float32), value.astype(np.float32)).astype(np.float32)
        return o, lse

    def compute_output(self, attention_inputs: AttentionInputs, is_benchmark):
        attn_out_bm = None
        attn_out_gt = None
        lse_bm = None
        lse_gt = None
        if not is_benchmark:
            attn_out_bm = np.zeros_like(attention_inputs.query) # tkd
        else:
            attn_out_gt = np.zeros_like(attention_inputs.query).astype(np.float32)

        input_layout = attention_inputs.aux_attrs.q_layout
        if input_layout == "TND":
            if not is_benchmark:
                lse_bm = np.zeros((attention_inputs.query.shape[0], attention_inputs.query.shape[1], 1), dtype=np.float32)
            else:
                lse_gt = np.zeros((attention_inputs.query.shape[0], attention_inputs.query.shape[1], 1), dtype=np.float32)
        elif input_layout == "BNSD" or input_layout == "BSND":
            if not is_benchmark:
                lse_bm = np.zeros((attention_inputs.query.shape[0], attention_inputs.query.shape[1], attention_inputs.query.shape[2], 1), dtype=np.float32)
            else:
                lse_gt = np.zeros((attention_inputs.query.shape[0], attention_inputs.query.shape[1], attention_inputs.query.shape[2], 1), dtype=np.float32)

        query = attention_inputs.query
        key = attention_inputs.key
        value = attention_inputs.value
        block_sparse_idx = attention_inputs.block_sparse_idx
        block_sparse_count = attention_inputs.block_sparse_count
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list
        q_dequant_scale = attention_inputs.q_dequant_scale # tkd
        k_dequant_scale = attention_inputs.k_dequant_scale # tkd
        v_dequant_scale = attention_inputs.v_dequant_scale # tkd

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        softmax_scale = attention_inputs.aux_attrs.scale
        inner_prec = attention_inputs.aux_attrs.inner_prec
        group_size = int(num_heads / kv_heads)
        attention_out_dtype = attention_inputs.aux_attrs.attention_out_dtype # tkd
        deq_scale_index = 0 # tkd
        already_deq_scale_time = 0 # tkd
        block_shape_x = block_sparse_shape[0]
        block_shape_y = block_sparse_shape[1] # tkd
        # loop_count = block_shape_y // 256 # tkd
        
        q_s_aligned = max(q_seqlen_list)
        kv_s_aligned = max(kv_seqlen_list)
        q_s_block_num_aligned = (q_s_aligned + block_shape_x - 1) // block_shape_x
        kv_s_block_num_aligned = (kv_s_aligned + block_shape_y - 1) // block_shape_y

        q_offsets_sum = [0]
        for i in range(len(q_seqlen_list) - 1):
            q_offsets_sum.append(q_offsets_sum[-1] + q_seqlen_list[i])

        kv_offsets_sum = [0]
        for i in range(len(kv_seqlen_list) - 1):
            kv_offsets_sum.append(kv_offsets_sum[-1] + kv_seqlen_list[i])

        for b in range(batch):
            q_s_block_num_aval = (q_seqlen_list[b] + block_shape_x - 1) // block_shape_x
            kv_s_block_num_aval = (kv_seqlen_list[b] + block_shape_y - 1) // block_shape_y
            for n1 in range(num_heads):
                n2 = int(n1 // group_size)
                for q_s_blk_idx in range(q_s_block_num_aval):
                    cur_q_s_blk_size = (q_seqlen_list[b] - q_s_blk_idx * block_shape_x) if q_s_blk_idx == (q_s_block_num_aval - 1) else block_shape_x
                    q_blk = None
                    q_deq_scale = q_dequant_scale[b, n1, q_s_blk_idx] # tkd
                    if q_layout == "TND":
                        t_offset = q_offsets_sum[b] + q_s_blk_idx * block_shape_x
                        q_blk = query[t_offset:t_offset + cur_q_s_blk_size, n1, :]
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_shape_x
                        q_blk = query[b, n1, s1_offset:s1_offset + cur_q_s_blk_size,:]
                    elif q_layout == "BSND":
                        s1_offset = q_s_blk_idx * block_shape_x
                        q_blk = query[b, s1_offset:s1_offset + cur_q_s_blk_size, n1, :]
                    q_blk_antiquant = q_blk.astype(np.float32) * q_deq_scale # tkd
                    cur_kv_s_rsvd_blk_count = block_sparse_count[b, n1, q_s_blk_idx]
                    # cur_kv_s_rsvd_blk_count = block_sparse_count[b, n1, q_s_blk_idx].item()
                    cur_kv_s_sparse_idx = block_sparse_idx[b, n1, q_s_blk_idx, :cur_kv_s_rsvd_blk_count].reshape(cur_kv_s_rsvd_blk_count).tolist()
                    key_gathered = None
                    value_gathered = None
                    k_deq_scale_gathered = None # tkd
                    v_deq_scale_gathered = None # tkd
                    key_gathered_antiquant = None # tkd
                    value_gathered_antiquant = None # tkd
                    # 使用列表收集所有部分
                    key_parts = []
                    value_parts = []
                    k_scale_parts = []
                    v_scale_parts = []
                    key_gathered_antiquant_parts = []
                    value_gathered_antiquant_parts = []

                    for kv_s_blk_idx in cur_kv_s_sparse_idx:
                        cur_kv_s_blk_size = (kv_seqlen_list[b] - kv_s_blk_idx * block_sparse_shape[1]) if kv_s_blk_idx == (kv_s_block_num_aval - 1) else block_sparse_shape[1]
                        cur_k_blk = None
                        cur_v_blk = None
                        cur_k_blk_antiquant = None
                        cur_v_blk_antiquant = None
                        cur_k_deq_scale_blk = k_dequant_scale[b, n2, kv_s_blk_idx]
                        cur_v_deq_scale_blk = v_dequant_scale[b, n2, kv_s_blk_idx]
                        
                        if kv_layout == "TND":
                            t_offset = sum(kv_seqlen_list[:b]) + kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[t_offset:t_offset + cur_kv_s_blk_size, n2, :]
                            cur_v_blk = value[t_offset:t_offset + cur_kv_s_blk_size, n2, :]
                            cur_k_blk = cur_k_blk.transpose(1, 0)
                            cur_k_blk_antiquant = key[t_offset:t_offset + cur_kv_s_blk_size, n2, :].astype(np.float32) * cur_k_deq_scale_blk
                            cur_v_blk_antiquant = value[t_offset:t_offset + cur_kv_s_blk_size, n2, :].astype(np.float32) * cur_v_deq_scale_blk
                            cur_k_blk_antiquant = cur_k_blk_antiquant.transpose(1, 0)
                        elif kv_layout == "BNSD":
                            s2_offset = kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[b, n2, s2_offset:s2_offset + cur_kv_s_blk_size, :]
                            cur_v_blk = value[b, n2, s2_offset:s2_offset + cur_kv_s_blk_size, :]
                            cur_k_blk = cur_k_blk.transpose(1, 0)
                            cur_k_blk_antiquant = key[b, n2, s2_offset:s2_offset + cur_kv_s_blk_size, :].astype(np.float32) * cur_k_deq_scale_blk
                            cur_v_blk_antiquant = value[b, n2, s2_offset:s2_offset + cur_kv_s_blk_size, :].astype(np.float32) * cur_v_deq_scale_blk
                            cur_k_blk_antiquant = cur_k_blk_antiquant.transpose(1, 0)
                        elif kv_layout == "BSND":
                            s2_offset = kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[b, s2_offset:s2_offset + cur_kv_s_blk_size, n2, :]
                            cur_v_blk = value[b, s2_offset:s2_offset + cur_kv_s_blk_size, n2, :]
                            cur_k_blk = cur_k_blk.transpose(1, 0)
                            cur_k_blk_antiquant = key[b, s2_offset:s2_offset + cur_kv_s_blk_size, n2, :].astype(np.float32) * cur_k_deq_scale_blk
                            cur_v_blk_antiquant = value[b, s2_offset:s2_offset + cur_kv_s_blk_size, n2, :].astype(np.float32) * cur_v_deq_scale_blk
                            cur_k_blk_antiquant = cur_k_blk_antiquant.transpose(1, 0)

                        # 收集到列表中（不立即拼接）
                        key_parts.append(cur_k_blk)
                        value_parts.append(cur_v_blk)
                        k_scale_parts.append(cur_k_deq_scale_blk)
                        v_scale_parts.append(cur_v_deq_scale_blk)
                        key_gathered_antiquant_parts.append(cur_k_blk_antiquant)
                        value_gathered_antiquant_parts.append(cur_v_blk_antiquant)

                    # 循环结束后，一次性拼接所有部分
                    if key_parts:
                        key_gathered = np.concatenate(key_parts, axis=1)
                        value_gathered = np.concatenate(value_parts, axis=0)
                        k_deq_scale_gathered = np.concatenate(k_scale_parts, axis=0)
                        v_deq_scale_gathered = np.concatenate(v_scale_parts, axis=0)
                        key_gathered_antiquant = np.concatenate(key_gathered_antiquant_parts, axis=1)
                        value_gathered_antiquant = np.concatenate(value_gathered_antiquant_parts, axis=0)
                    else:
                        key_gathered = None
                        value_gathered = None
                        k_deq_scale_gathered = None
                        v_deq_scale_gathered = None
                        key_gathered_antiquant = None
                        value_gathered_antiquant = None

                    if not is_benchmark:
                        attn_out_bm_slice, lse_bm_slice = self.ref_flash_block_sparse_attention(
                            q_blk, key_gathered, value_gathered, softmax_scale, inner_prec,
                            q_deq_scale,  # tkd
                            k_deq_scale_gathered,  # tkd
                            v_deq_scale_gathered,  # tkd
                            block_shape_y,  # tkd
                            attention_out_dtype
                        )  # tkd
                    else:
                        q_blk_antiquant = q_blk_antiquant  # 已经是 numpy 数组
                        key_gathered_antiquant = key_gathered_antiquant  # 已经是 numpy 数组
                        value_gathered_antiquant = value_gathered_antiquant  # 已经是 numpy 数组

                        attn_out_gt_slice, lse_gt_slice = self.ref_attention(
                            q_blk_antiquant, key_gathered_antiquant, value_gathered_antiquant,
                            softmax_scale, inner_prec
                        )  # tkd

                    # 根据 layout 和 is_benchmark 进行赋值
                    if q_layout == "TND":
                        t_offset = q_offsets_sum[b] + q_s_blk_idx * block_shape_x
                        if not is_benchmark:
                            # Benchmark 模式：写入 attn_out_bm
                            attn_out_bm[t_offset:t_offset + cur_q_s_blk_size, n1, :] = attn_out_bm_slice
                            lse_bm[t_offset:t_offset + cur_q_s_blk_size, n1, :] = lse_bm_slice.reshape(-1, 1) # tkd
                        else:
                            # 非 Benchmark 模式：写入 attn_out_gt
                            attn_out_gt[t_offset:t_offset + cur_q_s_blk_size, n1, :] = attn_out_gt_slice
                            lse_gt[t_offset:t_offset + cur_q_s_blk_size, n1, :] = lse_gt_slice.reshape(-1, 1) # tkd
                            
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_shape_x
                        if not is_benchmark:
                            # Benchmark 模式：写入 attn_out_bm
                            attn_out_bm[b, n1, s1_offset:s1_offset + cur_q_s_blk_size, :] = attn_out_bm_slice
                            lse_bm[b, n1, s1_offset:s1_offset + cur_q_s_blk_size, :] = lse_bm_slice.reshape(-1, 1)
                        else:
                            # 非 Benchmark 模式：写入 attn_out_gt
                            attn_out_gt[b, n1, s1_offset:s1_offset + cur_q_s_blk_size, :] = attn_out_gt_slice
                            lse_gt[b, n1, s1_offset:s1_offset + cur_q_s_blk_size, :] = lse_gt_slice.reshape(-1, 1)
                            
                    elif q_layout == "BSND":
                        s1_offset = q_s_blk_idx * block_shape_x
                        if not is_benchmark:
                            # Benchmark 模式：写入 attn_out_bm
                            attn_out_bm[b, s1_offset:s1_offset + cur_q_s_blk_size, n1, :] = attn_out_bm_slice
                            lse_bm[b, s1_offset:s1_offset + cur_q_s_blk_size, n1, :] = lse_bm_slice.reshape(-1, 1)
                        else:
                            # 非 Benchmark 模式：写入 attn_out_gt
                            attn_out_gt[b, s1_offset:s1_offset + cur_q_s_blk_size, n1, :] = attn_out_gt_slice
                            lse_gt[b, s1_offset:s1_offset + cur_q_s_blk_size, n1, :] = lse_gt_slice.reshape(-1, 1)

        return attn_out_bm, attn_out_gt, lse_bm, lse_gt

def tensor_to_numpy_array(t):
    if t is None:
        return None
    if t.dtype is torch.bfloat16:
        return t.to(torch.float32).numpy().astype(bfloat16)
    else:
        return t.numpy()

def numpy_array_to_tensor(arr):
    if arr is None:
        return None
    # 如果是 list，先转为 numpy array
    if isinstance(arr, list):
        arr = np.array(arr)
    # 如果是 bfloat16，先转 float32 再转 torch
    if arr.dtype.name == 'bfloat16':
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    # 其他类型直接转换
    return torch.from_numpy(arr)


@register("aclnn_blocksparseattentioninputprocess")
class BlockSparseAttentionInputProcess(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(BlockSparseAttentionInputProcess, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data):
        torch.manual_seed(10)
        torch.npu.synchronize()
        input_args = []
        output_packages = []

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)

        input_args.extend(output_packages)

        AclTensorPtr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)

        query = self.acl_tensor_to_torch(input_args[0]).to("npu")
        input_args[0] = self.torch_tensor_to_acl(query)
        key = self.acl_tensor_to_torch(input_args[1]).to("npu")
        input_args[1] = self.torch_tensor_to_acl(key)
        value = self.acl_tensor_to_torch(input_args[2]).to("npu")
        input_args[2] = self.torch_tensor_to_acl(value)
        blockSparseMask = self.acl_tensor_to_torch(input_args[3]).to("npu")
        input_args[3] = self.torch_tensor_to_acl(blockSparseMask)
        qDequantScale = self.acl_tensor_to_torch(input_args[9]).to("npu") # tkd
        input_args[9] = self.torch_tensor_to_acl(qDequantScale) # tkd
        kDequantScale = self.acl_tensor_to_torch(input_args[10]).to("npu") # tkd
        input_args[10] = self.torch_tensor_to_acl(kDequantScale) # tkd
        vDequantScale = self.acl_tensor_to_torch(input_args[11]).to("npu") # tkd
        input_args[11] = self.torch_tensor_to_acl(vDequantScale) # tkd

        self.input_layout = input_data.kwargs['qInputLayout']
        self.softmax_lse_flag = input_data.kwargs['softmaxLseFlag']

        self.qSeqlenList = input_data.kwargs['actualSeqLengths']
        if self.qSeqlenList is None:
            self.actseq_aval = 0
        else:
            self.actseq_aval = 1

        output_packages = []

        kv_heads = input_data.kwargs["numKeyValueHeads"]
        if self.input_layout == "TND":
            qhead = query.shape[1]
        elif self.input_layout == "BNSD":
            qhead = query.shape[1]
        elif self.input_layout == "BSND":
            qhead = query.shape[2]
        use_eff_rows = (qhead == kv_heads)
        atten_mask = input_data.kwargs.get("attenMask", None)
        if use_eff_rows and atten_mask is not None:
            block_eff_rows = self.acl_tensor_to_torch(input_args[4]).to("npu")
            input_args[4] = self.torch_tensor_to_acl(block_eff_rows)
            print("WANGCHAO DEBUG [use attenmask]")
        else:
            input_args[4] = null_tensor_ptr # attenMask
            print("WANGCHAO DEBUG [null attenmask]")
        input_args[8] = null_tensor_ptr # tkd blockTable
        # input_args[23] = null_tensor_ptr # tkd softmaxLse
        if input_data.kwargs['softmaxLseFlag'] == 1:
            input_args.pop()
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            input_args.pop()
            output_packages.append(input_args[-2])
        return input_args, output_packages

    def __call__(self):
        """算子调用逻辑"""
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(dtype=torch.float32)
            output.append(temp_output_pack)

        if self.input_layout == "BNSD":
            outputTemp = torch.zeros((output[0].shape[0], output[0].shape[1], output[0].shape[2], output[0].shape[3]), dtype=output[0].dtype)
            outputTensor = output[0]
            if self.qSeqlenList is not None:
                for i in range(output[0].shape[0]):
                    outputTemp[i, :, :self.qSeqlenList[i], :] = outputTensor[i, :, :self.qSeqlenList[i], :]
                print("WANGCHAO AclnnBaseApi ", outputTemp)
                if len(output) == 2:
                    lseTemp = torch.zeros((output[1].shape[0], output[1].shape[1], output[1].shape[2], 1), dtype=output[1].dtype)
                    lseTensor = output[1]
                    for i in range(output[1].shape[0]):
                        lseTemp[i, :, :self.qSeqlenList[i], :] = lseTensor[i, :, :self.qSeqlenList[i], :]
                    return [outputTemp, lseTemp]
                else:
                    return [outputTemp]
        elif self.input_layout == "BSND":
            outputTemp = torch.zeros((output[0].shape[0], output[0].shape[1], output[0].shape[2], output[0].shape[3]), dtype=output[0].dtype)
            outputTensor = output[0]
            if self.qSeqlenList is not None:
                for i in range(output[0].shape[0]):
                    outputTemp[i, :self.qSeqlenList[i], :, :] = outputTensor[i, :self.qSeqlenList[i], :, :]
                print("WANGCHAO AclnnBaseApi ", outputTemp)
                if len(output) == 2:
                    lseTemp = torch.zeros((output[1].shape[0], output[1].shape[1], output[1].shape[2], 1), dtype=output[1].dtype)
                    lseTensor = output[1]
                    for i in range(output[1].shape[0]):
                        lseTemp[i, :self.qSeqlenList[i], :, :] = lseTensor[i, :self.qSeqlenList[i], :, :]
                    return [outputTemp, lseTemp]
                else:
                    return [outputTemp]
        else:
            print("WANGCHAO AclnnBaseApi ", output[0])
            if len(output) == 2:
                return output[0], output[1]
            else:
                return output[0]

    # def get_cpp_func_signature_type(self):
    #     return "aclnnStatus aclnnBlockSparseAttentionV2GetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockSparseMask, const aclTensor *attenMaskOptional, const aclIntArray *blockShape, const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional, const aclTensor *vDequantScaleOptional, char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, int64_t nextTokens, int64_t softmaxLseFlag, const aclTensor *attentionOut, const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)"

@register("aclnn_blocksparseattention_v2")
class BlockSparseAttentionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query_golden = self.query.cpu()
        key_golden = self.key.cpu()
        value_golden = self.value.cpu()
        query_golden = tensor_to_numpy_array(query_golden.to(self.attention_out_dtype))
        key_golden = tensor_to_numpy_array(key_golden.to(self.attention_out_dtype))
        value_golden = tensor_to_numpy_array(value_golden.to(self.attention_out_dtype))

        q_dequant_scale_golden = tensor_to_numpy_array(self.q_dequant_scale.cpu()) # tkd
        k_dequant_scale_golden = tensor_to_numpy_array(self.k_dequant_scale.cpu()) # tkd
        v_dequant_scale_golden = tensor_to_numpy_array(self.v_dequant_scale.cpu()) # tkd

        testObj = TestBlockSparseAttentionGenBmGt()
        aux_attrs = testObj.AuxAttrs(
            self.num_heads, self.kv_heads, self.headDim, self.query_dtype,
            self.inner_precise, self.scale_value, self.q_input_layout, self.kv_input_layout, query_golden.dtype) # tkd
        attention_inputs = testObj.AttentionInputs(
            query_golden, key_golden, value_golden,
            self.block_sparse_idx, self.block_sparse_count, self.block_shape,
            self.q_seqlen_list, self.kv_seqlen_list,
            q_dequant_scale_golden, # tkd
            k_dequant_scale_golden, # tkd
            v_dequant_scale_golden, # tkd
            aux_attrs)
        attn_out_bm_np, attn_out_gt_np, lse_bm_np, lse_gt_np = testObj.compute_output(attention_inputs, self.task_result.is_benchmark_task)
        attn_out_bm = numpy_array_to_tensor(attn_out_bm_np)
        attn_out_gt = numpy_array_to_tensor(attn_out_gt_np)
        lse_bm = numpy_array_to_tensor(lse_bm_np)
        lse_gt = numpy_array_to_tensor(lse_gt_np)
        if not self.task_result.is_benchmark_task:
            # print("tkd attn_out_bm", attn_out_bm)
            # print("tkd lse_bm", lse_bm)
            # print("tkd attn_out_bm shape", attn_out_bm.shape)
            if self.softmax_lse_flag:
                return attn_out_bm.to(torch.float32), lse_bm.to(torch.float32)
            else:
                return attn_out_bm.to(torch.float32)
        else:
            # print("tkd attn_out_gt", attn_out_gt)
            # print("tkd lse_gt", lse_gt)
            # print("tkd attn_out_gt shape", attn_out_gt.shape)
            # gt_nan_count = torch.isnan(attn_out_gt).sum().item()
            # print("gt nan", gt_nan_count)
            if self.softmax_lse_flag:
                return attn_out_gt.to(torch.float32), lse_gt.to(torch.float32)
            else:
                return attn_out_gt.to(torch.float32)

    def init_by_input_data(self, input_data: InputDataset):
        inspect_kwargs(input_data.kwargs)

        self.query = input_data.kwargs["query"]
        self.query_dtype = self.query.dtype
        self.key = input_data.kwargs["key"]
        self.value = input_data.kwargs["value"]
        self.block_shape = input_data.kwargs["blockShape"]
        self.q_seqlen_list = input_data.kwargs["actualSeqLengths"]
        self.kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]
        self.q_input_layout = input_data.kwargs["qInputLayout"]
        self.kv_input_layout = input_data.kwargs["kvInputLayout"]
        self.kv_heads = input_data.kwargs["numKeyValueHeads"]
        self.scale_value = input_data.kwargs["scaleValue"]
        self.inner_precise = input_data.kwargs["innerPrecise"]
        self.q_dequant_scale = input_data.kwargs["qDequantScale"] # tkd
        self.k_dequant_scale = input_data.kwargs["kDequantScale"] # tkd
        self.v_dequant_scale = input_data.kwargs["vDequantScale"] # tkd
        self.attention_out_dtype = input_data.kwargs["attentionOut"].dtype # tkd
        self.softmax_lse_flag = input_data.kwargs["softmaxLseFlag"]
        random.seed(10)
        sparsity_ratio = random.uniform(0.0, 1.0)
        is_varied_sparsity = 0

        if self.q_input_layout == "TND":
            self.batch = len(self.q_seqlen_list)
            self.num_heads = self.query.shape[1]
            self.headDim = self.query.shape[2]
        elif self.q_input_layout == "BNSD":
            self.batch = self.query.shape[0]
            self.num_heads = self.query.shape[1]
            self.headDim = self.query.shape[3]
        elif self.q_input_layout == "BSND":
            self.batch = self.query.shape[0]
            self.num_heads = self.query.shape[2]
            self.headDim = self.query.shape[3]

        # self.block_sparse_mask = block_sparse_mask
        block_sparse_mask_np, self.block_sparse_idx, self.block_sparse_count = gen_block_sparse_pattern(
            self.q_seqlen_list, self.kv_seqlen_list,
            self.block_shape,
            self.batch, self.num_heads,
            sparsity_ratio, is_varied_sparsity)

        self.block_sparse_mask = numpy_array_to_tensor(block_sparse_mask_np)

        if self.device == "cpu":
            input_data.kwargs["blockSparseMask"] = self.block_sparse_mask
        if self.device == "pyaclnn" or self.device == "npu":
            input_data.kwargs["blockSparseMask"] = self.block_sparse_mask.npu()

        # 以下全部新增
        # GQA/MQA not supported with blockEffRows, leave framework-generated attenMask as-is
        s_block_x = self.block_shape[0]
        s_block_y = self.block_shape[1]
        max_q_block_num = block_sparse_mask_np.shape[2]
        max_kv_block_num = block_sparse_mask_np.shape[3]
        if self.num_heads == self.kv_heads:
            max_block_num = max(max_q_block_num, max_kv_block_num)
            block_eff_rows = torch.full(
                (self.batch, self.num_heads, max_block_num, 2),
                fill_value=-1,
                dtype=torch.int32
            )
            for i in range(self.batch):
                q_block_num = (self.q_seqlen_list[i] + s_block_x - 1) // s_block_x
                kv_block_num = (self.kv_seqlen_list[i] + s_block_y - 1) // s_block_y
                for j in range(self.num_heads):
                    for k in range(max_block_num):
                        if k < q_block_num:
                            block_eff_rows[i, j, k, 0] = np.random.randint(0, s_block_x + 1)
                        else:
                            block_eff_rows[i, j, k, 0] = s_block_x
                        if k < kv_block_num:
                            block_eff_rows[i, j, k, 1] = np.random.randint(0, s_block_y + 1)
                        else:
                            block_eff_rows[i, j, k, 1] = s_block_y
            if self.device == "cpu":
                input_data.kwargs["attenMask"] = block_eff_rows
            elif self.device in ("pyaclnn", "npu"):
                input_data.kwargs["attenMask"] = block_eff_rows.npu()