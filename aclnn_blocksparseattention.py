import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import numpy as np
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
import ctypes
from ml_dtypes import bfloat16
from dataclasses import dataclass
np.random.seed(1)

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
        aux_attrs: any
        block_eff_rows: any = None  # 新增，默认None
    
    def base_tile_mm(self, left, right, mm_k_tile):
        res = None
        # 此处与kernel代码中l0 buffer上的切分相同
        mm_k_loop = (left.shape[1] + mm_k_tile - 1) // mm_k_tile
        for mm1_k_loop_idx in range(mm_k_loop):
            sub_k = (left.shape[1] - mm1_k_loop_idx * mm_k_tile) if (mm1_k_loop_idx == mm_k_loop - 1) else mm_k_tile
            left_slice = left[:, mm1_k_loop_idx * mm_k_tile: mm1_k_loop_idx * mm_k_tile + sub_k]
            right_slice = right[mm1_k_loop_idx * mm_k_tile: mm1_k_loop_idx * mm_k_tile + sub_k, :]
            res_slice = np.matmul(left_slice, right_slice).astype(np.float32)
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

    def ref_flash_block_sparse_attention(self, query, key, value, softmax_scale, inner_prec, attn_mask=None):
        # key: [D, S2]
        # value: [S2, D]
        cur_kv_s_gathered = key.shape[1]
        kv_s_base_tile = 256 if inner_prec == 4 else 128
        interm_dtype_sm = np.float32 if inner_prec == 0 else query.dtype
        interm_dtype_re = query.dtype if inner_prec == 1 else np.float32
        softmax_scale = np.asarray(softmax_scale, dtype=query.dtype) if inner_prec == 4 else np.asarray(softmax_scale, dtype=np.float32)
        gl = None
        go = None
        for kv_s_start in range(0, cur_kv_s_gathered, kv_s_base_tile):
            cur_kv_s_tile = (cur_kv_s_gathered - kv_s_start) if (kv_s_start + kv_s_base_tile > cur_kv_s_gathered) else kv_s_base_tile
            key_tile = key[:, kv_s_start:kv_s_start + cur_kv_s_tile]
            value_tile = value[kv_s_start:kv_s_start + cur_kv_s_tile, :]
            # atten_mask暂不支持，逻辑待补充
            # q*k^t
            mm1_k_tile = 128
            qk_tile_res = self.base_tile_mm(query, key_tile, mm1_k_tile).astype(interm_dtype_sm)
            qk_tile_res = qk_tile_res * softmax_scale
            # online softmax
            if kv_s_start == 0:
                gm = None
            p, ll, dm, gm, lm = self.online_softmax(qk_tile_res, gm, kv_s_start == 0, interm_dtype_sm)
            p = p.astype(query.dtype)
            # p*v
            mm2_k_tile = 128
            lo = self.base_tile_mm(p, value_tile, mm2_k_tile).astype(interm_dtype_re)
            # rescale O
            go, gl = self.rescale_o(lo, ll, dm, go, gl, kv_s_start == 0, interm_dtype_re)
        go = go / gl
        go = go.astype(query.dtype) # [S1,D]
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32) # [S1]
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

    def compute_output(self, attention_inputs: AttentionInputs):
        attn_out_bm = np.zeros_like(attention_inputs.query)
        attn_out_gt = np.zeros_like(attention_inputs.query).astype(np.float32)
        
        query = attention_inputs.query
        key = attention_inputs.key
        value = attention_inputs.value
        block_sparse_idx = attention_inputs.block_sparse_idx
        block_sparse_count = attention_inputs.block_sparse_count
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list
        block_eff_rows = attention_inputs.block_eff_rows

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        softmax_scale = attention_inputs.aux_attrs.scale
        inner_prec = attention_inputs.aux_attrs.inner_prec
        group_size = int(num_heads / kv_heads)

        q_s_aligned = max(q_seqlen_list)
        kv_s_aligned = max(kv_seqlen_list)
        q_s_block_num_aligned = (q_s_aligned + block_sparse_shape[0] - 1) // block_sparse_shape[0]
        kv_s_block_num_aligned = (kv_s_aligned + block_sparse_shape[1] - 1) // block_sparse_shape[1]
        for b in range(batch):
            q_s_block_num_aval = (q_seqlen_list[b] + block_sparse_shape[0] - 1) // block_sparse_shape[0]
            kv_s_block_num_aval = (kv_seqlen_list[b] + block_sparse_shape[1] - 1) // block_sparse_shape[1]
            for n1 in range(num_heads):
                n2 = int(n1 // group_size)
                for q_s_blk_idx in range(q_s_block_num_aval):
                    # 获取当前q的稀疏分块
                    cur_q_s_blk_size = (q_seqlen_list[b] - q_s_blk_idx * block_sparse_shape[0]) if q_s_blk_idx == (q_s_block_num_aval - 1) else block_sparse_shape[0]
                    if block_eff_rows is not None:
                        effective_x = int(block_eff_rows[b, n1, q_s_blk_idx, 0])
                        cur_q_s_blk_size = min(cur_q_s_blk_size, effective_x)
                        if cur_q_s_blk_size <= 0:
                            continue
                    q_blk = None
                    if q_layout == "TND":
                        t_offset = sum(q_seqlen_list[:b]) + q_s_blk_idx * block_sparse_shape[0]
                        q_blk = query[t_offset:t_offset + cur_q_s_blk_size, n1, :] # numpy默认将整数索引降维
                        # print(q_blk.shape)
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_sparse_shape[0]
                        q_blk = query[b, n1, s1_offset:s1_offset + cur_q_s_blk_size,:] # numpy默认将整数索引降维
                    elif q_layout == "BSND":
                        s1_offset = q_s_blk_idx * block_sparse_shape[0]
                        q_blk = query[b, s1_offset:s1_offset + cur_q_s_blk_size, n1, :] # numpy默认将整数索引降维
                    # 获取这个q分块对应的kv分块的稀疏pattern
                    cur_kv_s_rsvd_blk_count = block_sparse_count[b, n1, q_s_blk_idx]
                    cur_kv_s_sparse_idx = block_sparse_idx[b, n1, q_s_blk_idx, :cur_kv_s_rsvd_blk_count].reshape(cur_kv_s_rsvd_blk_count).tolist()
                    # 获取将所有保留的kv块拼接起来的结果
                    key_gathered = None
                    value_gathered = None
                    count = 0
                    for kv_s_blk_idx in cur_kv_s_sparse_idx:
                        # cur_kv_s_blk_size = (kv_seqlen_list[b] - kv_s_blk_idx * block_sparse_shape[1]) if kv_s_blk_idx == (kv_s_block_num_aval - 1) else block_sparse_shape[1]
                        cur_kv_s_blk_size = (kv_seqlen_list[b] - kv_s_blk_idx * block_sparse_shape[1]) if kv_s_blk_idx == (kv_s_block_num_aval - 1) else block_sparse_shape[1]
                        if block_eff_rows is not None:
                            effective_y = int(block_eff_rows[b, n1, kv_s_blk_idx, 1])
                            cur_kv_s_blk_size = min(cur_kv_s_blk_size, effective_y)
                        if cur_kv_s_blk_size <= 0:
                            continue
                        cur_k_blk = None
                        cur_v_blk = None
                        if kv_layout == "TND":
                            t_offset = sum(kv_seqlen_list[:b]) + kv_s_blk_idx * block_sparse_shape[1]                        
                            cur_k_blk = key[t_offset:t_offset + cur_kv_s_blk_size, n2,:]
                            cur_v_blk = value[t_offset:t_offset + cur_kv_s_blk_size, n2,:]
                            cur_k_blk = np.transpose(cur_k_blk, (1, 0))
                        elif kv_layout == "BNSD":
                            s2_offset = kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[b, n2, s2_offset:s2_offset + cur_kv_s_blk_size,:]
                            cur_v_blk = value[b, n2, s2_offset:s2_offset + cur_kv_s_blk_size,:]
                            cur_k_blk = np.transpose(cur_k_blk, (1, 0))
                        elif kv_layout == "BSND":
                            s2_offset = kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[b, s2_offset:s2_offset + cur_kv_s_blk_size, n2, :]
                            cur_v_blk = value[b, s2_offset:s2_offset + cur_kv_s_blk_size, n2, :]
                            cur_k_blk = np.transpose(cur_k_blk, (1, 0))

                        if count == 0:
                            key_gathered = cur_k_blk
                            value_gathered = cur_v_blk
                        else:
                            key_gathered = np.concatenate([key_gathered, cur_k_blk], axis=1)
                            value_gathered = np.concatenate([value_gathered, cur_v_blk], axis=0)
                        count += 1
                    if key_gathered is None:
                        continue
                    attn_out_bm_slice, lse_bm_slice = self.ref_flash_block_sparse_attention(q_blk, key_gathered, value_gathered, softmax_scale, inner_prec)
                    attn_out_gt_slice, lse_gt_slice = self.ref_attention(q_blk, key_gathered, value_gathered, softmax_scale, inner_prec)

                    if q_layout == "TND":
                        t_offset = sum(q_seqlen_list[:b]) + q_s_blk_idx * block_sparse_shape[0]                        
                        attn_out_bm[t_offset:t_offset + cur_q_s_blk_size, n1, :] = attn_out_bm_slice
                        attn_out_gt[t_offset:t_offset + cur_q_s_blk_size, n1, :] = attn_out_gt_slice
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_sparse_shape[0]
                        attn_out_bm[b, n1, s1_offset:s1_offset + cur_q_s_blk_size, :] = attn_out_bm_slice
                        attn_out_gt[b, n1, s1_offset:s1_offset + cur_q_s_blk_size, :] = attn_out_gt_slice
                    elif q_layout == "BSND":
                        s1_offset = q_s_blk_idx * block_sparse_shape[0]
                        attn_out_bm[b, s1_offset:s1_offset + cur_q_s_blk_size, n1, :] = attn_out_bm_slice
                        attn_out_gt[b, s1_offset:s1_offset + cur_q_s_blk_size, n1, :] = attn_out_gt_slice

        return attn_out_bm, attn_out_gt

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
        # 打印输入
        print("AclnnBaseApi")
        # inspect_kwargs(input_data.kwargs)
        
        np.random.seed(10)
        torch.npu.synchronize()
        input_args = []  # 算子的入参列表
        output_packages = []  # 算子的出参数据包列表

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

        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)  # 把这个空指针类型转换为tensor指针类型

        # 将tensor重新to到npu上，防止流同步失败
        query = self.acl_tensor_to_torch(input_args[0]).to("npu")
        input_args[0] = self.torch_tensor_to_acl(query)
        key = self.acl_tensor_to_torch(input_args[1]).to("npu")
        input_args[1] = self.torch_tensor_to_acl(key)
        value = self.acl_tensor_to_torch(input_args[2]).to("npu")
        input_args[2] = self.torch_tensor_to_acl(value)
        blockSparseMask = self.acl_tensor_to_torch(input_args[3]).to("npu")
        input_args[3] = self.torch_tensor_to_acl(blockSparseMask)

        self.input_layout = input_data.kwargs['qInputLayout']

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
        
        input_args[8] = null_tensor_ptr # blockTable
        input_args[20] = null_tensor_ptr
        if input_data.kwargs['softmaxLseFlag'] == 1:
            input_args.pop()
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            input_args.pop()
            output_packages.append(input_args[-2])

        if use_eff_rows:
            for i in range(len(output_packages)):
                out_t = self.acl_tensor_to_torch(output_packages[i])
                if out_t is not None:
                    out_t.zero_()
        
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

        if output[0].dim() == 4:
            outputTemp = torch.zeros((output[0].shape[0], output[0].shape[1], output[0].shape[2], output[0].shape[3]), dtype=output[0].dtype)
            outputTensor = output[0]
            if self.qSeqlenList is not None:
                for i in range(output[0].shape[0]):
                    if self.input_layout == "BNSD":
                        outputTemp[i, :, :self.qSeqlenList[i], :] = outputTensor[i, :, :self.qSeqlenList[i], :]
                    elif self.input_layout == "BSND":
                        outputTemp[i, :self.qSeqlenList[i], :, :] = outputTensor[i, :self.qSeqlenList[i], :, :]
                print("WANGCHAO DEBUG aclnn [outputTemp]", [outputTemp])
                return [outputTemp]
        else:
            print("WANGCHAO DEBUG aclnn output[0]", output[0])
            return output[0]

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockSparseMask, const aclTensor *attenMaskOptional, const aclIntArray *blockShape, const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, int64_t nextTokens, int64_t softmaxLseFlag, const aclTensor *attentionOut, const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)"


@register("aclnn_blocksparseattention")
class BlockSparseAttentionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query_golden = tensor_to_numpy_array(self.query.cpu())
        key_golden = tensor_to_numpy_array(self.key.cpu())
        value_golden = tensor_to_numpy_array(self.value.cpu())

        # testObj = TestBlockSparseAttentionGenBmGt()
        # aux_attrs = testObj.AuxAttrs(
        #     self.num_heads, self.kv_heads, self.headDim, self.query_dtype,
        #     self.inner_precise, self.scale_value, self.q_input_layout, self.kv_input_layout)
        # attention_inputs = testObj.AttentionInputs(
        #     query_golden, key_golden, value_golden,
        #     self.block_sparse_idx, self.block_sparse_count, self.block_shape,
            # self.q_seqlen_list, self.kv_seqlen_list, aux_attrs)
        # 改后
        use_eff_rows = (self.num_heads == self.kv_heads)
        atten_mask = input_data.kwargs.get("attenMask", None)
        block_eff_rows_np = None
        if use_eff_rows and atten_mask is not None:
            if isinstance(atten_mask, torch.Tensor):
                block_eff_rows_np = atten_mask.cpu().numpy()
            elif isinstance(atten_mask, np.ndarray):
                block_eff_rows_np = atten_mask
        testObj = TestBlockSparseAttentionGenBmGt()
        aux_attrs = testObj.AuxAttrs(
            self.num_heads, self.kv_heads, self.headDim, self.query_dtype,
            self.inner_precise, self.scale_value, self.q_input_layout, self.kv_input_layout)        
        attention_inputs = testObj.AttentionInputs(
            query_golden, key_golden, value_golden,
            self.block_sparse_idx, self.block_sparse_count, self.block_shape,
            self.q_seqlen_list, self.kv_seqlen_list, aux_attrs,
            block_eff_rows=block_eff_rows_np)
        attn_out_bm_np, attn_out_gt_np = testObj.compute_output(attention_inputs)
        attn_out_bm = numpy_array_to_tensor(attn_out_bm_np)
        attn_out_gt = numpy_array_to_tensor(attn_out_gt_np)
        # attn_out_bm = torch.empty_like(numpy_array_to_tensor(attention_inputs.query))
        # attn_out_gt = torch.empty_like(numpy_array_to_tensor(attention_inputs.query)).to(torch.float32)
        if not self.task_result.is_benchmark_task:
            # 标杆
            print("WANGCHAO DEBUG attn_out_bm", attn_out_bm)
            return attn_out_bm.to(torch.float32)
        else:
            # 真值
            print("WANGCHAO DEBUG attn_out_gt", attn_out_gt)
            return attn_out_gt.to(torch.float32)

    def init_by_input_data(self, input_data: InputDataset):
        # 打印输入
        print("BaseApi")
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
        # 根据可控的稀疏率重新生成sparse pattern，此处稀疏率选取为(0, 1)
        random.seed(10)
        sparsity_ratio = random.uniform(0.0, 1.0)
        # sparsity_ratio = 1
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
        # GQA/MQA not supported with blockEffRows, leave framework-generated attenMask as-is

    