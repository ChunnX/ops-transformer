import os
import numpy as np
import torch
import time
import math
import random
import torch.nn as nn
import torch_npu
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
import torchair as tng
# import libs.tools as tools
# import libs.training.run_pytorchGraph as training
# from libs.tools import set_timeout
import ctypes
import torchair
#from en_dtypes import hifloat8

# def trans_np_array_float32_to_hifuint8(in_tensor: np.ndarray)-> np.ndarray:
#     return in_tensor.astype(hifloat8).view(np.uint8)

# def trans_np_array_hifuint8_to_float32(in_tensor: np.ndarray)-> np.ndarray:
#     return in_tensor.view(hifloat8).astype(np.float32)

def get_block_quant_scale(tensor, data_type, is_scale_ds=False, is_vcore_split=False, NEW_PER_BLOCK_SIZE=128):
    PER_VCORE_BLOCK_SIZE = 64
    GTYPE = torch.float32
    if data_type == torch.float8_e5m2:
        FP8_MAX = 57344.0
    elif data_type == torch.float8_e4m3fn:
        FP8_MAX = 448.0
    elif data_type == torch.uint8:
        FP8_MAX = 32768 #65504
    else:
        FP8_MAX = -1
        print(f"{data_type} Not support block quant")
        return None

    dim1 = tensor.shape[0]  # b
    dim2 = tensor.shape[1]  # n2
    dim4 = tensor.shape[2]  # s1
    dim5 = tensor.shape[3]  # d or s2
    if not is_scale_ds:
        scale_for_tensor = torch.ones([dim1, dim2, math.ceil(dim4 / NEW_PER_BLOCK_SIZE), 1]).to(GTYPE)
        for b in range(dim1):
            for n in range(dim2):
                for s in range(0, dim4, NEW_PER_BLOCK_SIZE):
                    # 取出128*D个数
                    s_start = s // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE
                    s_end = min(s // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE + NEW_PER_BLOCK_SIZE, dim4)
                    chunk = tensor[b, n, s_start:s_end, :]
                    max_val = torch.max(torch.abs(chunk))
                    epsilon = 1e-8
                    scale_for_tensor[b, n, s // NEW_PER_BLOCK_SIZE, 0] = FP8_MAX / (max_val + epsilon)
    else:
        if not is_vcore_split:
            if dim4 % PER_VCORE_BLOCK_SIZE == 0:
                scale_for_tensor = torch.ones(
                    [dim1, dim2, math.ceil(dim4 / PER_VCORE_BLOCK_SIZE), math.ceil(dim5 / NEW_PER_BLOCK_SIZE)]).to(GTYPE)
            else:
                scale_for_tensor = torch.ones(
                    [dim1, dim2, math.ceil(dim4 / PER_VCORE_BLOCK_SIZE) + 1, math.ceil(dim5 / NEW_PER_BLOCK_SIZE)]).to(
                    GTYPE)
            s1_tail = NEW_PER_BLOCK_SIZE if dim4 % NEW_PER_BLOCK_SIZE == 0 else dim4 % NEW_PER_BLOCK_SIZE
            for b in range(dim1):
                for n in range(dim2):
                    for s2 in range(0, dim5, NEW_PER_BLOCK_SIZE):
                        s2_start = s2 // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE
                        s2_end = min(s2 // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE + NEW_PER_BLOCK_SIZE, dim5)
                        for s1 in range(0, dim4 - s1_tail, PER_VCORE_BLOCK_SIZE):
                            s1_start = s1 // PER_VCORE_BLOCK_SIZE * PER_VCORE_BLOCK_SIZE
                            s1_end = min(s1_start + PER_VCORE_BLOCK_SIZE, dim4)
                            chunk = tensor[b, n, s1_start:s1_end, s2_start:s2_end]
                            max_val = torch.max(torch.abs(chunk))
                            scale_for_tensor[b, n, s1 // PER_VCORE_BLOCK_SIZE, s2 // NEW_PER_BLOCK_SIZE] = FP8_MAX / max_val

                        # 处理尾块 v0
                        s1_start = dim4 - s1_tail
                        s1_end = s1_start + math.ceil(s1_tail / 2)
                        chunk = tensor[b, n, s1_start:s1_end, s2_start:s2_end]
                        max_val = torch.max(torch.abs(chunk))
                        scale_for_tensor[
                            b, n, s1_start // PER_VCORE_BLOCK_SIZE, s2 // NEW_PER_BLOCK_SIZE] = FP8_MAX / max_val
                        v0_s1_start_idx = s1_start // PER_VCORE_BLOCK_SIZE
                        # 处理尾块 v1
                        s1_start = s1_end
                        s1_end = dim4
                        chunk = tensor[b, n, s1_start:s1_end, s2_start:s2_end]
                        max_val = torch.max(torch.abs(chunk))
                        scale_for_tensor[b, n, v0_s1_start_idx + 1, s2 // NEW_PER_BLOCK_SIZE] = FP8_MAX / max_val
        else:
            # S1方向区分V0与V1，shape=S1*D
            if dim4 % PER_VCORE_BLOCK_SIZE == 0:
                scale_for_tensor = torch.ones([dim1, dim2, math.ceil(dim4 / PER_VCORE_BLOCK_SIZE), 1]).to(GTYPE)
            else:
                scale_for_tensor = torch.ones([dim1, dim2, math.ceil(dim4 / PER_VCORE_BLOCK_SIZE) + 1, 1]).to(GTYPE)
            for b in range(dim1):
                for n in range(dim2):
                    for s in range(0, dim4, PER_VCORE_BLOCK_SIZE):
                        # 取出64*D个数
                        s_start = s // PER_VCORE_BLOCK_SIZE * PER_VCORE_BLOCK_SIZE
                        s_end = min(s_start + PER_VCORE_BLOCK_SIZE, dim4)
                        if s_end - start == PER_VCORE_BLOCK_SIZE:
                            chunk = tensor[b, n, s_start:s_end, :]
                            max_val = torch.max(torch.abs(chunk))
                            scale_for_tensor[b, n, s // PER_VCORE_BLOCK_SIZE, 0] = FP8_MAX / max_val
                        else:
                            s_end = s_start + math.ceil(s_end - s_start) / 2
                            chunk = tensor[b, n, s_start:s_end, :]
                            max_val = torch.max(torch.abs(chunk))
                            scale_for_tensor[b, n, s // PER_VCORE_BLOCK_SIZE, 0] = FP8_MAX / max_val
                            s_start = s_end
                            s_end = dim4
                            chunk = tensor[b, n, s_start:s_end, :]
                            max_val = torch.max(torch.abs(chunk))
                            scale_for_tensor[b, n, s // PER_VCORE_BLOCK_SIZE + 1, 0] = FP8_MAX / max_val
    return scale_for_tensor

def block_quant_or_dequant_single(tensor, scale, is_vcore_split=False, NEW_PER_BLOCK_SIZE=128):
    PER_VCORE_BLOCK_SIZE = 64
    GTYPE = torch.float32
    dim1 = tensor.shape[0]  # b
    dim2 = tensor.shape[1]  # n2
    dim3 = tensor.shape[2]  # s1
    dim4 = tensor.shape[3]  # d or s2
    quanted_tensor = torch.zeros([dim1, dim2, dim3, dim4]).to(GTYPE)
    if not is_vcore_split:
        for b in range(dim1):
            for n in range(dim2):
                for s in range(0, dim3, NEW_PER_BLOCK_SIZE):
                    # 取出128*D个数
                    s_start = s // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE
                    s_end = min(s // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE + NEW_PER_BLOCK_SIZE, dim3)

                    quanted_tensor[b, n, s_start: s_end, :] = tensor[b, n, s_start: s_end, :] * scale[
                        b, n, s // NEW_PER_BLOCK_SIZE, 0]
    else:
        s1_tail = NEW_PER_BLOCK_SIZE if dim3 % NEW_PER_BLOCK_SIZE == 0 else dim3 % NEW_PER_BLOCK_SIZE
        for b in range(dim1):
            for n in range(dim2):
                for s2 in range(0, dim4, NEW_PER_BLOCK_SIZE):
                    s2_start = s2 // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE
                    s2_end = min(s2_start + NEW_PER_BLOCK_SIZE, dim4)
                    for s1 in range(0, dim3 - s1_tail, PER_VCORE_BLOCK_SIZE):
                        # 取出64*128个数
                        s1_start = s1 // PER_VCORE_BLOCK_SIZE * PER_VCORE_BLOCK_SIZE
                        s1_end = min(s1_start + PER_VCORE_BLOCK_SIZE, dim3)
                        quanted_tensor[b, n, s1_start: s1_end, s2_start: s2_end] = tensor[b, n, s1_start: s1_end,
                                                                                   s2_start: s2_end] * scale[
                                                                                       b, n, s1 // PER_VCORE_BLOCK_SIZE, s2 // NEW_PER_BLOCK_SIZE]
                    # 处理尾块 v0
                    s1_start = dim3 - s1_tail
                    s1_end = s1_start + math.ceil(s1_tail / 2)
                    quanted_tensor[b, n, s1_start: s1_end, s2_start: s2_end] = tensor[b, n, s1_start: s1_end,
                                                                               s2_start: s2_end] * scale[
                                                                                   b, n, s1_start // PER_VCORE_BLOCK_SIZE, s2 // NEW_PER_BLOCK_SIZE]
                    v0_s1_start_idx = s1_start // PER_VCORE_BLOCK_SIZE
                    # 处理尾块 v1
                    s1_start = s1_end
                    s1_end = dim3
                    quanted_tensor[b, n, s1_start: s1_end, s2_start: s2_end] = tensor[b, n, s1_start: s1_end,
                                                                               s2_start: s2_end] * scale[
                                                                                   b, n, v0_s1_start_idx + 1, s2 // NEW_PER_BLOCK_SIZE]

    return quanted_tensor

def npu_fused_infer_attention_score_v2_functional(q1, k1, v1, qrope, krope, d_scale_q, d_scale_k, d_scale_v,
    N1, N2, scale, input_layout, query_dtype, key_dtype, value_dtype, out_dtype):
    out_all = torch.ops.npu.npu_fused_infer_attention_score_v2(q1,
        k1,
        v1,
        query_rope=qrope,
        key_rope=krope,
        dequant_scale_query=d_scale_q,
        dequant_scale_key=d_scale_k,
        dequant_scale_value=d_scale_v,
        num_query_heads=N1,
        num_key_value_heads=N2,
        softmax_scale=scale,
        input_layout=input_layout,
        query_quant_mode=3,
        key_quant_mode=0,
        value_quant_mode=0,
        query_dtype=query_dtype,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        dequant_scale_key_dtype=torch_npu.float32,
        dequant_scale_value_dtype=torch_npu.float32,
        out_dtype=out_dtype
        )
    return out_all

def FIA_HiF8():
    torch.manual_seed(1)
    device = torch.device('npu:0')
    layout = "BNSD"
    fp8_dtype = torch.uint8
    fp8_scale_dtype = torch.float32
    out_dtype=torch.bfloat16

    q_bnsd_fp16 = torch.from_numpy(np.random.uniform(low=-1, high=1, size=[1,8,8,512])).to(torch.float16).npu()
    k_bnsd_fp16 = torch.from_numpy(np.random.uniform(low=-1, high=1, size=[1,1,1024,512])).to(torch.float16).npu()
    v_bnsd_fp16 = torch.from_numpy(np.random.uniform(low=-1, high=1, size=[1,1,1024,512])).to(torch.float16).npu()
    q_rope = torch.from_numpy(np.random.uniform(low=-1, high=1, size=[1,8,8,64])).to(torch.bfloat16).npu()
    k_rope = torch.from_numpy(np.random.uniform(low=-1, high=1, size=[1,1,1024,64])).to(torch.bfloat16).npu()
    N1, N2, scale = q_bnsd_fp16.shape[1], k_bnsd_fp16.shape[1], 1/math.sqrt(q_bnsd_fp16.shape[-1])

    print("q_bnsd_fp16",q_bnsd_fp16.shape, "k_bnsd_fp16",k_bnsd_fp16.shape, "v_bnsd_fp16",v_bnsd_fp16.shape)
    q_bnsd_hif8, scale_q_bnsd = torch_npu.npu_dynamic_quant(q_bnsd_fp16, dst_type=torch_npu.hifloat8)
    k_bnsd_hif8, scale_k_bnsd = torch_npu.npu_dynamic_quant(k_bnsd_fp16, quant_mode="pertensor", dst_type=torch_npu.hifloat8)
    v_bnsd_hif8, scale_v_bnsd = torch_npu.npu_dynamic_quant(v_bnsd_fp16, quant_mode="pertensor", dst_type=torch_npu.hifloat8)
    print("q_bnsd_hif8",q_bnsd_hif8.shape, "k_bnsd_hif8",k_bnsd_hif8.shape, "v_bnsd_hif8",v_bnsd_hif8.shape)
    print("scale_q_bnsd",scale_q_bnsd.shape, "scale_k_bnsd",scale_k_bnsd.shape, "scale_v_bnsd",scale_v_bnsd.shape)

    # # 获取量化scale
    # scale_q_bnsd = get_block_quant_scale(q_bnsd_fp16, fp8_dtype, False, False, 128).to(fp8_scale_dtype)
    # scale_k_bnsd = get_block_quant_scale(k_bnsd_fp16, fp8_dtype, False, False, 256).to(fp8_scale_dtype)
    # scale_v_bnsd = get_block_quant_scale(v_bnsd_fp16, fp8_dtype, False, False, 256).to(fp8_scale_dtype)
    # 反量化因子
    dequantScale_query = 1 / scale_q_bnsd
    k_antiquant_scale = 1 / scale_k_bnsd
    v_antiquant_scale = 1 / scale_v_bnsd
    # # fp16 --> hif8
    # q_bnsd = block_quant_or_dequant_single(q_bnsd_fp16, scale_q_bnsd, False, 128).to(torch.float32).numpy()
    # k_bnsd = block_quant_or_dequant_single(k_bnsd_fp16, scale_k_bnsd, False, 256).to(torch.float32).numpy()
    # v_bnsd = block_quant_or_dequant_single(v_bnsd_fp16, scale_v_bnsd, False, 256).to(torch.float32).numpy()
    # q_np = trans_np_array_float32_to_hifuint8(q_bnsd)
    # k_np = trans_np_array_float32_to_hifuint8(k_bnsd)
    # v_np = trans_np_array_float32_to_hifuint8(v_bnsd)
    # q_bnsd_fp8 = torch.from_numpy(q_np)
    # k_bnsd_fp8 = torch.from_numpy(k_np)
    # v_bnsd_fp8 = torch.from_numpy(v_np)
    out_all = npu_fused_infer_attention_score_v2_functional(
        q_bnsd_hif8, k_bnsd_hif8, v_bnsd_hif8, q_rope, k_rope,
        dequantScale_query, k_antiquant_scale, v_antiquant_scale,
        N1, N2, scale, layout,
        torch_npu.hifloat8, torch_npu.hifloat8, torch_npu.hifloat8, out_dtype
        )
    print(out_all[0])

if __name__ == "__main__":
    FIA_HiF8()

