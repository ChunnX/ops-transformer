# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch.distributed as dist
import torch_npu

try:
    import torch_npu
except ImportError:
    pass

import ctypes
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


@register("function_aclnn_quant_matmul_allto_all")
class QuantMatmulAlltoAll(BaseApi):
    """
    Golden计算: NPU 小算子拼接
    """

    def __init__(self, task_result):
        super(QuantMatmulAlltoAll, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info

    def manual_all_to_all(self, x_splits_from_ranks, x_splits_to_ranks):
        # (world_size, m, n/world_size)
        for target_rank in range(self.dist_task_info.world_size):
            if target_rank == self.dist_task_info.rank:
                # 发给自己
                x_splits_from_ranks[self.dist_task_info.rank].copy_(
                    x_splits_to_ranks[target_rank]
                )
                # 从别的进程获取我要的张量
                for src_rank in range(self.dist_task_info.world_size):
                    if src_rank != self.dist_task_info.rank:
                        dist.recv(
                            tensor=x_splits_from_ranks[
                                src_rank
                            ],  # 接收数据的空张量（对应第src_rank个rank）
                            src=src_rank,  # 源rank
                            tag=src_rank * 1000
                            + self.dist_task_info.rank,  # 匹配发送方的标签（源rank*1000+当前rank）
                        )
            else:
                # 发给其他rank
                dist.send(
                    tensor=x_splits_to_ranks[
                        target_rank
                    ],  # 要发送的张量（给第target_rank个rank）
                    dst=target_rank,  # 目标rank
                    tag=self.dist_task_info.rank * 1000
                    + target_rank,  # 通信标签（避免数据混淆）
                )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        x = input_data.kwargs["x1"]  # (M, K)
        weight = input_data.kwargs["x2"]  # (K, N)
        bias = input_data.kwargs["biasOptional"]  # (N, )
        pertoken_scale = input_data.kwargs["x1Scale"]  # (M, )
        scale = input_data.kwargs["x2Scale"]  # (N, )
        transB = input_data.kwargs["transposeX2"]
        is_dequant = x.dtype == torch.int8
        output_type = x.dtype

        if is_dequant:
            output_type = (
                torch.float16 if (bias.dtype == torch.float16) else torch.bfloat16
            )

        if self.name == "cpu":
            """
            ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ CPU  真值 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            """
            x = x.cpu().to(torch.float32)
            weight = weight.cpu().to(torch.float32)
            bias = bias.cpu().to(torch.float32)
            if transB:
                weight = weight.T.contiguous()
            if is_dequant:
                output = torch.matmul(x, weight)
                scale_todim = torch.unsqueeze(scale.cpu(), 0)
                bias_todim = torch.unsqueeze(bias, 0)
                perTokenScale_todim = torch.unsqueeze(pertoken_scale.cpu(), 1)
                output = (output * scale_todim * perTokenScale_todim + bias_todim).to(
                    output_type
                )
            else:
                output = (torch.matmul(x, weight) + bias).to(output_type)

            # 1. split to world_size times chunk, with each (m, n/world_size)
            send_chunks = list(
                torch.chunk(output, world_size, dim=1)
            )  # (world_size, m, n/world_size)
            send_chunks = [c.contiguous() for c in send_chunks]
            # 2. recv_chunks shape: (world_size, m, n/world_size)
            recv_chunks = [
                torch.empty_like(
                    send_chunks[0], device=output.device, dtype=output.dtype
                )
                for _ in range(world_size)
            ]
            # 3. alltoall
            self.manual_all_to_all(recv_chunks, send_chunks)
            # 4. output shape: (m*world_size, n/world_size)
            y = torch.cat(recv_chunks, dim=0)

            return y

        if self.name == "npu_bm":
            """
            ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ NPU 小算子级联标杆 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            """
            if transB:
                weight = weight.T.contiguous()
            if is_dequant:  # 量化部分
                output = torch_npu.npu_quant_matmul(
                    x1=x,
                    x2=weight,
                    scale=scale,
                    pertoken_scale=pertoken_scale,
                    bias=bias,
                    output_dtype=output_type,
                )
            else:
                output = (torch.matmul(x, weight) + bias).to(output_type)

            # 1. split to world_size times chunk, with each (m, n/world_size)
            send_chunks = list(torch.chunk(output, world_size, dim=1))
            send_chunks = [c.contiguous() for c in send_chunks]
            # 2. output shape: (m*world_size, n/world_size)
            recv_chunks = [
                torch.empty_like(send_chunks[0], device=x.device, dtype=output_type)
                for _ in range(world_size)
            ]
            # 3. alltoall
            dist.all_to_all(recv_chunks, send_chunks)
            # 4. output
            y = torch.cat(recv_chunks, dim=0)

            return y

    @staticmethod
    def get_hcomm_info(rank_id):
        from torch.distributed.distributed_c10d import _get_default_group

        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            backend_obj = getattr(default_pg, "_get_backend", None)
            hcomm_info = backend_obj(torch.device("npu")).get_hccl_comm_name(rank_id)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank_id)
        return hcomm_info

    def init_by_input_data(self, input_data: InputDataset):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        device = "npu:" + str(rank_id)

        input_data.kwargs["x1"] = input_data.kwargs["x1"].to(device)
        input_data.kwargs["x2"] = input_data.kwargs["x2"].to(device)
        input_data.kwargs["biasOptional"] = input_data.kwargs["biasOptional"].to(device)

        # 量化处理
        is_dequant = input_data.kwargs["x1"].dtype == torch.int8

        if is_dequant:
            input_data.kwargs["x2Scale"] = input_data.kwargs["x2Scale"].to(device)
            input_data.kwargs["x1Scale"] = input_data.kwargs["x1Scale"].to(device)

        # 通信域配置
        if self.device == "pyaclnn" and dist.is_available():
            from torch.distributed.distributed_c10d import _get_default_group

            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcomm_info = default_pg._get_backend(
                    torch.device("npu")
                ).get_hccl_comm_name(rank_id)
            else:
                hcomm_info = default_pg.get_hccl_comm_name(rank_id)
            input_data.kwargs["group"] = default_pg
            input_data.kwargs["world_size"] = world_size


"""
    =================================================================================
    ==============================ACLNN输入特殊处理===================================
    =================================================================================
"""


@register("pyaclnn_aclnn_quant_matmul_allto_all")
class AclnnQuantMatmulAlltoAll(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super(AclnnQuantMatmulAlltoAll, self).__init__(task_result, backend)
        self.dist_task_info = task_result.dist_task_info

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def init_by_input_data(self, input_data: InputDataset):
        # 处理group
        rank_id = self.dist_task_info.rank
        self.group = input_data.kwargs["group"]
        input_data.kwargs["group"] = (
            input_data.kwargs["group"]
            ._get_backend(torch.device("npu"))
            .get_hccl_comm_name(rank_id)
        )

        # 确认是否量化
        is_dequant = input_data.kwargs["x1"].dtype == torch.int8

        # 抛弃 world_size
        input_data.kwargs.pop("world_size")
        # 转化为 AscendC
        input_args, output_packages = super().init_by_input_data(input_data)

        # offset, antiquant_scale, antiquant_offset, bias
        input_args[5] = self.get_null_tensor_pte()
        input_args[6] = self.get_null_tensor_pte()
        input_args[7] = self.get_null_tensor_pte()

        # scale, pertoken_scale
        if not is_dequant:
            input_args[3] = self.get_null_tensor_pte()
            input_args[4] = self.get_null_tensor_pte()

        return input_args, output_packages

    def get_null_tensor_pte(self):
        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型
        return null_tensor_ptr

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()


# ============================ HiF4 全量化数据生成与 golden ============================
# HiF4 格式: 64 个 S1P2(4-bit) 元素 + 32bits 三级共享元数据(1 个 hif4_scale, 4 字节)。
#   Level-1 E6M2 (Byte0): 全局 scale, exponent(6bit, bias=48) + mantissa(2bit)
#   Level-2 E1_8 (Byte1): 8 个 1-bit 微指数, 每 bit 覆盖 8 个元素, 0->2^0, 1->2^1
#   Level-3 E1_16 (Byte2~3): 16 个 1-bit 微指数, 每 bit 覆盖 4 个元素(Byte2 低 8 位在前, Byte3 高 8 位在后)
# 完整数值: V[i] = E6M2 * 2^(E1_8[i//8] + E1_16[i//4]) * S1P2[i]
#   S1P2: |Sign(1)|Integer(1)|Fraction(2)|, 幅值 = int + frac/4 (0.25 ~ 1.75), 2:1 打包存储


def gen_hif4_scale_meta(m, k_groups, seed=None):
    """生成 hif4_scale 三级元数据: Byte0=E6M2, Byte1=E1_8, Byte2=E1_16_lo, Byte3=E1_16_hi"""
    import numpy as np

    rng = np.random.default_rng(seed)
    meta = np.zeros((m, k_groups, 4), dtype=np.uint8)
    meta[:, :, 0] = rng.integers(0, 0x80, (m, k_groups))   # E6M2
    meta[:, :, 1] = rng.integers(0, 0x100, (m, k_groups))  # E1_8
    meta[:, :, 2] = rng.integers(0, 0x100, (m, k_groups))  # E1_16 低 8 位
    meta[:, :, 3] = rng.integers(0, 0x100, (m, k_groups))  # E1_16 高 8 位
    return meta


def hif4_dequant(data_packed, meta):
    """HiF4 反量化: V = E6M2 * 2^(E1_8[i//8] + E1_16[i//4]) * S1P2[i], 输出 FP32 [M, K]"""
    import numpy as np

    m, k_half = data_packed.shape
    k = k_half * 2
    data = np.asarray(data_packed, dtype=np.uint8)
    # 2:1 解包: 低 nibble 为偶数位元素, 高 nibble 为奇数位元素
    lo = (data & 0x0F).astype(np.int32)
    hi = ((data >> 4) & 0x0F).astype(np.int32)
    s1p2_code = np.empty((m, k), dtype=np.int32)
    s1p2_code[:, 0::2] = lo
    s1p2_code[:, 1::2] = hi
    # S1P2 解码: sign(1) + int(1) + frac(2)
    sign = np.where((s1p2_code >> 3) & 0x1 == 1, -1.0, 1.0)
    int_part = (s1p2_code >> 2) & 0x1
    frac_part = s1p2_code & 0x3
    s1p2_val = sign * (int_part.astype(np.float32) + frac_part.astype(np.float32) * 0.25)
    # E6M2 解码: exponent(6bit, bias=48) + mantissa(2bit)
    e6m2_code = meta[:, :, 0].astype(np.int32)
    e6m2_exp = (e6m2_code >> 2) & 0x3F
    e6m2_man = e6m2_code & 0x3
    e6m2_val = np.power(2.0, (e6m2_exp - 48).astype(np.float32)) * (1.0 + e6m2_man.astype(np.float32) * 0.25)
    # E1_8 / E1_16 微指数: 每 8/4 元素共享 1 bit
    e1_8_bits = np.unpackbits(meta[:, :, 1].astype(np.uint8), axis=-1)  # (m, k_groups, 8)
    e1_16 = (meta[:, :, 3].astype(np.uint16) << 8) | meta[:, :, 2].astype(np.uint16)
    e1_16_bits = np.zeros((m, k_groups, 16), dtype=np.float32)
    for bit in range(16):
        e1_16_bits[:, :, bit] = ((e1_16 >> (15 - bit)) & 0x1).astype(np.float32)
    # 逐元素展开为 [M, K]: 组内第 i 个元素
    idx = np.arange(k, dtype=np.int64) % 64
    exp_scale = (e1_8_bits[:, :, idx // 8] * 1.0 + e1_16_bits[:, :, idx // 4])  # (m, k_groups, k)?? -> 按 k 展平
    # 展平 k_groups 维: 元素 i 属于第 (i // 64) 组
    group_idx = np.arange(k, dtype=np.int64) // 64
    scale = e6m2_val[:, group_idx] * np.power(2.0, exp_scale[:, group_idx, np.arange(k)])
    return (scale * s1p2_val).astype(np.float32)


def gen_hif4_case(m, k, n, rank_num, trans_x2=True, y_dtype="bfloat16", seed=None):
    """生成 HiF4 全量化用例(数据/_scale 载体/反量化 golden)。
    约束: k 为偶数(无 32/64/128 对齐约束); n % rank_num == 0; 非转置时 n 为偶数; 不支持 bias(不生成)。
    golden = alltoall_simulate(dequant(x1) @ dequant(x2)).to(y_dtype)
    """
    import numpy as np
    import torch

    assert k % 2 == 0, "k must be even (4-bit 2:1 packing)"
    assert n % rank_num == 0, "n must be divisible by rank_num"
    if not trans_x2:
        assert n % 2 == 0, "n must be even when x2 is non-transposed"
    rng = np.random.default_rng(seed)
    k_groups = (k + 63) // 64
    # x1: (m, k) 沿 K 打包 -> (m, k//2) uint8 载体
    x1_packed = rng.integers(0, 0x100, (m, k // 2), dtype=np.uint8)
    x1_scale_meta = gen_hif4_scale_meta(m, k_groups, seed)                    # (m, k_groups, 4)
    # x2: 转置 (n, k) 沿 K 打包 -> (n, k//2); 非转置 (k, n) 沿 K 逐列打包 -> (k//2, n)
    if trans_x2:
        x2_packed = rng.integers(0, 0x100, (n, k // 2), dtype=np.uint8)
        x2_scale_meta = gen_hif4_scale_meta(n, k_groups, seed)                # (n, k_groups, 4)
        x2_deq = torch.from_numpy(hif4_dequant(x2_packed, x2_scale_meta))     # (n, k)
    else:
        x2_packed = rng.integers(0, 0x100, (k // 2, n), dtype=np.uint8)
        meta_nk = gen_hif4_scale_meta(n, k_groups, seed)                      # 先按 (n, k_groups) 生成
        x2_deq = torch.from_numpy(hif4_dequant(x2_packed.T.copy(), meta_nk))  # 转置为 (n, k//2) 后解包 -> (n, k)
        x2_scale_meta = meta_nk.transpose(1, 0, 2).copy()                     # 交付布局 (k_groups, n, 4)
    x1_deq = torch.from_numpy(hif4_dequant(x1_packed, x1_scale_meta))         # (m, k)
    output = torch.matmul(x1_deq, x2_deq.T)                                   # (m, n)
    # alltoall 模拟: 按目标卡切列 -> 按源卡拼接
    send_chunks = torch.chunk(output, rank_num, dim=1)                        # rank_num x (m, n/rank_num)
    golden = torch.cat(list(send_chunks), dim=0)                              # (rank_num*m, n/rank_num)
    golden = golden.to(getattr(torch, y_dtype))
    return {
        "x1_packed": x1_packed, "x2_packed": x2_packed,
        "x1_scale_meta": x1_scale_meta, "x2_scale_meta": x2_scale_meta,
        "golden": golden,
    }
