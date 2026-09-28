# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommChannelBuilderManager


TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
    15: torch.bfloat16,
    16: torch.quint4x2,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    285: torch.uint8,  # torch_npu.int4 use torch.uint8
    290: torch.uint8,  # torch_npu.hifloat8 use torch.uint8
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
    296: torch.uint8,  # torch_npu.float4_e2m1fn_x2 use torch.uint8
    297: torch.uint8,  # torch_npu.float4_e1m2fn_x2 use torch.uint8
}


class QuantAllReduceOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantAllReduceOpBuilder, self).__init__(
            "npu_quant_all_reduce", category="mc2"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/mc2/quant_all_reduce.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "npu_quant_all_reduce(Tensor context, Tensor x, Tensor scales, int hccl_buffer_size, "
            "int world_size, *, str? reduce_op='sum', int? output_dtype=None, int? x_dtype=None, "
            "int? scales_dtype=None) -> Tensor"
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def npu_quant_all_reduce_meta(
            context,
            x,
            scales,
            hccl_buffer_size,
            world_size,
            reduce_op="sum",
            output_dtype=None,
            x_dtype=None,
            scales_dtype=None,
        ):
            torch._check(
                x is not None,
                lambda: "x cannot be None, please input some value"
                + ops_error(ErrCode.TYPE),
            )
            torch._check(
                scales is not None,
                lambda: "scales cannot be None, please input some value"
                + ops_error(ErrCode.TYPE),
            )

            world_size_list = [2, 4, 8]
            torch._check(
                world_size in world_size_list,
                lambda: "world_size must be in "
                + str(world_size_list)
                + ", but actual value is: "
                + str(world_size)
                + ops_error(ErrCode.VALUE),
            )

            size = x.size()
            dtype = x.dtype
            if output_dtype is not None:
                dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[output_dtype]
            else:
                dtype = torch.bfloat16

            return torch.empty(size, dtype=dtype, device="meta")


# Instantiate the builder
quant_all_reduce_op_builder = QuantAllReduceOpBuilder()
quant_all_reduce_op_builder._ensure_initialized()


@impl(get_as_library(), quant_all_reduce_op_builder.name, "PrivateUse1")
def npu_quant_all_reduce(
    context,
    x,
    scales,
    hccl_buffer_size,
    world_size,
    reduce_op="sum",
    output_dtype=None,
    x_dtype=None,
    scales_dtype=None,
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    # Compiles/loads the .so file
    op_module = quant_all_reduce_op_builder.load()
    return op_module.npu_quant_all_reduce(
        context,
        x,
        scales,
        hccl_buffer_size,
        world_size,
        reduce_op,
        output_dtype,
        x_dtype,
        scales_dtype,
    )


class QuantAllReduceBuffer:
    """基于 CommChannelBuilderManager 的通信上下文持有类。

    - context 为 apace CommChannelBuilder(UBMEM) 创建的通道上下文（dataCtx/barrierCtx 布局），
      dataCtx.commBufferAddrs 为各 rank 的 HCCL 内置 buffer 地址（UB_MEM 协议，可直通 GM 读写），
      HCCL engine 按 ctx tag 幂等复用，win 区直接使用 HCCL 内置 buffer（HCCL_BUFFSIZE 定容）
    - 仅支持单机（单通信域）部署，跨机拓扑的远端地址不可直通 GM 访问
    """

    def __init__(
        self,
        group,
        world_size: int,
        x: torch.Tensor,
        scales: torch.Tensor,
    ):
        self.group = group
        self.world_size = world_size
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id, init_comm=False
        )

        self._builder = CommChannelBuilderManager(self.group_name)
        if not self._builder.is_single_server_topology():
            raise RuntimeError(
                "quant_all_reduce only supports single-server deployment "
                f"(all ranks in one communication domain), group: {self.group_name}. "
                "Cross-server topology is not supported."
            )
        # 固定 ctx tag：engine context 按 tag 幂等复用，重复调用仅查表返回，无重复建链
        # UBMEM data channel：远端内置 buffer 地址可直通 GM 读写（URMA channel 地址仅可作
        # WriteNbi/ReadNbi 参数，直写会导致对端读不到状态位而软同步死锁）
        self.context = self._builder.create_ubmem_context(self.group_name + "quant_all_reduce")
        self.hccl_buffer_size = self._builder.get_hccl_buffer_size()
        _check_hccl_buffer(x, scales, self.hccl_buffer_size)


def _align_up(value: int, base: int) -> int:
    return (value + base - 1) // base * base


def _check_hccl_buffer(x: torch.Tensor, scales: torch.Tensor, hccl_buffer_size: int) -> None:
    """校验 HCCL 内置 buffer 容量，公式与 host 侧 tiling CheckWindowSize 一致：

    每个 rank 的 win 区需求 = align32(x) + align32(scales) + 1MB 状态区，
    in/out 双分区（hcclBufferSize/2 切分）共需 2 份。
    """
    win_addr_align = 32
    mb_size = 1024 * 1024
    x_data_size = _align_up(x.numel() * x.element_size(), win_addr_align)
    scales_data_size = _align_up(scales.numel() * scales.element_size(), win_addr_align)
    required = 2 * (x_data_size + scales_data_size + mb_size)
    if required > hccl_buffer_size:
        raise ValueError(
            f"Communication window size ({required} bytes = 2 * (x {x_data_size} + "
            f"scales {scales_data_size} + state region {mb_size})) exceeds hccl_buffer_size "
            f"({hccl_buffer_size} bytes = {hccl_buffer_size / 1024 / 1024:.2f} MB). "
            "Please increase HCCL_BUFFSIZE environment variable."
        )


_buffer_cache: dict[str, QuantAllReduceBuffer] = {}
_context_data_cache: dict[tuple, list[int]] = {}


def _dynamo_disable(fn):
    # no-op兜底：dynamo不可用时保持原函数
    try:
        return torch._dynamo.disable(fn)
    except Exception:
        return fn


@_dynamo_disable
def _create_and_cache_buffer(
    group_name: str,
    group,
    world_size: int,
    x: torch.Tensor,
    scales: torch.Tensor,
) -> QuantAllReduceBuffer:
    """缓存未命中时创建 buffer。必须对 dynamo 透明(graph break)：
    context.cpu().tolist() 会把上千个 int32 展开成大量符号量，若被 trace，
    npu backend 将在 aten._local_scalar_dense 上编译失败。"""
    buffer = QuantAllReduceBuffer(group, world_size, x, scales)
    _buffer_cache[group_name] = buffer
    # 始终用最新 context 数据覆盖，供 GE 图模式经 get_cached_context_data 注入 Const
    _context_data_cache[(world_size, buffer.hccl_buffer_size)] = buffer.context.cpu().tolist()
    return buffer


def _get_or_create_buffer(
    group_name: str,
    group,
    world_size: int,
    x: torch.Tensor,
    scales: torch.Tensor,
) -> QuantAllReduceBuffer:
    """按 group 查找缓存：

    1. 命中时直接复用——本函数必须可被 dynamo 完整 trace（op 调用作为节点捕获后
       经 torchair 转 GE 图），禁止对命中路径加 @_dynamo_disable 导致断图退化 eager
    2. 未命中时走 _create_and_cache_buffer（dynamo 透明，graph break 后急执行）

    注：win 区为 HCCL 内置 buffer（HCCL_BUFFSIZE 定容），不存在"尺寸不足销毁重建"路径，
    容量不足在创建时由 _check_hccl_buffer 直接报错；engine context 按 tag 幂等复用，
    进程内无 python 侧持有的 device 资源需要主动释放。
    """
    buffer = _buffer_cache.get(group_name)
    if buffer is not None:
        return buffer
    return _create_and_cache_buffer(group_name, group, world_size, x, scales)


def get_cached_context_data(world_size: int, hccl_buffer_size: int) -> list[int]:
    return _context_data_cache.get((world_size, hccl_buffer_size))


def quant_all_reduce(
    x: torch.Tensor,
    scales: torch.Tensor,
    hcom: str,
    world_size: int,
    *,
    reduce_op: Optional[str] = "sum",
    output_dtype: Optional[int] = None,
    x_dtype: Optional[int] = None,
    scales_dtype: Optional[int] = None,
):
    group = torch.distributed.distributed_c10d._get_default_group()
    group_name = (
        hcom
        if isinstance(hcom, str)
        else group._get_backend(torch.device("npu")).get_hccl_comm_name(
            torch.distributed.get_rank(group), init_comm=False
        )
    )
    buffer = _get_or_create_buffer(group_name, group, world_size, x, scales)
    return torch.ops.cann_ops_transformer.npu_quant_all_reduce(
        buffer.context,
        x,
        scales,
        buffer.hccl_buffer_size,
        buffer.world_size,
        reduce_op=reduce_op,
        output_dtype=output_dtype,
        x_dtype=x_dtype,
        scales_dtype=scales_dtype,
    )
