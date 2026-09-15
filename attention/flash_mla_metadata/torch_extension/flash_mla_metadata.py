# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
from typing import List, Optional, Union

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

_METADATA_SIZE = 480  # FlashMla::kMlaMetadataElements, a fixed 1920-byte protocol.


def _check_metadata_dimensions(
    batch_size, q_seq_size, kv_seq_size, seq_q, seq_kv, input_layer, block_size
):
    """Check inference prerequisites using descriptors only, including FakeTensor."""
    tnd = input_layer in ("TND", "TND_NTD")
    if batch_size is None:
        torch._check(
            tnd and seq_q is not None,
            lambda: "omitted batch_size requires TND and actual_seq_lengths_q",
        )
        torch._check(
            seq_q.dim() == 1, lambda: "actual_seq_lengths_q must be one-dimensional"
        )
        batch_size = seq_q.numel()
    torch._check(0 <= batch_size <= 65536, lambda: "batch_size must be in [0, 65536]")
    for name, size, seq, limit in (
        ("q_seq_size", q_seq_size, seq_q, 4294967295),
        ("kv_seq_size", kv_seq_size, seq_kv, 2147483647),
    ):
        if size is None:
            torch._check(tnd, lambda: f"omitted {name} requires TND")
            torch._check(
                batch_size > 0, lambda: f"{name} must be explicit for an empty batch"
            )
            if name == "kv_seq_size":
                torch._check(
                    block_size == 0,
                    lambda: "kv_seq_size must be explicit for paged attention",
                )
            torch._check(
                seq is not None,
                lambda: f"omitted {name} requires its cumulative sequence tensor",
            )
            torch._check(
                seq.dim() == 1,
                lambda: "cumulative sequence tensors must be one-dimensional",
            )
            torch._check(
                seq.numel() == batch_size,
                lambda: "TND sequence item count must equal batch_size",
            )
        else:
            torch._check(
                0 <= size <= limit, lambda: f"{name} must be in [0, {limit}] or None"
            )
    return batch_size


def _metadata_meta(
    num_heads: int,
    batch_size: int,
    quant_mode: int,
    input_layer: str = "BSH",
    device: Union[torch.device, str] = "meta",
) -> torch.Tensor:
    torch._check(0 < num_heads <= 128, lambda: "num_heads must be in [1, 128]")
    torch._check(0 <= batch_size <= 65536, lambda: "batch_size must be in [0, 65536]")
    torch._check(
        quant_mode in (0, 1),
        lambda: "quant_mode must be 0 (nonquant) or 1 (INT8 full quantization)",
    )
    torch._check(
        input_layer
        in (
            "BSH",
            "BSND",
            "BNSD",
            "TND",
            "BSH_NBSD",
            "BSND_NBSD",
            "BNSD_NBSD",
            "TND_NTD",
        ),
        lambda: "unsupported input_layer",
    )
    return torch.empty((_METADATA_SIZE,), dtype=torch.int32, device=device)


class FlashMlaMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("flash_mla_metadata", category="attention")

    def sources(self) -> List[str]:
        return ["csrc/attention/flash_mla_metadata.cpp"]

    def include_paths(self) -> List[str]:
        # aclnn_common.h needs the SDK ACL enums before torch_npu's older headers.
        return [os.path.join(self.cann_path, "include")] + super().include_paths()

    def schema(self) -> str:
        return (
            "flash_mla_metadata(int num_heads, int? batch_size=None, int? q_seq_size=None, int? kv_seq_size=None, *, "
            "Tensor? actual_seq_lengths_q=None, Tensor? actual_seq_lengths_kv=None, "
            'str input_layer="BSH", int block_size=0, int sparse_mode=0, '
            "int pre_tokens=2147483647, int next_tokens=2147483647, bool has_atten_mask=False, "
            "int quant_mode=0, Device? device=None) -> Tensor"
        )

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::flash_mla_metadata")
        def flash_mla_metadata_meta(
            num_heads,
            batch_size=None,
            q_seq_size=None,
            kv_seq_size=None,
            *,
            actual_seq_lengths_q=None,
            actual_seq_lengths_kv=None,
            input_layer="BSH",
            block_size=0,
            sparse_mode=0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            has_atten_mask=False,
            quant_mode=0,
            device=None,
        ):
            batch_size = _check_metadata_dimensions(
                batch_size,
                q_seq_size,
                kv_seq_size,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                input_layer,
                block_size,
            )
            if device is None:
                if actual_seq_lengths_q is not None:
                    device = actual_seq_lengths_q.device
                elif actual_seq_lengths_kv is not None:
                    device = actual_seq_lengths_kv.device
                else:
                    device = "npu"
            return _metadata_meta(
                num_heads, batch_size, quant_mode, input_layer, device
            )


_flash_mla_metadata_op_builder = FlashMlaMetadataOpBuilder()
_flash_mla_metadata_op_builder._ensure_initialized()


# The composite registration also handles dense calls with both seq tensors absent.
@impl(get_as_library(), "flash_mla_metadata", "CompositeExplicitAutograd")
@impl(get_as_library(), "flash_mla_metadata", "PrivateUse1")
def _flash_mla_metadata(
    num_heads,
    batch_size=None,
    q_seq_size=None,
    kv_seq_size=None,
    *,
    actual_seq_lengths_q=None,
    actual_seq_lengths_kv=None,
    input_layer="BSH",
    block_size=0,
    sparse_mode=0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    has_atten_mask=False,
    quant_mode=0,
    device=None,
):
    resolved_batch_size = _check_metadata_dimensions(
        batch_size,
        q_seq_size,
        kv_seq_size,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        input_layer,
        block_size,
    )
    # A factory with no tensor arguments does not select the Meta key by itself.
    if device is not None and torch.device(device).type == "meta":
        return _metadata_meta(num_heads, resolved_batch_size, quant_mode, input_layer)
    return _flash_mla_metadata_op_builder.load().flash_mla_metadata(
        num_heads,
        batch_size,
        q_seq_size,
        kv_seq_size,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        input_layer,
        block_size,
        sparse_mode,
        pre_tokens,
        next_tokens,
        has_atten_mask,
        quant_mode,
        device,
    )


def flash_mla_metadata(
    num_heads: int,
    batch_size: Optional[int] = None,
    q_seq_size: Optional[int] = None,
    kv_seq_size: Optional[int] = None,
    *,
    actual_seq_lengths_q: Optional[torch.Tensor] = None,
    actual_seq_lengths_kv: Optional[torch.Tensor] = None,
    input_layer: str = "BSH",
    block_size: int = 0,
    sparse_mode: int = 0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    has_atten_mask: bool = False,
    quant_mode: int = 0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """在 NPU 上生成 FlashMla 调度信息。

    Args:
        num_heads: Q 头数，范围 [1, 128]。
        batch_size: 逻辑 batch，范围 [0, 65536]；None 时由 TND seq Q 的项数推导。
        q_seq_size: 非 TND 为 Q 存储 S，TND 为存储总 T；None 时由非空 batch 的
            TND seq Q 累计末项推导，读取设备值的操作在 AI CPU 执行。
        kv_seq_size: dense KV 存储 S/T；PA 为 block_table.shape[1] * block_size。
            None 时由非空 batch 的 dense TND seq KV 累计末项推导；PA 必须显式传入。
        actual_seq_lengths_q, actual_seq_lengths_kv: 可选连续一维 NPU INT64 实长；
            TND Q/dense KV 使用不含起始 0 的累计长度，PA KV 使用逐 batch 绝对长度。
        input_layer: 与 flash_mla 的 input_layout 对应，支持 BSH/BSND/BNSD/TND，
            以及前三者的 _NBSD 后缀和 TND_NTD；输出后缀不影响调度。
            dense KV 与 Q 布局相同，PA KV 的存储布局由 consumer 的张量描述确定。
        block_size: 非 PA 为 0，PA 为不大于 1024 的正 16 倍数；INT8 必须为 128。
        sparse_mode, pre_tokens, next_tokens: 与 flash_mla 保持一致的稀疏模式及窗口。
        has_atten_mask: 与 consumer 是否提供 atten_mask 保持一致。
        quant_mode: 整数 0 表示非量化（FP16/BF16），1 表示 INT8 全量化。
        device: 输出设备，默认取 seq 设备；两个 seq 均缺省时取当前 NPU。
            显式指定时必须与 seq 设备一致。

    Returns:
        新分配的 NPU INT32[480]，连续、零偏移、64B 对齐，保持到 consumer 执行完成。

    Host 不读取或复制 seq 的设备值；seq 更新后须重新调用，再在同一 stream 调用 flash_mla。
    显式尺寸优先并保留校验；无法推导的尺寸必须显式提供，0 表示空输入而非省略。
    INT8 要求 NZ KV、block_size=128、num_heads 为 2 的幂。完整约束见接口文档。
    """
    # Torch's dispatcher can coerce ScalarType objects to integer enum values.
    # Require the public argument to be an integer before schema conversion.
    torch._check(
        isinstance(quant_mode, int),
        lambda: "quant_mode must be an int: 0 (nonquant) or 1 (INT8 full quantization)",
    )
    return torch.ops.cann_ops_transformer.flash_mla_metadata(
        num_heads,
        batch_size,
        q_seq_size,
        kv_seq_size,
        actual_seq_lengths_q=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        input_layer=input_layer,
        block_size=block_size,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        has_atten_mask=has_atten_mask,
        quant_mode=quant_mode,
        device=device,
    )
