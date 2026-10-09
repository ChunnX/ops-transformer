# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class CompressorOpBuilder(OpBuilder):
    def __init__(self):
        super(CompressorOpBuilder, self).__init__("compressor", category="attention")

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/compressor.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return (
            "compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, Tensor ape, "
            "int cmp_ratio, *, "
            "Tensor? state_block_table=None, Tensor? cu_seqlens=None, Tensor? seqused=None, "
            "Tensor? start_pos=None, Tensor(b!)? x_cache=None, Tensor? x_cache_slot_mapping=None, "
            "int coff=1, int cache_mode=1, bool x_cache_update=True) -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for FakeTensor/shape inference.
        """

        @impl(get_as_library(), self.name, "Meta")
        def compressor_meta(
            x: torch.Tensor,
            wkv: torch.Tensor,
            wgate: torch.Tensor,
            state_cache: torch.Tensor,
            ape: torch.Tensor,
            cmp_ratio: int,
            *,
            state_block_table: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            seqused: Optional[torch.Tensor] = None,
            start_pos: Optional[torch.Tensor] = None,
            x_cache: Optional[torch.Tensor] = None,
            x_cache_slot_mapping: Optional[torch.Tensor] = None,
            coff: Optional[int] = 1,
            cache_mode: Optional[int] = 1,
            x_cache_update: Optional[bool] = True,
        ):
            cmp_kv, _, _ = _compressor_forward_fake(
                x,
                wkv,
                wgate,
                state_cache,
                ape,
                cmp_ratio,
                state_block_table,
                cu_seqlens,
                seqused,
                start_pos,
                x_cache,
                x_cache_slot_mapping,
                coff,
                cache_mode,
                x_cache_update,
            )
            return cmp_kv


compressor_op_builder = CompressorOpBuilder()
compressor_op_builder._ensure_initialized()


# ===========================================================================
# Register the stateful Compressor inference kernel
# ===========================================================================
def _compressor_forward_impl(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int,
    # state_block_table..x_cache stay Optional (None is a legal value) but must
    # be positional-or-keyword arguments without a default. The ADInplaceOrView
    # kernel of this mutable custom op bumps the version of the mutated args by
    # positional index (state_cache=3, x_cache=10). Older PyTorch dispatchers
    # drop a trailing run of positionals that equal their schema default, so a
    # defaulted x_cache let the args tuple shrink to length 10 and
    # ``increment_version(args[10])`` raised IndexError. Requiring explicit
    # values keeps the mutated index present for every caller.
    state_block_table: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    start_pos: Optional[torch.Tensor],
    x_cache: Optional[torch.Tensor],
    x_cache_slot_mapping: Optional[torch.Tensor] = None,
    coff: Optional[int] = 1,
    cache_mode: Optional[int] = 1,
    x_cache_update: Optional[bool] = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op_module = compressor_op_builder.load()
    return op_module.compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        x_cache,
        x_cache_slot_mapping,
        coff,
        cache_mode,
        x_cache_update,
        # The native forward is inference-only by contract: the gradient branch
        # is never enabled, regardless of the caller's autograd state.
        False,
    )


# Compressor is a stateful inference operator: it mutates state_cache and, when
# supplied, x_cache. PyTorch forbids autograd formulas on mutable ops, and this
# project does not implement Compressor training, so the single registration is
# the mutable inference kernel below.
_compressor_inference = torch.library.custom_op(
    "cann_ops_transformer::_compressor_inference",
    mutates_args=("state_cache", "x_cache"), device_types="npu"
)(_compressor_forward_impl)


@torch.library.register_fake("cann_ops_transformer::_compressor_inference")
def _compressor_forward_fake(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    x_cache: Optional[torch.Tensor] = None,
    x_cache_slot_mapping: Optional[torch.Tensor] = None,
    coff: Optional[int] = 1,
    cache_mode: Optional[int] = 1,
    x_cache_update: Optional[bool] = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = wkv.size(0) // coff
    coff_cmp = coff * cmp_ratio
    if x.dim() == 3:
        b = x.size(0)
        s = x.size(1)
        cmp_size = (s + cmp_ratio - 1) // cmp_ratio
        cmp_kv_size = (b, cmp_size, d)
        softmax_score_size = (b, cmp_size, coff_cmp, d)
        kv_size = (b, cmp_size, coff_cmp, d)
    else:
        b_size = cu_seqlens.size(0) - 1
        t = x.size(0)
        cmp_size = min(t, t // cmp_ratio + b_size)
        cmp_kv_size = (cmp_size, d)
        softmax_score_size = (cmp_size, coff_cmp, d)
        kv_size = (cmp_size, coff_cmp, d)

    cmp_kv_out = torch.empty(cmp_kv_size, dtype=x.dtype, device=x.device)
    softmax_score_out = torch.empty(
        softmax_score_size, dtype=torch.float32, device=x.device
    )
    kv_out = torch.empty(kv_size, dtype=torch.float32, device=x.device)
    return (cmp_kv_out, softmax_score_out, kv_out)


def _reject_grad_tracking_inputs(named_tensors) -> None:
    """Refuse grad-tracked inputs on this inference-only entry point."""
    offenders = [
        name
        for name, value in named_tensors
        if isinstance(value, torch.Tensor)
        and value.requires_grad
        and (value.is_floating_point() or value.is_complex())
    ]
    if offenders:
        raise ValueError(
            "compressor is an inference-only entry point and does not support "
            "autograd; input tensor(s) [{}] require grad. Run under "
            "torch.no_grad() or torch.inference_mode().".format(", ".join(offenders))
        )


@impl(get_as_library(), compressor_op_builder.name, "PrivateUse1")
@impl(get_as_library(), compressor_op_builder.name, "Autograd")
def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int,
    *,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    x_cache: Optional[torch.Tensor] = None,
    x_cache_slot_mapping: Optional[torch.Tensor] = None,
    coff: Optional[int] = 1,
    cache_mode: Optional[int] = 1,
    x_cache_update: Optional[bool] = True,
) -> torch.Tensor:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.

    This is the project's formal Compressor entry point and it is stateful
    inference only: it reads and writes ``state_cache`` (and ``x_cache`` when
    provided) in place. Grad-tracked inputs are rejected before the native
    kernel is reached; run under ``torch.no_grad()`` or
    ``torch.inference_mode()``.
    """
    if x is None or not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch.Tensor, got {}".format(type(x).__name__))
    if torch.is_grad_enabled():
        _reject_grad_tracking_inputs(
            (
                ("x", x),
                ("wkv", wkv),
                ("wgate", wgate),
                ("state_cache", state_cache),
                ("ape", ape),
                ("state_block_table", state_block_table),
                ("cu_seqlens", cu_seqlens),
                ("seqused", seqused),
                ("start_pos", start_pos),
                ("x_cache", x_cache),
                ("x_cache_slot_mapping", x_cache_slot_mapping),
            )
        )
    cmp_kv, _, _ = _compressor_inference(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        x_cache,
        x_cache_slot_mapping,
        coff,
        cache_mode,
        x_cache_update,
    )
    return cmp_kv
