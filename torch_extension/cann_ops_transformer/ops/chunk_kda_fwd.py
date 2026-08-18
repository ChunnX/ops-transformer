# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional, Sequence, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


ChunkKdaFwdOutputs = Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]


def _check_reserved_options(
    use_qk_l2norm_in_kernel: Optional[bool],
    use_beta_sigmoid_in_kernel: Optional[bool],
) -> None:
    if use_qk_l2norm_in_kernel:
        raise RuntimeError(
            "chunk_kda_fwd: use_qk_l2norm_in_kernel currently supports false only."
        )
    if use_beta_sigmoid_in_kernel:
        raise RuntimeError(
            "chunk_kda_fwd: use_beta_sigmoid_in_kernel currently supports false only."
        )


def _get_shape_info(
    q: torch.Tensor,
    v: torch.Tensor,
    layout: str,
) -> Tuple[int, int, int, int, int, bool]:
    if layout == "TND":
        return 1, q.size(0), q.size(2), v.size(1), v.size(2), True
    if layout == "NTD":
        return 1, q.size(1), q.size(2), v.size(0), v.size(2), True
    if layout == "BSND":
        return q.size(0), q.size(1), q.size(3), v.size(2), v.size(3), False
    if layout == "BNSD":
        return q.size(0), q.size(2), q.size(3), v.size(1), v.size(3), False
    raise RuntimeError(
        "chunk_kda_fwd: layout must be one of BSND, BNSD, TND and NTD."
    )


def _get_total_chunks(
    seqlen: int,
    chunk_size: int,
    cu_seqlens: Optional[Sequence[int]],
    chunk_indices: Optional[Sequence[int]],
) -> int:
    if chunk_indices is not None:
        return len(chunk_indices) // 2
    if cu_seqlens is None:
        return (seqlen + chunk_size - 1) // chunk_size
    return sum(
        (int(end) - int(start) + chunk_size - 1) // chunk_size
        for start, end in zip(cu_seqlens, cu_seqlens[1:])
    )


class ChunkKdaFwdOpBuilder(OpBuilder):
    def __init__(self):
        super(ChunkKdaFwdOpBuilder, self).__init__("chunk_kda_fwd")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/chunk_kda_fwd.cpp"]

    def schema(self) -> str:
        """PyTorch operator schema."""
        return (
            "chunk_kda_fwd("
            "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
            "float scale, int chunk_size=64, "
            "*, "
            'str layout="BSND", '
            "Tensor? initial_state=None, "
            "bool? output_final_state=False, "
            "int[]? cu_seqlens=None, "
            "int[]? chunk_indices=None, "
            "bool? safe_gate=False, "
            "float? lower_bound=None, "
            "bool? use_qk_l2norm_in_kernel=False, "
            "bool? use_gate_in_kernel=False, "
            "bool? use_beta_sigmoid_in_kernel=False, "
            "bool? use_exp2=True, "
            "Tensor? A_log=None, "
            "Tensor? dt_bias=None, "
            "bool? disable_recompute=False, "
            "bool? return_intermediate_states=False, "
            "bool? state_v_first=False"
            ") -> (Tensor, Tensor?, Tensor?, Tensor, Tensor, Tensor?, "
            "Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)"
        )

    def register_meta(self):
        """Register Meta implementation for shape and dtype inference."""

        @impl(AS_LIBRARY, self.name, "Meta")
        def chunk_kda_fwd_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            chunk_size: int = 64,
            *,
            layout: str = "BSND",
            initial_state: Optional[torch.Tensor] = None,
            output_final_state: Optional[bool] = False,
            cu_seqlens: Optional[List[int]] = None,
            chunk_indices: Optional[List[int]] = None,
            safe_gate: Optional[bool] = False,
            lower_bound: Optional[float] = None,
            use_qk_l2norm_in_kernel: Optional[bool] = False,
            use_gate_in_kernel: Optional[bool] = False,
            use_beta_sigmoid_in_kernel: Optional[bool] = False,
            use_exp2: Optional[bool] = True,
            A_log: Optional[torch.Tensor] = None,
            dt_bias: Optional[torch.Tensor] = None,
            disable_recompute: Optional[bool] = False,
            return_intermediate_states: Optional[bool] = False,
            state_v_first: Optional[bool] = False,
        ) -> ChunkKdaFwdOutputs:
            del k, g, beta, scale, safe_gate, lower_bound, use_exp2, A_log, dt_bias
            _check_reserved_options(
                use_qk_l2norm_in_kernel,
                use_beta_sigmoid_in_kernel,
            )
            if chunk_size not in (64, 128):
                raise RuntimeError(
                    "chunk_kda_fwd: chunk_size must be 64 or 128."
                )

            batch, seqlen, k_dim, hv, v_dim, is_rank3 = _get_shape_info(
                q, v, layout
            )
            seq_num = len(cu_seqlens) - 1 if cu_seqlens is not None else batch
            total_chunks = _get_total_chunks(
                seqlen,
                chunk_size,
                cu_seqlens,
                chunk_indices,
            )
            state_shape = (
                (seq_num, hv, v_dim, k_dim)
                if state_v_first
                else (seq_num, hv, k_dim, v_dim)
            )
            attn_shape = (
                (seqlen, hv, v_dim)
                if is_rank3
                else (batch, seqlen, hv, v_dim)
            )
            matrix_shape = (
                (hv, seqlen, chunk_size)
                if is_rank3
                else (batch, hv, seqlen, chunk_size)
            )
            k_shape = (
                (hv, seqlen, k_dim)
                if is_rank3
                else (batch, hv, seqlen, k_dim)
            )
            v_shape = (
                (hv, seqlen, v_dim)
                if is_rank3
                else (batch, hv, seqlen, v_dim)
            )
            if is_rank3:
                h_shape = (
                    (total_chunks, hv, v_dim, k_dim)
                    if state_v_first
                    else (total_chunks, hv, k_dim, v_dim)
                )
            else:
                h_shape = (
                    (batch, total_chunks, hv, v_dim, k_dim)
                    if state_v_first
                    else (batch, total_chunks, hv, k_dim, v_dim)
                )

            attn_out = v.new_empty(attn_shape)
            final_state = (
                torch.empty(state_shape, dtype=torch.float32, device=q.device)
                if output_final_state
                else None
            )
            gk_out = (
                torch.empty(k_shape, dtype=torch.float32, device=q.device)
                if not use_gate_in_kernel
                else None
            )
            aqk = q.new_empty(matrix_shape)
            akk = q.new_empty(matrix_shape)
            w = q.new_empty(k_shape) if disable_recompute else None
            u = q.new_empty(v_shape) if disable_recompute else None
            qg = q.new_empty(k_shape) if disable_recompute else None
            kg = q.new_empty(k_shape) if disable_recompute else None
            v_new = q.new_empty(v_shape) if disable_recompute else None
            h_out = (
                q.new_empty(h_shape) if return_intermediate_states else None
            )
            return (
                attn_out,
                final_state,
                gk_out,
                aqk,
                akk,
                w,
                u,
                qg,
                kg,
                v_new,
                h_out,
                initial_state,
            )


_chunk_kda_fwd_op_builder = ChunkKdaFwdOpBuilder()


@impl(AS_LIBRARY, _chunk_kda_fwd_op_builder.name, "PrivateUse1")
def _chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    *,
    layout: str = "BSND",
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False,
    cu_seqlens: Optional[Sequence[int]] = None,
    chunk_indices: Optional[Sequence[int]] = None,
    safe_gate: Optional[bool] = False,
    lower_bound: Optional[float] = None,
    use_qk_l2norm_in_kernel: Optional[bool] = False,
    use_gate_in_kernel: Optional[bool] = False,
    use_beta_sigmoid_in_kernel: Optional[bool] = False,
    use_exp2: Optional[bool] = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    disable_recompute: Optional[bool] = False,
    return_intermediate_states: Optional[bool] = False,
    state_v_first: Optional[bool] = False,
) -> ChunkKdaFwdOutputs:
    _check_reserved_options(
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
    )
    op_module = _chunk_kda_fwd_op_builder.load()
    return op_module.chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        chunk_size,
        layout,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        safe_gate,
        lower_bound,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        use_exp2,
        A_log,
        dt_bias,
        disable_recompute,
        return_intermediate_states,
        state_v_first,
    )


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    *,
    layout: str = "BSND",
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False,
    cu_seqlens: Optional[Sequence[int]] = None,
    chunk_indices: Optional[Sequence[int]] = None,
    safe_gate: Optional[bool] = False,
    lower_bound: Optional[float] = None,
    use_qk_l2norm_in_kernel: Optional[bool] = False,
    use_gate_in_kernel: Optional[bool] = False,
    use_beta_sigmoid_in_kernel: Optional[bool] = False,
    use_exp2: Optional[bool] = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    disable_recompute: Optional[bool] = False,
    return_intermediate_states: Optional[bool] = False,
    state_v_first: Optional[bool] = False,
) -> ChunkKdaFwdOutputs:
    """Execute chunk KDA forward and return optional backward intermediates."""
    return _chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        chunk_size,
        layout=layout,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        use_exp2=use_exp2,
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=disable_recompute,
        return_intermediate_states=return_intermediate_states,
        state_v_first=state_v_first,
    )
