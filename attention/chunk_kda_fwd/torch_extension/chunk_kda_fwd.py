"""ChunkKdaFwd 的 PyTorch 扩展入口；输出顺序与算子功能测试一致。"""
import math

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class ChunkKdaFwdOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("chunk_kda_fwd", category="attention")

    def sources(self):
        return ["csrc/attention/chunk_kda_fwd.cpp"]

    def schema(self):
        return (
            "chunk_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
            "Tensor? initial_state, bool output_final_state, bool return_intermediate_states, "
            "bool disable_recompute, bool safe_gate, float lower_bound, bool use_gate_in_kernel, "
            "Tensor? A_log, Tensor? dt_bias, int chunk_size, str layout, bool state_v_first, "
            "int[]? cu_seqlens, int[]? chunk_indices) -> Tensor?[]"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def meta(q, k, v, g, beta, scale, initial_state, output_final_state,
                 return_intermediate_states, disable_recompute, safe_gate, lower_bound,
                 use_gate_in_kernel, A_log, dt_bias, chunk_size, layout, state_v_first,
                 cu_seqlens, chunk_indices):
            rank3 = q.dim() == 3
            head_first = layout in ("BNSD", "NTD")
            batch = 1 if rank3 else q.shape[0]
            seqlen = q.shape[-2 if head_first else -3]
            hv, kd, vd = v.shape[-3 if head_first else -2], q.shape[-1], v.shape[-1]
            nseq = len(cu_seqlens) - 1 if cu_seqlens is not None else batch
            chunks = (sum((b - a + chunk_size - 1) // chunk_size
                          for a, b in zip(cu_seqlens, cu_seqlens[1:]))
                      if cu_seqlens is not None else (seqlen + chunk_size - 1) // chunk_size)
            prefix = [] if rank3 else [batch]
            state_dims = [vd, kd] if state_v_first else [kd, vd]
            def empty(shape, dtype=None):
                return q.new_empty(shape, dtype=dtype or q.dtype)
            out = [None] * 12
            out[0] = empty(prefix + [seqlen, hv, vd])
            if output_final_state:
                out[1] = empty([nseq, hv] + state_dims, torch.float32)
            for i in (3, 4):
                out[i] = empty(prefix + [hv, seqlen, chunk_size])
            if disable_recompute:
                out[2] = empty(prefix + [hv, seqlen, kd], torch.float32)
                for i, dim in ((5, kd), (6, vd), (7, kd), (8, kd), (9, vd)):
                    out[i] = empty(prefix + [hv, seqlen, dim])
            if return_intermediate_states or disable_recompute:
                out[10] = empty(prefix + [chunks, hv] + state_dims)
            return out


_builder = ChunkKdaFwdOpBuilder()
_builder._ensure_initialized()


@impl(get_as_library(), "chunk_kda_fwd", "PrivateUse1")
def _chunk_kda_fwd(*args):
    return _builder.load().chunk_kda_fwd(*args)


def _host_list(value, host_value=None):
    # 显式 host 副本避免变长元数据从 NPU 同步回读；与已有 pytest 调用约定一致。
    if host_value is not None:
        return list(host_value)
    if value is None:
        return None
    return value.detach().cpu().reshape(-1).tolist() if isinstance(value, torch.Tensor) else list(value)


def chunk_kda_fwd(q, k, v, g, beta, scale=None, *, initial_state=None,
                  output_final_state=False, return_intermediate_states=False,
                  disable_recompute=False, safe_gate=False, lower_bound=-5.0,
                  use_gate_in_kernel=False, A_log=None, dt_bias=None, chunk_size=64,
                  layout="BSND", state_v_first=False, cu_seqlens=None,
                  cu_seqlens_cpu=None, chunk_indices=None):
    """调用 CANN ChunkKdaFwd，返回 12 项 attention/state/中间量元组。

    q/k/v 为 BF16 或 FP16；g 为逐 K 维 FP32/BF16 gate，beta 为 FP32/BF16。
    layout 支持 BSND/BNSD/TND/NTD，K/V 必须为 [16,256] 内的 16 倍数。
    initial_state 为 FP32，[N,HV,K,V]（state_v_first=True 时末两维互换）。
    use_gate_in_kernel=True 时传入 raw g 和 A_log，可选 dt_bias；否则 g 是
    已激活的自然对数 gate。A_log/dt_bias 在入口提升为 FP32。
    chunk_size 为 64/128，scale 默认 K**-0.5。变长 cu_seqlens 为累计长度，
    可提供 cu_seqlens_cpu 避免回读；chunk_indices 为展平的序列/chunk 索引。
    返回顺序：attn、final_state、gk、Aqk、Akk、w、u、qg、kg、v_new、h、None。
    attn/Aqk/Akk 始终导出；其余由 output_final_state、disable_recompute 和
    return_intermediate_states 控制。输出布局详见算子的公开接口文档。
    """
    return tuple(torch.ops.cann_ops_transformer.chunk_kda_fwd(
        q, k, v, g, beta, scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]),
        initial_state, output_final_state, return_intermediate_states, disable_recompute,
        safe_gate, lower_bound, use_gate_in_kernel,
        A_log.float() if A_log is not None else None,
        dt_bias.float() if dt_bias is not None else None,
        chunk_size, layout, state_v_first, _host_list(cu_seqlens, cu_seqlens_cpu),
        _host_list(chunk_indices)))
