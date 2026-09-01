"""NPU kernel 后端抽象：run_attn 注册机制 + CPU fallback（仅流程验证）。

NPU 验收时经 backend 注册真实 kernel（AscendC+CATLASS，见设计文档 §3.2）；
未注册时回退到 CPU 参考（量化反解 + FP32 SDPA），只用于流程连通性调试，
精度/性能不作验收（与 sageAttentionTest 的 SAGEATTN_ALLOW_CPU_STUB 语义一致）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

_ATTN_BACKENDS: Dict[str, Callable[..., Any]] = {}


def register_attn_backend(name: str):
    """装饰器：注册 NPU kernel 后端。name 例：'npu'。"""
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _ATTN_BACKENDS[name] = fn
        return fn

    return deco


def _dequant_cpu(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, vm, tensor_layout, smooth_v, qk_quant_gran):
    """CPU 参考：把量化张量按 scale 反解回近似 float32 用于 SDPA（仅流程）。

    q/k: [B,H,S,D]（HND）或 [B,S,H,D]（NHD）；scale [B,H,G]，组映射与
    quant_qk_int8 一致（per_thread 为 stride-8 交错组，per_warp/per_block 为连续组）。
    """
    BLKQ, WARPQ, BLKK = 128, 32, 64
    if tensor_layout == "HND":
        b, hq, sq, d = q_int8.shape
        b2, hkv, sk, _ = k_int8.shape
        q_work, k_work = q_int8, k_int8
        q_s, k_s = q_scale, k_scale
    elif tensor_layout == "NHD":
        b, sq, hq, d = q_int8.shape
        b2, sk, hkv, _ = k_int8.shape
        q_work = q_int8.transpose(1, 2).contiguous()  # [B,Hq,Sq,D]
        k_work = k_int8.transpose(1, 2).contiguous()  # [B,Hk,Sk,D]
        q_s, k_s = q_scale, k_scale
    else:
        raise ValueError(tensor_layout)

    def _q_groups(slen: int, dev: torch.device) -> torch.Tensor:
        row = torch.arange(slen, device=dev)
        if qk_quant_gran == "per_thread":
            return (row // WARPQ) * 8 + (row % WARPQ) % 8
        return row // WARPQ  # per_warp

    def _k_groups(slen: int, dev: torch.device) -> torch.Tensor:
        row = torch.arange(slen, device=dev)
        if qk_quant_gran == "per_thread":
            return (row // BLKK) * 4 + ((row % BLKK) % 8) // 2
        return row // BLKK  # per_warp：K 恒 per-block

    def _broadcast_scale(x: torch.Tensor, s: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        # s [B,H,G] → 按行所属组 gather → [B,H,S] → [B,H,S,D]
        s_g = s[:, :, groups]  # [B,H,S]
        return x.float() * s_g.unsqueeze(-1)

    q_dec = _broadcast_scale(q_work, q_s, _q_groups(sq, q_work.device))
    k_dec = _broadcast_scale(k_work, k_s, _k_groups(sk, k_work.device))
    # v_fp8: [B,Hk,D,S_pad]；v_scale [B,Hk,D] → 乘到 S 维：unsqueeze(2) 得 [B,Hk,1,D]
    v_dec = v_fp8.float().transpose(2, 3) * v_scale.unsqueeze(2)  # [B,Hk,S,D]
    if smooth_v and vm is not None:
        v_dec = v_dec + vm.unsqueeze(2)  # vm [B,Hk,D] → [B,Hk,1,D]
    return q_dec, k_dec, v_dec


def _cpu_lse(scores: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """行 logsumexp（FP32），形状 [B,H,S]。"""
    lse = torch.logsumexp(scores.float() * sm_scale, dim=-1)
    return lse


def _cpu_attn(q_dec, k_dec, v_dec, is_causal, sm_scale, return_lse):
    """CPU 参考 attention（HND，BNSD 同形）。返回 (o, lse)。"""
    o = F.scaled_dot_product_attention(
        q_dec.float(), k_dec.float(), v_dec.float(),
        is_causal=is_causal, scale=sm_scale,
    )
    lse = None
    if return_lse:
        scores = torch.matmul(q_dec.float(), k_dec.float().transpose(-1, -2))  # [B,H,Sq,Sk]
        lse = _cpu_lse(scores, sm_scale)
    return o, lse


def run_attn(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    o: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    vm: Optional[torch.Tensor],
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    sm_scale: float,
    pv_accum_dtype: str,
    return_lse: bool,
    head_dim_og: int,
):
    """分发到已注册后端；无则 CPU fallback（仅流程调试）。"""
    for name in ("npu", "cpu_ref"):
        fn = _ATTN_BACKENDS.get(name)
        if fn is not None:
            return fn(
                q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm,
                tensor_layout=tensor_layout, is_causal=is_causal,
                qk_quant_gran=qk_quant_gran, sm_scale=sm_scale,
                pv_accum_dtype=pv_accum_dtype, return_lse=return_lse,
                head_dim_og=head_dim_og,
            )
    # CPU fallback（仅调试）
    q_dec, k_dec, v_dec = _dequant_cpu(
        q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, vm, tensor_layout,
        smooth_v=False, qk_quant_gran=qk_quant_gran,
    )
    # GQA：k/v 头数广播到 q 头数（HND/BNSD 语义，head 在 dim 1）
    if k_dec.shape[1] != q_dec.shape[1]:
        reps = q_dec.shape[1] // k_dec.shape[1]
        k_dec = torch.repeat_interleave(k_dec, reps, dim=1)
        v_dec = torch.repeat_interleave(v_dec, reps, dim=1)
    o_ref, lse = _cpu_attn(q_dec, k_dec, v_dec, is_causal, sm_scale, return_lse)
    if tensor_layout == "NHD":
        o_ref = o_ref.transpose(1, 2).contiguous()  # BNSD -> [B,Sq,Hq,D] 与 o 一致
    o.copy_(o_ref.to(o.dtype))
    return lse


def _register_npu_backend() -> None:
    """注册 NPU kernel 后端（AscendC+CATLASS，sageattn_fp8 编译产物）。

    后端函数实现见 sageattn_fp8 子包（kernel 侧）；此处仅登记分发入口。
    注册成功后 run_attn 将优先走 NPU kernel，不再走 CPU fallback。
    """
    try:
        from .sageattn_fp8_backend import npu_attn  # type: ignore
        register_attn_backend("npu")(npu_attn)
    except Exception:
        # NPU kernel 未就绪（未编译/未安装）时保持 CPU fallback，仅流程调试
        pass


_register_npu_backend()
