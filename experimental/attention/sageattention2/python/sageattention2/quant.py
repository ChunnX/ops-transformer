"""INT8 QK + FP8 V 量化（对标开源 sageattention/quant.py 语义的 CPU 参考实现）。

- per_thread：BLKQ=128 块内 8 个 scale，每组覆盖块内 stride-8 的 16 行
  （复刻 triton quant_query_per_thread_int8_kernel 的行映射）
- per_warp：Q 每 WARPQ=32 连续行一个 scale（BLKQ=128 内 4 个）
- K：恒 per-block，每 BLKK=64 连续行一个 scale；smooth_k 时融合减 km
- V：per-channel FP8 E4M3；scale = max|v|/scale_max；smooth_v 先减均值

NPU 验收时量化在 AscendC kernel 内实现，本模块仅用于 Host 侧流程验证与
数值对标（精度 golden 仍以 GPU sageattn_qk_int8_pv_fp8_cuda 为准）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

_INT8_MAX = 127.0
_EPS = 1e-7


def _group_scale_1d(src: torch.Tensor, groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """src: [B, H, S, D]；groups: [S]（每行所属 scale 组）→ scale [B, H, G] = max|abs|/127+eps。"""
    b, h, s, d = src.shape
    g = int(groups.max().item()) + 1
    src_flat = src.reshape(b * h, s, d).abs()
    scale = torch.empty(b * h, g, device=src.device, dtype=torch.float32)
    for gi in range(g):
        mask = groups == gi  # [S]
        if mask.any():
            scale[:, gi] = src_flat[:, mask, :].amax(dim=(1, 2)) / _INT8_MAX + _EPS
        else:
            scale[:, gi] = _EPS
    scale = scale.reshape(b, h, g)
    x_int8 = torch.empty_like(src, dtype=torch.int8)
    for gi in range(g):
        mask = groups == gi
        sel = src[:, :, mask, :].float() / scale[:, :, gi, None, None]
        x_int8[:, :, mask, :] = torch.where(sel >= 0, torch.floor(sel + 0.5), torch.ceil(sel - 0.5)).to(torch.int8)
    return x_int8, scale


def quant_qk_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor],
    tensor_layout: str = "HND",
    qk_quant_gran: str = "per_thread",
    BLKQ: int = 128,
    WARPQ: int = 32,
    BLKK: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """对标 per_warp_int8 / per_thread_int8：Q 按粒度量化，K 恒 per-block。

    返回 (q_int8, q_scale, k_int8, k_scale)，形状与开源一致。
    """
    if tensor_layout not in ("HND", "NHD"):
        raise ValueError(tensor_layout)
    if qk_quant_gran not in ("per_warp", "per_thread"):
        raise ValueError(qk_quant_gran)

    if tensor_layout == "HND":
        b, hq, sq, d = q.shape
        _, hkv, sk, _ = k.shape
    else:
        b, sq, hq, d = q.shape
        _, sk, hkv, _ = k.shape

    # ---- Q 量化（复刻开源 per_thread / per_warp 行→scale 组映射）----
    row = torch.arange(sq, device=q.device)
    if qk_quant_gran == "per_thread":
        # triton quant_query_per_thread_int8_kernel：BLK=WARPQ=32，
        # off_tld=pid%8，scale 索引 = (r//32)*8 + (r%8)（每 32 行子块 8 组，stride-8 交错）
        q_groups = (row // WARPQ) * 8 + (row % WARPQ) % 8
    else:  # per_warp
        # CUDA per_warp_int8：每 WARPQ=32 连续行一组
        q_groups = row // WARPQ

    # 统一按 HND 语义量化，NHD 先转置
    if tensor_layout == "NHD":
        q_t = q.transpose(1, 2).contiguous()  # [B, Hq, Sq, D]
    else:
        q_t = q
    q_int8_t, q_scale = _group_scale_1d(q_t, q_groups)
    if tensor_layout == "NHD":
        q_int8 = q_int8_t.transpose(1, 2).contiguous()
        q_scale = q_scale  # [B, Hq, G]
    else:
        q_int8 = q_int8_t

    # ---- K 量化（恒 per-block，BLKK 连续行一组；smooth_k 融合减均值）----
    if tensor_layout == "HND":
        k_t = k
        seq_ax, h_ax = 2, 1
    else:
        k_t = k.transpose(1, 2).contiguous()
        seq_ax, h_ax = 2, 1
    # ---- K 量化（分组随 qk_quant_gran：per_warp 每 BLKK=64 连续行一组；
    #      per_thread 每 64 行 4 组（复刻 quant_key_per_thread_int8_kernel：scale=(r//64)*4+((r%64)%8)//2））----
    row_k = torch.arange(sk, device=k.device)
    if qk_quant_gran == "per_thread":
        k_groups = (row_k // BLKK) * 4 + ((row_k % BLKK) % 8) // 2
    else:  # per_warp（K 恒 per-block）
        k_groups = row_k // BLKK

    if km is not None:
        # km 来自 core.py：HND=[B,Hk,1,D]，NHD=[B,1,Hk,D]；统一成 [B,Hk,1,D] 匹配 k_t
        km_s = km.transpose(1, 2).contiguous() if tensor_layout == "NHD" else km
        k_smoothed = k_t - km_s
    else:
        k_smoothed = k_t
    k_int8_t, k_scale = _group_scale_1d(k_smoothed, k_groups)
    if tensor_layout == "NHD":
        k_int8 = k_int8_t.transpose(1, 2).contiguous()
    else:
        k_int8 = k_int8_t

    return q_int8, q_scale, k_int8, k_scale


def quant_v_fp8(
    v: torch.Tensor,
    tensor_layout: str = "HND",
    scale_max: float = 448.0,
    smooth_v: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """V per-channel FP8 量化，对标 per_channel_fp8。

    返回 (v_fp8, v_scale, vm)：
    - HND: v_fp8 [B, Hk, D, S_pad64]（转置 + pad + 开源 permute 的简化连续视图）
    - v_scale [B, Hk, D]（FP32）；vm [B, Hk, D]（smooth_v 时非 None，FP32）
    """
    if tensor_layout == "HND":
        b, hkv, sk, d = v.shape
        padded_len = (sk + 63) // 64 * 64
        # 转置 [B,H,S,D] -> [B,H,D,S]，pad 到 64 倍数（开源另有 permute，NPU 实现自定）
        v_t = v.transpose(2, 3)
        if padded_len != sk:
            v_t = torch.nn.functional.pad(v_t, (0, padded_len - sk))
        reduce_dim = 3  # v_t 为 [B,Hk,D,S]，沿 S 维（最后一维）
    else:  # NHD
        b, sk, hkv, d = v.shape
        padded_len = (sk + 63) // 64 * 64
        v_t = v.permute(0, 2, 3, 1)  # [B, Hk, D, S]
        if padded_len != sk:
            v_t = torch.nn.functional.pad(v_t, (0, padded_len - sk))
        reduce_dim = 3

    # per-channel scale：沿 S 维取 max|abs|
    vm = None
    src = v_t
    if smooth_v:
        vm = v_t.float().mean(dim=reduce_dim, keepdim=True)  # [B,Hk,D,1]
        src = v_t.float() - vm
    else:
        src = v_t.float()

    v_scale = src.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(_EPS) / scale_max  # [B,Hk,D,1]
    v_scale = v_scale + 1e-30  # avoid 0 later
    v_fp8 = (src / v_scale).clamp(-scale_max, scale_max).to(torch.float8_e4m3fn)
    v_scale = v_scale.squeeze(reduce_dim).contiguous()  # [B, Hk, D]
    if vm is not None:
        vm = vm.squeeze(reduce_dim).contiguous()
    return v_fp8, v_scale, vm
