"""NPU 后端 — 按 npu_flash_attn.md 参数组调用。"""

import numpy as np
import torch
import torch_npu
from typing import Any, Dict

from .base import Backend

try:
    from npu_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

_AIC_CORE_NUM = 36
_AIV_CORE_NUM = 72
_FA_META_SIZE = 16
_FD_META_SIZE = 16
_HEADER_SIZE = 16

_FA_FIELDS = ["BN2_START", "M_START", "S2_START",
              "BN2_END", "M_END", "S2_END", "FIRST_FD_WS_IDX"]
_FD_FIELDS = ["BN2_IDX", "M_IDX", "WS_IDX",
              "WS_NUM", "M_START", "M_NUM"]


def print_metadata(metadata):
    arr = metadata.cpu().contiguous().to(torch.int32).numpy().astype(np.uint32)

    section_num = int(arr[0])
    is_fd = int(arr[1])
    m_base_size = int(arr[2])
    s2_base_size = int(arr[3])

    SEP = "=" * 78
    DASH = "-" * 78

    print(f"\n{SEP}")
    print(f"  [Metadata Header]")
    print(f"  sectionNum  = {section_num}")
    print(f"  isFD        = {is_fd}")
    print(f"  mBaseSize   = {m_base_size}")
    print(f"  s2BaseSize  = {s2_base_size}")

    fa_base = _HEADER_SIZE
    fd_base = _HEADER_SIZE + section_num * _AIC_CORE_NUM * _FA_META_SIZE

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(f"  [Section {sec}] FA Metadata — AIC cores  "
              f"(36 cores × 16 slots, 7 fields used)")
        print(DASH)
        cw = 15
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in _FA_FIELDS)
        print(hdr)
        print(f"  {'':<7}" + "".join(f"{'['+str(i)+']':>{cw}}" for i in range(len(_FA_FIELDS))))
        print(f"  {'─'*7}" + "─" * (cw * len(_FA_FIELDS)))
        for core in range(_AIC_CORE_NUM):
            off = fa_base + sec * _AIC_CORE_NUM * _FA_META_SIZE + core * _FA_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(_FA_FIELDS))]
            note = "  (inactive)" if all(v == 0 for v in vals) else ""
            print(f"  AIC{core:02d}  " + "".join(f"{v:>{cw}}" for v in vals) + note)

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(f"  [Section {sec}] FD Metadata — active AIV cores only  "
              f"(M_NUM > 0 shown, 72 cores × 16 slots, 6 fields used)")
        print(DASH)
        cw = 12
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in _FD_FIELDS)
        print(hdr)
        print(f"  {'':<7}" + "".join(f"{'['+str(i)+']':>{cw}}" for i in range(len(_FD_FIELDS))))
        print(f"  {'─'*7}" + "─" * (cw * len(_FD_FIELDS)))
        active = 0
        for core in range(_AIV_CORE_NUM):
            off = fd_base + sec * _AIV_CORE_NUM * _FD_META_SIZE + core * _FD_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(_FD_FIELDS))]
            if vals[5] == 0:
                continue
            print(f"  AIV{core:02d} " + "".join(f"{v:>{cw}}" for v in vals))
            active += 1
        if active == 0:
            print("  (no active FD cores)")

    print(f"{SEP}\n")


class NPUBackend(Backend):
    name = "npu"

    def __init__(self, device_id: int = 0, meta_only: bool = False):
        self._device = torch.device(f"npu:{device_id}")
        self._meta_only = meta_only
        if _HAS_NPU:
            torch.npu.set_device(device_id)

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return _HAS_NPU

    def compute(self, inputs: Dict[str, torch.Tensor],
                params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        from core.data import build_flash_attn_params

        meta_kwargs, kernel_kwargs, _ = build_flash_attn_params(
            params, self._device, inputs)

        metadata = npu_flash_attn_metadata(**meta_kwargs)

        if self._meta_only:
            print_metadata(metadata)
            bs = inputs["q"].shape[0]
            n1 = inputs["q"].shape[1] if inputs["q"].dim() >= 3 else params.get("N1", 1)
            s1 = inputs["q"].shape[2] if inputs["q"].dim() >= 4 else params.get("S1", 1)
            d = params["D"]
            dummy_out = torch.zeros(bs, n1, s1, d, dtype=inputs["q"].dtype, device="cpu")
            return {"out": dummy_out, "lse": None}

        out, lse = npu_flash_attn(inputs["q"], inputs["k"], inputs["v"],
                                  metadata=metadata, **kernel_kwargs)
        torch.npu.synchronize()
        return {"out": out, "lse": lse}
