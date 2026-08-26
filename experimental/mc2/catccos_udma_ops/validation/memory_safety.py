# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Opt-in HBM poisoning, red zones and mutation diagnostics for UDMA tests."""

from __future__ import annotations

import gc
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch


_SCAN_CHUNK_ELEMENTS = 1 << 22


@dataclass
class GuardedTensor:
    """A contiguous tensor view surrounded by deterministic red zones."""

    name: str
    tensor: torch.Tensor
    storage: torch.Tensor
    guard_elements: int
    guard_kind: str
    guard_value: Optional[float]

    @classmethod
    def allocate(
        cls,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: Any,
        name: str,
        guard_bytes: int = 16 << 20,
        guard_elements: Optional[int] = None,
        guard_kind: str = "nan",
        guard_value: Optional[float] = None,
    ) -> "GuardedTensor":
        if guard_elements is None:
            guard_elements = max(1, math.ceil(guard_bytes / torch.empty((), dtype=dtype).element_size()))
        if guard_elements <= 0:
            raise ValueError("guard_elements must be positive")
        if guard_kind not in {"nan", "exact"}:
            raise ValueError("guard_kind must be 'nan' or 'exact'")
        if guard_kind == "nan" and not dtype.is_floating_point:
            raise TypeError("NaN guards require a floating-point dtype")
        if guard_kind == "exact" and guard_value is None:
            raise ValueError("exact guards require guard_value")

        payload_elements = math.prod(int(value) for value in shape)
        storage = torch.empty(
            payload_elements + 2 * guard_elements, dtype=dtype, device=device
        )
        if guard_kind == "nan":
            storage[:guard_elements].fill_(float("nan"))
            storage[-guard_elements:].fill_(float("nan"))
        else:
            storage[:guard_elements].fill_(guard_value)
            storage[-guard_elements:].fill_(guard_value)
        tensor = storage[guard_elements:guard_elements + payload_elements].view(tuple(shape))
        return cls(name, tensor, storage, guard_elements, guard_kind, guard_value)

    def fill_payload(self, value: float) -> None:
        self.tensor.fill_(value)

    def reset_guards(self) -> None:
        prefix = self.storage[:self.guard_elements]
        suffix = self.storage[-self.guard_elements:]
        if self.guard_kind == "nan":
            prefix.fill_(float("nan"))
            suffix.fill_(float("nan"))
        else:
            prefix.fill_(self.guard_value)
            suffix.fill_(self.guard_value)

    def check(self) -> Dict[str, Any]:
        prefix = self.storage[:self.guard_elements]
        suffix = self.storage[-self.guard_elements:]
        if self.guard_kind == "nan":
            prefix_corrupt = int((~torch.isnan(prefix)).sum().item())
            suffix_corrupt = int((~torch.isnan(suffix)).sum().item())
        else:
            prefix_corrupt = int((prefix != self.guard_value).sum().item())
            suffix_corrupt = int((suffix != self.guard_value).sum().item())
        return {
            "name": self.name,
            "guard_kind": self.guard_kind,
            "guard_elements_each_side": self.guard_elements,
            "guard_bytes_each_side": self.guard_elements * self.storage.element_size(),
            "prefix_corrupted_elements": prefix_corrupt,
            "suffix_corrupted_elements": suffix_corrupt,
            "passed": prefix_corrupt == 0 and suffix_corrupt == 0,
        }


def poison_allocatable_hbm(
    device: Any,
    *,
    reserve_bytes: int = 2 << 30,
    chunk_bytes: int = 256 << 20,
) -> Dict[str, Any]:
    """Fill nearly all currently allocatable HBM with BF16 NaNs.

    Driver-reserved memory and allocations owned by other processes cannot be
    touched safely.  Releasing the poison tensors without emptying the caching
    allocator makes subsequent test tensors preferentially reuse NaN-filled
    blocks.
    """
    if reserve_bytes < 0 or chunk_bytes <= 0:
        raise ValueError("reserve_bytes must be non-negative and chunk_bytes positive")
    import torch_npu

    gc.collect()
    torch_npu.npu.empty_cache()
    free_before, total = torch_npu.npu.mem_get_info(device)
    target = max(0, int(free_before) - reserve_bytes)
    chunks = []
    allocated = 0
    allocation_error = None
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    while allocated < target:
        current = min(chunk_bytes, target - allocated)
        current -= current % element_size
        if current == 0:
            break
        try:
            chunk = torch.empty(current // element_size, dtype=torch.bfloat16, device=device)
            chunk.fill_(float("nan"))
            chunks.append(chunk)
            allocated += current
        except RuntimeError as error:
            allocation_error = str(error)
            break
    torch_npu.npu.synchronize()
    chunk_count = len(chunks)
    del chunks
    gc.collect()
    free_after_release, _ = torch_npu.npu.mem_get_info(device)
    return {
        "strategy": "poison_allocatable_hbm_then_reuse_caching_allocator",
        "free_bytes_before": int(free_before),
        "total_bytes": int(total),
        "reserve_bytes": reserve_bytes,
        "target_bytes": target,
        "poisoned_bytes": allocated,
        "chunk_bytes": chunk_bytes,
        "chunk_count": chunk_count,
        "free_bytes_after_release": int(free_after_release),
        "allocation_error": allocation_error,
        "passed": allocation_error is None and allocated == target,
        "limitation": (
            "Only memory safely allocatable by this process is poisoned; driver, SHMEM, "
            "and other-process allocations are excluded. Red zones cover adjacency."
        ),
    }


def nonfinite_summary(tensor: torch.Tensor, *, max_locations: int = 32) -> Dict[str, Any]:
    """Return exact nonfinite counts and a bounded set of multidimensional locations."""
    flat = tensor.detach().reshape(-1)
    nan_count = 0
    positive_inf_count = 0
    negative_inf_count = 0
    locations = []
    columns = int(tensor.shape[-1]) if tensor.dim() else 1
    for start in range(0, flat.numel(), _SCAN_CHUNK_ELEMENTS):
        chunk = flat[start:start + _SCAN_CHUNK_ELEMENTS]
        nan_mask = torch.isnan(chunk)
        positive_inf_mask = torch.isposinf(chunk)
        negative_inf_mask = torch.isneginf(chunk)
        nan_count += int(nan_mask.sum().item())
        positive_inf_count += int(positive_inf_mask.sum().item())
        negative_inf_count += int(negative_inf_mask.sum().item())
        if len(locations) < max_locations:
            mask = nan_mask | positive_inf_mask | negative_inf_mask
            indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
            for local_index in indices[:max_locations - len(locations)].cpu().tolist():
                flat_index = start + int(local_index)
                value = float(chunk[int(local_index)].float().item())
                locations.append({
                    "flat_index": flat_index,
                    "row": flat_index // columns,
                    "column": flat_index % columns,
                    "kind": "nan" if math.isnan(value) else ("positive_inf" if value > 0 else "negative_inf"),
                })
    total_nonfinite = nan_count + positive_inf_count + negative_inf_count
    return {
        "elements": flat.numel(),
        "nan": nan_count,
        "positive_inf": positive_inf_count,
        "negative_inf": negative_inf_count,
        "total_nonfinite": total_nonfinite,
        "first_locations": locations,
        "passed": total_nonfinite == 0,
    }


def sampled_fingerprint(tensor: torch.Tensor, *, max_elements: int = 1 << 20) -> Dict[str, Any]:
    """Create a bounded mutation fingerprint without cloning a model-sized input."""
    if max_elements <= 0:
        raise ValueError("max_elements must be positive")
    flat = tensor.detach().reshape(-1)
    step = max(1, math.ceil(flat.numel() / max_elements))
    sample = flat[::step][:max_elements].cpu().contiguous()
    return {
        "elements": flat.numel(),
        "sampled_elements": sample.numel(),
        "sample_step": step,
        "sha256": hashlib.sha256(sample.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def compare_fingerprints(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "before": before,
        "after": after,
        "passed": before == after,
    }
