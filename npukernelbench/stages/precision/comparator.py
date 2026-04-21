"""Tensor precision comparison — torch imported lazily inside functions."""

from __future__ import annotations

from typing import Tuple


def check_precision(
    ref_output,
    test_output,
    max_abs_error: float = 0.001,
    max_rel_error: float = 0.001,
) -> Tuple[int, "torch.Tensor", "torch.Tensor"]:
    """Compare reference and test outputs element-wise.

    Returns (is_accurate: 0|1, abs_diff_flat, rel_diff_flat).
    """
    import torch

    outputs = [ref_output] if not isinstance(ref_output, (list, tuple)) else list(ref_output)
    outputs_new = [test_output] if not isinstance(test_output, (list, tuple)) else list(test_output)

    all_abs, all_rel = [], []
    accurate = True

    for out, out_new in zip(outputs, outputs_new):
        if out_new.dtype == torch.bool:
            out = out.to(torch.int)
            out_new = out_new.to(torch.int)

        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)

        all_abs.append(abs_diff.reshape(-1))
        all_rel.append(rel_diff.reshape(-1))

        if (torch.any(~torch.isfinite(out)) or
            torch.any(~torch.isfinite(out_new)) or
            torch.any(~torch.isfinite(abs_diff)) or
            torch.any(~torch.isfinite(rel_diff))):
            accurate = False

        if ((abs_diff > max_abs_error) & (rel_diff > max_rel_error)).any():
            accurate = False

    all_abs_cat = torch.cat(all_abs)
    all_rel_cat = torch.cat(all_rel)

    return (1 if accurate else 0), all_abs_cat, all_rel_cat
