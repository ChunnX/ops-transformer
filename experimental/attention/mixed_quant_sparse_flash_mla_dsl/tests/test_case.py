"""MQSMLA NPU comparison cases (pytest entry: tests/test_mqsmla.py).

参数字典驱动：生成完整多 batch 输入 → CPU golden → NPU → 精度比较。
"""

from __future__ import annotations

import torch

from mqsmla_golden import (
    C, make_inputs, mqsmla_cpu_benchmark, pack_pa_layout,
)
from result_compare_method import check_result


MODES = ("ORI_SPARSE", "ORI_CMP_SPARSE")

DEFAULTS = {
    "quant_mode": 1,
    "B": 1,
    "S1": 2,
    "cu_seqlens_q": None,
    "seqused_q": None,
    "S2": 8192,
    "S2C": 2048,
    "K1": 128,
    "K2": 512,
    "omit_win_topk_length": False,
    "omit_cmp_topk_length": False,
    "win_topk_length": None,
    "cmp_topk_length": None,
    "ori_kv_topk_mode": "fullK",
    "cmp_kv_topk_mode": "fullK",
    "block_size1": 64,
    "block_size2": 64,
    "seed": 7,
    "dist": "norm",
    "kv_axis0_noncontiguous": False,
}


def fill_defaults(params):
    p = dict(DEFAULTS)
    p.update(params)
    return p


def build_case(params):
    """Generate one complete multi-batch case; CPU/NPU execution is separate."""
    p = fill_defaults(params)
    mode = p["template_run_mode"]
    if mode not in MODES:
        raise ValueError(f"template_run_mode must be one of {MODES}, got {mode!r}")
    _, ref = make_inputs(
        scenario=getattr(C.Scenario, mode), quant_mode=int(p["quant_mode"]),
        batch_size=int(p["B"]), seqlen_q=p["S1"], seqlen_kv=p["S2"],
        cmp_seqlen=p["S2C"], ori_topk=int(p["K1"]), cmp_topk=int(p["K2"]),
        seed=int(p["seed"]), dist=p["dist"],
        cu_seqlens_q=p["cu_seqlens_q"], seqused_q=p["seqused_q"],
        ori_topk_length=p['win_topk_length'],
        cmp_topk_length=p['cmp_topk_length'],
        ori_kv_topk_mode=p["ori_kv_topk_mode"],
        cmp_kv_topk_mode=p["cmp_kv_topk_mode"],
    )
    for side, ref_side in (("win", "ori"), ("cmp", "cmp")):
        if p.get(f"omit_{side}_topk_length") and ref.get(f"{ref_side}_topk_length") is not None:
            width = ref[f"{ref_side}_sparse_indices"].shape[-1]
            if bool((ref[f"{ref_side}_topk_length"] != width).any()):
                raise ValueError(f"omitting {side}_topk_length requires all K indices to be valid")
    if p.get("softmax_scale") is not None:
        ref["softmax_scale"] = float(p["softmax_scale"])
    pa = pack_pa_layout(ref, block_size=int(p["block_size1"]),
                        cmp_block_size=int(p["block_size2"]), seed=int(p["seed"]) + 97)
    case = {
        "mode": mode, "quant_mode": int(p["quant_mode"]),
        "b": ref["B"], "t1": ref["T1"], "n1": 64, "d": C.D,
        "q": ref["q"], "sinks": ref["sinks"], "scale": ref["softmax_scale"],
        "cu_seqlens_q": ref["cu_seqlens_q"],
        "seqused_q": ref["seqused_q"],
        "pa": pa["ori"], "pa_cmp": pa.get("cmp"),
        "idx_ori": ref["ori_sparse_indices"], "len_ori": ref["ori_topk_length"],
        "idx_cmp": ref.get("cmp_sparse_indices"), "len_cmp": ref.get("cmp_topk_length"),
    }
    return case, ref


def _npu_available():
    try:
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())
    except Exception:
        return False


def prepare_case(params):
    """Prepare CPU inputs and golden once; does not require an NPU or wheel."""
    p = fill_defaults(params)
    case, ref = build_case(p)
    expected, expected_lse = mqsmla_cpu_benchmark(ref, return_softmax_lse=True)
    return p, case, expected.reshape(-1, C.D), expected_lse


def run_params(params):
    run_prepared(*prepare_case(params))


def run_prepared(p, case, expected, expected_lse):
    """Execute prepared inputs; never regenerate inputs or golden."""
    mode = case["mode"]
    quant_mode = case["quant_mode"]
    if not _npu_available():
        raise RuntimeError("torch_npu/device unavailable")
    pa_ori = case["pa"]
    win_kv = _pa_tensor(
        pa_ori, C.KV_ROW_BYTES_WIN,
        axis0_noncontiguous=p["kv_axis0_noncontiguous"])
    pa_cmp = case["pa_cmp"]
    has_cmp = mode == "ORI_CMP_SPARSE"
    kwargs = {
        "q": case["q"].npu(),
        "win_kv": win_kv,
        "cmp_kv": _pa_tensor(
            pa_cmp, C.KV_ROW_BYTES_CMP,
            axis0_noncontiguous=p["kv_axis0_noncontiguous"]) if has_cmp else None,
        "win_sparse_indices": case["idx_ori"].reshape(case["t1"], 1, -1).npu(),
        "cmp_sparse_indices": case["idx_cmp"].reshape(case["t1"], 1, -1).npu() if has_cmp else None,
        "win_block_table": pa_ori["block_table"].npu(),
        "cmp_block_table": pa_cmp["block_table"].npu() if has_cmp else None,
        "cu_seqlens_q": case["cu_seqlens_q"].npu(),
        "seqused_q": case["seqused_q"].npu() if case.get("seqused_q") is not None else None,
        "seqused_win_kv": None,
        "seqused_cmp_kv": None,
        "win_topk_length": case["len_ori"].reshape(case["t1"], 1).npu(),
        "cmp_topk_length": (case["len_cmp"].reshape(case["t1"], 1).npu()
                            if has_cmp else None),
        "sinks": case["sinks"].npu(),
        "metadata": None,
        "quant_mode": quant_mode,
        "softmax_scale": case["scale"],
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "return_softmax_lse": p.get("return_softmax_lse", False),
    }
    for side in ("win", "cmp"):
        kv = kwargs[f"{side}_kv"]
        if kv is not None:
            print(f"{side}_kv shape={tuple(kv.shape)} stride={kv.stride()} "
                  f"storage_offset={kv.storage_offset()}")
    actual, actual_lse = _call_public(
        kwargs, mode, case["t1"],
        omit_win_length=p.get("omit_win_topk_length", False),
        omit_cmp_length=p.get("omit_cmp_topk_length", False))
    if kwargs["return_softmax_lse"]:
        torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
    result, percent = check_result(expected, actual)
    assert result == "Pass", f"strict BF16 comparator failed: {percent:.6f}%"
    # 大prefill输出可达16GiB，误差统计不要同时物化三份全量FP32输出。
    actual_flat, expected_flat = actual.reshape(-1), expected.reshape(-1)
    max_abs = 0.0
    for start in range(0, actual_flat.numel(), 1 << 20):
        diff = (actual_flat[start:start + (1 << 20)].float()
                - expected_flat[start:start + (1 << 20)].float()).abs()
        chunk_max = diff.max().item()
        max_abs = chunk_max if chunk_max != chunk_max else max(max_abs, chunk_max)
    print(f"[mqsmla test_case] NPU {mode} quant{quant_mode} B={case['b']} "
          f"T1={case['t1']} OK: max_abs={max_abs:.6g}")


def _pa_tensor(pa, row_bytes, *, axis0_noncontiguous=False):
    """先搬完整含 padding 的 storage，再在 NPU 上切视图，保留真实页距。"""
    row = pa["phys"]["row"]
    shape = (pa["num_phys_blocks"], pa["block_size"], 1, row_bytes)
    if not axis0_noncontiguous:
        return row.reshape(shape).npu()
    page_bytes = pa["block_size"] * row_bytes
    stride0 = page_bytes + C.KV_PAGE_PADDING_BYTES
    strides = (stride0, row_bytes, row_bytes, 1)
    # 0xA5 毒化页间无用数据；golden 保留原始逻辑 KV，不读取这份 storage。
    backing = torch.full((shape[0] * stride0,), 0xA5, dtype=torch.uint8)
    cpu_view = backing.as_strided(shape, strides)
    cpu_view.copy_(row.reshape(shape))
    result = backing.npu().as_strided(shape, strides)
    assert result.stride() == strides and result.storage_offset() == 0
    if shape[0] > 1:
        assert not result.is_contiguous()
    return result


def _call_public(kwargs, mode, t1, *, omit_win_length=False, omit_cmp_length=False):
    """Run metadata and attention through the installed wheel's torch.ops."""
    # Explicit import surfaces DSL registration failures instead of testing another implementation.
    import cann_ops_transformer.ops.attention.mixed_quant_sparse_flash_mla_dsl  # noqa: F401

    mixed_quant_sparse_flash_mla = torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla
    mixed_quant_sparse_flash_mla_metadata = torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla_metadata

    kwargs = dict(kwargs)
    if kwargs.get("metadata") is None:
        # Metadata also requires a zero length tensor for the inactive cmp side.
        cmp_lengths = kwargs["cmp_topk_length"]
        if mode == "ORI_SPARSE":
            cmp_lengths = torch.zeros((t1, 1), dtype=torch.int32,
                                      device=kwargs["q"].device)
        kwargs["metadata"] = mixed_quant_sparse_flash_mla_metadata(
            kwargs["win_topk_length"], cmp_lengths,
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            num_heads_q=C.N1, num_heads_kv=1, head_dim=C.D,
            quant_mode=kwargs["quant_mode"], has_cmp_kv=(mode == "ORI_CMP_SPARSE"))
    # Metadata keeps its required length inputs; attention can omit either side.
    if omit_win_length:
        kwargs.pop("win_topk_length")
    if omit_cmp_length:
        kwargs.pop("cmp_topk_length")
    want_lse = kwargs.get("return_softmax_lse", False)
    out, lse = mixed_quant_sparse_flash_mla(**kwargs)
    if want_lse:
        assert torch.isfinite(lse).all()
    else:
        assert lse.shape == (0,) and lse.dtype == torch.float32
    output = out.cpu().reshape(-1, C.D)
    return output, lse.cpu()
