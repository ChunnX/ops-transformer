"""Shared workloads and operator adapters for the MLA FD performance suite."""

import os
import json
from collections.abc import Mapping
from itertools import islice, product

KV_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
             4096, 8192, 16384, 32768, 65536, 131072)
KV_BOUNDARIES = (127, 129, 255, 257, 511, 513, 1023, 1025, 131071)
TOPK_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
QUERY_VALUES = (2, 4, 8, 16, 17, 32, 33, 64, 128, 512, 2048)
HEAD_VALUES = (1, 2, 4, 8, 16, 32, 64, 65, 96, 128)
MODES = ("SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE")
LAYOUTS = (("BSND", "BSND"), ("TND", "TND"),
           ("BSND", "PA_BBND"), ("TND", "PA_BBND"))


def _tops(length):
    return tuple(k for k in TOPK_VALUES if k <= length)


def _shapes(mode, batch, heads, query, kv, suite, **overrides):
    if query > kv:
        return
    has_cmp = mode in ("HCA", "CSA", "ORI_CMP_SPARSE")
    ratios = (4, 128) if has_cmp else (1,)
    for ratio in ratios:
        ckv, residual = divmod(kv, ratio) if has_cmp else (0, 0)
        ori_ks = _tops(kv) if mode in ("ORI_SPARSE", "ORI_CMP_SPARSE") else (0,)
        cmp_ks = (_tops(ckv) or (1,)) if mode in ("CSA", "ORI_CMP_SPARSE") else (0,)
        for k1, k in product(ori_ks, cmp_ks):
            case = dict(
                suite=suite, phase="decode" if query == 1 else "prefill",
                template_mode=mode, B=batch, N1=heads, S1=query, S2=kv,
                N2=1, D=512, K1=k1, K=k, CKV=ckv, cmp_ratio=ratio,
                residual=residual, layout_q="BSND", layout_kv="BSND",
                ori_mask_mode=4, cmp_mask_mode=3 if has_cmp else 0,
                ori_win_left=127, ori_win_right=0, block_size=128,
                page_order="contiguous", index_distribution="contiguous",
                actual_k1=k1, actual_k=k, return_softmax_lse=False,
                unsupported_reason="",
            )
            # Zero complete compressed tokens cannot supply a positive sparse TopK.
            # Keep this workload visible; never change KV or manufacture a token.
            if has_cmp and ckv == 0:
                case["unsupported_reason"] = "KV < cmp_ratio: no complete compressed token"
            case.update(overrides)
            yield case


def _representative(mode, batch, heads, query, kv, suite, **overrides):
    """Specialists keep one TopK capacity; the main matrix scans all capacities."""
    for case in _shapes(mode, batch, heads, query, kv, suite, **overrides):
        if case["K1"] == (max(_tops(kv)) if case["K1"] else 0) and case["K"] == (
            max(_tops(case["CKV"]) or (1,)) if case["K"] else 0
        ):
            yield case


def _matrix():
    for mode, batch, heads, query, kv in product(
        MODES, (1, 8, 32), (32, 64, 128), QUERY_VALUES, KV_VALUES
    ):
        suite = "PF-P0" if query in (16, 512, 2048) and heads in (32, 128) else "PF-P1"
        yield from _shapes(mode, batch, heads, query, kv, suite)
    for mode, batch, heads, kv in product(MODES, (1, 2, 8, 32), (32, 64, 128), KV_VALUES):
        yield from _shapes(mode, batch, heads, 1, kv, "DC")
    for mode, query, heads in product(MODES, (1, 16), HEAD_VALUES):
        yield from _representative(mode, 1, heads, query, 16384, "HEAD")
    for mode, query, kv in product(MODES, (1, 16), KV_BOUNDARIES):
        yield from _representative(mode, 1, 128, query, kv, "KV-BOUNDARY")
    # Also straddle compressed S2 block boundaries and each requested remainder.
    for mode, ratio, ckv in product(("HCA", "CSA", "ORI_CMP_SPARSE"), (4, 128),
                                    (1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024)):
        for residual in (0, 1, ratio - 1):
            kv = ckv * ratio + residual
            if kv > 131072:
                continue
            for case in _representative(mode, 1, 128, 1, kv, "CMP-BOUNDARY"):
                if case["cmp_ratio"] == ratio:
                    yield case
    for mode, query, kv, mask in product(MODES, (1, 16, 512),
                                        (128, 2048, 16384, 131072), (0, 3, 4)):
        cmp_masks = (0, 3) if mode == "ORI_CMP_SPARSE" else (
            (3,) if mode in ("CSA", "HCA") else (0,))
        for cmp_mask in cmp_masks:
            yield from _representative(
                mode, 1, 64, query, kv, "MASK", ori_mask_mode=mask,
                cmp_mask_mode=cmp_mask, ori_win_left=127 if mask == 4 else -1,
                ori_win_right=0 if mask == 4 else -1,
            )
    for mode, left, right in product(MODES, (0, 1, 16, 127, 128, 512, -1), (0, 1, -1)):
        yield from _representative(mode, 1, 64, 16, 16384, "WINDOW",
                                   ori_win_left=left, ori_win_right=right)
    for mode in ("CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"):
        for case in _representative(mode, 1, 64, 16, 16384, "ACTUAL-TOPK",
                                    ori_mask_mode=0, cmp_mask_mode=0,
                                    ori_win_left=-1, ori_win_right=-1):
            # CSA only permits cmp_mask_mode=3; vary original sparse paths here.
            if mode == "CSA":
                case["cmp_mask_mode"] = 3
            for divisor in (1024, 4, 2, 1):
                yield dict(case, actual_k1=max(1, case["K1"] // divisor) if case["K1"] else 0,
                           actual_k=max(1, case["K"] // divisor) if case["K"] else 0)
    for mode, distribution in product(MODES, ("contiguous", "random", "cross_page")):
        yield from _representative(mode, 1, 64, 16, 16384, "INDEX",
                                   index_distribution=distribution)
    for mode, query, batch, heads, kv in product(MODES, (1, 512), (1, 8), (64, 128),
                                                 (128, 2048, 16384, 65536, 131072)):
        if query == 512 and (batch, heads, kv) not in ((8, 64, 16384), (8, 64, 131072)):
            continue
        if query == 1 and (batch, heads) not in ((1, 128), (8, 64)):
            continue
        for lq, lk in LAYOUTS[1:]:
            for page in (("contiguous", "random") if lk == "PA_BBND" else ("contiguous",)):
                yield from _representative(mode, batch, heads, query, kv, "LAYOUT",
                                           layout_q=lq, layout_kv=lk, page_order=page)
    for mode, size in product(MODES, (64, 128, 256)):
        yield from _representative(mode, 1, 128, 1, 16384, "PAGE-SIZE",
                                   layout_kv="PA_BBND", block_size=size)
    for mode, query in product(MODES, (1, 512)):
        yield from _representative(mode, 1, 128, query, 131072, "LSE", return_softmax_lse=True)


def iter_fd_cases(phase=None, modes=MODES, layouts=None):
    for case in _matrix():
        if phase and case["phase"] != phase:
            continue
        if case["template_mode"] not in modes:
            continue
        if layouts and (case["layout_q"], case["layout_kv"]) not in layouts:
            continue
        keys = ("suite", "template_mode", "B", "N1", "S1", "S2", "cmp_ratio",
                "K1", "K", "residual", "layout_q", "layout_kv", "ori_mask_mode", "cmp_mask_mode",
                "ori_win_left", "ori_win_right", "block_size", "page_order",
                "index_distribution", "actual_k1", "actual_k", "return_softmax_lse")
        case["case_id"] = "_".join(str(case[key]) for key in keys)
        yield case


def adapt_case(case, operator, variant):
    import torch

    params = dict(case)
    b, q, kv, ckv = (case[k] for k in ("B", "S1", "S2", "CKV"))
    has_cmp = case["template_mode"] in ("HCA", "CSA", "ORI_CMP_SPARSE")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(variant, torch.uint8)
    params.update(
        testcase_name=case["case_id"], Testcase_Name=case["case_id"], variant=variant,
        template_run_mode=case["template_mode"], actlen_mode="full", S1EQS2=False,
        q_type=dtype if operator != "mqsmla" else torch.bfloat16,
        ori_kv_type=dtype if operator == "smla" else torch.float8_e4m3fn,
        cmp_kv_type=(dtype if operator == "smla" else torch.float8_e4m3fn) if has_cmp else None,
        softmax_scale=512 ** -0.5, isSink=True, topk_value_mode=1,
        batch_consistency=False, T1=b*q, T2=b*kv, T3=b*ckv,
        block_num1=b*((kv+case["block_size"]-1)//case["block_size"]),
        block_num2=b*((ckv+case["block_size"]-1)//case["block_size"]) if has_cmp else None,
        block_size1=case["block_size"], block_size2=case["block_size"] if has_cmp else None,
        seqused_q=[q]*b, seqused_ori_kv=[kv]*b, seqused_cmp_kv=[ckv]*b if has_cmp else None,
        cu_seqlens_q=[i*q for i in range(b+1)],
        cu_seqlens_ori_kv=[i*kv for i in range(b+1)],
        cu_seqlens_cmp_kv=[i*ckv for i in range(b+1)] if has_cmp else None,
        cmp_residual_kv=[case["residual"]]*b if has_cmp else None,
        ori_topk=0 if operator == "smla" else case["K1"], cmp_topk=case["K"],
        ori_kv_topk_mode="fullK", cmp_kv_topk_mode="fullK",
        ori_sparse_indices_mode="full", cmp_sparse_indices_mode="full",
        ori_topk_length=None, cmp_topk_length=None,
        q_datarange=[-2, 2], ori_kv_datarange=[-2, 2], cmp_kv_datarange=[-2, 2],
    )
    if operator == "qsmla":
        params.update(quant_mode=1, q_descale_val=1.0, ori_kv_descale_val=1.0, cmp_kv_descale_val=1.0)
    elif operator == "mqsmla":
        params.update(quant_mode=int(variant[-1]), rope_head_dim=64, tile_size=64)
    # Every value is an independent one-element parameter axis, including lists.
    return {key: [value] for key, value in params.items()}


class ParamMatrix(Mapping):
    """Reiterable lazy mapping; explicit slicing keeps pytest collection bounded."""

    def __init__(self, operator, start=0, limit=None):
        self.operator, self.start, self.limit = operator, start, limit
        self.variants = {"smla": ("fp16", "bf16"), "qsmla": ("hif8",),
                         "mqsmla": ("quant1", "quant2")}[operator]

    def items(self):
        stop = None if self.limit is None else self.start + self.limit
        manifest = os.environ.get("FD_CASE_MANIFEST")
        if manifest:
            with open(manifest, encoding="utf-8") as source:
                cases = json.load(source)
        else:
            cases = islice(iter_fd_cases(), self.start, stop)
        for case in cases:
            for variant in self.variants:
                yield case["case_id"] + "/" + variant, adapt_case(case, self.operator, variant)

    def values(self):
        return (params for _, params in self.items())

    def __iter__(self):
        return (key for key, _ in self.items())

    def __len__(self):
        return sum(1 for _ in self.items())

    def __getitem__(self, key):
        for name, params in self.items():
            if name == key:
                return params
        raise KeyError(key)


def build_test_params(operator="smla", limit=None):
    start = int(os.environ.get("FD_CASE_START", "0"))
    configured = os.environ.get("FD_CASE_LIMIT")
    if configured is not None:
        limit = int(configured)
    if start < 0 or (limit is not None and limit <= 0):
        raise ValueError("FD_CASE_START must be >= 0 and FD_CASE_LIMIT must be > 0")
    return ParamMatrix(operator, start, limit)
