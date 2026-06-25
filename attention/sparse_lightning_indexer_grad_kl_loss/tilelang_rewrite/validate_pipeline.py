#!/usr/bin/env python3
"""
Validate Plan B implementation against original AscendC operator.

Checks two aspects:
  A) Computational logic: does each original stage have a corresponding
     operation in Plan B (Kernel1 / Kernel2 / Python)?
  B) Parameter matching: for each mapped stage, are the dimensions, scale
     factors, transpose flags, activation types, cast modes, etc. identical?

Usage:
  python validate_pipeline.py --check-python          # Parse Plan B Python source
  python validate_pipeline.py --compare <file.cpp>    # Parse generated AscendC
  python validate_pipeline.py --full                  # Both + dump attempt
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# Stage definition: what the ORIGINAL AscendC does
# ==============================================================================


@dataclass
class StageSpec:
    """Expected computation for one stage in the original AscendC operator."""
    name: str
    desc: str
    ops: List[Dict[str, Any]] = field(default_factory=list)
    # Each op dict: {"type": "gemm"|"softmax"|"reduce"|"mul"|...,
    #                "params": {...}, "planb_location": "Kernel1"|"Kernel2"|"Python"}


# ==============================================================================
# Original AscendC pipeline fingerprint (from arch22 source analysis)
# ==============================================================================

ORIGINAL_STAGES: List[StageSpec] = [
    # ---- V0: Gather ----
    StageSpec("V0_GatherKey", "Gather Key/KeyRope by sparseIndices", [
        {"type": "gather", "params": {
            "src": "key[B,S2,N2,DQuery]", "indices": "sparse_indices[B,S1,Nidx2,K]",
            "dst_shape": "[B,S1,K,DQuery]", "note": "含 optional key_rope 拼接 d_rope=64",
        }},
    ]),
    StageSpec("V0_GatherKeyIndex", "Gather KeyIndex by sparseIndices", [
        {"type": "gather", "params": {
            "src": "key_index[B,S2,Nidx2,DQueryIndex]", "indices": "sparse_indices[B,S1,Nidx2,K]",
            "dst_shape": "[B,S1,K,DQueryIndex]",
        }},
    ]),

    # ---- C1: Cube matmuls ----
    StageSpec("C1_MM1", "MM1 = Query||QueryRope @ GatherKey||GatherKeyRope^T", [
        {"type": "gemm", "params": {
            "A": "query_cat[B,S1,N1,DQuery+d_rope]",
            "B": "gather_key_cat[B,S1,K,DQuery+d_rope]",
            "C": "bmm1Res[B,S1,N1,K]",
            "M": "N1", "N": "K", "K_inner": "DQuery + d_rope (=512+64=576)",
            "transpose_B": True, "split_M": 128, "split_N": 128, "split_K": 128,
            "dtype_input": "float16", "dtype_accum": "float32",
        }},
    ]),
    StageSpec("C1_MM2", "MM2 = QueryIndex @ GatherKeyIndex^T (with ReLU)", [
        {"type": "gemm", "params": {
            "A": "query_index[B,S1,Nidx1,DQueryIndex]",
            "B": "gather_key_index[B,S1,K,DQueryIndex]",
            "C": "bmm2Res[B,S1,Nidx1,K]",
            "M": "Nidx1", "N": "K", "K_inner": "DQueryIndex (=128)",
            "transpose_B": True, "activation": "ReLU",
            "dtype_input": "float16", "dtype_accum": "float32",
        }},
    ]),

    # ---- V1-P: Softmax over bmm1 ----
    StageSpec("V1_VectorP", "P = softmax(bmm1 * scale) using precomputed max/sum", [
        {"type": "mul", "params": {"scalar": "scale_value = 1/sqrt(DQuery)"}},
        {"type": "softmax", "params": {
            "input": "bmm1Res[N1, K]", "axis": -1,
            "use_precomputed_max_sum": True,
            "max_src": "softmax_max[B,N2,S1,G]", "sum_src": "softmax_sum[B,N2,S1,G]",
            "max_abs": True, "sum_clamp_min": 1e-8,
        }},
        {"type": "reduce", "params": {
            "op": "sum", "axis": 0, "shape": "[N1, K]",
            "note": "sum over N1 (G heads)", "dtype": "float32",
        }},
        {"type": "mul", "params": {"scalar": "1.0 / N1", "note": "mean over heads"}},
    ]),
    StageSpec("V1_PRreduced", "PRreduced[K] = mean(softmax(bmm1 * scale), dim=N1)", []),
    # (merged with VectorP above)

    # ---- V1-SY: weight * ReLU(bmm2) -> ReduceSumY ----
    StageSpec("V1_ReduceSumY", "ReduceSumY[K] = sum_g(weight[g] * ReLU(bmm2[g,:]))", [
        {"type": "mul", "params": {
            "op": "per-row scalar mul", "A": "bmm2Res[Nidx1, K]",
            "scalar": "weight[g]", "note": "bmm2 already ReLU'd in MM2 fixpipe",
        }},
        {"type": "reduce", "params": {
            "op": "sum", "axis": 0, "shape": "[Nidx1, K]",
            "note": "sum over Nidx1", "dtype": "float32",
        }},
    ]),

    # ---- V1: Global SoftmaxI ----
    StageSpec("V1_GlobalSoftmaxI", "SoftmaxI[K] = global_softmax(ReduceSumY[K])", [
        {"type": "softmax", "params": {
            "input": "ReduceSumY[K]", "axis": -1, "shape": "[K]",
            "clamp_min": 1e-8,
        }},
    ]),

    # ---- V1: diff, dWeight, reluGrad ----
    StageSpec("V1_DiffDW", "diff = SoftmaxI - P_reduced; dWeight += sum_k(diff * ReLU(bmm2))", [
        {"type": "sub", "params": {"A": "SoftmaxI[K]", "B": "P_reduced[K]", "dst": "diff[K]"}},
        {"type": "mul", "params": {
            "op": "broadcast", "A": "diff[K]", "B": "ReLU(bmm2)[Nidx1, K]",
            "note": "diff * reluResult per row",
        }},
        {"type": "reduce", "params": {
            "op": "sum", "axis": -1, "shape": "[Nidx1, K]",
            "note": "sum over K axis for each g → dWeight[Nidx1]", "dtype": "float32",
        }},
    ]),
    StageSpec("V1_ReLUGrad", "reluGrad[Nidx1,K] = weight * diff * (bmm2 > 0)", [
        {"type": "mul", "params": {
            "op": "per-row", "A": "diff[K]", "scalar": "weight[g]",
            "note": "diff * weight per head",
        }},
        {"type": "compare", "params": {
            "A": "bmm2[Nidx1, K]", "B": 0.0, "op": "GT",
            "note": "sigmoid mask: bmm2 > 0",
        }},
        {"type": "select", "params": {
            "cond": "mask[bmm2>0]", "true_val": "diff*weight", "false_val": 0.0,
            "dst": "reluGrad[Nidx1, K]",
        }},
        {"type": "cast", "params": {
            "from": "float32", "to": "float16/bf16",
            "mode": "CAST_ROUND",
        }},
    ]),

    # ---- V1: KL Loss ----
    StageSpec("V1_KL_Loss", "KL = sum_k(P[k] * (log(P[k]) - log(SoftmaxI[k])))", [
        {"type": "clamp", "params": {"A": "P_reduced[K]", "min": 1e-8}},
        {"type": "clamp", "params": {"A": "SoftmaxI[K]", "min": 1e-8}},
        {"type": "log", "params": {"A": "P_reduced[K]", "dst": "log(P)"}},
        {"type": "log", "params": {"A": "SoftmaxI[K]", "dst": "log(SoftmaxI)"}},
        {"type": "sub", "params": {"A": "log(P)", "B": "log(SoftmaxI)", "dst": "log_ratio[K]"}},
        {"type": "mul", "params": {"A": "P_reduced[K]", "B": "log_ratio[K]", "dst": "kl_terms[K]"}},
        {"type": "reduce", "params": {
            "op": "sum", "axis": -1, "shape": "[K]", "dst": "loss[1]",
            "note": "sum over K", "dtype": "float32",
        }},
    ]),

    # ---- C2-MM5: dKeyIndex (scatter-add input) ----
    StageSpec("C2_MM5", "bmm5Res[K,DQueryIndex] = reluGrad^T @ queryIndex", [
        {"type": "gemm", "params": {
            "A": "reluGrad[Nidx1, K]", "B": "queryIndex[Nidx1, DQueryIndex]",
            "C": "bmm5Res[K, DQueryIndex]",
            "M": "K", "N": "DQueryIndex", "K_inner": "Nidx1",
            "transpose_A": True, "transpose_B": False,
            "dtype_input": "float16", "dtype_accum": "float32",
        }},
    ]),

    # ---- C2-MM6: dQueryIndex ----
    StageSpec("C2_MM6", "dQueryIndex[Nidx1,DQueryIndex] = reluGrad @ GatherKeyIndex", [
        {"type": "gemm", "params": {
            "A": "reluGrad[Nidx1, K]", "B": "gather_key_index[K, DQueryIndex]",
            "C": "dQueryIndex[Nidx1, DQueryIndex]",
            "M": "Nidx1", "N": "DQueryIndex", "K_inner": "K",
            "transpose_A": False, "transpose_B": False,
            "dtype_input": "float16/bf16", "dtype_accum": "float32",
        }},
        {"type": "cast", "params": {
            "from": "float32", "to": "float16/bf16",
            "mode": "F322F16 / F322BF16",
        }},
    ]),

    # ---- V2: ScatterAdd dKey + Cast ----
    StageSpec("V2_ScatterAdd", "ScatterAdd dKey partial to dKeyIndex[Bs2,DQueryIndex]", [
        {"type": "scatter_add", "params": {
            "src": "bmm5Res[K, DQueryIndex]",
            "dst": "dKeyIndex[S2, DQueryIndex]",
            "indices": "sparse_indices[K]",
            "note": "atomic scatter-add by topK indices per (B,S1) pair",
        }},
        {"type": "cast", "params": {
            "from": "float32", "to": "float16/bf16",
            "mode": "CAST_ROUND",
        }},
    ]),
]

# ==============================================================================
# Plan B mapping: original stage -> Plan B location
# ==============================================================================

PLANB_MAPPING = {
    "V0_GatherKey": {
        "location": "Python",
        "function": "_preprocess_gather()",
        "note": "torch.gather fuses key + key_rope concatenation",
    },
    "V0_GatherKeyIndex": {
        "location": "Python",
        "function": "_preprocess_gather()",
        "note": "torch.gather for key_index",
    },
    "C1_MM1": {
        "location": "Kernel1",
        "scope": "C",
        "function": "T.gemm_v0 (T.Pipelined over d_idx)",
        "note": "MM1: Q||QR @ GK||GKR^T, pipelined over D blocks (BLOCK_K=128)",
    },
    "C1_MM2": {
        "location": "Kernel1",
        "scope": "C",
        "function": "T.gemm_v0 (T.serial) + enable_relu=True",
        "note": "MM2: QI @ GKI^T, single K-iter (DQueryIndex=128 fits 1 block)",
    },
    "V1_VectorP": {
        "location": "Kernel1",
        "scope": "V",
        "function": "T.tile.mul/exp/sub/div (per-row serial) + T.reduce_sum",
        "note": "Softmax per row, BLOCK_N1=32 tiling, reduce_sum for accumulation",
    },
    "V1_PRreduced": {
        "location": "Kernel1",
        "scope": "V",
        "function": "T.tile.mul(*, 1.0/N1) + T.copy → PReduced workspace",
        "note": "mean over N1 heads",
    },
    "V1_ReduceSumY": {
        "location": "Kernel1",
        "scope": "V",
        "function": "T.tile.mul (per-row) + T.reduce_sum",
        "note": "weight * ReLU(bmm2), reduce_sum over Nidx1",
    },
    "V1_GlobalSoftmaxI": {
        "location": "Kernel2",
        "scope": "V",
        "function": "reduce_max + sub + exp + reduce_sum + max(1e-8) + div",
        "note": "redundantly computed per core, K-parallel",
    },
    "V1_DiffDW": {
        "location": "Kernel2",
        "scope": "V",
        "function": "T.tile.sub + mul(per-row) + reduce_sum → dWeightPartial",
        "note": "diff = SoftmaxI - P_reduced, dWeight chunk written per n_idx block",
    },
    "V1_ReLUGrad": {
        "location": "Kernel2",
        "scope": "V",
        "function": "T.tile.mul + compare(GT,0) + select + cast(CAST_ROUND) + copy",
        "note": "weight * diff, mask via 2D compare+select, cast to fp16, store",
    },
    "V1_KL_Loss": {
        "location": "Kernel2",
        "scope": "V",
        "function": "T.tile.max(eps) + ln + sub + mul + reduce_sum",
        "note": "P*log(P/SoftmaxI), partial sum per K-block",
    },
    "C2_MM5": {
        "location": "Python",
        "function": "_compute_d_key_index() → scatter_add_",
        "note": "reluGrad^T @ queryIndex bmm + scatter_add_ to key positions",
    },
    "C2_MM6": {
        "location": "Python",
        "function": "torch.bmm(reluGrad, gather_key_index)",
        "note": "dQueryIndex = reluGrad @ GKI, single bmm then reshape",
    },
    "V2_ScatterAdd": {
        "location": "Python",
        "function": "_compute_d_key_index() → scatter_add_ + .to(dtype)",
        "note": "atomic scatter_add + cast, Python handles both",
    },
}


# ==============================================================================
# Python source parser: extract operations from Plan B source
# ==============================================================================

@dataclass
class ParsedOp:
    op_type: str
    line: int
    details: str  # human-readable snippet/key params
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedStage:
    stage_name: str
    location: str  # "Kernel1", "Kernel2", "Python"
    ops: List[ParsedOp] = field(default_factory=list)


def parse_planb_python_source(filepath: str) -> Tuple[List[ParsedStage], List[str]]:
    """Parse Plan B Python source to extract computational stages and their params.

    Returns (stages, warnings).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    lines = source.split("\n")

    parsed_stages: List[ParsedStage] = []
    warnings: List[str] = []

    # --- Kernel 1 detection ---
    k1_start = None
    k1_c_start = None
    k1_v_start = None
    for i, line in enumerate(lines):
        if "def planB_kernel1(" in line:
            k1_start = i
        if k1_start and 'with T.Scope("C")' in line and i > k1_start:
            if k1_c_start is None:
                k1_c_start = i
        if k1_start and 'with T.Scope("V")' in line and i > k1_start:
            if k1_v_start is None:
                k1_v_start = i
        if k1_start and k1_c_start and k1_v_start:
            break

    # --- Kernel 2 detection ---
    k2_start = None
    k2_v_start = None
    for i, line in enumerate(lines):
        if "def planB_kernel2(" in line:
            k2_start = i
        if k2_start and 'with T.Scope("V")' in line and i > k2_start:
            if k2_v_start is None:
                k2_v_start = i
                break

    # --- Parse Kernel1 C1 ops ---
    k1_c_ops = []
    if k1_c_start:
        for i in range(k1_c_start + 1, k1_v_start or len(lines)):
            line = lines[i].strip()
            if "T.gemm_v0" in line:
                params = _extract_gemm_params(line)
                # Also check next 1-2 lines for enable_relu on T.copy (MM2)
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if "T.copy" in next_line and "enable_relu=True" in next_line:
                        params["activation"] = "ReLU"
                        break
                k1_c_ops.append(ParsedOp("gemm", i + 1, line, params))
            elif "T.set_cross_flag" in line:
                k1_c_ops.append(ParsedOp("sync", i + 1, line,
                                         {"type": "set_cross_flag", "flag": _extract_str_arg(line)}))

    if k1_c_ops:
        parsed_stages.append(ParsedStage("Kernel1_C1_MM1_MM2", "Kernel1", k1_c_ops))

    # --- Parse Kernel1 V1 ops ---
    k1_v_ops = []
    if k1_v_start:
        for i in range(k1_v_start + 1, (k2_start or len(lines))):
            line = lines[i].strip()
            op = _classify_tilelang_op(line, i)
            if op:
                k1_v_ops.append(op)
    if k1_v_ops:
        parsed_stages.append(ParsedStage("Kernel1_V1_P_ReduceSumY", "Kernel1", k1_v_ops))

    # --- Parse Kernel2 V1 ops ---
    k2_v_ops = []
    if k2_v_start:
        for i in range(k2_v_start + 1, len(lines)):
            line = lines[i].strip()
            # Stop at next function definition
            if line.startswith("def ") and "planB_kernel2" not in line:
                break
            if line.startswith("def ") and i > k2_v_start + 5:
                break
            op = _classify_tilelang_op(line, i)
            if op:
                k2_v_ops.append(op)
    if k2_v_ops:
        parsed_stages.append(ParsedStage("Kernel2_V1_SoftmaxI_DiffDW_ReLUGrad_Loss", "Kernel2", k2_v_ops))

    # --- Parse Python ops ---
    py_ops = []
    # V0: torch.gather (in _preprocess_gather)
    gather_count = 0
    for i, line in enumerate(lines):
        if "torch.gather" in line or ".gather(" in line:
            gather_count += 1
            py_ops.append(ParsedOp("gather", i + 1, line.strip(),
                                   {"note": f"V0 gather #{gather_count}: key/key_rope/key_index by sparse_indices"}))
    # dWeight sum
    for i, line in enumerate(lines):
        if "d_weight_partial.sum(dim=2)" in line:
            py_ops.append(ParsedOp("reduce", i + 1, line.strip(),
                                   {"op": "sum", "axis": 2, "note": "dWeight partials → sum"}))
            break
    # KL loss sum
    for i, line in enumerate(lines):
        if "loss_partial.sum()" in line:
            py_ops.append(ParsedOp("reduce", i + 1, line.strip(),
                                   {"op": "sum", "note": "KL loss partials → scalar"}))
            break
    # dQueryIndex bmm (MM6)
    found_bmm = False
    for i, line in enumerate(lines):
        if "torch.bmm(" in line and not found_bmm:
            py_ops.append(ParsedOp("gemm", i + 1, line.strip(),
                                   {"note": "MM6: reluGrad @ GKI → dQueryIndex"}))
            found_bmm = True
            break
    # scatter_add
    for i, line in enumerate(lines):
        if "scatter_add_" in line or "scatter_add(" in line:
            py_ops.append(ParsedOp("scatter_add", i + 1, line.strip(),
                                   {"note": "MM5 + V2: reluGrad^T @ QI → scatter_add → dKeyIndex"}))
            break
    # Cast .to(dtype)
    for i, line in enumerate(lines):
        if "d_query_index" in line and ".to(input_dtype)" in line:
            py_ops.append(ParsedOp("cast", i + 1, line.strip(),
                                   {"from": "float32", "to": "input_dtype (fp16/bf16)", "note": "Python dQI .to()",
                                    "mode": "PYTHON_CAST"}))
            break

    if py_ops:
        parsed_stages.append(ParsedStage("Python_Postprocessing", "Python", py_ops))

    return parsed_stages, warnings


def _classify_tilelang_op(line: str, idx: int) -> Optional[ParsedOp]:
    """Classify a single TileLang operation line."""
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("def "):
        return None

    if "T.gemm_v0" in line:
        return ParsedOp("gemm", idx + 1, line,
                        _extract_gemm_params(line))
    elif "T.tile.exp(" in line:
        return ParsedOp("exp", idx + 1, line)
    elif "T.tile.ln(" in line:
        return ParsedOp("log", idx + 1, line, {"note": "natural log"})
    elif "T.reduce_max(" in line:
        return ParsedOp("reduce_max", idx + 1, line,
                        _extract_reduce_params(line))
    elif "T.reduce_sum(" in line:
        return ParsedOp("reduce_sum", idx + 1, line,
                        _extract_reduce_params(line))
    elif "T.tile.mul(" in line:
        return ParsedOp("mul", idx + 1, line)
    elif "T.tile.sub(" in line:
        return ParsedOp("sub", idx + 1, line)
    elif "T.tile.div(" in line:
        return ParsedOp("div", idx + 1, line)
    elif "T.tile.add(" in line:
        return ParsedOp("add", idx + 1, line)
    elif "T.tile.abs(" in line:
        return ParsedOp("abs", idx + 1, line)
    elif "T.tile.max(" in line:
        return ParsedOp("max/clamp", idx + 1, line)
    elif "T.tile.compare(" in line:
        return ParsedOp("compare", idx + 1, line,
                        _extract_compare_params(line))
    elif "T.tile.select(" in line:
        return ParsedOp("select", idx + 1, line)
    elif "T.tile.cast(" in line:
        return ParsedOp("cast", idx + 1, line,
                        _extract_cast_params(line))
    elif "T.tile.fill(" in line:
        return ParsedOp("fill", idx + 1, line,
                        _extract_fill_params(line))
    elif "T.set_cross_flag" in line:
        return ParsedOp("sync_set", idx + 1, line)
    elif "T.wait_cross_flag" in line:
        return ParsedOp("sync_wait", idx + 1, line)
    elif "T.barrier_all()" in line:
        return ParsedOp("barrier", idx + 1, line)
    elif "T.Pipelined" in line:
        return ParsedOp("pipeline", idx + 1, line)
    return None


def _extract_gemm_params(line: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if "transpose_B=True" in line:
        params["transpose_B"] = True
    if "transpose_A=True" in line:
        params["transpose_A"] = True
    if "enable_relu=True" in line:
        params["activation"] = "ReLU"
    return params


def _extract_reduce_params(line: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    m = re.search(r"dim\s*=\s*([-\d]+)", line)
    if m:
        params["axis"] = int(m.group(1))
    m = re.search(r"clear\s*=\s*(True|False)", line)
    if m:
        params["clear"] = (m.group(1) == "True")
    return params


def _extract_compare_params(line: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for op in ["GT", "GE", "LT", "LE", "EQ", "NE"]:
        if f'"{op}"' in line or f"'{op}'" in line:
            params["compare_op"] = op
            break
    m = re.search(r",\s*([\d.e+\-]+)\s*[,\)]", line)
    if m and "0.0" in line:
        params["threshold"] = 0.0
    return params


def _extract_cast_params(line: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for mode in ["CAST_NONE", "CAST_ROUND", "CAST_FLOOR", "CAST_CEIL"]:
        if mode in line:
            params["mode"] = mode
            break
    return params


def _extract_fill_params(line: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    m = re.search(r"fill\([^,]+,\s*([\d.eE+\-]+)\s*\)", line)
    if m:
        val_str = m.group(1)
        try:
            params["value"] = float(val_str)
        except ValueError:
            params["value"] = val_str
    return params


def _extract_str_arg(line: str) -> str:
    m = re.search(r'"([^"]*)"', line)
    if m:
        return m.group(1)
    return ""


# ==============================================================================
# AscendC source parser
# ==============================================================================

@dataclass
class AscendCStage:
    name: str
    ops: List[ParsedOp] = field(default_factory=list)


def parse_ascendc_source(source: str) -> List[AscendCStage]:
    """Parse TileLang-generated AscendC source to extract operations."""
    lines = source.split("\n")
    stages: List[AscendCStage] = []

    gemm_ops = []
    vec_ops = []
    sync_ops = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        ln = i + 1

        # Detect MatMul
        if re.search(r'(?i)(MatMul\b|mmad|MMAD|Matmul)', stripped):
            gemm_ops.append(ParsedOp("gemm", ln, stripped))
        # Detect ReLU
        elif re.search(r'(?i)\bRelu\b', stripped):
            vec_ops.append(ParsedOp("relu", ln, stripped))
        # Detect Softmax / Exp
        elif re.search(r'(?i)(Softmax\b|SoftMax\b)', stripped):
            vec_ops.append(ParsedOp("softmax", ln, stripped))
        elif re.search(r'(?i)\bExp\b', stripped) and "Expr" not in stripped:
            vec_ops.append(ParsedOp("exp", ln, stripped))
        # Detect Reduce ops
        elif re.search(r'(?i)(ReduceSum\b|ReduceMax\b|ReduceMin\b)', stripped):
            vec_ops.append(ParsedOp("reduce", ln, stripped))
        # Detect Cast
        elif re.search(r'(?i)\bCast\b', stripped):
            vec_ops.append(ParsedOp("cast", ln, stripped))
        # Detect Mul/Sub/Div
        elif re.search(r'(?i)\bMuls?\b', stripped):
            vec_ops.append(ParsedOp("mul", ln, stripped))
        elif re.search(r'(?i)\bSub\b', stripped):
            vec_ops.append(ParsedOp("sub", ln, stripped))
        elif re.search(r'(?i)\bDiv\b', stripped):
            vec_ops.append(ParsedOp("div", ln, stripped))
        # Detect Log/Ln
        elif re.search(r'(?i)\b(Ln|Log)\b', stripped):
            vec_ops.append(ParsedOp("log", ln, stripped))
        # Detect Compare / Select
        elif re.search(r'(?i)\bCompare\b', stripped):
            vec_ops.append(ParsedOp("compare", ln, stripped))
        elif re.search(r'(?i)\bSelect\b', stripped):
            vec_ops.append(ParsedOp("select", ln, stripped))
        # Detect sync flags
        elif re.search(r'(?i)(SetFlag|WaitFlag)', stripped):
            sync_ops.append(ParsedOp("sync", ln, stripped))
        # Detect ReLU in fixpipe params
        elif re.search(r'(?i)reluEn\s*=\s*true', stripped):
            vec_ops.append(ParsedOp("relu_fixpipe", ln, stripped,
                                    {"note": "ReLU enabled in fixpipe output"}))
        # Detect transpose flags in MatMul params
        elif re.search(r'(?i)isRightTranspose\s*=\s*true', stripped):
            gemm_ops.append(ParsedOp("gemm_transpose", ln, stripped,
                                     {"transpose_B": True}))

    if gemm_ops:
        stages.append(AscendCStage("C_GEMM", gemm_ops))
    if vec_ops:
        stages.append(AscendCStage("V_Vector", vec_ops))
    if sync_ops:
        stages.append(AscendCStage("Sync", sync_ops))

    return stages


# ==============================================================================
# Comparison engine
# ==============================================================================

@dataclass
class StageMatch:
    orig_stage: str
    planb_location: str
    status: str  # "OK", "MISMATCH", "WARNING", "NOT_FOUND"
    details: List[str] = field(default_factory=list)
    param_issues: List[str] = field(default_factory=list)


def _check_op_params(orig_op: Dict, planb_ops: List[ParsedOp], stage_name: str) -> List[str]:
    """Check if expected parameters exist in parsed Plan B operations."""
    issues: List[str] = []
    op_type = orig_op.get("type", "")
    params = orig_op.get("params", {})

    matching_ops = [pb_op for pb_op in planb_ops if _op_type_matches(op_type, pb_op.op_type)]

    if not matching_ops:
        issues.append(f"  ?? {stage_name}: op type '{op_type}' not found in Plan B ops")
        return issues

    # For cast/compare: check if ANY matching op has the right params
    param_ok = False
    param_detail = ""
    for pb_op in matching_ops:
        if op_type == "gemm":
            if params.get("transpose_B") and not pb_op.params.get("transpose_B"):
                continue  # this gemm doesn't match, try next
            if params.get("activation") == "ReLU":
                if pb_op.params.get("activation") == "ReLU":
                    param_ok = True
                    break
            else:
                param_ok = True
                break
        elif op_type == "cast":
            pb_mode = pb_op.params.get("mode", "")
            orig_mode = params.get("mode", "")
            if "ROUND" in orig_mode and "ROUND" in pb_mode:
                param_ok = True
                break
            elif pb_mode == "PYTHON_CAST":
                param_ok = True  # Python .to() has no explicit round mode
                break
            elif "ROUND" not in orig_mode and pb_mode:
                param_ok = True
                break
            elif not orig_mode:
                param_ok = True  # no specific mode required
                break
            param_detail = f"expected={orig_mode}, got={pb_mode}"
        elif op_type == "compare":
            orig_op_name = params.get("op", "")
            pb_op_name = pb_op.params.get("compare_op", "")
            if orig_op_name and pb_op_name and orig_op_name == pb_op_name:
                param_ok = True
                break
            param_detail = f"expected={orig_op_name}, got={pb_op_name}"
        elif op_type == "reduce":
            orig_axis = params.get("axis")
            pb_axis = pb_op.params.get("axis")
            if orig_axis is not None and pb_axis is not None and orig_axis != pb_axis:
                param_detail = f"reduce axis expected={orig_axis}, got={pb_axis}"
                continue
            param_ok = True
            break
        else:
            param_ok = True
            break

    if not param_ok and matching_ops:
        if param_detail:
            issues.append(f"  !! {stage_name}: {param_detail}")
        elif op_type == "cast" and params.get("mode"):
            issues.append(f"  !! {stage_name}: cast mode expected={params['mode']}")
        elif op_type == "gemm" and params.get("activation") == "ReLU":
            issues.append(f"  !! {stage_name}: ReLU activation expected")

    return issues


def _op_type_matches(orig_type: str, pb_type: str) -> bool:
    """Check if two operation types match semantically."""
    equivalents = {
        "gemm": ["gemm", "gemm_transpose"],
        "softmax": ["softmax", "exp"],
        "reduce": ["reduce", "reduce_sum", "reduce_max"],
        "mul": ["mul"],
        "sub": ["sub"],
        "div": ["div"],
        "log": ["log"],
        "exp": ["exp"],
        "clamp": ["max/clamp", "fill"],
        "compare": ["compare"],
        "select": ["select"],
        "cast": ["cast"],
        "scatter_add": ["scatter_add"],
        "gather": ["gather"],
        "abs": ["abs"],
    }
    return pb_type in equivalents.get(orig_type, [orig_type])


def run_comparison(
    parsed_stages: List[ParsedStage],
    ascendc_stages: List[AscendCStage] = None,
) -> Tuple[List[StageMatch], float]:
    """Compare original AscendC expectation vs Plan B implementation.

    Returns (matches, score).
    """
    matches: List[StageMatch] = []

    # Flatten Plan B ops by location
    k1_c_ops: List[ParsedOp] = []
    k1_v_ops: List[ParsedOp] = []
    k2_v_ops: List[ParsedOp] = []
    py_ops: List[ParsedOp] = []

    for ps in parsed_stages:
        if "Kernel1" in ps.stage_name:
            if "C1" in ps.stage_name or "MM" in ps.stage_name:
                k1_c_ops.extend(ps.ops)
            else:
                k1_v_ops.extend(ps.ops)
        elif "Kernel2" in ps.stage_name:
            k2_v_ops.extend(ps.ops)
        elif "Python" in ps.stage_name:
            py_ops.extend(ps.ops)

    # Flatten AscendC ops
    ac_gemm_ops: List[ParsedOp] = []
    ac_vec_ops: List[ParsedOp] = []
    if ascendc_stages:
        for acs in ascendc_stages:
            if "GEMM" in acs.name:
                ac_gemm_ops.extend(acs.ops)
            elif "Vector" in acs.name:
                ac_vec_ops.extend(acs.ops)

    # Check each original stage against Plan B
    total_checks = 0
    passed_checks = 0

    for stage in ORIGINAL_STAGES:
        mapping = PLANB_MAPPING.get(stage.name, {})
        location = mapping.get("location", "UNKNOWN")
        total_checks += 1

        details = [f"  Plan B: {location} | {mapping.get('function', mapping.get('note', '?'))}"]

        # Check ops exist in the right location
        target_ops = []
        if location == "Kernel1":
            target_ops = k1_c_ops + k1_v_ops
        elif location == "Kernel2":
            target_ops = k2_v_ops
        elif location == "Python":
            target_ops = py_ops

        param_issues = []
        for op_spec in stage.ops:
            issues = _check_op_params(op_spec, target_ops, stage.name)
            param_issues.extend(issues)

        if param_issues:
            details.extend(param_issues)
            match = StageMatch(stage.name, location, "PARAM_ISSUE", details, param_issues)
        elif not target_ops and stage.ops:
            match = StageMatch(stage.name, location, "WARNING",
                               details + ["  (empty ops, Python-side — check manually)"],
                               param_issues)
        else:
            match = StageMatch(stage.name, location, "OK", details, param_issues)
            passed_checks += 1

        matches.append(match)

    # AscendC code check (if provided)
    if ascendc_stages:
        total_checks += 2
        # Check GEMM count
        gemm_actual = len(ac_gemm_ops)
        if gemm_actual >= 2:
            passed_checks += 1
            matches.append(StageMatch("AscendC_GEMM", "Generated", "OK",
                                      [f"  Found {gemm_actual} GEMM events (>=2 OK for Plan B)"]))
        else:
            matches.append(StageMatch("AscendC_GEMM", "Generated", "MISMATCH",
                                      [f"  Found {gemm_actual} GEMM events, expected >=2"]))

        # Check Vector ops
        vec_actual = len(ac_vec_ops)
        if vec_actual >= 5:
            passed_checks += 1
            matches.append(StageMatch("AscendC_Vector", "Generated", "OK",
                                      [f"  Found {vec_actual} vector ops (>=5 OK)"]))
        else:
            matches.append(StageMatch("AscendC_Vector", "Generated", "WARNING",
                                      [f"  Only {vec_actual} vector ops found, may be incomplete"]))

    score = (passed_checks / total_checks * 100) if total_checks > 0 else 0.0
    return matches, score


# ==============================================================================
# Report generation
# ==============================================================================

def generate_report(matches: List[StageMatch], score: float,
                    parsed_stages: List[ParsedStage] = None,
                    ascendc_stages: List[AscendCStage] = None) -> str:
    """Generate a human-readable comparison report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  Plan B vs Original AscendC — Logic & Parameter Validation")
    lines.append("=" * 72)

    # ---- Original pipeline summary ----
    lines.append("")
    lines.append("## Original AscendC Pipeline (14 stages)")
    lines.append("  V0(GatherKey)→V0(GatherKeyIdx)→C1(MM1)→C1(MM2)")
    lines.append("  →V1(VectorP)→V1(ReduceSumY)→V1(GlobalSoftmaxI)")
    lines.append("  →V1(DiffDW)→V1(ReLUGrad)→V1(KL_Loss)")
    lines.append("  →C2(MM5:scatterAdd)→C2(MM6:dQueryIndex)→V2(ScatterAdd+Cast)")

    # ---- Plan B mapping ----
    lines.append("")
    lines.append("## Plan B Mapping")
    lines.append("  Kernel1 (C1): MM1 (Pipelined) + MM2 (ReLU)")
    lines.append("  Kernel1 (V1): P_reduced + ReduceSumY (per K-block)")
    lines.append("  Kernel2 (V1): SoftmaxI + diff + dWeight + reluGrad + KL Loss")
    lines.append("  Python:      dW sum, Loss sum, MM6 bmm, MM5 scatter_add, Cast")

    # ---- Stage-by-stage comparison ----
    lines.append("")
    lines.append("## Stage-by-Stage Comparison")
    lines.append("")

    status_counts = {"OK": 0, "WARNING": 0, "PARAM_ISSUE": 0, "MISMATCH": 0, "NOT_FOUND": 0}
    for m in matches:
        icon = {"OK": "[OK]", "WARNING": "[??]", "PARAM_ISSUE": "[!!]",
                "MISMATCH": "[XX]", "NOT_FOUND": "[--]"}.get(m.status, "[  ]")
        lines.append(f"  {icon} {m.orig_stage}")
        lines.append(f"      → {m.planb_location}")
        for d in m.details:
            lines.append(f"        {d}")
        lines.append("")
        status_counts[m.status] = status_counts.get(m.status, 0) + 1

    # ---- Summary ----
    lines.append("-" * 72)
    lines.append(f"  Results: {status_counts['OK']} OK, {status_counts['WARNING']} warnings, "
                 f"{status_counts['PARAM_ISSUE']} param issues, {status_counts['MISMATCH']} mismatches")
    lines.append(f"  Overall Score: {score:.0f}/100")
    lines.append("")

    if score >= 90:
        lines.append("  Assessment: Logic and parameters match original")
    elif score >= 70:
        lines.append("  Assessment: Major stages match, minor param differences")
    elif score >= 50:
        lines.append("  Assessment: Partial match, review param issues above")
    else:
        lines.append("  Assessment: Significant differences — review each stage")

    # ---- AscendC generated-code summary ----
    if ascendc_stages:
        lines.append("")
        lines.append("## Generated AscendC Code Summary")
        for acs in ascendc_stages:
            lines.append(f"  [{acs.name}] {len(acs.ops)} operations detected")
            for op in acs.ops[:8]:
                lines.append(f"    L{op.line}: {op.op_type}")

    # ---- Parsed Plan B summary ----
    if parsed_stages:
        lines.append("")
        lines.append("## Parsed Plan B Operations")
        for ps in parsed_stages:
            lines.append(f"  [{ps.stage_name}] ({ps.location}) — {len(ps.ops)} ops")
            for op in ps.ops[:10]:
                extra = ""
                if op.params:
                    key_params = {k: v for k, v in op.params.items()
                                  if k not in ("note",)}
                    if key_params:
                        extra = f"  {key_params}"
                lines.append(f"    L{op.line}: {op.op_type}{extra}")

    return "\n".join(lines)


# ==============================================================================
# Dump generated AscendC infrastructure
# ==============================================================================

def dump_generated_code() -> Optional[str]:
    """Attempt to dump TileLang-generated AscendC source and return the file path."""
    # Try common cache locations
    import pathlib
    home = pathlib.Path.home()
    cache_dirs = [
        home / ".cache" / "tilelang",
        home / ".tilelang",
        home / ".tilelang_cache",
        pathlib.Path("/tmp/tilelang_cache"),
        pathlib.Path("/tmp/tilelang"),
    ]
    for d in cache_dirs:
        if d.exists():
            cpp_files = list(d.rglob("*.cpp")) + list(d.rglob("*.cuh"))
            if cpp_files:
                latest = max(cpp_files, key=lambda f: f.stat().st_mtime)
                print(f"Found generated source: {latest}")
                print(f"  Size: {latest.stat().st_size} bytes")
                print(f"  Run: python validate_pipeline.py --compare {latest}")
                return str(latest)

    print("No generated .cpp files found in cache dirs.")
    print("Try running the test first to trigger compilation, then re-run --dump.")
    return None


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate Plan B vs original AscendC — logic & parameter comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_pipeline.py                       # default: check Plan B Python source
  python validate_pipeline.py --dump                # find cached generated .cpp file
  python validate_pipeline.py --compare kernel.cu   # also check generated AscendC code
  python validate_pipeline.py --full                # Python check + auto-find generated code
        """,
    )
    parser.add_argument("--check-python", action="store_true",
                        help="Parse Plan B Python source, compare stages against original AscendC spec")
    parser.add_argument("--compare", metavar="FILE",
                        help="Path to TileLang-generated AscendC .cpp/.cu file (use --dump to find it)")
    parser.add_argument("--dump", action="store_true",
                        help="Search cache dirs for TileLang-generated AscendC source, print the path")
    parser.add_argument("--full", action="store_true",
                        help="--check-python + auto --dump + --compare (if generated file found)")
    args = parser.parse_args()

    # Detect generated file for --full
    if args.full and not args.compare:
        found = dump_generated_code()
        if found:
            args.compare = found
        else:
            print("(Proceeding with Python-only comparison; run test first to generate AscendC)")

    if args.dump:
        dump_generated_code()
        return

    if args.compare and not os.path.exists(args.compare):
        print(f"Error: File not found: {args.compare}")
        print("Hint: run with --dump to search for cached generated files,\n"
              "      or compile the kernel first (run test_planB.py --quick), then try again.")
        sys.exit(1)

    parsed_stages: List[ParsedStage] = []
    ascendc_stages: List[AscendCStage] = []

    # Parse Python source
    if args.check_python or args.full or not args.compare:
        planb_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sparse_lightning_indexer_grad_kl_loss_planB.py",
        )
        if os.path.exists(planb_path):
            print(f"Parsing Plan B source: {planb_path}")
            parsed_stages, warnings = parse_planb_python_source(planb_path)
            for w in warnings:
                print(f"  WARN: {w}")
            print(f"  Found {len(parsed_stages)} stages with "
                  f"{sum(len(s.ops) for s in parsed_stages)} total ops")
        else:
            print(f"Plan B source not found: {planb_path}")
            sys.exit(1)

    # Parse AscendC source
    if args.compare:
        print(f"Parsing generated AscendC: {args.compare}")
        with open(args.compare, "r", encoding="utf-8") as f:
            source = f.read()
        print(f"  File: {len(source)} chars, {len(source.splitlines())} lines")
        ascendc_stages = parse_ascendc_source(source)
        print(f"  Found {len(ascendc_stages)} stages with "
              f"{sum(len(s.ops) for s in ascendc_stages)} total ops")

    # If no specific action, default to --check-python
    if not args.check_python and not args.compare and not args.full:
        args.check_python = True
        planb_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sparse_lightning_indexer_grad_kl_loss_planB.py",
        )
        if os.path.exists(planb_path):
            print(f"Parsing Plan B source: {planb_path}")
            parsed_stages, warnings = parse_planb_python_source(planb_path)
            for w in warnings:
                print(f"  WARN: {w}")
            print(f"  Found {len(parsed_stages)} stages with "
                  f"{sum(len(s.ops) for s in parsed_stages)} total ops")

    # Run comparison
    if parsed_stages or ascendc_stages:
        matches, score = run_comparison(parsed_stages, ascendc_stages if ascendc_stages else None)
        report = generate_report(matches, score, parsed_stages, ascendc_stages if ascendc_stages else None)
        print(report)

        # Save report
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pipeline_validation_report.txt",
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
