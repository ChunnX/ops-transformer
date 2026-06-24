#!/usr/bin/env python3
"""
Validate that TileLang-generated AscendC code matches the original AscendC
operator's pipeline structure.

Usage:
    # Step 1: Dump the generated AscendC source
    TL_DUMP=1 python validate_pipeline.py --dump

    # Step 2: Compare generated vs original pipeline
    python validate_pipeline.py --compare <generated.cpp>

    # Step 3: Full auto (dump + compare in one shot)
    python validate_pipeline.py --auto
"""

import argparse
import os
import re
import sys
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ==============================================================================
# Expected pipeline structure from original AscendC
# ==============================================================================

# The original AscendC kernel (MainProcess) has this pipeline:
#
#   V0:  Gather Key/KeyRope/KeyIndex by sparseIndices
#        -> CrossCoreSetFlag(SYNC_V0_TO_C1_P_FLAG) / SYNC_V0_TO_C1_SY_FLAG
#   C1:  MM1 = Query||QueryRope @ GatherKey||GatherKeyRope^T
#        MM2 = QueryIndex @ GatherKeyIndex^T (with ReLU)
#        -> CrossCoreSetFlag(SYNC_C1_TO_V1_P_FLAG) / SYNC_C1_TO_V1_SY_FLAG
#   V1:  P = softmax(bmm1 * scale) using precomputed max/sum
#        P_reduced = mean(P over G heads)
#        ReduceSumY = sum_g(weight[g] * ReLU(bmm2[g,:]))
#        SoftmaxI = global_softmax(ReduceSumY)
#        diff = SoftmaxI - P_reduced
#        dWeight[g] = sum_k(diff[k] * ReLU(bmm2[g,k]))
#        reluGrad[g,k] = weight[g] * diff[k] * (bmm2[g,k] > 0)
#        KL_Loss = sum_k(P_reduced[k] * (log(P[k]) - log(SoftmaxI[k])))
#        -> CrossCoreSetFlag(SYNC_V1_TO_C2_DW_FLAG)
#   C2:  MM5 = reluGrad^T @ QueryIndex  (scatterAdd dKeyIndex)
#        MM6 = reluGrad @ GatherKeyIndex^T  (dQueryIndex)
#        -> CrossCoreSetFlag(SYNC_C2_TO_V2_SA_FLAG)
#   V2:  ScatterAdd dKeyIndex from scatter buffer
#        Cast dQueryIndex FP32 -> FP16
#
# Since TileLang lacks gather/scatter, the TileLang rewrite maps:
#   V0 -> Python torch.gather preprocessing
#   V1 -> TileLang V Scope 1
#   C1 -> TileLang C Scope 1
#   C2 -> TileLang C Scope 2  (only MM6; MM5 -> Python scatter_add_ postprocessing)
#   V2 -> TileLang V Scope 2  (only Cast)

# Expected computational stages in order
EXPECTED_PIPELINE = [
    # (stage, description, required_patterns, forbidden_patterns)
    ("C1_MM1", "MM1: Query||QueryRope @ GatherKey||GatherKeyRope^T",
     ["gemm", "transpose"], []),
    ("C1_MM2", "MM2: QueryIndex @ GatherKeyIndex^T with ReLU",
     ["gemm", "relu"], []),
    ("V1_P", "VectorP: softmax(bmm1 * scale) using precomputed max/sum",
     ["exp", "softmax", "div"], []),
    ("V1_P_reduced", "P_reduced = mean(P over G heads)",
     ["reduce", "mean"], []),
    ("V1_SY", "ReduceSumY = sum_g(weight[g] * ReLU(bmm2[g,:]))",
     ["reduce_sum", "reduce"], []),
    ("V1_GlobalSoftmaxI", "SoftmaxI = global_softmax(ReduceSumY)",
     ["exp", "reduce_max"], []),
    ("V1_DiffDW", "diff = SoftmaxI - P_reduced; dWeight, reluGrad",
     ["sub", "compare", "select"], []),
    ("V1_KL_Loss", "KL_Loss = sum_k(P[k] * (log(P[k]) - log(SoftmaxI[k])))",
     ["ln", "log", "reduce_sum"], []),
    ("C2_MM6", "MM6: reluGrad @ GatherKeyIndex^T",
     ["gemm", "transpose"], []),
    ("V2_Cast", "Cast dQueryIndex FP32 -> FP16",
     ["cast"], []),
]

# Key sync patterns expected in TileLang-generated code
# TileLang's C/V separation should produce CrossCoreSetFlag/CrossCoreWaitFlag pairs
EXPECTED_SYNC_PAIRS = [
    # (set_flag_pattern, wait_flag_pattern, description)
    ("SetFlag", "WaitFlag", "C1 complete -> V1 start (MM1+MM2 done, Vector can begin)"),
    ("SetFlag", "WaitFlag", "V1 complete -> C2 start (dW/reluGrad done, MM6 can begin)"),
    ("SetFlag", "WaitFlag", "C2 complete -> V2 start (MM6 done, Cast can begin)"),
]

# MM parameters from original: M_SPLIT=128, N_SPLIT=128, K_SPLIT=128
EXPECTED_MM_PARAMS = {
    "M_SPLIT_SIZE": 128,
    "N_SPLIT_SIZE": 128,
    "K_SPLIT_SIZE": 128,
}

# Original sync flag values (from common.h)
# TileLang may not use the same flag numbers, but should have C->V pairs
ORIGINAL_SYNC_FLAGS = {
    "V0_TO_C1_P": [0, 1],
    "V0_TO_C1_SY": [2, 3],
    "C1_TO_V1_P": [4, 5],
    "C1_TO_V1_SY": [6, 7],
    "V1_TO_C2_DW": [8, 9],
    "C2_TO_V2_SA": [10, 11],
}


# ==============================================================================
# Pattern extraction from AscendC source
# ==============================================================================

@dataclass
class PipelineStage:
    """A detected stage in the generated code."""
    name: str
    line_start: int
    line_end: int
    patterns: List[str] = field(default_factory=list)


@dataclass
class PipelineReport:
    """Full comparison report."""
    generated_file: str
    stages_detected: List[PipelineStage] = field(default_factory=list)
    missing_stages: List[str] = field(default_factory=list)
    sync_pairs_detected: List[Tuple[int, int, str]] = field(default_factory=list)
    mm_params_detected: Dict[str, int] = field(default_factory=dict)
    gemm_count: int = 0
    relu_count: int = 0
    softmax_count: int = 0
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0


def parse_generated_code(source: str, filename: str = "<generated>") -> PipelineReport:
    """Parse TileLang-generated AscendC source and extract pipeline structure."""
    report = PipelineReport(generated_file=filename)
    lines = source.split("\n")

    # ---- Count key operations ----
    report.gemm_count = len(re.findall(r'(?i)MatMul\b|gemm|mmad|MMAD', source))
    report.relu_count = len(re.findall(r'(?i)Relu\b|relu|RELU', source))
    report.softmax_count = len(re.findall(r'(?i)Softmax\b|softmax|SOFTMAX', source))

    # ---- Detect sync patterns ----
    set_flag_matches = list(re.finditer(r'(?i)CrossCoreSetFlag|SetFlag\b', source))
    wait_flag_matches = list(re.finditer(r'(?i)CrossCoreWaitFlag|WaitFlag\b', source))

    for i, sf in enumerate(set_flag_matches):
        sf_line = source[:sf.start()].count("\n") + 1
        # Find the next WaitFlag after this SetFlag
        for wf in wait_flag_matches:
            if wf.start() > sf.start():
                wf_line = source[:wf.start()].count("\n") + 1
                report.sync_pairs_detected.append((sf_line, wf_line, "C→V sync"))
                break

    if len(set_flag_matches) < 2:
        report.warnings.append(
            f"Expected >=2 SetFlag/WaitFlag pairs (C→V transitions), "
            f"found {len(set_flag_matches)} SetFlag, {len(wait_flag_matches)} WaitFlag"
        )

    # ---- Detect MM split sizes ----
    for name, expected in EXPECTED_MM_PARAMS.items():
        # Look for split size constants in generated code
        m = re.search(rf'{name}\s*=\s*(\d+)', source)
        if not m:
            # Also check for numeric literals near gemm calls
            m = re.search(rf'(\d+)\s*//.*{name.lower()}', source)
        if m:
            report.mm_params_detected[name] = int(m.group(1))

    # ---- Detect pipeline stages ----
    # Look for the core computation patterns in order

    # C1: MatMul (GEMM) calls
    gemm_lines = []
    for m in re.finditer(r'(?i)(MatMul|gemm|MMAD|matmul)', source):
        gemm_lines.append(source[:m.start()].count("\n") + 1)

    # V1: Softmax / Exp
    softmax_lines = []
    for m in re.finditer(r'(?i)(Softmax|Exp\b|softmax|exp\b)', source):
        softmax_lines.append(source[:m.start()].count("\n") + 1)

    # V1: Reduce
    reduce_lines = []
    for m in re.finditer(r'(?i)(Reduce|reduce)', source):
        reduce_lines.append(source[:m.start()].count("\n") + 1)

    # V1: KL Loss (Log/Ln)
    log_lines = []
    for m in re.finditer(r'(?i)(Ln\b|Log\b|ln\b|log\b)', source):
        log_lines.append(source[:m.start()].count("\n") + 1)

    # C2: Cast
    cast_lines = []
    for m in re.finditer(r'(?i)(Cast|cast|Fixpipe|fixpipe)', source):
        cast_lines.append(source[:m.start()].count("\n") + 1)

    # Detect the ordering: GEMM -> Softmax -> Reduce -> Log -> GEMM -> Cast
    # This matches the expected pipeline

    # Stage 1: First set of GEMMs (C1)
    if gemm_lines:
        first_gemm = min(gemm_lines)
        last_gemm_before_v1 = max([l for l in gemm_lines if l < (softmax_lines[0] if softmax_lines else 999999)] or gemm_lines)
        report.stages_detected.append(PipelineStage(
            "C1_GEMM", first_gemm, last_gemm_before_v1,
            [f"GEMM calls at lines: {gemm_lines[:6]}..."] if len(gemm_lines) > 6 else [f"GEMM calls at lines: {gemm_lines}"]
        ))

    # Stage 2: Softmax/Exp (V1 start)
    if softmax_lines:
        report.stages_detected.append(PipelineStage(
            "V1_Softmax", min(softmax_lines), max(softmax_lines),
            [f"Softmax/Exp at lines: {softmax_lines}"]
        ))

    # Stage 3: Reduce ops (V1 middle)
    if reduce_lines:
        report.stages_detected.append(PipelineStage(
            "V1_Reduce", min(reduce_lines), max(reduce_lines),
            [f"Reduce at lines: {reduce_lines}"]
        ))

    # Stage 4: Log/Ln ops (V1 KL Loss)
    if log_lines:
        report.stages_detected.append(PipelineStage(
            "V1_KL_Loss", min(log_lines), max(log_lines),
            [f"Log/Ln at lines: {log_lines}"]
        ))

    # Stage 5: Second set of GEMMs (C2)
    post_v1_gemm_lines = [l for l in gemm_lines if softmax_lines and l > max(softmax_lines)]
    if post_v1_gemm_lines:
        report.stages_detected.append(PipelineStage(
            "C2_GEMM", min(post_v1_gemm_lines), max(post_v1_gemm_lines),
            [f"Post-V1 GEMM calls at lines: {post_v1_gemm_lines}"]
        ))

    # Stage 6: Cast ops (V2)
    if cast_lines:
        report.stages_detected.append(PipelineStage(
            "V2_Cast", min(cast_lines), max(cast_lines),
            [f"Cast at lines: {cast_lines}"]
        ))

    # ---- Check for missing stages ----
    stage_names = [s.name for s in report.stages_detected]
    if "C1_GEMM" not in stage_names:
        report.missing_stages.append("C1: No GEMM (MM1/MM2) detected")
    if "V1_Softmax" not in stage_names:
        report.missing_stages.append("V1: No Softmax/Exp detected")
    if "V1_Reduce" not in stage_names:
        report.missing_stages.append("V1: No Reduce ops detected")
    if "V1_KL_Loss" not in stage_names:
        report.warnings.append("V1: No explicit Log/Ln found (may be using intrinsics)")
    if "C2_GEMM" not in stage_names:
        report.missing_stages.append("C2: No post-V1 GEMM (MM6) detected")
    if "V2_Cast" not in stage_names:
        report.warnings.append("V2: No explicit Cast found (may be inlined)")

    # ---- Score the match ----
    score = 0.0
    # Correct GEMM count: at least 3 (MM1, MM2, MM6)
    if report.gemm_count >= 3:
        score += 25
    elif report.gemm_count >= 2:
        score += 15
        report.warnings.append(f"GEMM count={report.gemm_count}, expected >=3 (MM1, MM2, MM6)")
    else:
        report.warnings.append(f"GEMM count={report.gemm_count}, expected >=3 (MM1, MM2, MM6)")

    # ReLU present
    if report.relu_count >= 1:
        score += 15
    else:
        report.warnings.append("No ReLU activation detected (needed for MM2 and reluGrad)")

    # Softmax present
    if report.softmax_count >= 1:
        score += 15
    else:
        report.warnings.append("No Softmax detected (needed for P and SoftmaxI)")

    # Sync pairs present
    if len(report.sync_pairs_detected) >= 2:
        score += 15
    elif len(report.sync_pairs_detected) >= 1:
        score += 10
        report.warnings.append(f"Only {len(report.sync_pairs_detected)} sync pairs, expected >=2 (C1→V1, V1→C2, C2→V2)")
    else:
        report.warnings.append("No C→V sync pairs detected")

    # Correct stage ordering (GEMM before Softmax before GEMM before Cast)
    gemm_positions = [s.line_start for s in report.stages_detected if "GEMM" in s.name]
    softmax_pos = [s.line_start for s in report.stages_detected if "Softmax" in s.name]
    cast_pos = [s.line_start for s in report.stages_detected if "Cast" in s.name]

    ordering_ok = True
    if gemm_positions and softmax_pos:
        if not any(g < min(softmax_pos) for g in gemm_positions):
            ordering_ok = False
            report.warnings.append("Ordering error: No GEMM before Softmax (C1 before V1)")
    if gemm_positions and softmax_pos and len(gemm_positions) >= 2:
        if not any(g > max(softmax_pos) for g in gemm_positions):
            ordering_ok = False
            report.warnings.append("Ordering error: No GEMM after Softmax (C2 after V1)")
    if cast_pos and (softmax_pos or gemm_positions):
        last_compute = max(softmax_pos + gemm_positions)
        if not any(c > last_compute for c in cast_pos):
            ordering_ok = False
            report.warnings.append("Ordering error: No Cast after compute stages (V2 at end)")

    if ordering_ok:
        score += 20

    # No missing stages
    if not report.missing_stages:
        score += 10

    report.score = score
    return report


# ==============================================================================
# Comparison against original AscendC source
# ==============================================================================

def analyze_original_pipeline() -> dict:
    """Analyze the original AscendC source to extract the expected pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arch_dir = os.path.join(base_dir, "op_kernel", "arch22")

    result = {
        "sync_flags": ORIGINAL_SYNC_FLAGS,
        "mm_params": EXPECTED_MM_PARAMS,
    }

    # Read base.h to extract MainProcess structure
    base_h = os.path.join(arch_dir, "sparse_lightning_indexer_grad_kl_loss_base.h")
    if os.path.exists(base_h):
        with open(base_h, "r") as f:
            content = f.read()

        # Extract the MainProcess pipeline calls
        pipeline_calls = []
        for m in re.finditer(
            r'(?:ComputeMm|ProcessVector|vectorService\.|matmulService\.)(\w+)',
            content
        ):
            pipeline_calls.append(m.group(0))

        result["original_pipeline_calls"] = pipeline_calls

        # Count the sequence
        v0_count = sum(1 for c in pipeline_calls if "Vector0" in c)
        mm1_count = sum(1 for c in pipeline_calls if "Mm1" in c)
        mm2_count = sum(1 for c in pipeline_calls if "Mm2" in c)
        v1_count = sum(1 for c in pipeline_calls if "Vector1" in c)
        mm5_count = sum(1 for c in pipeline_calls if "Mm5" in c)
        mm6_count = sum(1 for c in pipeline_calls if "Mm6" in c)
        v2_count = sum(1 for c in pipeline_calls if "Vector2" in c)

        result["pipeline_counts"] = {
            "V0(gather)": v0_count,
            "C1_MM1": mm1_count,
            "C1_MM2": mm2_count,
            "V1(softmax/dW/Loss)": v1_count,
            "C2_MM5(scatterAdd_dK)": mm5_count,
            "C2_MM6(dQueryIndex)": mm6_count,
            "V2(scatterAdd_copy)": v2_count,
        }

    return result


def compare_pipelines(report: PipelineReport, original_info: dict) -> str:
    """Generate a human-readable comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("Pipeline Structure Comparison")
    lines.append("TileLang-Generated vs Original AscendC")
    lines.append("=" * 70)

    # ---- Original pipeline ----
    lines.append("\n## Original AscendC Pipeline (MainProcess)")
    lines.append("")
    lines.append("  V0(gather) -> C1(MM1+MM2) -> V1(P/SY/dW/reluGrad/Loss)")
    lines.append("  -> C2(MM5+MM6) -> V2(ScatterAdd+Cast)")
    lines.append("")
    if "pipeline_counts" in original_info:
        lines.append("  Stage counts in MainProcess loop:")
        for stage, count in original_info["pipeline_counts"].items():
            lines.append(f"    {stage}: {count} call(s)")
    lines.append("")
    lines.append("  Sync flags: C->V via CrossCoreSetFlag/WaitFlag pairs")
    lines.append("  MM params: M_SPLIT=128, N_SPLIT=128, K_SPLIT=128")
    lines.append("  Core model: AIC (Cube) + AIV (Vector) separate cores")
    lines.append("  Decomposition: (B, S1) pairs across cores; K-axis in serial")

    # ---- Generated pipeline ----
    lines.append("\n## TileLang-Generated AscendC Pipeline")
    lines.append(f"  Source: {report.generated_file}")
    lines.append("")
    lines.append("  TileLang model: Single kernel, C Scope + V Scope within one core")
    lines.append("  Decomposition: (B, S1) pairs across cores; K-axis in T.serial")

    if report.stages_detected:
        lines.append("\n  Detected stages (in source order):")
        for stage in report.stages_detected:
            lines.append(
                f"    [{stage.name}] lines {stage.line_start}-{stage.line_end}"
            )
            for p in stage.patterns[:3]:
                lines.append(f"      {p}")

    # ---- Comparison metrics ----
    lines.append("\n## Comparison Metrics")

    lines.append(f"\n  GEMM operations:")
    lines.append(f"    Expected: >=3 (MM1, MM2, MM6)")
    lines.append(f"    Detected: {report.gemm_count}")
    lines.append(f"    Status: {'OK' if report.gemm_count >= 3 else 'MISMATCH'}")

    lines.append(f"\n  ReLU activations:")
    lines.append(f"    Expected: >=1 (MM2 ReLU output, reluGrad mask)")
    lines.append(f"    Detected: {report.relu_count}")
    lines.append(f"    Status: {'OK' if report.relu_count >= 1 else 'MISMATCH'}")

    lines.append(f"\n  Softmax operations:")
    lines.append(f"    Expected: >=1 (P softmax + SoftmaxI)")
    lines.append(f"    Detected: {report.softmax_count}")
    lines.append(f"    Status: {'OK' if report.softmax_count >= 1 else 'MISMATCH'}")

    lines.append(f"\n  C→V Sync pairs:")
    lines.append(f"    Expected: >=2 (C1→V1, V1→C2, C2→V2)")
    lines.append(f"    Detected: {len(report.sync_pairs_detected)}")
    for sf_line, wf_line, desc in report.sync_pairs_detected:
        lines.append(f"      SetFlag(line {sf_line}) -> WaitFlag(line {wf_line}): {desc}")
    lines.append(f"    Status: {'OK' if len(report.sync_pairs_detected) >= 2 else 'MISMATCH'}")

    if report.mm_params_detected:
        lines.append(f"\n  MM split sizes:")
        for name, val in report.mm_params_detected.items():
            expected = EXPECTED_MM_PARAMS.get(name, "?")
            lines.append(f"    {name}: {val} (expected {expected})")
    else:
        lines.append(f"\n  MM split sizes: Not explicitly detected in generated code")

    # ---- Missing stages ----
    if report.missing_stages:
        lines.append(f"\n## MISSING STAGES:")
        for s in report.missing_stages:
            lines.append(f"  FAIL: {s}")

    # ---- Warnings ----
    if report.warnings:
        lines.append(f"\n## Warnings:")
        for w in report.warnings:
            lines.append(f"  WARN: {w}")

    # ---- Score ----
    lines.append(f"\n## Structural Match Score: {report.score:.0f}/100")
    if report.score >= 80:
        lines.append("  Assessment: Pipeline structure closely matches original")
    elif report.score >= 60:
        lines.append("  Assessment: Major stages present, some details differ")
    elif report.score >= 40:
        lines.append("  Assessment: Partial match, significant structural differences")
    else:
        lines.append("  Assessment: Poor match, pipeline may be substantially different")

    # ---- Key differences from original ----
    lines.append("\n## Known Differences (by design)")
    lines.append("  1. V0(gather): Moved to Python torch.gather preprocessing")
    lines.append("  2. MM5(scatterAdd dKeyIndex): Moved to Python scatter_add_ postprocessing")
    lines.append("  3. AIC/AIV dual-core -> TileLang single-core C+V Scopes")
    lines.append("  4. K decomposition across cores -> serial T.serial within one core")

    return "\n".join(lines)


# ==============================================================================
# Dump generated AscendC code
# ==============================================================================

def dump_generated_code(config: dict = None) -> Optional[str]:
    """Try to dump TileLang-generated AscendC source code.

    Tries multiple methods:
    1. Environment variable TL_DUMP=1
    2. Checking common cache directories
    3. Calling .get_source() on the compiled kernel
    """
    if config is None:
        config = {
            "B": 1, "S1": 1, "S2": 2048, "N1": 128, "N2": 1,
            "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
            "d_rope": 64, "K": 2048,
        }

    # Method 1: Try running with TL_DUMP=1
    print("[Method 1] Trying TL_DUMP=1 environment variable...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(script_dir, "dump_kernel.py")

    # We'll create a minimal script that triggers compilation
    dump_code = '''
import os
import sys
sys.path.insert(0, r"{script_dir}")

# Try to enable dumping
for env_var in ["TL_DUMP", "TL_DEBUG", "TILELANG_DUMP", "TILELANG_DEBUG",
                "ASCENDC_DUMP", "TVM_DUMP", "TL_DUMP_ASCENDC"]:
    os.environ[env_var] = "1"

import tilelang
import tilelang.language as T

# Check available PassConfigKey values
print("=== PassConfigKey attributes ===")
for attr in dir(tilelang.PassConfigKey):
    if not attr.startswith("_"):
        print(f"  tilelang.PassConfigKey.{attr}")

# Check for dump-related attributes in tilelang module
print("\\n=== tilelang module attributes ===")
for attr in dir(tilelang):
    if any(kw in attr.lower() for kw in ["dump", "debug", "source", "cache", "build"]):
        print(f"  tilelang.{attr}")

# Check common cache directories
import pathlib
home = pathlib.Path.home()
cache_dirs = [
    home / ".cache" / "tilelang",
    home / ".tilelang",
    home / ".tilelang_cache",
    pathlib.Path("/tmp/tilelang_cache"),
    pathlib.Path("/tmp/tilelang"),
]
print("\\n=== Cache directory check ===")
for d in cache_dirs:
    if d.exists():
        print(f"  EXISTS: {{d}}")
        for f in d.rglob("*"):
            if f.is_file():
                print(f"    {{f}} ({{f.stat().st_size}} bytes)")
    else:
        print(f"  NOT FOUND: {{d}}")

# Now try to compile a kernel
print("\\n=== Compiling kernel ===")
from tilelang_rewrite.sparse_lightning_indexer_grad_kl_loss import (
    sparse_lightning_indexer_grad_kl_loss,
    _clear_caches,
)

import torch
import math

B, S1, S2, N1, N2 = 1, 1, 2048, 128, 1
Nidx1, Nidx2 = 64, 1
DQuery, DQueryIndex, d_rope, K = 512, 128, 64, 2048
G = N1 // N2

torch.manual_seed(42)
dtype = torch.float16
device = "npu"

query = torch.randn(B, S1, N1, DQuery, dtype=dtype).to(device)
key = torch.randn(B, S2, N2, DQuery, dtype=dtype).to(device)
query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=dtype).to(device)
key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=dtype).to(device)
query_rope = torch.randn(B, S1, N1, d_rope, dtype=dtype).to(device)
key_rope = torch.randn(B, S2, N2, d_rope, dtype=dtype).to(device)
weights = torch.randn(B, S1, Nidx1, dtype=dtype).to(device)
sparse_indices = torch.zeros(B, S1, Nidx2, K, dtype=torch.int32).to(device)

s2_real_len = min(S2, K)
indices = torch.randperm(S2)[:s2_real_len]
for s in range(S1):
    sparse_indices[0, s, 0, :s2_real_len] = indices.to(torch.int32).to(device)
    if s2_real_len < K:
        sparse_indices[0, s, 0, s2_real_len:] = -1

scale_value = 1.0 / math.sqrt(DQuery)
scale_tensor = torch.tensor(scale_value, dtype=torch.float32)

# Compute softmax max/sum
with torch.no_grad():
    k_mqa = key.squeeze(2)
    k_rope_mqa = key_rope.squeeze(2)
    q_full = torch.cat([query, query_rope], dim=-1)
    k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
    k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
    sm_max_list, sm_sum_list = [], []
    for s in range(S1):
        row = torch.matmul(q_full[:, s:s+1], k_full_t)
        row_scaled = row.float() * scale_value
        row_max = row_scaled.max(dim=-1).values
        row_sum = torch.exp(row_scaled - row_max.unsqueeze(-1)).sum(dim=-1)
        sm_max_list.append(row_max)
        sm_sum_list.append(row_sum)
    sm_max = torch.cat(sm_max_list, dim=1)
    sm_sum = torch.cat(sm_sum_list, dim=1)
    softmax_max = sm_max.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float().to(device)
    softmax_sum = sm_sum.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float().to(device)

torch.npu.synchronize()

_clear_caches()

# Try to call the kernel
print("Calling kernel...")
try:
    d_qi, d_ki, dw, loss = sparse_lightning_indexer_grad_kl_loss(
        query, key, query_index, key_index,
        weights, sparse_indices,
        softmax_max, softmax_sum,
        scale_value,
        query_rope=query_rope,
        key_rope=key_rope,
        layout_query="BSND",
        actual_seq_lengths_query=[S1]*B,
        actual_seq_lengths_key=[S2]*B,
        sparse_mode=3,
    )
    torch.npu.synchronize()
    print("Kernel executed successfully")
except Exception as e:
    print(f"Kernel execution failed: {{e}}")
    import traceback
    traceback.print_exc()

# After compilation, check for generated files again
print("\\n=== Post-compilation cache check ===")
for d in cache_dirs:
    if d.exists():
        print(f"  EXISTS: {{d}}")
        for f in sorted(d.rglob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file():
                print(f"    {{f}} ({{f.stat().st_size}} bytes, mtime={{f.stat().st_mtime}})")
                if f.suffix in [".cpp", ".h", ".cuh", ".cc"]:
                    print(f"      *** ASCENDC SOURCE FILE ***")
                    try:
                        content = f.read_text()
                        print(f"      First 500 chars:")
                        print(f"      {{content[:500]}}")
                    except:
                        pass

# Check for kernel object methods
print("\\n=== Kernel function attributes ===")
print("(Will be printed by validate_pipeline.py wrapper)")
'''.format(script_dir=script_dir)

    dump_script = os.path.join(script_dir, "_dump_kernel_temp.py")
    with open(dump_script, "w") as f:
        f.write(dump_code)

    try:
        # Run on NPU device (use python directly since we're on the NPU machine)
        result = subprocess.run(
            [sys.executable, dump_script],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "TL_DUMP": "1", "TL_DEBUG": "1",
                 "TILELANG_DUMP": "1", "TILELANG_DEBUG": "1"},
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:2000])

        # Search for generated .cpp files in stdout
        for line in result.stdout.split("\n"):
            if ".cpp" in line or ".cuh" in line:
                print(f"  Found potential source: {line}")

    finally:
        if os.path.exists(dump_script):
            os.remove(dump_script)

    return None  # No automatic extraction; user should check output


def inspect_compiled_kernel_object():
    """Generate a script that inspects the compiled kernel object's attributes."""
    code = '''
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TL_DUMP", "1")

import tilelang
from tilelang_rewrite.sparse_lightning_indexer_grad_kl_loss import (
    sparse_lightning_indexer_grad_kl_loss as tl_func,
    _clear_caches, _kernel_cache,
)

# Check what tilelang.jit returns
print("=== Exploring @tilelang.jit return value ===")

# Monkey-patch to capture the compiled function
import tilelang.language as T
original_jit = tilelang.jit

def inspecting_jit(*args, **kwargs):
    print(f"@tilelang.jit called with: out_idx={kwargs.get('out_idx')}, "
          f"workspace_idx={kwargs.get('workspace_idx')}")
    def decorator(fn):
        result = original_jit(*args, **kwargs)(fn)
        print(f"Compiled object type: {type(result)}")
        print(f"Compiled object dir: {[a for a in dir(result) if not a.startswith('_')]}")
        # Check for useful methods
        for meth in ["get_source", "get_kernel_source", "get_build_config",
                     "export_library", "get_code", "dump", "source"]:
            if hasattr(result, meth):
                print(f"  HAS METHOD: {meth}")
        return result
    return decorator

tilelang.jit = inspecting_jit

# Now trigger compilation
import torch, math

B, S1, S2 = 1, 1, 2048
N1, N2, Nidx1, Nidx2 = 128, 1, 64, 1
DQuery, DQueryIndex, d_rope, K = 512, 128, 64, 2048
G = N1 // N2
dtype = torch.float16
device = "npu"

torch.manual_seed(42)
query = torch.randn(B, S1, N1, DQuery, dtype=dtype).to(device)
key = torch.randn(B, S2, N2, DQuery, dtype=dtype).to(device)
query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=dtype).to(device)
key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=dtype).to(device)
query_rope = torch.randn(B, S1, N1, d_rope, dtype=dtype).to(device)
key_rope = torch.randn(B, S2, N2, d_rope, dtype=dtype).to(device)
weights = torch.randn(B, S1, Nidx1, dtype=dtype).to(device)
sparse_indices = torch.zeros(B, S1, Nidx2, K, dtype=torch.int32).to(device)
indices = torch.randperm(S2)[:min(S2, K)]
for s in range(S1):
    sparse_indices[0, s, 0, :min(S2, K)] = indices.to(torch.int32).to(device)
    if min(S2, K) < K:
        sparse_indices[0, s, 0, min(S2, K):] = -1

scale_value = 1.0 / math.sqrt(DQuery)
with torch.no_grad():
    k_mqa = key.squeeze(2); k_rope_mqa = key_rope.squeeze(2)
    q_full = torch.cat([query, query_rope], dim=-1)
    k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
    k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
    sm_max_list, sm_sum_list = [], []
    for s in range(S1):
        row = torch.matmul(q_full[:, s:s+1], k_full_t)
        rs = row.float() * scale_value
        sm_max_list.append(rs.max(dim=-1).values)
        sm_sum_list.append(torch.exp(rs - rs.max(dim=-1).values.unsqueeze(-1)).sum(dim=-1))
    sm_max = torch.cat(sm_max_list, dim=1)
    sm_sum = torch.cat(sm_sum_list, dim=1)
    softmax_max = sm_max.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float().to(device)
    softmax_sum = sm_sum.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float().to(device)

torch.npu.synchronize()
_clear_caches()

try:
    result = tl_func(
        query, key, query_index, key_index, weights, sparse_indices,
        softmax_max, softmax_sum, scale_value,
        query_rope=query_rope, key_rope=key_rope,
        layout_query="BSND",
        actual_seq_lengths_query=[S1]*B,
        actual_seq_lengths_key=[S2]*B,
        sparse_mode=3,
    )
    torch.npu.synchronize()
    print(f"Result shapes: dQI={result[0].shape}, dKI={result[1].shape}, "
          f"dW={result[2].shape}, loss={result[3].shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Restore original jit
tilelang.jit = original_jit
print("\\n=== Done ===")
'''
    return code


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate TileLang-generated AscendC pipeline against original"
    )
    parser.add_argument("--compare", metavar="FILE",
                        help="Compare generated AscendC .cpp file against original")
    parser.add_argument("--dump", action="store_true",
                        help="Attempt to dump generated AscendC source")
    parser.add_argument("--inspect", action="store_true",
                        help="Generate script to inspect compiled kernel object")
    parser.add_argument("--auto", action="store_true",
                        help="Full auto: dump then compare (requires NPU)")
    args = parser.parse_args()

    # ---- Mode: Inspect kernel object ----
    if args.inspect:
        code = inspect_compiled_kernel_object()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_inspect_kernel.py")
        with open(out_path, "w") as f:
            f.write(code)
        print(f"Inspection script written to: {out_path}")
        print("Run this on the NPU machine to discover how to get generated source.")
        return

    # ---- Mode: Dump generated code ----
    if args.dump:
        dump_generated_code()
        return

    # ---- Mode: Auto (dump then compare) ----
    if args.auto:
        print("Step 1: Attempting to dump generated code...")
        dump_generated_code()
        # The dump function will print instructions for finding the generated file
        print("\nAfter finding the generated .cpp file, run:")
        print(f"  python {__file__} --compare <path/to/generated.cpp>")
        return

    # ---- Mode: Compare ----
    if args.compare:
        if not os.path.exists(args.compare):
            print(f"Error: File not found: {args.compare}")
            sys.exit(1)

        with open(args.compare, "r") as f:
            generated_source = f.read()

        print(f"Analyzing: {args.compare}")
        print(f"File size: {len(generated_source)} chars, {len(generated_source.splitlines())} lines")

        report = parse_generated_code(generated_source, args.compare)
        original_info = analyze_original_pipeline()

        comparison = compare_pipelines(report, original_info)
        print(comparison)

        # Save report
        report_path = args.compare + ".pipeline_report.txt"
        with open(report_path, "w") as f:
            f.write(comparison)
        print(f"\nReport saved to: {report_path}")
        return

    # ---- Default: print help ----
    parser.print_help()
    print("\nQuick start:")
    print(f"  1. First, discover how to dump: python {__file__} --inspect")
    print(f"  2. Run the generated inspect script on NPU machine")
    print(f"  3. Find the generated .cpp file")
    print(f"  4. Compare: python {__file__} --compare <generated.cpp>")


if __name__ == "__main__":
    main()
