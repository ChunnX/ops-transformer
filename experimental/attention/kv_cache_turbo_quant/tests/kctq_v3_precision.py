#!/usr/bin/env python3
"""kctq_v3_precision.py — KCTQ 端到端 decode 精度（V3，双路径 × 双口径）。

口径来源：`docs/verification.md` §2/§3 指定的参考实现 `docs/evidence/tools/verify4_precision.py`
（复现《调研分析报告》§5 的数字）与 golden 的编码语义。decode 公式：

    x̂_mse = ‖x‖ · (c[idx] @ H)                     主量化重构（MSE-only 路径）
    x̂     = x̂_mse + sqrt(π/2) · γ/d · (signs @ S)  完整 decode（MSE+QJL 路径）

其中 H = rotation_matrix（golden 的 `rotated = unit @ rotation.T` ⇒ 反变换为 `c @ H`）、
c = centroid 表、idx = 主编码、signs = ±1（由 1-bit 打包的 `1[p >= 0]` 位还原）、
S = qjl_matrix、‖x‖/γ = bf16 输出。

指标（双口径并列，主口径加粗）：
    relMSE    = E‖x−x̂‖² / E‖x‖²                       （报告 §5：3-bit MSE-only 0.0339）
    内积 p95  归一化口径 = p95( |⟨q,x̂⟩−⟨q,x⟩| / (‖q‖‖x‖) )  （阈值 < 0.1，报告 3-bit 0.0603）
    内积 p95  字面口径   = p95( |⟨q,x̂⟩−⟨q,x⟩| / |⟨q,x⟩| )   （对照列，病态归一化，报告 3-bit 2.26）

判定（按《调研分析报告》结论 C5：relMSE<0.05 仅 3-bit MSE-only 路径成立，2-bit 目标数学不可达）：
    * bits ∈ {3,4}：要求 relMSE(MSE-only) < 0.05 且 归一化口径 p95(含 QJL) < 0.1
    * bits = 2     ：口径不适用（数学不可达），只报告实测值（不影响 RESULT）

用法：
    python tools/pipelines/kctq_v3_precision.py --workdir .runtime/kctq_case_t2048 \
        --rot data/rotation_matrix.bin --qjl data/qjl_matrix.bin \
        --tokens 2048 --heads 8 --bits 3 [--samples 2048] [--json out.json]
输出：末行 RESULT: PASS / FAIL（仅针对上述判定）；`--json` 落盘结构化结果。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np

HEAD_DIM = 128
SEED = 20260923
# 报告 §5 参考值（同口径、不同数据/实现），用于对照：{bits: (relMSE_mse_only, p95_norm_mse_qjl)}
REPORT_REF = {2: (0.1159, 0.1116), 3: (0.0339, 0.0603), 4: (0.0093, 0.0316)}


def find_repo_root(start: Path) -> Path:
    path = start.resolve()
    while path != path.parent:
        if (path / "AGENTS.md").exists():
            return path
        path = path.parent
    return start.resolve()


def load_centroids(golden_path: str | None = None) -> dict:
    """码本从任务包 golden.py 读取：--golden > $KCTQ_GOLDEN > 仓库内 9月社区任务-*/golden.py。"""
    if golden_path:
        golden = Path(golden_path)
    else:
        env = os.environ.get("KCTQ_GOLDEN")
        golden = (Path(env) if env
                  else sorted(find_repo_root(Path(__file__).parent).glob("9月社区任务-*/golden.py"))[0])
    tree = ast.parse(golden.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "CENTROIDS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("golden.py 中未找到 CENTROIDS")


def bf16_to_fp32(raw: np.ndarray) -> np.ndarray:
    """bfloat16 位模式（uint16）→ fp32（含 f32 语义的精确扩展）。"""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load_kv(path: Path, rows: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != rows * HEAD_DIM:
        raise SystemExit(f"kv_vectors.bin 元素数 {raw.size} != {rows * HEAD_DIM}")
    return bf16_to_fp32(raw).reshape(rows, HEAD_DIM).astype(np.float64)


def unpack_bits(raw: np.ndarray, bit_width: int, lanes: int) -> np.ndarray:
    """C2 语义：每 8 通道一组、组内 LSB 优先小端。"""
    idx = np.zeros(lanes, dtype=np.int64)
    chunk_bytes = bit_width  # 8 通道 * b bits / 8 = b 字节
    groups = lanes // 8
    words = raw[: groups * chunk_bytes].reshape(groups, chunk_bytes).astype(np.int64)
    packed = np.zeros(groups, dtype=np.int64)
    for byte in range(chunk_bytes):
        packed |= words[:, byte] << (8 * byte)
    mask = (1 << bit_width) - 1
    for lane in range(8):
        idx[lane::8] = (packed >> (bit_width * lane)) & mask
    return idx


def unpack_signs(raw: np.ndarray, lanes: int) -> np.ndarray:
    """`quant_qjl`：每 8 通道一位，LSB 优先；位=1 → 1[p>=0] 为真 → +1，否则 -1。"""
    groups = lanes // 8
    packed = raw[:groups].astype(np.int64)
    signs = np.zeros(lanes, dtype=np.float64)
    for lane in range(8):
        bit = (packed >> lane) & 1
        signs[lane::8] = np.where(bit == 1, 1.0, -1.0)
    return signs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="含 quant_*.bin 与 kv_vectors.bin 的目录")
    ap.add_argument("--kv", default=None, help="kv_vectors.bin 路径（默认取 workdir 下同名文件）")
    ap.add_argument("--rot", required=True)
    ap.add_argument("--qjl", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--bits", type=int, required=True, choices=(2, 3, 4))
    ap.add_argument("--samples", type=int, default=2048, help="内积口径的随机 q 条数")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--golden", default=None,
                    help="任务包 golden.py 路径（默认 $KCTQ_GOLDEN 或仓库内搜索）")
    ap.add_argument("--json", default=None, help="结构化结果输出路径")
    args = ap.parse_args()

    rows = args.tokens * args.heads
    workdir = Path(args.workdir)
    kv_path = Path(args.kv) if args.kv else workdir / "kv_vectors.bin"
    x = load_kv(kv_path, rows)
    idx_raw = np.fromfile(workdir / "quant_idx.bin", dtype=np.uint8)
    qjl_raw = np.fromfile(workdir / "quant_qjl.bin", dtype=np.uint8)
    norm = bf16_to_fp32(np.fromfile(workdir / "quant_norm.bin", dtype=np.uint16)).astype(np.float64)
    gamma = bf16_to_fp32(np.fromfile(workdir / "quant_gamma.bin", dtype=np.uint16)).astype(np.float64)
    rot = np.fromfile(args.rot, dtype=np.float32).reshape(HEAD_DIM, HEAD_DIM).astype(np.float64)
    qjl_mat = np.fromfile(args.qjl, dtype=np.float32).reshape(HEAD_DIM, HEAD_DIM).astype(np.float64)

    idx = unpack_bits(idx_raw, args.bits, rows * HEAD_DIM).reshape(rows, HEAD_DIM)
    signs = unpack_signs(qjl_raw, rows * HEAD_DIM).reshape(rows, HEAD_DIM)
    centroids = np.asarray(load_centroids(args.golden)[args.bits], dtype=np.float64)
    x_hat_mse = norm[:, None] * (centroids[idx] @ rot)
    x_hat = x_hat_mse + np.sqrt(np.pi / 2.0) * (gamma / HEAD_DIM)[:, None] * (signs @ qjl_mat)

    x_norm2 = (x * x).sum(axis=1)
    rel_mse_only = float(((x - x_hat_mse) ** 2).sum(axis=1).mean() / x_norm2.mean())
    rel_mse_qjl = float(((x - x_hat) ** 2).sum(axis=1).mean() / x_norm2.mean())

    rng = np.random.default_rng(args.seed)
    sample_rows = rng.choice(rows, size=min(args.samples, rows), replace=False)
    q = rng.standard_normal((sample_rows.size, HEAD_DIM))
    q_norm = np.linalg.norm(q, axis=1)
    xs = x[sample_rows]
    xs_norm = np.sqrt((xs * xs).sum(axis=1))
    ip_true = (q * xs).sum(axis=1)
    err_mse = np.abs((q * x_hat_mse[sample_rows]).sum(axis=1) - ip_true)
    err_qjl = np.abs((q * x_hat[sample_rows]).sum(axis=1) - ip_true)
    p95_norm_mse_only = float(np.percentile(err_mse / (q_norm * xs_norm), 95))
    p95_norm_qjl = float(np.percentile(err_qjl / (q_norm * xs_norm), 95))
    p95_literal_qjl = float(np.percentile(err_qjl / np.abs(ip_true), 95))

    ref_rel, ref_p95 = REPORT_REF[args.bits]
    applicable = args.bits in (3, 4)
    ok_rel = rel_mse_only < 0.05
    ok_p95 = p95_norm_qjl < 0.1
    passed = (ok_rel and ok_p95) if applicable else True

    print(f"=== KCTQ V3 精度（decode 双路径 × 双口径） ===")
    print(f"workdir={workdir} rows={rows} bits={args.bits} "
          f"（内积抽样 {sample_rows.size} 行，seed={args.seed}）")
    print(f"{'指标':<34}{'实测':>12}{'报告参考':>12}{'阈值':>10}")
    print(f"{'relMSE  (MSE-only，主口径)':<34}{rel_mse_only:>12.4f}{ref_rel:>12.4f}{'<0.05':>10}")
    print(f"{'relMSE  (MSE+QJL，对照)':<34}{rel_mse_qjl:>12.4f}{'—':>12}{'—':>10}")
    print(f"{'内积 p95 归一化（含 QJL，主口径)':<34}{p95_norm_qjl:>12.4f}{ref_p95:>12.4f}{'<0.1':>10}")
    print(f"{'内积 p95 归一化（MSE-only）':<34}{p95_norm_mse_only:>12.4f}{'—':>12}{'—':>10}")
    print(f"{'内积 p95 字面口径（含 QJL，对照)':<34}{p95_literal_qjl:>12.4f}{'—':>12}{'—':>10}")
    if not applicable:
        print("注：bits=2 的精度目标数学上不可达（《调研分析报告》结论 C5）→ 只报告实测值，不参与判定。")
    print(f"判定：relMSE(MSE-only) {'PASS' if ok_rel else 'FAIL'}、"
          f"p95 归一化(含 QJL) {'PASS' if ok_p95 else 'FAIL'}"
          f"{'（bits=2 不适用）' if not applicable else ''}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "workdir": str(workdir), "rows": rows, "bits": args.bits,
            "relMseMseOnly": rel_mse_only, "relMseWithQjl": rel_mse_qjl,
            "p95NormWithQjl": p95_norm_qjl, "p95NormMseOnly": p95_norm_mse_only,
            "p95LiteralWithQjl": p95_literal_qjl,
            "referenceRelMse": ref_rel, "referenceP95Norm": ref_p95,
            "applicable": applicable, "pass": passed,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
