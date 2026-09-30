#!/usr/bin/env python3
"""kctq_v3_precision.py — KCTQ 端到端 decode 精度（V3，按任务书 §3.2 字面口径统计）。

口径（任务书 §3.2 原文 + 官方 2026-09-28 discussions/14 补充说明）：
    重构（主量化 + 1-bit QJL 残差完整重构）：
        x̂ = (‖x‖·centroid[idx] + γ·sqrt(π/2)/d·(sign @ S)) @ H
    其中 H = rotation_matrix（golden 的 `rotated = unit @ rotation.T` ⇒ 反变换乘 H）、
    S = qjl_matrix、sign = ±1（由 1-bit 打包位还原）、‖x‖/γ 为算子 bf16 输出。

指标（阈值均取自任务书 §3.2 字面式，另含官方为 2-bit 单列的逐向量阈值）：
    ① relMSE          = Σ‖x−x̂‖² / Σ‖x‖²                      阈值 < 0.05
    ② 内积 p95        = p95( |<q,x̂>−<q,x>| / max(|<q,x>|,1e-6) )  阈值 < 0.1（**字面分母**）
    ③ 逐向量 NMSE max = max_i ‖x_i−x̂_i‖²/‖x_i‖²               阈值：3/4-bit < 0.05；2-bit < 0.10
    ④ 逐向量 NMSE P95                                         阈值：2-bit < 0.06
    对照行：仅主量化（不含 QJL 残差）的 relMSE —— 用于展示 QJL 的作用，不参与判定。

q 的来源：`--ip-query`（官方件）优先；缺省按官方说明自采 q = BF16(BF16(x) + 0.5·N(0,1))。

用法：
    python3 kctq_v3_precision.py --workdir <out_dir> --rot <H.bin> --qjl <S.bin> \
        --tokens T --heads H --bits B [--kv <kv.bin>] [--ip-query <ip_query.bin>] \
        [--golden <golden.py>] [--json <out.json>]
输出：末行 RESULT: PASS / FAIL；`--json` 落盘结构化结果。
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


def bf16_round_round_half_even(a: np.ndarray) -> np.ndarray:
    """fp64 → BF16 舍入（round-to-nearest-even）再回到 fp64。

    用于按官方说明自采 q = BF16(BF16(x) + 0.5·N(0,1))；bf16 = fp32 高 16 位，
    加 0x7FFF + lsb 后截断即得 RNE。
    """
    bits = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (bits >> np.uint64(16)) & np.uint64(1)
    rounded = (bits + np.uint64(0x7FFF) + lsb) & np.uint64(0xFFFF0000)
    return rounded.astype(np.uint32).view(np.float32).astype(np.float64)


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
    ap.add_argument("--ip-query", default=None,
                    help="官方 ip_query.bin（bf16）；缺省按 q = BF16(x + 0.5*N(0,1)) 自采")
    ap.add_argument("--rot", required=True)
    ap.add_argument("--qjl", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--bits", type=int, required=True, choices=(2, 3, 4))
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

    # ---- 重构：主量化 + QJL 残差完整重构（官方 2026-09-28 口径）----
    #     算子侧 gamma 已含 ‖x‖ 因子：x̂ = (‖x‖·primary + γ·sqrt(π/2)/d·(sign@S)) @ H
    primary = centroids[idx]                                        # [rows, d]（旋转域）
    r_hat = np.sqrt(np.pi / 2.0) / HEAD_DIM * (signs @ qjl_mat)     # [rows, d]（旋转域）
    x_hat = (norm[:, None] * primary + gamma[:, None] * r_hat) @ rot
    x_hat_mse_only = (norm[:, None] * primary) @ rot                # 对照：仅主量化

    # ---- 指标① 相对 MSE（aggregate）与指标③/④ 逐向量 NMSE 分布 ----
    err2 = ((x - x_hat) ** 2).sum(axis=1)                           # ‖x_i − x̂_i‖²
    x2 = (x * x).sum(axis=1)                                        # ‖x_i‖²
    rel_mse = float(err2.sum() / x2.sum())                          # aggregate（任务书式）
    vec_nmse = err2 / np.maximum(x2, 1e-30)
    vec_max = float(vec_nmse.max())
    vec_p95 = float(np.percentile(vec_nmse, 95))
    rel_mse_mse_only = float(((x - x_hat_mse_only) ** 2).sum() / x2.sum())

    # ---- 指标② 内积误差（任务书字面分母；q 用官方件或按官方说明自采）----
    if args.ip_query:
        q_raw = np.fromfile(args.ip_query, dtype=np.uint16)
        if q_raw.size != rows * HEAD_DIM:
            raise SystemExit(f"ip_query 元素数 {q_raw.size} != {rows * HEAD_DIM}")
        q = bf16_to_fp32(q_raw).reshape(rows, HEAD_DIM)
        q_src = f"官方件 {Path(args.ip_query).name}"
    else:
        rng = np.random.default_rng(args.seed)
        q = bf16_round_round_half_even(x + 0.5 * rng.standard_normal((rows, HEAD_DIM)))
        q_src = f"自采 BF16(BF16(x)+0.5·N(0,1))，seed={args.seed}"
    ip_true = (q * x).sum(axis=1)
    ip_hat = (q * x_hat).sum(axis=1)
    ip_err = np.abs(ip_hat - ip_true) / np.maximum(np.abs(ip_true), 1e-6)
    ip_p95 = float(np.percentile(ip_err, 95))

    # ---- 判定（任务书 §3.2 字面式 + 官方为 2-bit 单列的逐向量阈值）----
    bits2 = args.bits == 2
    checks = {"relMSE < 0.05": rel_mse < 0.05,
              "内积 p95 < 0.1（字面分母）": ip_p95 < 0.1,
              "逐向量 max < " + ("0.10" if bits2 else "0.05"): vec_max < (0.10 if bits2 else 0.05)}
    if bits2:
        checks["逐向量 P95 < 0.06"] = vec_p95 < 0.06
    passed = all(checks.values())

    print(f"=== KCTQ V3 精度（任务书 §3.2 字面口径；规范 = 完整重构） ===")
    print(f"workdir={workdir} rows={rows} bits={args.bits}（内积用全部 {rows} 行）")
    print(f"q 生成：{q_src}；分母 max(|<q,x>|,1e-6)")
    print(f"{'指标':<40}{'实测':>12}{'阈值':>10}  判定")
    print(f"{'relMSE (完整重构 主量化+QJL)':<40}{rel_mse:>12.6f}{'<0.05':>10}  "
          f"{'PASS' if rel_mse < 0.05 else 'FAIL'}")
    print(f"{'relMSE (仅主量化，对照不判定)':<40}{rel_mse_mse_only:>12.6f}{'—':>10}  —")
    print(f"{'内积 p95 (任务书字面分母)':<40}{ip_p95:>12.6f}{'<0.1':>10}  "
          f"{'PASS' if ip_p95 < 0.1 else 'FAIL'}")
    print(f"{'逐向量 NMSE max':<40}{vec_max:>12.6f}{('<0.10' if bits2 else '<0.05'):>10}  "
          f"{'PASS' if (vec_max < (0.10 if bits2 else 0.05)) else 'FAIL'}")
    if bits2:
        print(f"{'逐向量 NMSE P95':<40}{vec_p95:>12.6f}{'<0.06':>10}  "
              f"{'PASS' if vec_p95 < 0.06 else 'FAIL'}")
    else:
        print(f"{'逐向量 NMSE P95（参考）':<40}{vec_p95:>12.6f}{'—':>10}  —")
    print("判定：" + "、".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in checks.items()))

    if args.json:
        Path(args.json).write_text(json.dumps({
            "workdir": str(workdir), "rows": rows, "bits": args.bits,
            "qSource": q_src,
            "relMse": rel_mse, "relMseMseOnly": rel_mse_mse_only,
            "ipP95LiteralDenom": ip_p95, "ipDenominator": "max(|<q,x>|,1e-6)",
            "vecNmseMax": vec_max, "vecNmseP95": vec_p95,
            "checks": checks, "pass": passed,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
