#!/usr/bin/env python3
"""Run one QFA HIF8 performance case.

This intentionally only prepares HIF8 Q/K/V and calls the real
``quant_flash_attn_metadata -> quant_flash_attn`` path.  It does not run a
CPU golden or compare outputs.

Usage::

    python invoke_npu_qfa_hif8_perf.py [qS] [kvS] [device]

The defaults match the BSA performance case: ``qS=kvS=4096`` and ``D=128``.
When ``ASCEND_RT_VISIBLE_DEVICES`` is set to one physical device, pass the
logical device id (normally 0).
"""

from __future__ import annotations

import logging
import os
import sys
import sysconfig
from pathlib import Path

# CANN's environment script may prepend its own older Python package.  Prefer
# the package installed in the active Python environment, which contains the
# QFA torch wrappers.
sys.path.insert(0, sysconfig.get_paths()["purelib"])

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)


PERF_DIR = Path(__file__).resolve().parent
REPO_ROOT = PERF_DIR.parents[4]
QFA_COMMON = REPO_ROOT / "attention/quant_flash_attn/tests/pytest/fia_fullquant_hif8_test/common"
sys.path.insert(0, str(QFA_COMMON))
import quant_flash_attn_golden as golden  # noqa: E402


def _int_arg(index: int, default: int) -> int:
    return int(sys.argv[index]) if len(sys.argv) > index else default


def main() -> None:
    q_seq = _int_arg(1, 4096)
    kv_seq = _int_arg(2, 4096)
    device = _int_arg(3, int(os.environ.get("DEVICE", "0")))
    if q_seq <= 0 or kv_seq <= 0:
        raise ValueError("qS and kvS must be positive")

    # Keep this case aligned with the QFA HIF8 golden implementation.
    golden.DEVICE_ID = device
    golden.B = 1
    golden.N_q = 1
    golden.N_kv = 1
    golden.D = 128
    golden.CU_SEQLENS_Q = [0, q_seq]
    golden.CU_SEQLENS_KV = [0, kv_seq]
    golden.SEQUSED_Q = [q_seq]
    golden.SEQUSED_KV = [kv_seq]
    golden.MAX_SEQLEN_Q = q_seq
    golden.MAX_SEQLEN_KV = kv_seq
    golden.INPUT_LAYOUT = "TND"
    golden.Q_SCALE_LAYOUT = "BSND"
    golden.SPARSE_MODE = 0
    golden.SOFTMAX_SCALE = 128 ** -0.5
    golden.P_SCALE = 1.0
    golden.ENABLE_LSE = False
    golden.logger.setLevel(logging.WARNING)

    q, k, v, q_scale, k_scale, v_scale, p_scale, qr, kr, block_table = golden.generate_data()
    torch.npu.synchronize()
    golden.npu_hif8_fa(
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        golden.CU_SEQLENS_Q,
        golden.CU_SEQLENS_KV,
        golden.SEQUSED_Q,
        golden.SEQUSED_KV,
        golden.MAX_SEQLEN_Q,
        golden.MAX_SEQLEN_KV,
        block_table,
        qr,
        kr,
    )
    torch.npu.synchronize()
    print(f"QFA HIF8 done: B=1 Nq=1 Nkv=1 qS={q_seq} kvS={kv_seq} D=128 device={device}")


if __name__ == "__main__":
    main()
