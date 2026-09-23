"""C++ 孪生 vs Python golden 对拍。"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "golden"))
from cpu_golden import encode  # noqa: E402


def main() -> int:
    bin_dir = ROOT / "data"
    subprocess.check_call(
        ["/usr/local/python3.12.13/bin/python3", str(ROOT / "gen_selftest_matrices.py")],
        cwd=str(ROOT),
    )
    twin = ROOT / "cpu_encode_twin"
    subprocess.check_call(
        ["g++", "-O2", "-std=c++17", "-o", str(twin), str(ROOT / "cpu_encode_twin.cpp")],
    )
    subprocess.check_call([str(twin)], cwd=str(ROOT))

    rotation = np.fromfile(bin_dir / "rotation_matrix.bin", dtype=np.float32).reshape(128, 128)
    qjl = np.fromfile(bin_dir / "qjl_matrix.bin", dtype=np.float32).reshape(128, 128)
    x = np.array([((i % 17) - 8) * 0.1 for i in range(128)], dtype=np.float32).reshape(1, 1, 128)
    idx, qj, norm, gamma = encode(x, rotation, qjl, 3)

    with open(bin_dir / "twin_smoke_out.bin", "rb") as f:
        idx_c = np.frombuffer(f.read(48), dtype=np.uint8)
        qj_c = np.frombuffer(f.read(16), dtype=np.uint8)
        norm_c, gamma_c = struct.unpack("<ff", f.read(8))

    if not np.array_equal(idx.reshape(-1), idx_c):
        print("FAIL idx mismatch")
        print("py", idx.reshape(-1)[:8], "c", idx_c[:8])
        return 1
    if not np.array_equal(qj.reshape(-1), qj_c):
        print("FAIL qjl mismatch")
        return 1
    if abs(float(norm.reshape(-1)[0]) - norm_c) > 1e-5 or abs(float(gamma.reshape(-1)[0]) - gamma_c) > 1e-5:
        print("FAIL norm/gamma", float(norm), norm_c, float(gamma), gamma_c)
        return 1
    print("PASS python↔c++ twin byte match (smoke vector)")
    print("ALL PASS compare_twin_vs_python")
    return 0


if __name__ == "__main__":
    sys.exit(main())
