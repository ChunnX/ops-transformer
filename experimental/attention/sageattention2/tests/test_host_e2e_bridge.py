# test_host_e2e_bridge.py — 内核宿主差分桥（M2a）：C++ 内核全链（KV prep → 注意力主循环 →
# epilogue）的输出 vs sageattn_reference 逐配置对拍。
#
# 补齐评审缺口（v27 §一.1）：此前 pytest 全部跑在 reference.py 上，op_kernel 的 C++ 实现零执行
# 覆盖；本测试把 sa2_kernel.cpp 的宿主仿真路径真正跑起来（含 P0-1 双 Die 全 KV head 覆盖回归、
# GQA 组映射、causal 掩码、pad 几何）。
#
# TC-05 数值化（todo §四）：test_host_e2e_tc05_pad_numeric 把 d=48→64 / d=96→128 的内核 pad 路径
# （pad 量化 + epilogue 裁剪）纳入对拍，per_thread 与 per_warp 各一档。
#
# 已知口径差（对拍用容差而非逐位，见 sa2_host_e2e.cpp 头注）：per_thread 的 S 舍入点与
# FP16 scale 存储。两者均为 M4.3 L-A golden 反推收口项。
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from sageattention2.reference import sageattn_reference

TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parent
GXX = shutil.which("g++")
pytestmark = pytest.mark.skipif(GXX is None, reason="g++ not available (host bridge needs it)")


def _compile(out_dir: Path) -> Path:
    binary = out_dir / "sa2_host_e2e"
    cmd = [GXX, "-std=c++17", "-O2", "-Wall", "-Wextra",
           "-I", str(REPO_DIR / "op_kernel"),
           "-I", str(REPO_DIR / "sim"),
           "-I", str(REPO_DIR / "op_host"),
           str(TEST_DIR / "sa2_host_e2e.cpp"), "-o", str(binary)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"
    return binary


def _load_fp16(path: Path) -> torch.Tensor:
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float16).clone()


def _load_f32(path: Path) -> torch.Tensor:
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32).clone()


def _parse_meta(path: Path) -> dict:
    fields = path.read_text().strip().split()
    out = {}
    for item in fields:
        key, _, val = item.partition("=")
        out[key] = val
    return out


def _run_binary(binary: Path, tmp: Path, tag: str) -> Path:
    """执行 e2e 二进制（单次运行导出全部用例的 q/k/v/out/lse 字节），返回输出前缀。"""
    prefix = tmp / f"e2e_{tag}"
    proc = subprocess.run([str(binary), str(prefix)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"e2e binary failed:\n{proc.stdout}\n{proc.stderr}"
    return prefix


def _validate_case(case_prefix: Path, label: str) -> None:
    """单用例对拍：读回内核导出的输入字节（与内核逐位一致，强于同 seed）送 reference 直算，
    容差与既有 host 断言口径一致（逐元素 tol=0.02+0.02|ref|、失配率<1%、max_abs<0.2、lse<0.05）。"""
    meta = _parse_meta(case_prefix.with_name(case_prefix.name + ".meta.txt"))
    b, hq, hk = int(meta["b"]), int(meta["hq"]), int(meta["hk"])
    sq, sk, d = int(meta["sq"]), int(meta["sk"]), int(meta["d"])
    gran = meta["gran"]
    is_causal = meta["causal"] == "1"
    accum = {"0": "fp32", "1": "fp32+fp32", "2": "fp32+fp16"}[meta["accum"]]

    q = _load_fp16(Path(str(case_prefix) + ".q.bin")).view(b, hq, sq, d)
    k = _load_fp16(Path(str(case_prefix) + ".k.bin")).view(b, hk, sk, d)
    v = _load_fp16(Path(str(case_prefix) + ".v.bin")).view(b, hk, sk, d)
    out_kernel = _load_fp16(Path(str(case_prefix) + ".out.bin")).view(b, hq, sq, d)
    lse_kernel = _load_f32(Path(str(case_prefix) + ".lse.bin")).view(b, hq, sq)

    out_ref, lse_ref = sageattn_reference(
        q, k, v, tensor_layout="HND", is_causal=is_causal, qk_quant_gran=gran,
        sm_scale=1.0 / (d ** 0.5), pv_accum_dtype=accum, smooth_k=True, return_lse=True)

    diff = (out_kernel.float() - out_ref.float()).abs()
    tol = 0.02 + 0.02 * out_ref.float().abs()
    mismatch = (diff > tol).float().mean().item()
    max_abs = diff.max().item()
    assert mismatch < 0.01, f"[{label}] out mismatch ratio {mismatch:.4f} (max_abs={max_abs:.4f})"
    assert max_abs < 0.2, f"[{label}] out max_abs {max_abs:.4f} too large"
    lse_diff = (lse_kernel - lse_ref).abs().max().item()
    assert lse_diff < 0.05, f"[{label}] lse max_abs {lse_diff:.4f} too large"


def test_host_e2e_bridge(tmp_path: Path) -> None:
    """C++ 内核宿主全链 vs reference.py：两个用例（GQA/per_thread/fp32+fp16、奇hk/per_warp/causal/pad）。"""
    binary = _compile(tmp_path)
    prefix = _run_binary(binary, tmp_path, "run")
    _validate_case(prefix.with_name(prefix.name + "_gqa"), "gqa")

    # 第二用例（奇 hk + per_warp + causal + d=48 pad + 尾块）逐配置对拍
    _validate_case(prefix.with_name(prefix.name + "_odd"), "odd")


def test_host_e2e_tc05_pad_numeric(tmp_path: Path) -> None:
    """TC-05 数值化（todo §四）：head_dim pad 路径数值对拍，补「仅形状」缺口。

    d=48（pad→64，per_thread/fp32+fp16/GQA4:2）与 d=96（pad→128，per_warp/fp32）经内核
    pad 量化 + epilogue 裁剪回原 D 后，与 sageattn_reference 直接按 D=48/96 计算的结果对拍
    （容差与既有 host 断言口径一致；输入为内核导出的同一份字节）。
    """
    binary = _compile(tmp_path)
    prefix = _run_binary(binary, tmp_path, "tc05")
    _validate_case(prefix.with_name(prefix.name + "_pad48"), "tc05/d48_per_thread")
    _validate_case(prefix.with_name(prefix.name + "_pad96"), "tc05/d96_per_warp")
