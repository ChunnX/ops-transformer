#!/usr/bin/env python3
from pathlib import Path

ROOT = Path("/home/npu_user5/xuxuepeng/ops-transformer_0703")


def rw(rel):
    p = ROOT / rel
    raw = p.read_bytes()
    nl = b"\r\n" if b"\r\n" in raw else b"\n"
    text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8")
    return p, text, nl


def wr(p, text, nl):
    out = text.encode("utf-8")
    if nl == b"\r\n":
        out = out.replace(b"\n", b"\r\n")
    p.write_bytes(out)


p, text, nl = rw("attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_mxfp8.hpp")
repls = [
    (
        """        copyL1ToL0AMx(
            l0ATensor[l0ABufId],
            l1ATensor[l1ABufId],""",
        """        copyL1ToL0AMx(
            l0ATensor[l0ABufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
            l1ATensor[l1ABufId].template ReinterpretCast<fp8_e4m3fn_t>(),""",
    ),
    (
        """        copyL1ToL0BMx(
            l0BTensor[l0BBufId],
            l1BTensor[l1BBufId],""",
        """        copyL1ToL0BMx(
            l0BTensor[l0BBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
            l1BTensor[l1BBufId].template ReinterpretCast<fp8_e4m3fn_t>(),""",
    ),
    (
        """             LoadData(l0ATensor[pvL0abBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                     l1ATensor[l1ABufId].template ReinterpretCast<fp8_e4m3fn_t>(),""",
        """             LoadData(l0ATensor[pvL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                     l1ATensor[l1ABufId].template ReinterpretCast<fp8_e4m3fn_t>(),""",
    ),
]
for a, b in repls:
    if a not in text:
        raise SystemExit(f"pv: pattern not found:\n{a}")
    text = text.replace(a, b, 1)
wr(p, text, nl)
print("patched pv mxfp8 LoadData dst types")

p, text, nl = rw("attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_mxfp8.hpp")
repls = [
    (
        """            l0BTensor[qkL0abBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
            l1BTensor[qBufId].template ReinterpretCast<fp8_e4m3fn_t>(),""",
        """            l0BTensor[qkL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
            l1BTensor[qBufId].template ReinterpretCast<fp8_e4m3fn_t>(),""",
    ),
    (
        """            l0ATensor[qkL0abBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
            l1ATensor[kBufId].template ReinterpretCast<fp8_e4m3fn_t>(),""",
        """            l0ATensor[qkL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
            l1ATensor[kBufId].template ReinterpretCast<fp8_e4m3fn_t>(),""",
    ),
]
for a, b in repls:
    if a not in text:
        raise SystemExit(f"qk: pattern not found:\n{a}")
    text = text.replace(a, b, 1)
wr(p, text, nl)
print("patched qk mxfp8 LoadData dst types")
print("ok")
