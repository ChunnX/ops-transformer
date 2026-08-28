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


p, text, nl = rw("attention/block_sparse_attention/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp")
repls = [
    ("struct CopyL1ToL0AMxA5", "struct CopyL1ToL0AMxFp8A5"),
    ("CopyL1ToL0AMxA5()", "CopyL1ToL0AMxFp8A5()"),
    ("struct CopyL1ToL0BMxA5", "struct CopyL1ToL0BMxFp8A5"),
    ("CopyL1ToL0BMxA5()", "CopyL1ToL0BMxFp8A5()"),
    ("struct CopyL1ToL0AMxQKA5", "struct CopyL1ToL0AMxFp8QKA5"),
    ("CopyL1ToL0AMxQKA5()", "CopyL1ToL0AMxFp8QKA5()"),
    ("struct CopyL1ToL0BMxQKA5", "struct CopyL1ToL0BMxFp8QKA5"),
    ("CopyL1ToL0BMxQKA5()", "CopyL1ToL0BMxFp8QKA5()"),
]
for a, b in repls:
    if a not in text:
        raise SystemExit(f"missing {a}")
    text = text.replace(a, b)
wr(p, text, nl)
print("renamed structs in copy_l1_to_l0_mx_fp8_a5.hpp")

p, text, nl = rw("attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_mxfp8.hpp")
text = text.replace("Tile::CopyL1ToL0AMxA5", "Tile::CopyL1ToL0AMxFp8A5")
text = text.replace("Tile::CopyL1ToL0BMxA5", "Tile::CopyL1ToL0BMxFp8A5")
wr(p, text, nl)
print("patched pv mxfp8")

p, text, nl = rw("attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_mxfp8.hpp")
text = text.replace("Tile::CopyL1ToL0AMxQKA5", "Tile::CopyL1ToL0AMxFp8QKA5")
text = text.replace("Tile::CopyL1ToL0BMxQKA5", "Tile::CopyL1ToL0BMxFp8QKA5")
wr(p, text, nl)
print("patched qk mxfp8")

old = (
    "        AscendC::DataCopyL1ToUB<ElementLocalGlobalMax, VEC0>(peerGlobalMax, localGlobalMax0, intriParams);\n"
    "        AscendC::DataCopyL1ToUB<ElementLocalGlobalMax, VEC1>(peerGlobalMax, localGlobalMax1, intriParams);"
)
new = (
    "        // beta.1 无 DataCopyL1ToUB<T,subBlockId>；按 subBlock 各拷自己的半面。\n"
    "        if (AscendC::GetSubBlockIdx() == VEC0) {\n"
    "            AscendC::DataCopy(peerGlobalMax, localGlobalMax0, intriParams);\n"
    "        } else {\n"
    "            AscendC::DataCopy(peerGlobalMax, localGlobalMax1, intriParams);\n"
    "        }"
)
for rel in (
    "attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_copy_global_max_l1_to_ub_arch35_mxfp4.hpp",
    "attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_copy_global_max_l1_to_ub_arch35_mxfp8.hpp",
):
    p, text, nl = rw(rel)
    if old not in text:
        raise SystemExit(f"DataCopyL1ToUB block not found in {rel}")
    wr(p, text.replace(old, new, 1), nl)
    print("patched", Path(rel).name)

print("ok")
