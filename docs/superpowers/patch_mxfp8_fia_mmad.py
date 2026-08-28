#!/usr/bin/env python3
"""Align BSA mxfp8 QK/PV Mmad L0 types with FIA MatmulFullMX.

FIA: LoadData dest AND Mmad operands are mx_fp8_e4m3_t.
BSA previously only recast LoadData dest; Mmad still used fp8_e4m3fn_t,
so cube did raw e4m3 matmul (S overflows fp16 → inf).
"""
from pathlib import Path

ROOT = Path("/home/npu_user5/xuxuepeng/ops-transformer_0703")


def rw(rel, old, new):
    p = ROOT / rel
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern not found in {rel}:\n{old[:180]!r}")
    if text.count(old) != 1:
        raise SystemExit(f"pattern count={text.count(old)} in {rel}")
    p.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", rel)


rw(
    "attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_mxfp8.hpp",
    """        auto l0ALayout = tla::MakeLayout<ElementB, LayoutTagL0A>(n, kCur);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[qkL0aBufId], l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<ElementA, LayoutTagL0B>(kCur, qsActBaseTileAlign16);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[qkL0bBufId], l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(n, qsActBaseTileAlign16);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[qkL0cBufId], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, n, qsActBaseTileAlign16, kCur, initC);""",
    """        // FIA MatmulFullMX: Mmad 操作数必须是 mx_fp8_e4m3_t（与 LoadData L0 dest 一致），
        // 否则 cube 按普通 e4m3 乘加，S 在 fp16 fixpipe 溢出成 inf。
        auto l0ALayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0A>(n, kCur);
        auto l0ATensorTla = tla::MakeTensor(
            l0ATensor[qkL0aBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
            l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0B>(kCur, qsActBaseTileAlign16);
        auto l0BTensorTla = tla::MakeTensor(
            l0BTensor[qkL0bBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
            l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(n, qsActBaseTileAlign16);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[qkL0cBufId], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, n, qsActBaseTileAlign16, kCur, initC);""",
)

# PV file uses 5-space indent on this block
pv = (ROOT / "attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_mxfp8.hpp").read_text(encoding="utf-8")
old_pv = """         auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(m, k);
         auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayout, Arch::PositionL0A{});
         auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
         auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayout, Arch::PositionL0B{});
         auto l0CLayout = tla::MakeLayoutL0C(m, n);
         auto l0CTensorTla = tla::MakeTensor(l0CTensor[0], l0CLayout, Arch::PositionL0C{});
         tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, m, n, k, initC);"""
new_pv = """         auto l0ALayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0A>(m, k);
         auto l0ATensorTla = tla::MakeTensor(
             l0ATensor[l0ABufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
             l0ALayout, Arch::PositionL0A{});
         auto l0BLayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0B>(k, n);
         auto l0BTensorTla = tla::MakeTensor(
             l0BTensor[l0BBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
             l0BLayout, Arch::PositionL0B{});
         auto l0CLayout = tla::MakeLayoutL0C(m, n);
         auto l0CTensorTla = tla::MakeTensor(l0CTensor[0], l0CLayout, Arch::PositionL0C{});
         tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, m, n, k, initC);"""
if old_pv not in pv:
    raise SystemExit("PV MatmulPV pattern not found")
(ROOT / "attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_mxfp8.hpp").write_text(
    pv.replace(old_pv, new_pv, 1), encoding="utf-8")
print("patched pv MatmulPV")

rw(
    "attention/block_sparse_attention/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp",
    "        loadMxDataParams.srcStride = (scaleK + 1) / 2; // full D scale row half\n        loadMxDataParams.dstStride = scaleRowHalf;",
    "        // FIA LoadDataToL0AMx: srcStride = yStep = (K+63)>>5>>1，不是 (scaleK+1)/2\n        loadMxDataParams.srcStride = scaleRowHalf;\n        loadMxDataParams.dstStride = scaleRowHalf;",
)

rw(
    "attention/block_sparse_attention/op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp",
    "        loadMxDataParams.srcStride = (scaleK + 1) / 2;\n        loadMxDataParams.dstStride = scaleRowHalf;",
    "        loadMxDataParams.srcStride = scaleRowHalf;\n        loadMxDataParams.dstStride = scaleRowHalf;",
)

rw(
    "attention/block_sparse_attention/op_kernel/attn_infra/gemm/tile_common/copy_gm_to_l1_mx_scale_dn2nz_a5.hpp",
    "        dn2nzParams.dstNzNStride = 1;\n        dn2nzParams.dstNzMatrixStride = 0;",
    "        dn2nzParams.dstNzNStride = 1;\n        // FIA CopySingleMXScaleDNToNZ: dstNzMatrixStride = nValue（half 列数）\n        dn2nzParams.dstNzMatrixStride = dn2nzParams.nValue;",
)

print("done")
