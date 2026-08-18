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


p, text, nl = rw("attention/block_sparse_attention/op_kernel/arch35/kernel_utils.hpp")
old_tail = '''}

#include "../attn_infra/epilogue/block/bsa_block_epilogue.hpp"
'''
new_tail = '''}

namespace MXFP8Kernel {
static constexpr uint32_t TILE_GROUP_N = 8;
using TaskInfo = MXFP4Kernel::TaskInfo;
using TileInfo = MXFP4Kernel::TileInfo;
using BatchOffsetInfo = MXFP4Kernel::BatchOffsetInfo;
}

#include "../attn_infra/epilogue/block/bsa_block_epilogue.hpp"
'''
if old_tail not in text:
    raise SystemExit("kernel_utils: include insertion point not found")
text = text.replace(old_tail, new_tail, 1)

old_ns = '''namespace MXFP8Kernel {
static constexpr uint32_t TILE_GROUP_N = 8;
using TaskInfo = MXFP4Kernel::TaskInfo;
using TileInfo = MXFP4Kernel::TileInfo;
using BatchOffsetInfo = MXFP4Kernel::BatchOffsetInfo;
using BsaFullQuantKernelParamsArch35 = MXFP4Kernel::BsaFullQuantKernelParamsArch35;
__aicore__ inline uint32_t GetCurQSTileNum(int64_t curQSeqlen, uint32_t blockShapeX, uint32_t qBaseTile)
{
    return MXFP4Kernel::GetCurQSTileNum(curQSeqlen, blockShapeX, qBaseTile);
}
}  // namespace MXFP8Kernel
'''
new_ns = '''namespace MXFP8Kernel {
using BsaFullQuantKernelParamsArch35 = BsaKernelArch35::BsaFullQuantKernelParamsArch35;
__aicore__ inline uint32_t GetCurQSTileNum(int64_t curQSeqlen, uint32_t blockShapeX, uint32_t qBaseTile)
{
    return BsaKernelArch35::GetCurQSTileNum(curQSeqlen, blockShapeX, qBaseTile);
}
}  // namespace MXFP8Kernel
'''
if old_ns not in text:
    raise SystemExit("kernel_utils: trailing MXFP8Kernel block not found")
text = text.replace(old_ns, new_ns, 1)
wr(p, text, nl)
print("patched kernel_utils.hpp")

qualify = [
    ("attention/block_sparse_attention/op_kernel/arch35/block_sparse_attention_kernel_arch35_mxfp8_full_quant.h",
     "TILE_GROUP_N", "MXFP8Kernel::TILE_GROUP_N"),
    ("attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/block_epilogue_compute_pscale_arch35_mxfp8.hpp",
     "TILE_GROUP_N", "MXFP8Kernel::TILE_GROUP_N"),
    ("attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o_arch35_reg_high_prec_mxfp8.hpp",
     "TILE_GROUP_N", "MXFP8Kernel::TILE_GROUP_N"),
    ("attention/block_sparse_attention/op_kernel/arch35/block_sparse_attention_kernel_arch35_mxfp4_full_quant.h",
     "TILE_GROUP_N", "MXFP4Kernel::TILE_GROUP_N"),
    ("attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/block_epilogue_compute_pscale_arch35_mxfp4.hpp",
     "TILE_GROUP_N", "MXFP4Kernel::TILE_GROUP_N"),
    ("attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o_arch35_reg_high_prec_mxfp4.hpp",
     "TILE_GROUP_N", "MXFP4Kernel::TILE_GROUP_N"),
]
for rel, old, new in qualify:
    p, text, nl = rw(rel)
    n = text.count(old)
    # Don't rewrite already-qualified or comments like `// TILE_GROUP_N`
    parts = []
    i = 0
    replaced = 0
    while True:
        j = text.find(old, i)
        if j < 0:
            parts.append(text[i:])
            break
        before = text[max(0, j - 20):j]
        if "MXFP4Kernel::" in before[-13:] or "MXFP8Kernel::" in before[-13:]:
            parts.append(text[i:j + len(old)])
            i = j + len(old)
            continue
        # skip comment-only occurrences
        line_start = text.rfind("\n", 0, j) + 1
        line = text[line_start:text.find("\n", j)]
        code = line.split("//", 1)[0]
        if old not in code:
            parts.append(text[i:j + len(old)])
            i = j + len(old)
            continue
        parts.append(text[i:j] + new)
        i = j + len(old)
        replaced += 1
    wr(p, "".join(parts), nl)
    print(f"qualified {replaced} TILE_GROUP_N in {Path(rel).name}")

print("ok")
