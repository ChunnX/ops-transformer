#!/usr/bin/env python3
"""Fix mxfp8 include-guard collisions with mxfp4, and add make_wrapper."""
from pathlib import Path

ROOT = Path("/home/npu_user5/xuxuepeng/ops-transformer_0703")


def read_text(p: Path):
    raw = p.read_bytes()
    nl = b"\r\n" if b"\r\n" in raw else b"\n"
    text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8")
    return text, nl


def write_text(p: Path, text: str, nl: bytes):
    out = text.encode("utf-8")
    if nl == b"\r\n":
        out = out.replace(b"\n", b"\r\n")
    p.write_bytes(out)


def replace_once(text, old, new, label):
    if old not in text:
        raise SystemExit(f"{label}: pattern not found:\n{old!r}")
    return text.replace(old, new, 1)


# --- include guards ---
p = ROOT / "attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/mxfp8_vf/vf_common_def.h"
text, nl = read_text(p)
text = text.replace("#ifndef VF_COMMON_DEF_H_\n#define VF_COMMON_DEF_H_\n",
                    "#ifndef VF_COMMON_DEF_MXFP8_H_\n#define VF_COMMON_DEF_MXFP8_H_\n")
text = text.replace("#endif // VF_COMMON_DEF_H_", "#endif // VF_COMMON_DEF_MXFP8_H_")
write_text(p, text, nl)
print("patched", p.name)

p = ROOT / "attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/mxfp8_vf/vf_mm1_res_pre_padding_align_kvs32_multi.h"
text, nl = read_text(p)
text = text.replace(
    "#ifndef VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_H\n#define VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_H\n",
    "#ifndef VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_MXFP8_H\n#define VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_MXFP8_H\n",
)
write_text(p, text, nl)
print("patched", p.name)

p = ROOT / "attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/mxfp8_vf/vf_mm1_res_pre_padding_align_kvs32_multi_qs64.h"
text, nl = read_text(p)
text = text.replace(
    "#ifndef VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_H\n#define VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_H\n",
    "#ifndef VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_MXFP8_H\n#define VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_QS64_MXFP8_H\n",
)
write_text(p, text, nl)
print("patched", p.name)

# --- make_wrapper ---
p = ROOT / "torch_extension/cann_ops_transformer/ops/csrc/block_sparse_attention.cpp"
text, nl = read_text(p)
if "TensorWrapper make_wrapper(" in text:
    print("make_wrapper already present")
else:
    needle = "const int64_t FLOAT8_E8M0 = static_cast<int64_t>(293);\n"
    insert = needle + """
TensorWrapper make_wrapper(const at::Tensor &tensor, const int64_t real_dtype)
{
	TensorWrapper wrapper = {tensor, ACL_DT_UNDEFINED};
	auto physical = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())];
	if (physical == ACL_UINT8) {
		if (real_dtype == FLOAT4_E2M1) {
			wrapper.dtype = ACL_FLOAT4_E2M1;
		} else if (real_dtype == FLOAT8_E8M0) {
			wrapper.dtype = ACL_FLOAT8_E8M0;
		}
	} else if (real_dtype == FLOAT8_E8M0) {
		wrapper.dtype = ACL_FLOAT8_E8M0;
	} else {
		wrapper.dtype = physical;
	}
	return wrapper;
}

"""
    text = replace_once(text, needle, insert, "make_wrapper")
    write_text(p, text, nl)
    print("patched make_wrapper")

print("ok")
