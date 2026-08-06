import os
import re
import glob

ROOT = r"D:\code\ops-transformer"

CATEGORY_MAP = {
    # attention
    "compressor": "attention",
    "dense_lightning_indexer_kl_loss_grad": "attention",
    "dense_lightning_indexer_softmax_lse": "attention",
    "flash_attn": "attention",
    "indexer_quant_cache": "attention",
    "kv_compress_epilog": "attention",
    "lightning_indexer": "attention",
    "mixed_quant_sparse_flash_mla": "attention",
    "quant_compressor": "attention",
    "quant_flash_attn": "attention",
    "quant_lightning_indexer": "attention",
    "quant_sparse_flash_mla": "attention",
    "scatter_pa_kv_cache_with_k_scale": "attention",
    "sparse_flash_mla": "attention",
    "sparse_flash_mla_grad": "attention",
    "sparse_flash_mla_softmax_l1_norm": "attention",
    "sparse_lightning_indexer_kl_loss_grad": "attention",
    "stem_oam_prep_paged_kv": "attention",
    "stem_oam_prep_varlen_q": "attention",
    # gmm
    "grouped_matmul_activation_quant": "gmm",
    # mamba
    "causal_conv1d_fn": "mamba",
    "causal_conv1d_update": "mamba",
    # mc2
    "comm_context": "mc2",
    "elastic_buffer": "mc2",
    "mega_moe": "mc2",
    # deep_ep.py contains dispatch+combine, both mc2
    "deep_ep": "mc2",
    # mhc
    "mhc_post": "mhc",
    "mhc_post_backward": "mhc",
    "mhc_pre_sinkhorn": "mhc",
    "mhc_pre_sinkhorn_backward": "mhc",
    # moe
    "moe_finalize_routing": "moe",
    "moe_token_permute": "moe",
    # posembedding
    "inplace_partial_rotary_mul": "posembedding",
    "inplace_partial_rotary_mul_backward": "posembedding",
    "qkv_rms_norm_rope_cache_with_k_scale": "posembedding",
}

def process_file(filepath):
    # Derive the base name (operator name) from the file path
    basename = os.path.splitext(os.path.basename(filepath))[0]
    category = CATEGORY_MAP.get(basename)
    if category is None:
        print(f"  SKIP (no category): {basename}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    modified = content

    # 1. Combine/replace the import lines
    # Pattern A: two separate lines for OpBuilder and AS_LIBRARY
    pattern_a = re.compile(
        r"from cann_ops_transformer\.op_builder\.builder import OpBuilder\n"
        r"from cann_ops_transformer\.op_builder\.builder import AS_LIBRARY"
    )
    if pattern_a.search(modified):
        modified = pattern_a.sub(
            "from cann_ops_transformer.op_builder import OpBuilder, get_as_library",
            modified,
        )
        print(f"  [import] split -> combined")

    # Pattern B: single line: from cann_ops_transformer.op_builder.builder import OpBuilder, AS_LIBRARY
    pattern_b = re.compile(
        r"from cann_ops_transformer\.op_builder\.builder import (OpBuilder,\s*AS_LIBRARY|AS_LIBRARY,\s*OpBuilder)"
    )
    if pattern_b.search(modified):
        modified = pattern_b.sub(
            "from cann_ops_transformer.op_builder import OpBuilder, get_as_library",
            modified,
        )
        print(f"  [import] single line -> combined")

    # Pattern C: only OpBuilder (no AS_LIBRARY), like comm_context.py
    pattern_c = re.compile(
        r"from cann_ops_transformer\.op_builder\.builder import OpBuilder$",
        re.MULTILINE,
    )
    if pattern_c.search(modified):
        modified = pattern_c.sub(
            "from cann_ops_transformer.op_builder import OpBuilder, get_as_library",
            modified,
        )
        print(f"  [import] OpBuilder only -> combined")

    # Pattern D: AS_LIBRARY only (some files import them on separate non-adjacent lines)
    pattern_d = re.compile(
        r"from cann_ops_transformer\.op_builder\.builder import AS_LIBRARY\n"
    )
    if pattern_d.search(modified):
        modified = pattern_d.sub("", modified)
        print(f"  [import] removed standalone AS_LIBRARY line")

    # Pattern E: remaining single import of OpBuilder from old path
    if "from cann_ops_transformer.op_builder.builder import OpBuilder" in modified:
        modified = modified.replace(
            "from cann_ops_transformer.op_builder.builder import OpBuilder",
            "from cann_ops_transformer.op_builder import OpBuilder, get_as_library",
        )
        print(f"  [import] fallback replace")

    # 2. Replace all remaining AS_LIBRARY references with get_as_library()
    if "AS_LIBRARY" in modified:
        modified = modified.replace("AS_LIBRARY", "get_as_library()")
        print(f"  [AS_LIBRARY] replaced all references")

    # 3. Add category to super().__init__()
    # Pattern: super(ClassName, self).__init__("op_name")
    pattern_super1 = re.compile(
        r'(super\(\w+,\s*self\)\.__init__\("([^"]+)"\))'
    )
    # Pattern: super().__init__("op_name")  
    pattern_super2 = re.compile(
        r'(super\(\)\.__init__\("([^"]+)"\))'
    )

    def add_category(m):
        full = m.group(1)
        op = m.group(2)
        if ', category=' in full:
            return full
        return full[:-1] + f', category="{category}")'

    modified, count1 = pattern_super1.subn(add_category, modified)
    modified, count2 = pattern_super2.subn(add_category, modified)
    if count1 + count2 > 0:
        print(f"  [category] {count1 + count2} super().__init__ updated")

    # 4. Update sources path: ops/csrc/X.cpp -> csrc/category/X.cpp
    pattern_src = re.compile(r'ops/csrc/([^"]+\.cpp)')
    def fix_src(m):
        return f"csrc/{category}/{m.group(1)}"
    modified, count_src = pattern_src.subn(fix_src, modified)
    if count_src > 0:
        print(f"  [sources] {count_src} path(s) updated")

    # 5. Add _ensure_initialized() after builder instantiation
    # Pattern: <var> = <ClassName>()
    # or: <var> = <ClassName>(...)
    pattern_builder = re.compile(
        r'^(\w+)\s*=\s*(\w+OpBuilder|MoeDistributeDispatchOpBuilder|MoeDistributeCombineOpBuilder'
        r'|MhcPostOpBuilder|MhcPreSinkhornOpBuilder|MhcPreSinkhornBackwardOpBuilder'
        r'|QkvRmsNormRopeCacheWithKScaleOpBuilder)\(\)$',
        re.MULTILINE,
    )

    def add_ensure(m):
        return m.group(0) + f"\n{m.group(1)}._ensure_initialized()"

    modified, count_ensure = pattern_builder.subn(add_ensure, modified)

    # Also handle patterns for named constructors: _xxx_op_builder = XxxOpBuilder()
    pattern_builder2 = re.compile(
        r'^(_\w+_op_builder|_?\w+_op_builder)\s*=\s*(\w+OpBuilder'
        r'|MoeDistributeDispatchOpBuilder|MoeDistributeCombineOpBuilder'
        r'|QkvRmsNormRopeCacheWithKScaleOpBuilder)\(\)$',
        re.MULTILINE,
    )
    def add_ensure2(m):
        if "_ensure_initialized()" in m.group(0):
            return m.group(0)
        return m.group(0) + f"\n{m.group(1)}._ensure_initialized()"
    modified, count_ensure2 = pattern_builder2.subn(add_ensure2, modified)
    
    total_ensure = count_ensure + count_ensure2
    if total_ensure > 0:
        print(f"  [ensure] {total_ensure} builder(s) updated")

    if modified == content:
        print(f"  NO CHANGES")
        return

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(modified)


def main():
    pattern = os.path.join(ROOT, "*", "*", "torch_extension", "*.py")
    files = [f for f in glob.glob(pattern) if os.path.basename(f) != "__init__.py"]

    print(f"Found {len(files)} .py files to process\n")

    for f in sorted(files):
        rel = os.path.relpath(f, ROOT)
        print(f"{rel}:")
        process_file(f)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
