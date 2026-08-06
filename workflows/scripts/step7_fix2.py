import os
import re
import glob

ROOT = r"D:\code\ops-transformer"

def fix_duplicates(filepath):
    """Remove consecutive duplicate _ensure_initialized() lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed = []
    prev_line = ""
    for line in lines:
        if "_ensure_initialized()" in line and "_ensure_initialized()" in prev_line:
            continue
        fixed.append(line)
        prev_line = line.rstrip()

    if len(fixed) != len(lines):
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(fixed)
        return True
    return False

def fix_multiline_super(filepath, category):
    """Fix multi-line super().__init__ to add category."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern: super(X, self).__init__(\n    "op_name"
    # Should become: super(X, self).__init__(\n    "op_name", category="cat"
    pattern = re.compile(
        r'(super\(\w+,\s*self\)\.__init__\(\s*"([^"]+)"\s*)$',
        re.MULTILINE,
    )
    fixed = pattern.sub(
        lambda m: m.group(1) + f', category="{category}"',
        content,
    )
    
    if fixed != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(fixed)
        return True
    return False

def fix_deep_ep_import(filepath):
    """Fix deep_ep.py's relative comm_context import."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    old = "from .comm_context import CommContextManager"
    new = "from cann_ops_transformer.ops.mc2.common import CommContextManager"
    if old in content:
        content = content.replace(old, new)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False

def fix_mega_moe_import(filepath):
    """Fix mega_moe.py's relative comm_context import."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    old = "from .comm_context import CommContextManager"
    new = "from cann_ops_transformer.ops.mc2.common import CommContextManager"
    if old in content:
        content = content.replace(old, new)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False

def fix_sources_args(filepath, category):
    """Fix sources() that return split across lines like return ["ops/csrc/xxx.cpp"]"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    old_pattern = re.compile(r'ops/csrc/([^"\s]+\.cpp)')
    new_content = old_pattern.sub(f'csrc/{category}/\\1', content)
    
    if new_content != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False

CATEGORY_MAP = {
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
    "grouped_matmul_activation_quant": "gmm",
    "causal_conv1d_fn": "mamba",
    "causal_conv1d_update": "mamba",
    "comm_context": "mc2",
    "elastic_buffer": "mc2",
    "mega_moe": "mc2",
    "deep_ep": "mc2",
    "mhc_post": "mhc",
    "mhc_post_backward": "mhc",
    "mhc_pre_sinkhorn": "mhc",
    "mhc_pre_sinkhorn_backward": "mhc",
    "moe_finalize_routing": "moe",
    "moe_token_permute": "moe",
    "inplace_partial_rotary_mul": "posembedding",
    "inplace_partial_rotary_mul_backward": "posembedding",
    "qkv_rms_norm_rope_cache_with_k_scale": "posembedding",
}

def main():
    pattern = os.path.join(ROOT, "*", "*", "torch_extension", "*.py")
    files = [f for f in glob.glob(pattern) if os.path.basename(f) != "__init__.py"]
    files = sorted(files)

    print("=== Fix 1: Remove duplicate _ensure_initialized() ===")
    for f in files:
        if fix_duplicates(f):
            print(f"  FIXED DUP: {os.path.relpath(f, ROOT)}")

    print("\n=== Fix 2: Multi-line super().__init__ ===")
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        cat = CATEGORY_MAP.get(basename)
        if cat and fix_multiline_super(f, cat):
            print(f"  FIXED SUPER: {os.path.relpath(f, ROOT)}")

    print("\n=== Fix 3: Cross-module import paths ===")
    for f in files:
        if fix_deep_ep_import(f):
            print(f"  FIXED IMPORT: {os.path.relpath(f, ROOT)} (comm_context)")
        if fix_mega_moe_import(f):
            print(f"  FIXED IMPORT: {os.path.relpath(f, ROOT)} (comm_context)")

    print("\n=== Fix 4: Remaining multi-line sources ===")
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        cat = CATEGORY_MAP.get(basename)
        if cat and fix_sources_args(f, cat):
            print(f"  FIXED SOURCE: {os.path.relpath(f, ROOT)}")

    # Check for remaining AS_LIBRARY in distributed files
    print("\n=== Final check: AS_LIBRARY references ===")
    remaining = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            if "AS_LIBRARY" in fp.read():
                remaining.append(os.path.relpath(f, ROOT))
    if remaining:
        for r in remaining:
            print(f"  REMAINING: {r}")
    else:
        print("  None - all clear!")

if __name__ == "__main__":
    main()
