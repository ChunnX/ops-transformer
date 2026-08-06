$OPS = "D:\code\ops-transformer\torch_extension\cann_ops_transformer\ops"
$ROOT = "D:\code\ops-transformer"
$SRC_CSRC = "$OPS\csrc"
$SRC_GC = "$OPS\graph_convert"

$ops = @(
    @("attention", "compressor", "gc"),
    @("attention", "dense_lightning_indexer_kl_loss_grad", ""),
    @("attention", "dense_lightning_indexer_softmax_lse", ""),
    @("attention", "flash_attn", "gc"),
    @("attention", "indexer_quant_cache", "gc"),
    @("attention", "kv_compress_epilog", "gc"),
    @("attention", "lightning_indexer", "gc"),
    @("attention", "mixed_quant_sparse_flash_mla", "gc"),
    @("attention", "quant_compressor", "gc"),
    @("attention", "quant_flash_attn", "gc"),
    @("attention", "quant_lightning_indexer", "gc"),
    @("attention", "quant_sparse_flash_mla", ""),
    @("attention", "scatter_pa_kv_cache_with_k_scale", "gc"),
    @("attention", "sparse_flash_mla", "gc"),
    @("attention", "sparse_flash_mla_grad", ""),
    @("attention", "sparse_flash_mla_softmax_l1_norm", ""),
    @("attention", "sparse_lightning_indexer_kl_loss_grad", ""),
    @("attention", "stem_oam_prep_paged_kv", "gc"),
    @("attention", "stem_oam_prep_varlen_q", "gc"),
    @("gmm", "grouped_matmul_activation_quant", ""),
    @("mc2", "mega_moe", "gc"),
    @("mc2", "moe_distribute_combine", "gc"),
    @("mhc", "mhc_post", ""),
    @("mhc", "mhc_post_backward", ""),
    @("mhc", "mhc_pre_sinkhorn", "gc"),
    @("mhc", "mhc_pre_sinkhorn_backward", "gc"),
    @("moe", "moe_finalize_routing", "gc"),
    @("moe", "moe_token_permute", ""),
    @("posembedding", "inplace_partial_rotary_mul", "gc"),
    @("posembedding", "qkv_rms_norm_rope_cache_with_k_scale", "")
)

$count = 0
foreach ($op in $ops) {
    $cat = $op[0].ToString()
    $name = $op[1].ToString()
    $hasGc = ($op[2].ToString() -eq "gc")

    $te = "$ROOT\$cat\$name\torch_extension"
    $cs = "$te\csrc"

    New-Item -ItemType Directory -Path $cs -Force | Out-Null
    Copy-Item "$OPS\$name.py" "$te\$name.py" -Force
    Copy-Item "$SRC_CSRC\$name.cpp" "$cs\$name.cpp" -Force

    if ($hasGc) {
        Copy-Item "$SRC_GC\graph_convert_$name.py" "$te\graph_convert_$name.py" -Force
        $init = "__all__ = [`"$name`", `"convert_$name`"]`n`nfrom .$name import $name`nfrom .graph_convert_$name import convert_$name`n"
    } else {
        $init = "__all__ = [`"$name`"]`n`nfrom .$name import $name`n"
    }
    Set-Content -Path "$te\__init__.py" -Value $init
    $count++
    Write-Output "OK: $cat/$name"
}
Write-Output "Done: $count ops"
