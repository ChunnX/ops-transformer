@echo off
set OPS=D:\code\ops-transformer\torch_extension\cann_ops_transformer\ops
set ROOT=D:\code\ops-transformer
set SRC_CSRC=%OPS%\csrc
set SRC_GC=%OPS%\graph_convert

call :copydir attention compressor gc
call :copydir attention dense_lightning_indexer_kl_loss_grad ""
call :copydir attention dense_lightning_indexer_softmax_lse ""
call :copydir attention flash_attn gc
call :copydir attention indexer_quant_cache gc
call :copydir attention kv_compress_epilog gc
call :copydir attention lightning_indexer gc
call :copydir attention mixed_quant_sparse_flash_mla gc
call :copydir attention quant_compressor gc
call :copydir attention quant_flash_attn gc
call :copydir attention quant_lightning_indexer gc
call :copydir attention quant_sparse_flash_mla ""
call :copydir attention scatter_pa_kv_cache_with_k_scale gc
call :copydir attention sparse_flash_mla gc
call :copydir attention sparse_flash_mla_grad ""
call :copydir attention sparse_flash_mla_softmax_l1_norm ""
call :copydir attention sparse_lightning_indexer_kl_loss_grad ""
call :copydir attention stem_oam_prep_paged_kv gc
call :copydir attention stem_oam_prep_varlen_q gc
call :copydir gmm grouped_matmul_activation_quant ""
call :copydir mc2 mega_moe gc
call :copydir mc2 moe_distribute_combine gc
call :copydir mhc mhc_post ""
call :copydir mhc mhc_post_backward ""
call :copydir mhc mhc_pre_sinkhorn gc
call :copydir mhc mhc_pre_sinkhorn_backward gc
call :copydir moe moe_finalize_routing gc
call :copydir moe moe_token_permute ""
call :copydir posembedding inplace_partial_rotary_mul gc
call :copydir posembedding qkv_rms_norm_rope_cache_with_k_scale ""

echo Done: 30 standard ops
goto :eof

:copydir
set CAT=%1
set NAME=%2
set GC=%3
set TE=%ROOT%\%CAT%\%NAME%\torch_extension
set CS=%TE%\csrc

mkdir "%CS%" 2>nul

copy /Y "%OPS%\%NAME%.py" "%TE%\%NAME%.py" >nul
copy /Y "%SRC_CSRC%\%NAME%.cpp" "%CS%\%NAME%.cpp" >nul

if "%GC%"=="gc" (
    copy /Y "%SRC_GC%\graph_convert_%NAME%.py" "%TE%\graph_convert_%NAME%.py" >nul
    echo __all__ = ["%NAME%", "convert_%NAME%"] > "%TE%\__init__.py"
    echo. >> "%TE%\__init__.py"
    echo from .%NAME% import %NAME% >> "%TE%\__init__.py"
    echo from .graph_convert_%NAME% import convert_%NAME% >> "%TE%\__init__.py"
) else (
    echo __all__ = ["%NAME%"] > "%TE%\__init__.py"
    echo. >> "%TE%\__init__.py"
    echo from .%NAME% import %NAME% >> "%TE%\__init__.py"
)
echo OK: %CAT%/%NAME%
goto :eof
