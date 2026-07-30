add_op_to_compiled_list()

option(BSA_ENABLE_FP8_RESCALE_PROBE "Enable FP8 PV rescale mad probe" OFF)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        block_sparse_attention_def.cpp
    )
endif()

set(_BSA_KERNEL_OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -fpermissive
)
if (BSA_ENABLE_FP8_RESCALE_PROBE)
    list(APPEND _BSA_KERNEL_OPTIONS -DBSA_PV_RESCALE_PROBE_FP8)
endif()
add_ops_compile_options(
        OP_NAME BlockSparseAttention
        OPTIONS ${_BSA_KERNEL_OPTIONS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE block_sparse_attention ACLNNTYPE aclnn_exclude)
endif()
