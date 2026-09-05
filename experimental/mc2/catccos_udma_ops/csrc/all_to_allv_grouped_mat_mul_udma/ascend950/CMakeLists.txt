set(ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_ASC_FLAGS
    "--npu-arch=dav-3510 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false ${CATCCOS_UDMA_ASCEND_FLAGS}")
add_sources("${ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_ASC_FLAGS}")
