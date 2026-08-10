set(ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_ASC_FLAGS
    "--npu-arch=dav-3510 --cce-aicore-input-parameter-size=4096 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false")
add_sources("${ALL_TO_ALLV_GROUPED_MAT_MUL_UDMA_ASC_FLAGS}")
