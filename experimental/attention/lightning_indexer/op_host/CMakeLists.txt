add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE lightning_indexer_def.cpp)
endif()

# The source project uses the Ascend C default kernel options (including auto-sync).
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE lightning_indexer ACLNNTYPE aclnn)
endif()
