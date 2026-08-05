# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(grouped_matmul_arch35_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/grouped_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/arch35/grouped_int4_quant_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp)

    AddOpTestCase(
        grouped_matmul
        "Ascend950PR_9599"
        "-DKERNELUT=1 -DBLAZE_UT=1 -DDTYPE_X=int8_t -DDTYPE_WEIGHT=AscendC::int4b_t -DDTYPE_BIAS=float -DDTYPE_SCALE=uint64_t -DDTYPE_Y=half -DORIG_DTYPE_X=DT_INT8 -DORIG_DTYPE_WEIGHT=DT_INT4 -DORIG_DTYPE_SCALE=DT_UINT64 -DORIG_DTYPE_Y=DT_FLOAT16"
        "${grouped_matmul_arch35_tiling_files}"
        "${CMAKE_CURRENT_SOURCE_DIR}/gmm_s8s4_v5_kernel_stub.cpp")

    target_include_directories(grouped_matmul_Ascend950PR_9599_tiling_tmp PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${OPTENSOR_SOURCE_PATH}/include)
    target_sources(grouped_matmul_Ascend950PR_9599_tiling_tmp PRIVATE
        $<TARGET_OBJECTS:opbase_util_objs>)

    target_include_directories(grouped_matmul_Ascend950PR_9599_cases_obj PRIVATE
        ${OPTENSOR_SOURCE_PATH}/include
        ${OPTENSOR_SOURCE_PATH}/include/tensor_api/include
        ${OPTENSOR_SOURCE_PATH}/include/tensor_api
        ${OPTENSOR_SOURCE_PATH}/tests/ut/op_kernel
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/c_api
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/impl/c_api
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/impl/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_kernel)
endif()
