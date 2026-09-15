# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 算子自己的tiling文件路径
    set(moe_tutel_combine_x_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/moe_tutel_combine_x_tiling_arch35.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/moe_tutel_combine_x_tiling_base.cpp
    )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本。须使用 CANN cpudebug 模拟库提供的标签(950 系为 Ascend950PR_9599, 见
    #         $ASCEND_HOME_PATH/tools/cpudebug/cmake/targets-tikicpulib.cmake), 与真机型号无关
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(moe_tutel_combine_x "Ascend950PR_9599" "-DDTYPE_Y_GRAD=float" "${moe_tutel_combine_x_tiling_files}")

    # kernel 入口为 op_kernel/moe_tutel_combine_x_apt.cpp(带 _apt 后缀, UT 框架按
    # op_kernel/{op_name}.cpp 无法 glob 到), 由测试编译单元直接包含 kernel 源文件
    # (见 test_moe_tutel_combine_x.cpp 顶部 #include), 故此处不传 kernel 文件参数。

    # tiling 源码(moe_tutel_combine_x_tiling_arch35.h)包含 CANN 平台头
    # platform/platform_ascendc.h(读取 coreNum / ubSize), 该头所在的
    # ${ASCEND_DIR}/{arch}-linux/asc/include/tiling 不在 ut.cmake 为 tiling_tmp
    # 目标设置的默认 include 集中(本算子是仓库首个在 tiling 中使用平台信息查询的
    # kernel UT 算子)。参照 block_attn_res_prepare 在 AddOpTestCase 之后按需增补
    # tiling_tmp 目标属性的范式补齐(x86_64 与 aarch64 两种布局都给出)。
    set(moe_tutel_combine_x_tiling_tmp "moe_tutel_combine_x_Ascend950PR_9599_tiling_tmp")
    if(TARGET ${moe_tutel_combine_x_tiling_tmp})
        target_include_directories(${moe_tutel_combine_x_tiling_tmp} PRIVATE
            ${ASCEND_DIR}/x86_64-linux/asc/include/tiling
            ${ASCEND_DIR}/aarch64-linux/asc/include/tiling
        )
    endif()
endif()
