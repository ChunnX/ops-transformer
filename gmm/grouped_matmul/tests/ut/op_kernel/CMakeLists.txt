# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # tiling 源码引用仓库根目录相对路径头文件（如 gmm/common/op_host/log_format_util.h），补充根目录 include 路径
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
    # 算子自己的tiling文件路径
    set(grouped_matmul_tiling_files
        # opbase log.cpp：tiling 源码使用 OP_LOGE_FOR_* 系列宏（展开为 opbase 的 REPORT 函数及 Ops::Base::ToString），
        # kernel UT 独立链接 tiling_tmp.so 时需要其实现
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/opbase/src/op_common/log/log.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/op_tiling/grouped_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本，多个以分号分隔，例如："Ascend950PR_9599;AscendB1"
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串，例如："-DDTYPE_X=float"，多个使用空格分隔，例如："-DDTYPE_X=float -DDTYPE_Y=float"
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(grouped_matmul "ascend910B1" "-DDTYPE_X=half -DDTYPE_WEIGHT=half -DDTYPE_BIAS=half -DDTYPE_SCALE=uint64_t -DDTYPE_Y=half -DORIG_DTYPE_X=DT_FLOAT16 -DORIG_DTYPE_WEIGHT=DT_FLOAT16 -DORIG_DTYPE_Y=DT_FLOAT16" "${grouped_matmul_tiling_files}")
endif()