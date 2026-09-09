# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18.4)
require_pypto_pro(pdist)
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    # aclnn_exclude: the hand-written op_api/ layer replaces the autogen aclnn_pdist.cpp, so the def
    # joins op_host_aclnnExc (autogen only generates for defs listed in op_host_aclnn).
    target_sources(op_host_aclnnExc PRIVATE
    pdist_def.cpp
    )
    # PyPTO kernel: runs codegen at configure time and exposes PYPTO_GEN_DIR for the host tiling
    # force-include (PdistTiling_tiling.h / PdistTilingKey_tilingkey.h). Must be called BEFORE
    # add_modules_sources(...) reads PYPTO_GEN_DIR.
    enable_pypto_kernel(pdist)
endif()

# ascend950 条件编译——缺少此项会导致 kernel binary 未编译
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME Pdist
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -Wno-deprecated-declarations
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME Pdist
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME Pdist
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        # op_api 与 op_host 同级：手写两段式 aclnn 实现（aclnn_exclude：不参与 autogen 生成）
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE pdist
        ACLNNTYPE aclnn_exclude
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
    # PyPTO wrapper (dynamic/pdist.py) locates the DSL kernel next to itself at
    # op_kernel/pdist.py (hardcoded in ascendc_impl_build.py's PYPTO_COMPILE_OP_API), and the
    # online compile additionally needs the codegen artifacts (pdist_pypto_infer.cpp for
    # KernelInfoInfer + the tiling headers) in the same directory. ${pdist_pypto_gen_dir} is
    # the enable_pypto_kernel codegen output; together with the DSL this mirrors the opc
    # staging layout (binary/<soc>/src/pdist/op_kernel/).
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/pdist.py
                  ${pdist_pypto_gen_dir}/pdist_pypto_infer.cpp
                  ${pdist_pypto_gen_dir}/PdistTiling_tiling.h
                  ${pdist_pypto_gen_dir}/PdistTilingKey_tilingkey.h
            DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}/op_kernel
            OPTIONAL)
endif()
