# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
# ---------------- CANN 版本检查 ----------------
# 参考 cmake/config.cmake 中 CANN_VERSION 的获取方式（check_version_compatible.py 读取 CANN 包 version.info），
# 优先使用 config.cmake 输出的 CANN_VERSION（已转小写），否则直接读取 version.info 的 Version= 完整版本号；
# 版本为 9.2.0-beta.1 / 9.2.0-beta.2 / 9.2.0 时跳过本算子编译，其余版本正常编译。
message(STATUS "[quant_flash_attn_grad][DEBUG] CANN_VERSION='${CANN_VERSION}', ASCEND_CANN_PACKAGE_PATH='${ASCEND_CANN_PACKAGE_PATH}'")
set(_qfag_excluded_versions "9.2.0-beta.1;9.2.0-beta.2;9.2.0")
set(_qfag_cann_version "")
if(DEFINED CANN_VERSION AND NOT "${CANN_VERSION}" STREQUAL "")
    set(_qfag_cann_version "${CANN_VERSION}")
else()
    foreach(_qfag_ver_file
            "${ASCEND_CANN_PACKAGE_PATH}/toolkit/version.info"
            "${ASCEND_CANN_PACKAGE_PATH}/version.info"
            "${ASCEND_CANN_PACKAGE_PATH}/opp/version.info"
            "${ASCEND_CANN_PACKAGE_PATH}/compiler/version.info")
        message(STATUS "[quant_flash_attn_grad][DEBUG] try version.info: ${_qfag_ver_file}")
        if(EXISTS "${_qfag_ver_file}")
            file(READ "${_qfag_ver_file}" _qfag_ver_content)
            message(STATUS "[quant_flash_attn_grad][DEBUG] version.info 原始内容:\n${_qfag_ver_content}")
            string(REPLACE "\r" "" _qfag_ver_content "${_qfag_ver_content}")
            string(REPLACE "\n" ";" _qfag_ver_lines "${_qfag_ver_content}")
            foreach(_qfag_line IN LISTS _qfag_ver_lines)
                if(_qfag_line MATCHES "^Version=")
                    string(SUBSTRING "${_qfag_line}" 8 -1 _qfag_cann_version)
                    message(STATUS "[quant_flash_attn_grad][DEBUG] 匹配到行: '${_qfag_line}' -> 版本号: '${_qfag_cann_version}'")
                    break()
                endif()
            endforeach()
            if(NOT "${_qfag_cann_version}" STREQUAL "")
                break()
            endif()
        endif()
    endforeach()
endif()

string(TOLOWER "${_qfag_cann_version}" _qfag_cann_version)
message(STATUS "[quant_flash_attn_grad][DEBUG] 最终解析版本号: '${_qfag_cann_version}'，排除列表: ${_qfag_excluded_versions}")
if("${_qfag_cann_version}" STREQUAL "")
    message(WARNING "[quant_flash_attn_grad] 无法获取 CANN 版本号，版本排除检查未生效，本算子将继续编译")
endif()
if(_qfag_cann_version IN_LIST _qfag_excluded_versions)
    message(STATUS "算子quant_flash_attn_grad不支持在cann version=${_qfag_cann_version}场景下编译，已跳过编译")
    set_property(GLOBAL PROPERTY quant_flash_attn_grad_CANN_VERSION_SKIPPED TRUE)
    set_property(GLOBAL APPEND PROPERTY CANN_VERSION_SKIPPED_OPS "quant_flash_attn_grad（cann=${_qfag_cann_version}不支持）")
    return()
endif()

require_pypto_pro(quant_flash_attn_grad)
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        quant_flash_attn_grad_def.cpp
    )
    add_modules_sources(OPTYPE quant_flash_attn_grad ACLNNTYPE aclnn_inner)
    enable_pypto_kernel(quant_flash_attn_grad)
endif()

# ascend950 条件编译—缺少此项会导致 kernel binary 未编译
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    # PyPTO kernel: run codegen at configure time and exposes PYPTO_GEN_DIR for the host tiling -I below.
    # Must be called BEFORE add_modules_sources(...) (which reads PYPTO_GEN_DIR).
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(OPTYPE quant_flash_attn_grad ACLNNTYPE aclnn_inner)
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()
