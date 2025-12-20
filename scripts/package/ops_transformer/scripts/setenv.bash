#!/bin/bash
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

param_mult_ver=$1
REAL_SHELL_PATH=`realpath ${BASH_SOURCE[0]}`
CANN_PATH=$(cd $(dirname ${REAL_SHELL_PATH})/../../ && pwd)
if [ -d "${CANN_PATH}/ops_transformer" ] && [ -d "${CANN_PATH}/../latest" ]; then
    INSATLL_PATH=$(cd $(dirname ${REAL_SHELL_PATH})/../../../ && pwd)
    if [ -L "${INSATLL_PATH}/latest/ops_transformer" ]; then
        _ASCEND_OPS_TRANSFORMER_PATH=`cd ${CANN_PATH}/ops_transformer && pwd`
        if [ "$param_mult_ver" = "multi_version" ]; then
            _ASCEND_OPS_TRANSFORMER_PATH=`cd ${INSATLL_PATH}/latest/ops_transformer && pwd`
        fi
    fi
elif [ -d "${CANN_PATH}/ops_transformer" ]; then
    _ASCEND_OPS_TRANSFORMER_PATH=`cd ${CANN_PATH}/ops_transformer && pwd`
fi  

export ASCEND_OPS_TRANSFORMER_PATH=${_ASCEND_OPS_TRANSFORMER_PATH}

library_path="${_ASCEND_OPS_TRANSFORMER_PATH}/lib64"
ld_library_path="${LD_LIBRARY_PATH}"
num=$(echo ":${ld_library_path}:" | grep ":${library_path}:" | wc -l)
if [ "${num}" -eq 0 ]; then
    if [ "-${ld_library_path}" = "-" ]; then
        export LD_LIBRARY_PATH="${library_path}"
    else
        export LD_LIBRARY_PATH="${library_path}:${ld_library_path}"
    fi
fi

