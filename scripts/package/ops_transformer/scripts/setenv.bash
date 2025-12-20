#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# all package set same environment ASCEND_OPP_PATH

REAL_SHELL_PATH=`realpath ${BASH_SOURCE[0]}`
MULTI_VERSION=$1
CANN_PATH=$(cd $(dirname ${REAL_SHELL_PATH})/../../../../ && pwd)
if [ -d "${CANN_PATH}/opp" ]; then
    INSATLL_PATH=$(cd $(dirname ${REAL_SHELL_PATH})/../../../../../ && pwd)
    _ASCEND_OPP_PATH="${CANN_PATH}/opp"
    if [ "$MULTI_VERSION" = "multi_version" ]; then
        _ASCEND_OPP_PATH="${INSATLL_PATH}/cann/opp"
    fi
fi  

export ASCEND_OPP_PATH=${_ASCEND_OPP_PATH}

pylib_path="${_ASCEND_OPP_PATH}/python/site-packages/"
if [ -d ${pylib_path} ];then
    export PYTHONPATH=$PYTHONPATH:${pylib_path}
fi

library_path="${_ASCEND_OPP_PATH}/cann/lib64"
ld_library_path="${LD_LIBRARY_PATH}"
num=$(echo ":${ld_library_path}:" | grep ":${library_path}:" | wc -l)
if [ "${num}" -eq 0 ]; then
    if [ "-${ld_library_path}" = "-" ]; then
        export LD_LIBRARY_PATH="${library_path}"
    else
        export LD_LIBRARY_PATH="${library_path}:${ld_library_path}"
    fi
fi

