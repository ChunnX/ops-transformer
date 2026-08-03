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
#
# soc_validator.sh 库的单元测试脚本
# 运行方式: bash tests/test_soc_validator.sh
# 判定标准: 全部断言输出 PASS 且脚本退出码为 0; 任一断言 FAIL 则脚本退出码非 0。
# 该脚本不依赖任何外部测试框架, 使用纯 bash 断言实现。

set -euo pipefail

# 定位脚本所在目录与仓库根目录, 保证在任何工作目录下执行都能正确 source 库文件
TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "${TEST_DIR}")

# source 待测库文件, 若 source 失败则脚本直接退出
# shellcheck source=scripts/util/soc_validator.sh
source "${REPO_ROOT}/scripts/util/soc_validator.sh"

# 失败断言计数器
FAIL_COUNT=0

# -----------------------------------------------------------------------------------------------------------
# 断言工具函数
# -----------------------------------------------------------------------------------------------------------

# 断言函数实际退出码与期望退出码一致
assert_rc() {
    local desc="$1"
    local expected="$2"
    local actual="$3"
    if [[ "${expected}" == "${actual}" ]]; then
        echo "PASS: ${desc}"
    else
        echo "FAIL: ${desc} (期望退出码: ${expected}, 实际退出码: ${actual})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# 断言两个字符串相等
assert_eq() {
    local desc="$1"
    local expected="$2"
    local actual="$3"
    if [[ "${expected}" == "${actual}" ]]; then
        echo "PASS: ${desc}"
    else
        echo "FAIL: ${desc}"
        echo "  期望: ${expected}"
        echo "  实际: ${actual}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# 被测函数执行后的实际退出码, 由 run_capture_rc 写入
RUN_RC=0

# 执行任意函数并在当前 shell 中捕获其退出码(兼容 set -e 场景, 输出丢弃)
# 注意: 必须在当前 shell 中执行被测函数, 否则函数内设置的全局变量(如 INVALID_SOC_ITEM)
#       会因子 shell 隔离而无法被后续断言读取
run_capture_rc() {
    set +e
    "$@" >/dev/null 2>&1
    RUN_RC=$?
    set -e
}

# -----------------------------------------------------------------------------------------------------------
# 用例 1: soc_is_supported 对 7 个受支持 SoC 均应返回 0
# -----------------------------------------------------------------------------------------------------------
VALID_SOC_LIST=("ascend910b" "ascend910_93" "ascend950" "ascend310p" "kirinx90" "kirin9030" "mc62")
for soc in "${VALID_SOC_LIST[@]}"; do
    run_capture_rc soc_is_supported "${soc}"
    assert_rc "soc_is_supported \"${soc}\" 返回 0" "0" "${RUN_RC}"
done

# -----------------------------------------------------------------------------------------------------------
# 用例 2: soc_is_supported 对非法输入均应返回 1
# -----------------------------------------------------------------------------------------------------------
INVALID_SOC_LIST=("123" "ascend_950" "ascend910b3" "")
for soc in "${INVALID_SOC_LIST[@]}"; do
    run_capture_rc soc_is_supported "${soc}"
    assert_rc "soc_is_supported \"${soc}\" 返回 1" "1" "${RUN_RC}"
done

# -----------------------------------------------------------------------------------------------------------
# 用例 3: soc_is_supported 大小写不敏感, 且自动裁剪前后空白
# -----------------------------------------------------------------------------------------------------------
run_capture_rc soc_is_supported "ASCEND950"
assert_rc "soc_is_supported \"ASCEND950\" 返回 0 (大小写不敏感)" "0" "${RUN_RC}"

run_capture_rc soc_is_supported " ascend950 "
assert_rc "soc_is_supported \" ascend950 \" 返回 0 (前后空白被裁剪)" "0" "${RUN_RC}"

# -----------------------------------------------------------------------------------------------------------
# 用例 4: validate_soc_list 对全合法列表应返回 0
# -----------------------------------------------------------------------------------------------------------
run_capture_rc validate_soc_list "ascend910b,ascend950"
assert_rc "validate_soc_list \"ascend910b,ascend950\" 返回 0" "0" "${RUN_RC}"

run_capture_rc validate_soc_list "ascend310p;ascend910b"
assert_rc "validate_soc_list \"ascend310p;ascend910b\" 返回 0" "0" "${RUN_RC}"

run_capture_rc validate_soc_list "ascend910b, ascend950 ; mc62"
assert_rc "validate_soc_list \"ascend910b, ascend950 ; mc62\" 返回 0 (混合分隔符与空白)" "0" "${RUN_RC}"

# -----------------------------------------------------------------------------------------------------------
# 用例 5: validate_soc_list 对含非法项的列表应返回 1 并通过 INVALID_SOC_ITEM 报告非法项
# -----------------------------------------------------------------------------------------------------------
run_capture_rc validate_soc_list "ascend910b,123"
assert_rc "validate_soc_list \"ascend910b,123\" 返回 1" "1" "${RUN_RC}"
assert_eq "INVALID_SOC_ITEM 报告 123" "123" "${INVALID_SOC_ITEM}"

run_capture_rc validate_soc_list "ascend950,ascend_950"
assert_rc "validate_soc_list \"ascend950,ascend_950\" 返回 1" "1" "${RUN_RC}"
assert_eq "INVALID_SOC_ITEM 报告 ascend_950" "ascend_950" "${INVALID_SOC_ITEM}"

# -----------------------------------------------------------------------------------------------------------
# 用例 6: validate_soc_list 对空输入与纯空白输入应返回 1
# -----------------------------------------------------------------------------------------------------------
run_capture_rc validate_soc_list ""
assert_rc "validate_soc_list \"\" 返回 1" "1" "${RUN_RC}"

run_capture_rc validate_soc_list "   "
assert_rc "validate_soc_list \"   \" 返回 1 (纯空白输入)" "1" "${RUN_RC}"

# -----------------------------------------------------------------------------------------------------------
# 用例 7: get_supported_soc_text 应输出全部 7 个受支持 SoC, 以逗号加空格分隔
# -----------------------------------------------------------------------------------------------------------
expected_soc_text="ascend910b, ascend910_93, ascend950, ascend310p, kirinx90, kirin9030, mc62"
actual_soc_text="$(get_supported_soc_text)"
assert_eq "get_supported_soc_text 输出全部 7 个受支持 SoC" "${expected_soc_text}" "${actual_soc_text}"

# -----------------------------------------------------------------------------------------------------------
# 汇总测试结果
# -----------------------------------------------------------------------------------------------------------
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    echo ""
    echo "测试失败: ${FAIL_COUNT} 个断言未通过"
    exit 1
fi

echo ""
echo "所有断言全部通过"
exit 0
