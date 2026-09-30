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

# ====================== 配置区======================
# 需要读取的用例excel表格路径，如下：
PATH1="./excel/*"
# 用例pt的文件存放路径，如下：
PATH2="./pt_path"

source /opt/qsc/InstallCann/CANN9.2.0_0819/ascend-toolkit/latest/set_env.sh
source /opt/qsc/InstallCann/CANN9.2.0_0819/ascend-toolkit/latest/vendors/cmpbit_transformer/bin/set_env.bash

# 确保 cyh conda 环境(torch_npu 26.2.0-beta.1, 支持 deterministic level=3)生效:
# 直接 `bash test_run.sh` 时子shell不会带 conda 环境, python3 会落到系统解释器
# (系统 torch_npu 2.10.0 不支持 level=3, 且 msprof 导出子进程会报 libsqlite3.so ImportError)。
if ! [[ "$(command -v python3)" == *"/envs/cyh/"* ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate cyh
fi

# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source /usr/local/Ascend/ascend-toolkit/vendors/cmpbit_transformer/bin/set_env.bash
# 脚本路径
Compressor_PT_SAVE_SCRIPT="./batch/compressor_pt_save.py"
TEST_Compressor_BATCH_SCRIPT="test_compressor_batch.py"
REPLACE_PATH_SCRIPT="./batch/replace_path.py"
TEST_Compressor_SINGLE_SCRIPT="test_compressor_single.py"
TEST_Compressor_BITLEVEL_SCRIPT="test_compressor_bitlevel.py"
BC_PERF_SCRIPT="bc_perf_runner.py"
BC_PERF_ITERS=50
BC_PERF_WARMUP=5

# ====================== 执行区======================

# 自动加载提供 Compressor 算子的 vendor 包环境(设置 ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH)。
# 说明: 算子run包安装在后, runtime 需要 vendor 包的 set_env.bash 才能找到 kernel 注册,
#       否则报 EZ1001 "binary_info_config.json does not support opType [Compressor]"。
#       仅加载 binary_info_config.json 中含 Compressor 的 vendor, 不影响其他 vendor 包。
# setup_vendor_env() {
#     local vendors_root="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/vendors"
#     if [ ! -d "$vendors_root" ]; then
#         vendors_root="/usr/local/Ascend/ascend-toolkit/latest/vendors"
#     fi
#     local sourced=0
#     local env_script vendor_dir
#     while IFS= read -r env_script; do
#         vendor_dir=$(dirname "$(dirname "$env_script")")
#         if grep -rql '"Compressor"' "$vendor_dir/op_impl" 2>/dev/null; then
#             echo "[test_run] source vendor env: $env_script"
#             source "$env_script"
#             sourced=1
#         fi
#     done < <(find "$vendors_root" -maxdepth 3 -name set_env.bash 2>/dev/null | sort)
#     if [ "$sourced" -eq 0 ]; then
#         echo "[test_run][警告] 未找到提供 Compressor 算子的 vendor 包环境脚本, 可能报 EZ1001"
#         echo "  请确认已安装算子run包(bash build.sh --pkg --ops=compressor --vendor_name=xxx), 并手动执行:"
#         echo "  source ${vendors_root}/<vendor_name>_transformer/bin/set_env.bash"
#     fi
# }

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    # setup_vendor_env
    python3 -m pytest -rA -s $TEST_Compressor_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 位级一致性调测(跨模板/动态D分块悬崖/位置一致性)
run_bitlevel() {
    echo "===== 执行位级一致性调测 ====="
    # setup_vendor_env
    python3 -m pytest -rA -s $TEST_Compressor_BITLEVEL_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

xingneng(){
    echo "===== msprof性能测试 ====="
    msprof python3 -m pytest test_compressor_bitlevel.py -m ci -s
}

# batch一致性性能采集: 4组采集(FULL_LOAD B=32 / NORMAL B=64, 各自 level=3 与 level=0),
# 每组 warmup=5 + iters=50 次串行直调, msprof 输出分目录存放于 ./prof_bc/
# 采集校验: op_summary 中须有 (warmup+iters) 条 Compressor 实例, 不足则重试(共享设备偶发采集失败)
bc_perf_check(){
    local dir="$1"
    local f need
    need=$((BC_PERF_ITERS + BC_PERF_WARMUP))
    f=$(ls "$dir"/PROF_*/mindstudio_profiler_output/op_summary*.csv 2>/dev/null | head -1)
    [ -n "$f" ] || return 1
    [ "$(grep -ci compressor "$f")" -ge "$need" ] || return 1
    return 0
}

bc_perf(){
    local OUT="$PWD/prof_bc"
    local B L try DEV TS
    # 4组必须同设备; 共享机器上设备可能被其他任务占用导致挂死, 可用 BC_PERF_DEVICE 指定空闲设备
    DEV="${BC_PERF_DEVICE:-0}"
    export CMP_TEST_DEVICE_ID="$DEV"
    TS=$(date +%Y%m%d_%H%M%S)
    # 不清空 OUT: 保留历史结果; 仅对本次重采的组, 先把旧目录备份为 <组>.bak_<时间戳> 再覆盖
    mkdir -p "$OUT"
    for B in 4096 8192; do
        for L in 0 3; do
            if [ -d "$OUT/B${B}_L${L}" ]; then
                mv "$OUT/B${B}_L${L}" "$OUT/B${B}_L${L}.bak_${TS}"
                echo "[bc_perf] 旧结果已备份: B${B}_L${L}.bak_${TS}"
            fi
            for try in 1 2 3; do
                echo "===== [bc_perf] msprof 采集: B=$B level=$L device=$DEV 第${try}次 ====="
                { echo "[env] python3=$(command -v python3) device=$DEV"; } > "$OUT/B${B}_L${L}.${TS}.try${try}.log" 2>&1
                # 单组超过8分钟视为挂死(设备被占/死锁), 强制中止进入重试
                timeout -k 30 480 msprof --application="python3 $BC_PERF_SCRIPT --batch $B --level $L --iters $BC_PERF_ITERS --warmup $BC_PERF_WARMUP" \
                    --output="$OUT/B${B}_L${L}" >> "$OUT/B${B}_L${L}.${TS}.try${try}.log" 2>&1
                if bc_perf_check "$OUT/B${B}_L${L}"; then
                    echo "[bc_perf] B=$B L=$L 采集校验通过 (>= $((BC_PERF_ITERS + BC_PERF_WARMUP)) 条 Compressor 实例)"
                    break
                fi
                echo "[bc_perf][警告] B=$B L=$L 第${try}次采集数据不完整, 重试..."
                rm -rf "$OUT/B${B}_L${L}"
                sleep 5
            done
            bc_perf_check "$OUT/B${B}_L${L}" || echo "[bc_perf][错误] B=$B L=$L 三次采集均失败, 请查看 $OUT/B${B}_L${L}.*.try*.log"
        done
    done
    echo "===== [bc_perf] 4 组采集完成, 输出目录: $OUT ====="
    for d in "$OUT"/B32_L0 "$OUT"/B32_L3 "$OUT"/B64_L0 "$OUT"/B64_L3; do
        if bc_perf_check "$d"; then echo "  OK   $d"; else echo "  FAIL $d"; fi
    done
}


# 用例批量生成调试
run_batch() {
    echo "===== 执行用例批量生成测试 ====="
    # setup_vendor_env

    echo -e "\n===== 第一步：执行compressor_pt_save.py ====="
    python3 $Compressor_PT_SAVE_SCRIPT $PATH1 $PATH2
    if [ $? -ne 0 ]; then
        echo "警告：compressor_pt_save.py 执行失败，将使用已有pt文件继续"
    fi

    echo -e "\n===== 第二步：替换test_compressor_batch.py中的路径 ====="
    python3 $REPLACE_PATH_SCRIPT $TEST_Compressor_BATCH_SCRIPT $PATH2
    if [ $? -ne 0 ]; then
        echo "替换路径失败，退出"
        exit 1
    fi

    # 获取Excel文件路径（取第一个匹配的文件）
    shopt -s nullglob
    EXCEL_FILES=( $PATH1 )
    shopt -u nullglob
    if [ ${#EXCEL_FILES[@]} -gt 0 ]; then
        EXCEL_FILE="${EXCEL_FILES[0]}"
        export TEST_CASE_EXCEL="$EXCEL_FILE"
    else
        unset TEST_CASE_EXCEL
    fi

    echo -e "\n===== 第三步：执行pytest命令 (Excel: $TEST_CASE_EXCEL)====="
    python3 -m pytest -rA -s $TEST_Compressor_BATCH_SCRIPT -v -m ci
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    cp test_compressor_batch.py.bak test_compressor_batch.py

    echo -e "\n=====执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明:"
    echo "  single    执行单算子用例调测"
    echo "  bitlevel  执行位级一致性调测(跨模板/动态D分块悬崖/batch位置一致性)"
    echo "  batch     执行用例批量生成调试"
    echo "  bc_perf   batch一致性性能采集(FULL_LOAD B=32 / NORMAL B=64, 各 level=3 与 level=0, 各50次取平均)"
    echo "  help      显示本帮助信息"
    echo "示例:"
    echo "  $0 single    # 执行single模式"
    echo "  $0 bitlevel  # 执行bitlevel模式"
    echo "  $0 batch   # 执行batch模式"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/bitlevel/batch/help）"
    show_help
    exit 1
fi

# 根据参数执行相应函数
case "$1" in
    single)
        run_single
        ;;
    bitlevel)
        run_bitlevel
        ;;
    batch)
        run_batch
        ;;
    help)
        show_help
        ;;
    xingneng)
        xingneng
        ;;
    bc_perf)
        bc_perf
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/bitlevel/batch/help"
        show_help
        exit 1
        ;;
esac

exit 0

#
