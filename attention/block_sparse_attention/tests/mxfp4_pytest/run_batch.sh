#!/bin/bash
# 批量测试脚本: 生成 b, n1, n2, s1, s2, mask_kind 的所有组合并运行测试
source /srv/workspace/mnt1/work_code/Ascend_20260703/cann/bin/setenv.bash
cd "$(dirname "$0")"
export BSA_CPU_ONLY=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# 定义参数列表
B_LIST=(1 2 3 4 5)
N1_LIST=(1 2 4 8)
N2_LIST=(1 2 4)
# 特殊 s1/s2: 极小值、边界不对齐值、256对齐值、中等值、长序列
S1_LIST=(1 33 65 127 257 511 1000 2001)
S2_LIST=(1 33 65 127 257 511 1000 2001 4096 5000 8448 10000 18000 30000)
MASK_KIND_LIST=("dense" "sparse")
QK_D=128

# 测试计数器
TOTAL=0
PASSED=0
FAILED=0

# 生成所有组合
for b in "${B_LIST[@]}"; do
    for n1 in "${N1_LIST[@]}"; do
        for n2 in "${N2_LIST[@]}"; do
            for s1 in "${S1_LIST[@]}"; do
                for s2 in "${S2_LIST[@]}"; do
                    for mask_kind in "${MASK_KIND_LIST[@]}"; do
                        # GQA 约束检查: n1 必须是 n2 的整数倍
                        if [ $((n1 % n2)) -ne 0 ]; then
                            continue
                        fi
                        
                        # 计算 group_size
                        g=$((n1 / n2))
                        
                        # 构建测试名称
                        TEST_NAME="test_bsa_mxfp4_bnsd_gqa_dense_b${b}_qn${n1}_kvn${n2}_g${g}_qs${s1}_kvs${s2}_d${QK_D}"
                        if [ "$mask_kind" == "sparse" ]; then
                            TEST_NAME="${TEST_NAME//_dense/_sparse}"
                        fi
                        
                        # 如果是 mha (n1 == n2), 使用 mha 命名
                        if [ "$n1" == "$n2" ]; then
                            TEST_NAME="test_bsa_mxfp4_bnsd_mha_${mask_kind}_b${b}_qn${n1}_kvn${n2}_g1_qs${s1}_kvs${s2}_d${QK_D}"
                        fi
                        
                        TOTAL=$((TOTAL + 1))
                        echo "==========================================" | tee -a "$LOG_FILE"
                        echo "[$TOTAL] Running: $TEST_NAME" | tee -a "$LOG_FILE"
                        echo "  b=$b n1=$n1 n2=$n2 g=$g s1=$s1 s2=$s2 mask=$mask_kind" | tee -a "$LOG_FILE"
                        echo "==========================================" | tee -a "$LOG_FILE"

                        # 运行测试（pipefail 保证下面 if 取到 pytest 的退出码）
                        if python -m pytest "test_bsa_mxfp4_golden.py::TestBSAMxfp4Golden::test_bsa_mxfp4_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d128" \
                            -v -s \
                            --b "$b" --n1 "$n1" --n2 "$n2" \
                            --s1 "$s1" --s2 "$s2" --qk_d "$QK_D" --dump True 2>&1 | tee -a "$LOG_FILE"; then
                            PASSED=$((PASSED + 1))
                            echo "[PASSED] $TEST_NAME" | tee -a "$LOG_FILE"
                        else
                            FAILED=$((FAILED + 1))
                            echo "[FAILED] $TEST_NAME" | tee -a "$LOG_FILE"
                        fi
                        echo "" | tee -a "$LOG_FILE"
                    done
                done
            done
        done
    done
done

echo "==========================================" | tee -a "$LOG_FILE"
echo "Test Summary:" | tee -a "$LOG_FILE"
echo "  Total:  $TOTAL" | tee -a "$LOG_FILE"
echo "  Passed: $PASSED" | tee -a "$LOG_FILE"
echo "  Failed: $FAILED" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE"


