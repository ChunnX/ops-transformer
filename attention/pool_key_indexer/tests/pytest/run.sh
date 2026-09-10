#!/bin/bash
# PoolKeyIndexer pytest 套件运行脚本
#
# 用法:
#   bash run.sh                      # 全部套件(精度 + PA block_size 2048 + GE 图, 含冒烟)
#   bash run.sh smoke                # 冒烟套件(eager/torch.ops/torchair GE/aclgraph 可用性)
#   bash run.sh precision            # 精度套件(定点 + 泛化 + 缺陷回归 + 边界计数)
#   bash run.sh pabs                 # PA block_size 2048 专项(精度+边界+负向)
#   bash run.sh ge                   # GE 图套件(定点 + 泛化)
#   bash run.sh aclgraph             # aclgraph/NPUGraph 图套件(捕获/回放 + 定点/泛化 + PA 变体)
#   bash run.sh perf                 # 性能套件(算子 vs NPU 组合标杆)
#   bash run.sh smoke precision      # 组合多个套件
#   bash run.sh -- <pytest args>     # 透传 pytest 参数
#
# 环境变量:
#   PKI_DEVICE_ID=2                  NPU 设备号(默认 0; 多用户环境选空闲设备)
#   PKI_GE_GENERAL_MAX=5             GE 泛化用例子集规模(冒烟)
#   pytest test_precision.py --ref-device npu   # 切换 NPU 组合标杆
#
# 示例:
#   PKI_DEVICE_ID=2 bash run.sh smoke
#   bash run.sh -- "test_precision.py -k bsnd_pa --ref-device npu -x"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUITES=("$@")
if [ "${SUITES[0]:-}" == "--" ]; then
    # 透传模式: bash run.sh -- "args..."
    PYTEST_ARGS="${SUITES[@]:1}"
    exec python -m pytest $PYTEST_ARGS -v -s --tb=short 2>&1 | tee pytest_output.log
fi

# 默认全部功能套件(冒烟 -> 精度 -> PA block_size 2048 专项 -> GE 图 -> aclgraph 图)
if [ ${#SUITES[@]} -eq 0 ]; then
    SUITES=(smoke precision pabs ge aclgraph)
fi

declare -A SUITE_FILES=(
    [smoke]="test_smoke.py"
    [precision]="test_precision.py"
    [pabs]="test_pa_block_size.py"
    [ge]="test_ge_graph.py"
    [aclgraph]="test_aclgraph.py test_aclgraph_pa.py"
)

TARGETS=()
for s in "${SUITES[@]}"; do
    if [ -n "${SUITE_FILES[$s]:-}" ]; then
        for f in ${SUITE_FILES[$s]}; do
            TARGETS+=("$f")
        done
    elif [ "$s" == "perf" ]; then
        TARGETS+=("perf")
    else
        echo "未知套件: $s (可选: smoke precision pabs ge aclgraph perf)"
        exit 1
    fi
done

echo "============================================"
echo " PoolKeyIndexer pytest 测试 (device=${PKI_DEVICE_ID:-0})"
echo " 套件: ${SUITES[@]}"
echo "============================================"

# 确保 cann_ops_transformer 可 import
export PYTHONPATH="${PYTHONPATH:-}:/usr/local/lib/python3.10/site-packages"

for t in "${TARGETS[@]}"; do
    if [ "$t" == "perf" ]; then
        echo "---- 性能套件: perf_compare.py (算子 vs NPU 组合标杆) ----"
        python perf_compare.py 2>&1 | tee -a pytest_output.log
    else
        echo "---- 套件: $t ----"
        python -m pytest "$t" -v -s --tb=short 2>&1 | tee -a pytest_output.log
    fi
done

echo "============================================"
echo " 测试完成，日志: pytest_output.log"
echo "============================================"
