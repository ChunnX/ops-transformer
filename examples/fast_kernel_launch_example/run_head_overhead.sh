#!/bin/bash
# ----------------------------------------------------------------------------
# Kernel Launch 头开销一键测试脚本
#
# 用法:
#   source <CANN路径>/set_env.sh
#   bash run_head_overhead.sh
#
# 原理:
#   通过 sed 修改源码中的核数参数，构建满核 wheel，
#   用 msprof capture 模式采集 101 次循环的头开销数据，最后汇总输出。
#
# 可选环境变量:
#   NPU_SOC_VERSION  编译目标 SoC (默认: npu-smi 自动检测)
#   OUTPUT_DIR       采集数据输出目录 (默认: ./head_overhead_results)
#   RUN_ADD          是否跑 add (默认: 1)
#   RUN_GMM          是否跑 grouped_matmul (默认: 1)
# ----------------------------------------------------------------------------
set -e

cd "$(dirname "$0")"

SOC_VERSION="${NPU_SOC_VERSION:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/head_overhead_results}"
LAUNCH_COUNT=101
RUN_ADD="${RUN_ADD:-1}"
RUN_GMM="${RUN_GMM:-1}"

# 自动检测平台: npu-smi info 输出的芯片名映射到 SOC_VERSION
if [ -z "$SOC_VERSION" ]; then
    CHIP_NAME=$(npu-smi info 2>/dev/null | grep -o 'Ascend[0-9A-Za-z_]*' | head -1)
    case "$CHIP_NAME" in
        Ascend950*)    SOC_VERSION="ascend950" ;;
        Ascend910)     SOC_VERSION="ascend910_93" ;;
        Ascend910_93*) SOC_VERSION="ascend910_93" ;;
        Ascend910B1)   SOC_VERSION="ascend910b" ;;
        *) echo "[ERROR] 无法识别芯片: $CHIP_NAME，请手动设置 NPU_SOC_VERSION"; exit 1 ;;
    esac
fi

# 根据 SOC_VERSION 选择源码目录后缀
case "$SOC_VERSION" in
    ascend950)    SRC_SUFFIX="ascend950" ;;
    ascend910_93) SRC_SUFFIX="ascend910_93" ;;
    ascend910b)   SRC_SUFFIX="ascend910b" ;;
    *) echo "[ERROR] 不支持的 NPU_SOC_VERSION: $SOC_VERSION"; exit 1 ;;
esac

echo "============================================"
echo "  Kernel Launch 头开销测试"
echo "============================================"
echo "  Python:          $(which python3)"
echo "  NPU_SOC_VERSION: $SOC_VERSION"
echo "  SRC_SUFFIX:      $SRC_SUFFIX"
echo "  Output:          $OUTPUT_DIR"
echo "============================================"
echo ""

# 检查 CANN 环境
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "[ERROR] ASCEND_HOME_PATH 未设置，请先 source <CANN路径>/set_env.sh"
    exit 1
fi
echo "[CHECK] CANN 环境: $ASCEND_HOME_PATH"

# 检查 NPU
if ! command -v npu-smi &>/dev/null; then
    echo "[ERROR] npu-smi 不可用"
    exit 1
fi
echo "[CHECK] NPU 设备:"
npu-smi info | head -10
echo ""

# 检查 torch_npu
python3 -c "import torch, torch_npu; torch.npu._lazy_init(); print('[CHECK] torch_npu OK')" 2>/dev/null || {
    echo "[ERROR] torch_npu 不可用或无法初始化 NPU"
    echo "        需要 torch_npu>=2.7.1.post4 (支持 Ascend950)"
    exit 1
}
echo ""

# 查询设备核数（从 CANN platform_config ini 读取）
SOC_NAME=$(python3 -c "
import ctypes; lib=ctypes.CDLL('libascendcl.so'); lib.aclInit(None); lib.aclrtSetDevice(0)
fn=lib.aclrtGetSocName; fn.restype=ctypes.c_char_p; print(fn().decode())
" 2>/dev/null)
ARCH_SUFFIX="$(uname -m)-linux"
PLATFORM_DIR="${ASCEND_HOME_PATH}/${ARCH_SUFFIX}/data/platform_config"
INI_FILE="${PLATFORM_DIR}/${SOC_NAME}.ini"
if [ -f "$INI_FILE" ]; then
    CUBE_CORES=$(grep -i "cube_core_cnt" "$INI_FILE" 2>/dev/null | cut -d= -f2 | tr -d ' ')
    VECTOR_CORES=$(grep -i "vector_core_cnt" "$INI_FILE" 2>/dev/null | cut -d= -f2 | tr -d ' ')
fi
# 按平台给默认值: A3(aiv48/aic24), A5(aiv56/aic28)
if [ "$SOC_VERSION" = "ascend910_93" ]; then
    CUBE_CORES="${CUBE_CORES:-24}"
    VECTOR_CORES="${VECTOR_CORES:-48}"
else
    CUBE_CORES="${CUBE_CORES:-28}"
    VECTOR_CORES="${VECTOR_CORES:-56}"
fi
echo "[CHECK] 设备: ${SOC_NAME:-?}  Cube 核数: ${CUBE_CORES}  Vector 核数: ${VECTOR_CORES}"
echo ""

mkdir -p "$OUTPUT_DIR"

ADD_CPP="csrc/add/${SRC_SUFFIX}/add.cpp"
GMM_CPP="csrc/grouped_matmul/${SRC_SUFFIX}/grouped_matmul.cpp"

# 备份原始源码
cp "$ADD_CPP"  "${ADD_CPP}.orig"
cp "$GMM_CPP"  "${GMM_CPP}.orig"

cleanup() {
    # 恢复原始源码
    [ -f "${ADD_CPP}.orig" ] && mv "${ADD_CPP}.orig" "$ADD_CPP"
    [ -f "${GMM_CPP}.orig" ] && mv "${GMM_CPP}.orig" "$GMM_CPP"
    echo "[CLEANUP] 源码已恢复"
}
trap cleanup EXIT

# 排除 incre_flash_attention (IFA 在部分 CANN 版本有 aclprofTensor 编译问题)
IFA_CMAKE="csrc/incre_flash_attention/${SRC_SUFFIX}/CMakeLists.txt"
if [ -f "$IFA_CMAKE" ]; then
    mv "$IFA_CMAKE" "${IFA_CMAKE}.bak"
    echo "[SETUP] 已临时排除 incre_flash_attention"
fi

# ============================================================
# 构建函数: 接收核数参数, sed 改源码, 构建+安装 wheel
# ============================================================
build_wheel() {
    local add_cores=$1 gmm_cores=$2
    if [ -z "$gmm_cores" ]; then gmm_cores="$add_cores"; fi
    echo "[BUILD] 构建 (add=${add_cores}核, gmm=${gmm_cores}核)..."

    # 从原始备份恢复，再 sed 改核数（保证多次调用都从干净状态开始）
    cp "${ADD_CPP}.orig" "$ADD_CPP"
    cp "${GMM_CPP}.orig" "$GMM_CPP"

    # add: numBlocks = std::min(coreNum, ...) => 固定值
    sed -i "s/int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) \/ MIN_ELEMS_PER_CORE);/int64_t numBlocks = ${add_cores};/" "$ADD_CPP"

    # add: <<<48, ...>>> 或 <<<64, ...>>> => <<<${add_cores}, ...>>>  (launch 核数也必须改，否则 numBlocks 不生效)
    sed -i "s/<<<[0-9]*,/<<<${add_cores},/g" "$ADD_CPP"

    # gmm: blockDim = tilingData.gmmBaseParams.get_coreNum() => 固定值
    sed -i "s/uint32_t blockDim = tilingData.gmmBaseParams.get_coreNum();/uint32_t blockDim = ${gmm_cores}; tilingData.gmmBaseParams.coreNum = ${gmm_cores};/" "$GMM_CPP"

    # check: 确认核数修改结果
    echo "[CHECK] add  numBlocks: $(grep 'numBlocks = ' "$ADD_CPP" | head -1)"
    echo "[CHECK] add  launch<<<: $(grep '<<<' "$ADD_CPP" | head -1 | grep -o '<<<[0-9]*')"
    echo "[CHECK] gmm  blockDim:  $(grep 'blockDim = ' "$GMM_CPP" | head -1)"

    export NPU_SOC_VERSION="$SOC_VERSION"
    python3 setup.py clean --quiet 2>/dev/null
    python3 -m build --wheel --no-isolation 2>&1 | tail -1
    python3 -m pip install dist/*.whl --force-reinstall --no-deps --quiet 2>/dev/null
    echo "[BUILD] wheel 安装完成"
    echo ""
}

# ============================================================
# 采集函数 (capture 模式: NPUGraph capture + replay)
# ============================================================
SCAN_REPLAY_PY="$(pwd)/scan_replay.py"
if [ ! -f "$SCAN_REPLAY_PY" ]; then
    echo "[ERROR] scan_replay.py 不存在: $SCAN_REPLAY_PY"
    exit 1
fi

collect_add() {
    local label=$1 outdir=$2
    echo "[COLLECT] 采集 add ($label, capture 模式)..."
    rm -rf "$outdir"
    OP_TYPE=add msprof --output="$outdir" python3 "$SCAN_REPLAY_PY" 2>&1 | tail -1
}

collect_gmm() {
    local label=$1 outdir=$2
    echo "[COLLECT] 采集 grouped_matmul ($label, capture 模式)..."
    rm -rf "$outdir"
    # gemm 悬垂引用修复后 capture 可用; Cube 核 capture 模式 msprof 可正常生成 op_summary
    OP_TYPE=gemm msprof --output="$outdir" python3 "$SCAN_REPLAY_PY" 2>&1 | tail -1
}

# ============================================================
# 满核: 构建 + 采集
# add 用 Vector 核数, gmm 用 Cube 核数
# ============================================================
echo "========== 满核 =========="
build_wheel "${VECTOR_CORES}" "${CUBE_CORES}"
if [ "$RUN_ADD" = "1" ]; then
    collect_add "满核" "$OUTPUT_DIR/add_full"
    echo ""
fi
if [ "$RUN_GMM" = "1" ]; then
    collect_gmm "满核" "$OUTPUT_DIR/gmm_full"
    echo ""
fi

# ============================================================
# 汇总分析
# ============================================================
echo "========== 汇总 =========="
python3 - "$OUTPUT_DIR" "$SOC_VERSION" << 'PYEOF'
import csv, statistics, sys, os, glob

output_dir = sys.argv[1]
soc_version = sys.argv[2]
if soc_version == "ascend950":
    add_cores, gmm_cores = 56, 28
else:
    add_cores, gmm_cores = 48, 24

def analyze_add(prof_dir, label):
    csvs = glob.glob(os.path.join(prof_dir, "**", "op_summary_*.csv"), recursive=True)
    if not csvs:
        print(f"  {label}: [未找到 op_summary CSV]")
        return None
    with open(csvs[0]) as f:
        rows = list(csv.DictReader(f))
    kr = [r for r in rows if 'add_kernel' in r.get('Op Name', '')]
    if not kr:
        print(f"  {label}: [未找到 add_kernel 数据]")
        return None
    durs = [float(r['Task Duration(us)'].strip().rstrip('\t')) for r in kr[2:]]
    result = {'label': label, 'mean': statistics.mean(durs)}
    print(f"  {label:12s} steady mean={result['mean']:.3f}us  n={len(durs)}")
    return result

def analyze_gmm(prof_dir, label):
    csvs = glob.glob(os.path.join(prof_dir, "**", "op_summary_*.csv"), recursive=True)
    if csvs:
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        kr = [r for r in rows if 'groupedmatmul' in r.get('Op Name', '')]
        if not kr:
            print(f"  {label}: [op_summary 中未找到 groupedmatmul 数据]")
            return None
        all_durs = [float(r['Task Duration(us)'].strip().rstrip('\t')) for r in kr]
    else:
        csvs = glob.glob(os.path.join(prof_dir, "**", "OpBasicInfo_*.csv"), recursive=True)
        if not csvs:
            print(f"  {label}: [未找到 op_summary 或 OpBasicInfo CSV]")
            return None
        all_durs = []
        for csv_path in sorted(csvs):
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    try:
                        all_durs.append(float(r.get('Task Duration(us)', '0').strip()))
                    except: pass
    if not all_durs:
        print(f"  {label}: [无有效数据]")
        return None
    steady = all_durs[2:]
    result = {'label': label, 'mean': statistics.mean(steady)}
    print(f"  {label:12s} steady mean={result['mean']:.3f}us  n={len(steady)}")
    return result

print()
print("============================================")
print("  头开销测试结果 (稳态=跳过前2次)")
print("============================================")
print()

results = []
if os.path.isdir(os.path.join(output_dir, 'add_full')):
    print("add (Vector):")
    r = analyze_add(os.path.join(output_dir, 'add_full'), 'add 满核')
    if r: results.append(('add', 'Vector', add_cores, r['mean']))
    print()
if os.path.isdir(os.path.join(output_dir, 'gmm_full')):
    print("grouped_matmul (Cube):")
    r = analyze_gmm(os.path.join(output_dir, 'gmm_full'), 'gmm 满核')
    if r: results.append(('grouped_matmul', 'Cube', gmm_cores, r['mean']))
    print()

# 生成汇总表
summary_path = os.path.join(output_dir, 'summary.csv')
with open(summary_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(['算子', '类型', '核数', '稳态mean(us)'])
    for op, typ, cores, mean in results:
        w.writerow([op, typ, cores, f"{mean:.3f}"])

print(f"汇总 CSV: {summary_path}")
print()
print("============================================")
print("  完成!")
print("============================================")
PYEOF
