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
#   NPU_SOC_VERSION  编译目标 SoC (默认: ascend910_93)
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
    if [ -z "$CHIP_NAME" ]; then
        # npu-smi 不可用时（如 host 侧 dcmi 初始化失败），回退用 aclrtGetSocName 探测
        CHIP_NAME=$(python3 -c "
import ctypes; lib=ctypes.CDLL('libascendcl.so'); lib.aclInit(None); lib.aclrtSetDevice(0)
fn=lib.aclrtGetSocName; fn.restype=ctypes.c_char_p; print(fn().decode())
" 2>/dev/null)
    fi
    # A3 = Ascend 910C = 910_93 (dav-2201); A2 = 910B 系列
    case "$CHIP_NAME" in
        Ascend950*)    SOC_VERSION="ascend950" ;;
        Ascend910C* | Ascend910_93*) SOC_VERSION="ascend910_93" ;;
        Ascend910B*)   SOC_VERSION="ascend910b" ;;
        Ascend910)     SOC_VERSION="ascend910_93" ;;
        *) echo "[ERROR] 无法识别芯片: $CHIP_NAME，请手动设置 NPU_SOC_VERSION (如 ascend950/ascend910_93)"; exit 1 ;;
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
# 按平台给默认值: A5(aiv56/aic28), A3/910C 与 A2/910B(aiv48/aic24)
if [ "$SOC_VERSION" = "ascend950" ]; then
    CUBE_CORES="${CUBE_CORES:-28}"
    VECTOR_CORES="${VECTOR_CORES:-56}"
else
    CUBE_CORES="${CUBE_CORES:-24}"
    VECTOR_CORES="${VECTOR_CORES:-48}"
fi
# SYS_CNT 频率由 scan_replay.py/汇总段按算子取默认 (A5: AIV=283, AIC≈587), 不再导出覆盖;
# 如需手工指定可 export SYS_CNT_FREQ_MHZ (整体覆盖两算子)
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
# 采集函数 (capture 模式: NPUGraph capture + replay + cycle dump)
# ============================================================
SCAN_REPLAY_PY="$(pwd)/scan_replay.py"
PARSE_STARS_PY="$(pwd)/parse_stars_log.py"
if [ ! -f "$SCAN_REPLAY_PY" ]; then
    echo "[ERROR] scan_replay.py 不存在: $SCAN_REPLAY_PY"
    exit 1
fi
if [ ! -f "$PARSE_STARS_PY" ]; then
    echo "[ERROR] parse_stars_log.py 不存在: $PARSE_STARS_PY"
    exit 1
fi

collect_add() {
    local label=$1 outdir=$2
    echo "[COLLECT] 采集 add ($label, capture 模式 + T0/T1 打点)..."
    rm -rf "$outdir"
    OP_TYPE=add msprof --output="$outdir" python3 "$SCAN_REPLAY_PY" \
        > "$outdir.stdout.log" 2> "$outdir.stderr.log" || true
    grep -aE "valid blocks|T1-T0:|WARN" "$outdir.stdout.log" | head -3
}

collect_gmm() {
    local label=$1 outdir=$2
    echo "[COLLECT] 采集 grouped_matmul ($label, capture 模式 + T0/T1 打点)..."
    rm -rf "$outdir"
    # gemm 悬垂引用修复后 capture 可用; Cube 核 capture 模式 msprof 可正常生成 op_summary
    OP_TYPE=gemm msprof --output="$outdir" python3 "$SCAN_REPLAY_PY" \
        > "$outdir.stdout.log" 2> "$outdir.stderr.log" || true
    grep -aE "valid blocks|T1-T0:|WARN" "$outdir.stdout.log" | head -3
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
# 汇总分析 (TaskDuration + STARS 三指标: T0-stars / T1-T0 / T1-stars)
# ============================================================
echo "========== 汇总 =========="
python3 - "$OUTPUT_DIR" "$SOC_VERSION" "$PARSE_STARS_PY" << 'PYEOF'
import bisect, csv, statistics, sys, os, glob, re, subprocess, struct

output_dir = sys.argv[1]
soc_version = sys.argv[2]
parse_stars_py = sys.argv[3]
if soc_version == "ascend950":
    add_cores, gmm_cores = 56, 28
else:
    add_cores, gmm_cores = 48, 24  # A3/910C 与 A2/910B 同为 48 AIV / 24 AIC
# SYS_CNT 频率 (A5 实测 1000MHz, 时钟速率直测, AIV/AIC 同频全局计数器; 历史 283/587 为误标定)
freq_add = float(os.environ.get("SYS_CNT_FREQ_MHZ", "1000"))
freq_gmm = float(os.environ.get("SYS_CNT_FREQ_MHZ", "1000"))

def parse_stars_dispatches(prof_dir):
    """解析 PROF 目录下的 stars_soc.data, 返回排序的 dispatch sys_cnt 列表"""
    data_dirs = glob.glob(os.path.join(prof_dir, "**", "device_*", "data"), recursive=True)
    if not data_dirs:
        return []
    out_csv = os.path.join(output_dir, f"stars_{os.path.basename(prof_dir)}.csv")
    try:
        subprocess.run([sys.executable, parse_stars_py, "-p", data_dirs[0], "--output", out_csv],
                       capture_output=True, check=True)
        disp = []
        for row in csv.DictReader(open(out_csv)):
            try:
                if int(row["func type"]) == 0:
                    disp.append(int(row["sys cnt"]))
            except (ValueError, KeyError):
                continue
        disp.sort()
        return disp
    except Exception:
        return []

BLOCK_RE = re.compile(r"ReplayLast Block (\d+): T0=(\d+) T1=(\d+) T2=(\d+)")

def parse_cycle_blocks(stderr_log):
    """解析 scan_replay.py dump 的每 block T0/T1/T2 明细"""
    recs = []
    if not os.path.exists(stderr_log):
        return recs
    for line in open(stderr_log, errors="ignore"):
        m = BLOCK_RE.search(line)
        if m:
            recs.append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return recs

def three_metrics(prof_dir, op, freq):
    """T0-stars / T1-T0 / T1-stars / T2-stars 四指标 (kernel 打点 + STARS 联合)
    每个时间戳独立最近邻匹配 dispatch（免疫 D2H 回读拍次错位）:
    T0/T1/T2 即使回读为不同拍的值, 各自匹配同拍 dispatch 后单值指标自洽。
    跨 slot 差值(T1-T0/T2-T0)仅同拍可信, 单调软校验兜底。"""
    stderr_log = f"{prof_dir}.stderr.log"
    recs = parse_cycle_blocks(stderr_log)
    if not recs:
        return None
    disp = parse_stars_dispatches(prof_dir)
    if not disp:
        return None

    def match(ts):
        # 只向前匹配 (dispatch <= ts): back-to-back 排队时下一个 kernel 的 dispatch
        # 会早于本 kernel 结束下发, 最近邻会把 T2 配到下一个 dispatch 上 (d 为负)
        i = bisect.bisect_right(disp, ts) - 1
        if i < 0:
            return -1
        return ts - disp[i]

    t0s, t1t0, t1s, t2s, t2t0 = [], [], [], [], []
    for a, b, b2 in recs:
        d0 = match(a)
        if 50 <= d0 <= 20000:
            t0s.append(d0)
        d1 = match(b)
        if 50 <= d1 <= 20000:
            t1s.append(d1)
        d2 = match(b2)
        if 50 <= d2 <= 20000:
            t2s.append(d2)
        if a < b <= b2:  # 同拍单调: 跨 slot 差值才可信
            t1t0.append(b - a)
            t2t0.append(b2 - a)
    if not t0s:
        return None
    # T2 校验口径 (TaskDur 同口径): TaskDuration 由最慢完成 block 决定,
    # mean(所有 block 平均) 天然偏小; 末拍簇 (T0 聚类) 内取 max(T2),
    # 与簇 dispatch 之差才可与 TaskDur 比较 (实测 add: +0.145us 吻合)
    # 窗口取 1.5us*freq: 固定 cyc 数在高频核 (AIC 587MHz) 上会跨拍混入上一拍 block,
    # 导致 dp 匹配到上拍 dispatch (实测 gmm 曾因此偏大 1.78us)
    # dp 取簇内各 T0 独立前向匹配的最大 dispatch: 混入的旧拍 block 匹配值更小, max 淹没之
    # T0/T1/T2 统一 max 口径 (墙钟语义, 与 TaskDur 同口径): TaskDur 由最慢核决定,
    # 故 T1-stars 应取 max(T1)——所有核都进入计算的时刻; mean 是单核平均视角会偏小
    # (实测 add mean 0.63 vs max 0.88us; Cube 核启动本身错峰, gmm T0 spread 0.6us)
    # max 对 D2H 旧拍回读天然稳健: 旧拍时间戳更小, max 自动淹没之
    win = int(1.5 * freq)
    t0max = max(r[0] for r in recs)
    cluster = [r for r in recs if r[0] >= t0max - win and r[2] > 0]
    t2_stars_chk = None
    t1_stars_max = None
    t0_stars_max = None
    if len(cluster) >= max(4, len(recs) // 4):
        dps = [match(r[0]) for r in cluster]
        dps = [d for d in dps if 0 < d <= 20000]
        if dps:
            # 各 T0 的前向匹配 dispatch 时刻 = T0 - d0
            dp = max(r[0] - d for r, d in zip(cluster, dps) if 0 < d <= 20000)
            t0_stars_max = (max(r[0] for r in cluster) - dp) / freq
            t1_stars_max = (max(r[1] for r in cluster if r[1] > 0) - dp) / freq
            t2_stars_chk = (max(r[2] for r in cluster) - dp) / freq
    return {
        'n': len(t0s),
        'n_t1': len(t1s),
        'n_t2': len(t2s),
        # 差值指标用中位数: 混合回读(拍次错位)产生的离群差值不拉偏均值
        't1_t0': statistics.median(t1t0) / freq if t1t0 else None,
        't2_t0': statistics.median(t2t0) / freq if t2t0 else None,
        # 墙钟口径 (末拍簇 max): 与 TaskDur/ffts 校验同口径
        't0_stars': t0_stars_max if t0_stars_max is not None else (statistics.mean(t0s) / freq if t0s else None),
        't1_stars': t1_stars_max if t1_stars_max is not None else (statistics.mean(t1s) / freq if t1s else None),
        't2_stars': t2_stars_chk if t2_stars_chk is not None else (statistics.mean(t2s) / freq if t2s else None),
        # 单核参考口径 (mean)
        't0_stars_mean': statistics.mean(t0s) / freq if t0s else None,
        't1_stars_mean': statistics.mean(t1s) / freq if t1s else None,
    }

def parse_ffts_pmu(prof_dir):
    """解析 ffts_profile.data (STARS 每核 PMU, david 脚本 prof_pmu_david_analyze.py 的结构):
    返回记录列表, 关键字段 pmu_start/pmu_end (每核启动/结束的 SYS_CNT 权威时间戳)
    与 ai_total_cycle (核计算 cycles 总和)"""
    data_dirs = glob.glob(os.path.join(prof_dir, "**", "device_*", "data"), recursive=True)
    if not data_dirs:
        return []
    fmt = "HH" + "HH" + "Q" + "I" * 4 + "Q" * 12  # 256B/record
    sz = struct.calcsize(fmt)
    recs = []
    for name in os.listdir(data_dirs[0]):
        if "ffts_profile.data" not in name or "done" in name:
            continue
        try:
            with open(os.path.join(data_dirs[0], name), "rb") as fd:
                ctx = fd.read()
        except OSError:
            continue
        for i in range(len(ctx) // sz):
            f = struct.unpack(fmt, ctx[i*sz:(i+1)*sz])
            recs.append({"func_type": f[0] & 63, "core_type": f[6] & 1,
                         "block_id": f[7] >> 16, "ai_total_cycle": f[4],
                         "pmu_start": f[19], "pmu_end": f[20]})
    return recs

def ffts_check(prof_dir, freq, taskdur):
    """ffts PMU + kernel 打点 + dispatch 三方联合校验:
    ① dispatch->pmu_start = 核启动延迟 (权威, 同域 SYS_CNT)
    ② kernel T2 -> 最近 pmu_end = 硬件尾 (MTE 排空; kernel 打点无法覆盖, 这就是
       T2-stars 不可能等于 TaskDur 的物理原因)
    ③ dispatch(末拍) -> pmu_end max ≈ TaskDur + STARS 完成上报 (正确校验口径)"""
    pmu = parse_ffts_pmu(prof_dir)
    recs = parse_cycle_blocks(f"{prof_dir}.stderr.log")
    disp = parse_stars_dispatches(prof_dir)
    if not pmu or not recs or not disp:
        return None
    ends = sorted(r["pmu_end"] for r in pmu if r["pmu_end"] > 0)
    if not ends:
        return None
    # kernel 末拍簇 (T0 聚类, 窗口 1.5us, 与 three_metrics 同口径)
    t0max = max(r[0] for r in recs)
    kcluster = [r for r in recs if r[0] >= t0max - int(1.5 * freq) and r[2] > 0]
    if not kcluster:
        return None
    t2_max = max(r[2] for r in kcluster)
    # 末拍 dispatch: 簇内 T0 前向匹配的最大 dispatch 时刻
    dps = []
    for r in kcluster:
        j = bisect.bisect_right(disp, r[0]) - 1
        if j >= 0:
            dps.append(disp[j])
    if not dps:
        return None
    dp = max(dps)
    # 核启动延迟: dp 之后最早的 pmu_start
    starts_after = [r["pmu_start"] for r in pmu if r["pmu_start"] > dp]
    launch_delay = (min(starts_after) - dp) / freq if starts_after else None
    # 硬件尾: 用全量 pmu_end 最大值 (末拍最慢核结束), 与 T2 末拍最大值做差。
    # 注意 ffts 是滚动采样 (每拍仅记部分核), 末拍样本可能很少, 不做 per-block 配对
    # (曾因 min(e>=T2) 配到邻拍样本产生 ~2.2us 假尾); 全局 max 两端口径一致且稳定
    pmuend_max = max(ends)
    hw_tail = (pmuend_max - t2_max) / freq
    # 校验: dispatch -> pmu_end max vs TaskDur (差值 = STARS 完成上报段)
    pmuend_stars = (pmuend_max - dp) / freq
    return {
        "launch_delay": launch_delay,
        "hw_tail": hw_tail,
        "pmuend_stars": pmuend_stars,
        "diff_vs_taskdur": pmuend_stars - taskdur,
    }

def analyze_add(prof_dir, label):
    csvs = glob.glob(os.path.join(prof_dir, "**", "op_summary_*.csv"), recursive=True)
    if not csvs:
        print(f"  {label}: [未找到 op_summary CSV]")
        return None, None
    with open(csvs[0]) as f:
        rows = list(csv.DictReader(f))
    kr = [r for r in rows if 'add_kernel' in r.get('Op Name', '')]
    if not kr:
        print(f"  {label}: [未找到 add_kernel 数据]")
        return None, None
    durs = [float(r['Task Duration(us)'].strip().rstrip('\t')) for r in kr[2:]]
    result = {'label': label, 'mean': statistics.mean(durs)}
    return result, three_metrics(prof_dir, 'add', freq_add)

def analyze_gmm(prof_dir, label):
    csvs = glob.glob(os.path.join(prof_dir, "**", "op_summary_*.csv"), recursive=True)
    if csvs:
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        kr = [r for r in rows if 'groupedmatmul' in r.get('Op Name', '')]
        if not kr:
            print(f"  {label}: [op_summary 中未找到 groupedmatmul 数据]")
            return None, None
        all_durs = [float(r['Task Duration(us)'].strip().rstrip('\t')) for r in kr]
    else:
        csvs = glob.glob(os.path.join(prof_dir, "**", "OpBasicInfo_*.csv"), recursive=True)
        if not csvs:
            print(f"  {label}: [未找到 op_summary 或 OpBasicInfo CSV]")
            return None, None
        all_durs = []
        for csv_path in sorted(csvs):
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    try:
                        all_durs.append(float(r.get('Task Duration(us)', '0').strip()))
                    except: pass
    if not all_durs:
        print(f"  {label}: [无有效数据]")
        return None, None
    steady = all_durs[2:]
    result = {'label': label, 'mean': statistics.mean(steady)}
    return result, three_metrics(prof_dir, 'gemm', freq_gmm)

def fmt(v):
    return f"{v:.3f}" if v is not None else "N/A"

print()
print("============================================")
print("  头开销测试结果 (稳态=跳过前2次)")
print("  STARS 四指标 (max 口径=墙钟, 与 TaskDur 同语义由最慢核决定):")
print("  max(T0)-stars=调度+硬件启动  T1-T0=单核软件头(median)  max(T1)-stars=总头开销  max(T2)-stars=kernel末指令")
print("  ffts PMU 权威校验: max(T2)+硬件尾 == dispatch->最慢核结束 (自洽闭环, 不依赖 TaskDur)")
print("============================================")
print()

results = []
if os.path.isdir(os.path.join(output_dir, 'add_full')):
    print("add (Vector):")
    r, m = analyze_add(os.path.join(output_dir, 'add_full'), 'add 满核')
    if r:
        print(f"  {r['label']:12s} TaskDur steady mean={r['mean']:.3f}us")
        if m:
            print(f"  {'':12s} max(T0)-stars={fmt(m['t0_stars'])}us  T1-T0={fmt(m['t1_t0'])}us  "
                  f"max(T1)-stars={fmt(m['t1_stars'])}us  max(T2)-stars={fmt(m['t2_stars'])}us  "
                  f"[单核mean: T0={fmt(m['t0_stars_mean'])} T1={fmt(m['t1_stars_mean'])}]  (n={m['n']} blocks)")
        fm = ffts_check(os.path.join(output_dir, 'add_full'), freq_add, r['mean'])
        if fm:
            print(f"  {'':12s} [ffts PMU] dispatch->核启动={fm['launch_delay']:.3f}us  "
                  f"T2->核结束(硬件尾)={fm['hw_tail']:.3f}us  dispatch->最慢核结束={fm['pmuend_stars']:.3f}us")
        results.append(('add', 'Vector', add_cores, r['mean'], m))
    print()
if os.path.isdir(os.path.join(output_dir, 'gmm_full')):
    print("grouped_matmul (Cube):")
    r, m = analyze_gmm(os.path.join(output_dir, 'gmm_full'), 'gmm 满核')
    if r:
        print(f"  {r['label']:12s} TaskDur steady mean={r['mean']:.3f}us")
        if m:
            print(f"  {'':12s} max(T0)-stars={fmt(m['t0_stars'])}us  T1-T0={fmt(m['t1_t0'])}us  "
                  f"max(T1)-stars={fmt(m['t1_stars'])}us  max(T2)-stars={fmt(m['t2_stars'])}us  "
                  f"[单核mean: T0={fmt(m['t0_stars_mean'])} T1={fmt(m['t1_stars_mean'])}]  (n={m['n']} blocks)")
        fm = ffts_check(os.path.join(output_dir, 'gmm_full'), freq_gmm, r['mean'])
        if fm:
            print(f"  {'':12s} [ffts PMU] dispatch->核启动={fm['launch_delay']:.3f}us  "
                  f"T2->核结束(硬件尾)={fm['hw_tail']:.3f}us  dispatch->最慢核结束={fm['pmuend_stars']:.3f}us")
        results.append(('grouped_matmul', 'Cube', gmm_cores, r['mean'], m))
    print()

# 生成汇总表
summary_path = os.path.join(output_dir, 'summary.csv')
with open(summary_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(['算子', '类型', '核数', 'TaskDur稳态mean(us)', 'T0-stars(us)', 'T1-T0(us)', 'T1-stars(us)', 'T2-stars(us)'])
    for op, typ, cores, mean, m in results:
        if m:
            w.writerow([op, typ, cores, f"{mean:.3f}", fmt(m['t0_stars']),
                        fmt(m['t1_t0']), fmt(m['t1_stars']), fmt(m['t2_stars'])])
        else:
            w.writerow([op, typ, cores, f"{mean:.3f}", "N/A", "N/A", "N/A", "N/A"])

print(f"汇总 CSV: {summary_path}")
print()
print("============================================")
print("  完成!")
print("============================================")
PYEOF
