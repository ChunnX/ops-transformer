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
#   NPU_SOC_VERSION  编译目标 SoC (默认: 自动检测)
#   OUTPUT_DIR       采集数据输出目录 (默认: ./head_overhead_results)
#   RUN_ADD          是否跑 add (默认: 1)
#   RUN_GMM          是否跑 grouped_matmul (默认: 1)
#   REPLAY_SHOTS     replay 多拍次数 (默认: 5, 传给 scan_replay.py; 各拍独立计算指标取 median)
#
# 输出 (全部归档到 OUTPUT_DIR, 原始/中间/结果数据):
#   {add,gmm}_full/            msprof 原始数据 (op_summary/stars_soc.data/ffts_profile.data)
#   {add,gmm}_full.{stdout,stderr}.log  采集原始日志 (含每 block T0/T1/T2 dump)
#   stars_{add,gmm}_full.csv   STARS dispatch/complete 事件表
#   blocks_{add,gmm}.csv       每 block T0/T1/T2 + 前向匹配 dispatch (中间数据)
#   seg1_pairs_{add,gmm}.csv   段1 有效配对明细 (dispatch -> kick)
#   seg2_pairs_{add,gmm}.csv   段2 有效配对明细 (kick -> T0)
#   summary_{soc}.csv          汇总表: 段1/段2/段3/头开销/T2-stars 各 min/mean/max
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
    # 恢复被排除算子的 CMakeLists (排除时 mv 成 .bak)
    for _op in incre_flash_attention $INCOMPAT_OPS; do
        _c="csrc/${_op}/${SRC_SUFFIX}/CMakeLists.txt"
        [ -f "${_c}.bak" ] && mv "${_c}.bak" "$_c"
    done
    echo "[CLEANUP] 源码已恢复"
}
trap cleanup EXIT

# 排除 incre_flash_attention (IFA 在部分 CANN 版本有 aclprofTensor 编译问题)
IFA_CMAKE="csrc/incre_flash_attention/${SRC_SUFFIX}/CMakeLists.txt"
if [ -f "$IFA_CMAKE" ]; then
    mv "$IFA_CMAKE" "${IFA_CMAKE}.bak"
    echo "[SETUP] 已临时排除 incre_flash_attention"
fi

# 排除 CANN 8.5 不兼容算子 (TPipe::DestroyWithoutPipeAll 不存在, 会拖垮整 wheel 构建;
# 与头开销测试无关, 仅在源码存在时排除)
INCOMPAT_OPS="moe_distribute_combine_v2 moe_distribute_dispatch_v2 incre_flash_attention_meta update_context"
for _op in $INCOMPAT_OPS; do
    _c="csrc/${_op}/${SRC_SUFFIX}/CMakeLists.txt"
    if [ -f "$_c" ]; then
        mv "$_c" "${_c}.bak"
        echo "[SETUP] 已临时排除 ${_op} (CANN 8.5 不兼容)"
    fi
done

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

# 头开销核心指标为 max(T1)-stars (墙钟口径)。单轮采集内部即 101 次 launch
# (capture 101 次 + replay 整图回放: TaskDur 取稳态 mean 跳过前 2 次冷启动,
# 打点数据取末次 replay 的满核 blocks), 无需外部多轮循环;
# 3 轮独立运行实测 max(T1)-stars: add 0.907±0.037us, gmm 2.160±0.106us

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
SHOT_RE = re.compile(r"Shot(\d+) Block (\d+): T0=(\d+) T1=(\d+) T2=(\d+)")

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

def parse_cycle_shots(stderr_log):
    """解析 scan_replay.py 多拍 dump: {shot_id: [(T0, T1, T2), ...]}
    多拍时每拍是独立完整 replay 的末次 launch 快照, 各拍独立算指标后取 median"""
    shots = {}
    if not os.path.exists(stderr_log):
        return shots
    for line in open(stderr_log, errors="ignore"):
        m = SHOT_RE.search(line)
        if m:
            shots.setdefault(int(m.group(1)), []).append(
                (int(m.group(3)), int(m.group(4)), int(m.group(5))))
    return shots

def parse_cycle_shots_bid(stderr_log):
    """解析 scan_replay.py 多拍 dump (带 block_id): {shot_id: {bid: (T0, T1, T2)}}"""
    shots = {}
    if not os.path.exists(stderr_log):
        return shots
    for line in open(stderr_log, errors="ignore"):
        m = SHOT_RE.search(line)
        if m:
            shots.setdefault(int(m.group(1)), {})[int(m.group(2))] = (
                int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return shots

def seg2_match(prof_dir, freq):
    """段2 (T0 - Taicore_kick_start) dispatch 分组时间邻近配对:
    ffts_profile.data 无核 id 字段 (f[7]>>16 恒 0, f[2] 为环形缓冲槽位非核 id),
    且 blocks dump 的 shot 分组含 D2H 回读错位的跨拍 T0, 不能按 shot 聚类
    (gmm Cube 核间错峰达 3~29k cyc, 1.5us 簇只收 8/28 核)。
    正确分组: T0 按其前向匹配的 dispatch 归组 (同拍 = 同 dispatch),
    每个 pmu_start(=某核 kick) ① 前向匹配所属 dispatch ② 组内最近 T0
    ③ 2us 窗口 (back-to-back launch 间隔 ≈ TaskDur>2us, 窗口内必同拍)。
    返回 [(kick, t0, diff_cyc)]"""
    pmu = parse_ffts_pmu(prof_dir)
    disp = parse_stars_dispatches(prof_dir)
    shots = parse_cycle_shots_bid(f"{prof_dir}.stderr.log")
    if not pmu or not disp or not shots:
        return []
    # T0 按前向匹配的 dispatch 归组 (真拍分组, 免疫 D2H 跨拍脏 T0)
    by_disp = {}
    for sid in shots:
        for bid, (t0, _, _) in shots[sid].items():
            i = bisect.bisect_right(disp, t0) - 1
            if i >= 0:
                by_disp.setdefault(disp[i], []).append(t0)
    if not by_disp:
        return []
    disps = sorted(by_disp)
    for v in by_disp.values():
        v.sort()
    pmu_starts = sorted(r["pmu_start"] for r in pmu if r["pmu_start"] > 0)
    win = int(2 * freq)
    pairs = []
    for ps in pmu_starts:
        j = bisect.bisect_right(disps, ps) - 1
        if j < 0:
            continue
        t0s = by_disp[disps[j]]
        k = bisect.bisect_right(t0s, ps)
        if k < len(t0s):
            d = t0s[k] - ps
            if 0 < d <= win:
                pairs.append((ps, t0s[k], d))
    return pairs

def seg2_stats(prof_dir, freq):
    """段2 (T0 - Taicore_kick_start) 统计: 时间邻近配对取 min/mean/max
    (核间有快有慢; pmu 滚动采样, 每拍仅部分核有 pmu 记录)"""
    pairs = seg2_match(prof_dir, freq)
    if not pairs:
        return None
    vals = [d / freq for _, _, d in pairs]
    return {"n": len(vals), "min": min(vals),
            "mean": statistics.mean(vals), "max": max(vals)}

def dump_intermediates(prof_dir, tag, freq):
    """归档计算过程原始/中间数据到结果目录:
    blocks_{tag}.csv   每 block T0/T1/T2 + 前向匹配的 dispatch sys cnt
    seg1_pairs_{tag}.csv  段1 有效配对 (dispatch -> pmu_start) 明细
    seg2_pairs_{tag}.csv  段2 有效配对 (pmu_start -> T0) 明细"""
    if not os.path.isdir(prof_dir):
        return
    disp = parse_stars_dispatches(prof_dir)
    shots = parse_cycle_shots_bid(f"{prof_dir}.stderr.log")
    pmu = parse_ffts_pmu(prof_dir)
    d_hi = int(20 * freq)
    if disp and shots:
        def fwd(ts):
            i = bisect.bisect_right(disp, ts) - 1
            return disp[i] if i >= 0 else ''
        with open(os.path.join(output_dir, f'blocks_{tag}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['shot', 'block', 'T0', 'T1', 'T2',
                        'T0_matched_dispatch', 'T1_matched_dispatch', 'T2_matched_dispatch'])
            for sid in sorted(shots):
                for bid, (a, b, b2) in sorted(shots[sid].items()):
                    w.writerow([sid, bid, a, b, b2, fwd(a), fwd(b), fwd(b2)])
    if disp and pmu:
        with open(os.path.join(output_dir, f'seg1_pairs_{tag}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dispatch_syscnt', 'pmu_start_kick', 'kick-stars_cyc', 'kick-stars_us'])
            for r in pmu:
                j = bisect.bisect_right(disp, r["pmu_start"]) - 1
                if j >= 0:
                    d = r["pmu_start"] - disp[j]
                    if 0 < d <= d_hi:
                        w.writerow([disp[j], r["pmu_start"], d, round(d / freq, 6)])
    if pmu and shots:
        pairs = seg2_match(prof_dir, freq)
        with open(os.path.join(output_dir, f'seg2_pairs_{tag}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pmu_start_kick', 'T0', 'T0-kick_cyc', 'T0-kick_us'])
            for ps, t0, d in pairs:
                w.writerow([ps, t0, d, round(d / freq, 6)])

def three_metrics(prof_dir, op, freq):
    """T0-stars / T1-T0 / T1-stars / T2-stars 四指标 (kernel 打点 + STARS 联合)
    多拍模式: scan_replay.py 每拍独立 dump (Shot<r>), 各拍独立计算全部指标后
    对每指标取 median——单拍快照受 icache 冷热/核启动错峰/HBM 干扰 (~10-18% 波动),
    median 滤除离群拍 (预期轮间波动 <3%)。无 Shot 数据时回退单拍 (ReplayLast)。
    每个时间戳独立最近邻匹配 dispatch（免疫 D2H 回读拍次错位）:
    T0/T1/T2 即使回读为不同拍的值, 各自匹配同拍 dispatch 后单值指标自洽。
    跨 slot 差值(T1-T0/T2-T0)仅同拍可信, 单调软校验兜底。"""
    stderr_log = f"{prof_dir}.stderr.log"
    shots = parse_cycle_shots(stderr_log)
    if shots:  # 多拍: {shot_id: recs}
        shot_list = [shots[k] for k in sorted(shots)]
    else:      # 单拍兼容 (旧版 scan_replay.py 只 dump ReplayLast)
        recs = parse_cycle_blocks(stderr_log)
        if not recs:
            return None
        shot_list = [recs]
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

    def match_idx(ts):
        # 返回 (dispatch 下标, 差值): 供跨 slot 差值做同拍校验
        # (T0/T1/T2 回读可能来自不同拍, 差值 T1-T0 仅当两者匹配同一条 dispatch 才可信)
        i = bisect.bisect_right(disp, ts) - 1
        if i < 0:
            return -1, -1
        return i, ts - disp[i]

    # 匹配窗口按频率缩放: A5 1GHz 下 50cyc=0.05us/20000cyc=20us;
    # 固定 cyc 阈值在 A3 50MHz (1cyc=20ns) 下会把 T0-stars(<1us) 全部过滤掉
    d_lo = max(2, int(0.05 * freq))
    d_hi = int(20 * freq)

    def one_shot(recs):
        """单拍指标计算 (返回 dict 或 None)。recs 为该拍全部 block 的 (T0,T1,T2)"""
        t0s, t1t0, t1s, t2s, t2t0 = [], [], [], [], []
        for a, b, b2 in recs:
            d0 = match(a)
            if d_lo <= d0 <= d_hi:
                t0s.append(d0)
            d1 = match(b)
            if d_lo <= d1 <= d_hi:
                t1s.append(d1)
            d2 = match(b2)
            if d_lo <= d2 <= d_hi:
                t2s.append(d2)
            if a < b <= b2:  # 同拍单调: 跨 slot 差值才可信
                # 同拍校验: T0/T1/T2 须匹配同一条 dispatch (D2H 回读拍次错位时
                # T0 来自上一拍, 差值可达 launch 间隔 ~44us, 直接污染 min/mean/max)
                i0, dd0 = match_idx(a)
                i1, dd1 = match_idx(b)
                i2, dd2 = match_idx(b2)
                if i0 >= 0 and i0 == i1:
                    t1t0.append(b - a)
                if i0 >= 0 and i0 == i2:
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
            dps = [d for d in dps if 0 < d <= d_hi]
            if dps:
                # 各 T0 的前向匹配 dispatch 时刻 = T0 - d0
                dp = max(r[0] - d for r, d in zip(cluster, dps) if 0 < d <= d_hi)
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
            # 核间快慢分布 (min=最快核 / mean=核平均 / max=最慢核):
            # 段3 T1-T0、头开销 T1-stars、T2-stars 的三组统计
            't1_t0_min': min(t1t0) / freq if t1t0 else None,
            't1_t0_mean': statistics.mean(t1t0) / freq if t1t0 else None,
            't1_t0_max': max(t1t0) / freq if t1t0 else None,
            't1_stars_min': min(t1s) / freq if t1s else None,
            't1_stars_max_blk': max(t1s) / freq if t1s else None,
            't2_stars_min': min(t2s) / freq if t2s else None,
            't2_stars_mean': statistics.mean(t2s) / freq if t2s else None,
            't2_stars_max_blk': max(t2s) / freq if t2s else None,
        }

    per_shot = [m for m in (one_shot(recs) for recs in shot_list) if m]
    if not per_shot:
        return None
    if len(per_shot) == 1:
        out = dict(per_shot[0])
        out['shots'] = 1
        return out
    # 多拍: 各指标独立取 median
    keys = ['t0_stars', 't1_t0', 't1_stars', 't2_stars', 't2_t0',
            't0_stars_mean', 't1_stars_mean',
            't1_t0_min', 't1_t0_mean', 't1_t0_max',
            't1_stars_min', 't1_stars_max_blk',
            't2_stars_min', 't2_stars_mean', 't2_stars_max_blk']
    out = {'shots': len(per_shot)}
    for k in keys:
        vals = [m[k] for m in per_shot if m.get(k) is not None]
        out[k] = statistics.median(vals) if vals else None
    out['n'] = per_shot[-1]['n']
    out['n_t1'] = per_shot[-1]['n_t1']
    out['n_t2'] = per_shot[-1]['n_t2']
    # 各拍 t1_stars 逐拍值 (stdout 观测拍间波动)
    out['t1_stars_per_shot'] = [round(m['t1_stars'], 3) for m in per_shot if m.get('t1_stars') is not None]
    return out

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
    ③ dispatch(末拍) -> pmu_end max ≈ TaskDur + STARS 完成上报 (正确校验口径)
    另附 段1(Taicore_kick_start - Tstars) 全量分布: 每个 pmu_start 前向匹配
    dispatch, 全部有效配对的 min/median/p90/max (纯硬件固定开销, 核间无错峰)"""
    pmu = parse_ffts_pmu(prof_dir)
    recs = parse_cycle_blocks(f"{prof_dir}.stderr.log")
    disp = parse_stars_dispatches(prof_dir)
    if not pmu or not recs or not disp:
        return None
    ends = sorted(r["pmu_end"] for r in pmu if r["pmu_end"] > 0)
    if not ends:
        return None
    # === 段1 全量分布: Taicore_kick_start(每个 pmu_start) 前向匹配 dispatch ===
    seg1 = []
    seg1_win = int(20 * freq)
    for r in pmu:
        j = bisect.bisect_right(disp, r["pmu_start"]) - 1
        if j >= 0:
            d = r["pmu_start"] - disp[j]
            if 0 < d <= seg1_win:
                seg1.append(d / freq)
    seg1_dist = None
    if seg1:
        s = sorted(seg1)
        seg1_dist = {
            "n": len(s),
            "min": s[0],
            "median": statistics.median(s),
            "mean": statistics.mean(s),
            "p90": s[int(len(s) * 0.9)],
            "max": s[-1],
        }
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
        "seg1_dist": seg1_dist,
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
print("  头开销分段测试结果 (稳态=跳过前2次; 多拍各拍独立计算取 median)")
print("  分段口径 (T0=取回首指令):")
print("    段1 调度开销   = Taicore_kick_start - Tstars")
print("    段2 硬件头     = T0 - Taicore_kick_start")
print("    段3 软件头     = T1 - T0")
print("    头开销(总)     = T1 - Tstars")
print("  核间有快有慢, 每指标三组统计: min=最快核 / mean=核平均 / max=最慢核(墙钟,与TaskDur同语义)")
print("  ffts PMU 校验: max(T2)+硬件尾 == dispatch->最慢核结束 (自洽闭环, 不依赖 TaskDur)")
print("============================================")
print()

def seg_line(name, formula, mn, mean, mx, extra=""):
    return (f"  {name:16s} {formula:14s} min={fmt(mn):>7s}  mean={fmt(mean):>7s}  max={fmt(mx):>7s} us{extra}")

def print_op_result(op, typ, cores, r, m, fm, s2):
    shots = m.get('shots', 1) if m else 1
    print(f"{op} ({typ}, {cores}核, {shots}拍):")
    print(f"  {'TaskDur':16s} {'稳态mean':14s} = {r['mean']:.3f} us")
    if not m:
        print("  [无 kernel 打点数据]")
        print()
        return
    seg1 = fm.get('seg1_dist') if fm else None
    if seg1:
        print(seg_line("段1 调度开销", "kick-stars", seg1['min'], seg1['mean'], seg1['max'],
                       f"  (n={seg1['n']}, median={seg1['median']:.3f}, 核间无错峰)"))
    else:
        print(seg_line("段1 调度开销", "kick-stars", None, None, None))
    if s2:
        print(seg_line("段2 硬件头", "T0-kick", s2['min'], s2['mean'], s2['max'],
                       f"  (n={s2['n']} dispatch分组配对, ffts 滚动采样)"))
    else:
        print(seg_line("段2 硬件头", "T0-kick", None, None, None, "  (无 ffts pmu 配对)"))
    print(seg_line("段3 软件头", "T1-T0", m.get('t1_t0_min'), m.get('t1_t0_mean'),
                   m.get('t1_t0_max'), f"  (median={fmt(m['t1_t0'])})"))
    print(seg_line("头开销(总)", "T1-stars", m.get('t1_stars_min'), m.get('t1_stars_mean'),
                   m['t1_stars']))
    print(seg_line("kernel末指令", "T2-stars", m.get('t2_stars_min'), m.get('t2_stars_mean'),
                   m['t2_stars']))
    per = m.get('t1_stars_per_shot')
    if per and len(per) > 1:
        print(f"  逐拍 max(T1)-stars: {'/'.join(f'{v:.3f}' for v in per)}  (n={m['n']} blocks)")
    if fm:
        # launch_delay 可能为 None (A3 低频分辨率, dispatch->核启动差可能为 0)
        ld = fm['launch_delay'] if fm['launch_delay'] is not None else 0.0
        print(f"  [ffts PMU] dispatch->核启动={ld:.3f}us  T2->核结束(硬件尾)={fm['hw_tail']:.3f}us  "
              f"dispatch->最慢核结束={fm['pmuend_stars']:.3f}us")
    print()

results = []
if os.path.isdir(os.path.join(output_dir, 'add_full')):
    prof = os.path.join(output_dir, 'add_full')
    r, m = analyze_add(prof, 'add 满核')
    if r:
        fm = ffts_check(prof, freq_add, r['mean'])
        s2 = seg2_stats(prof, freq_add)
        dump_intermediates(prof, 'add', freq_add)
        print_op_result('add', 'Vector', add_cores, r, m, fm, s2)
        results.append(('add', 'Vector', add_cores, r, m, fm, s2))
if os.path.isdir(os.path.join(output_dir, 'gmm_full')):
    prof = os.path.join(output_dir, 'gmm_full')
    r, m = analyze_gmm(prof, 'gmm 满核')
    if r:
        fm = ffts_check(prof, freq_gmm, r['mean'])
        s2 = seg2_stats(prof, freq_gmm)
        dump_intermediates(prof, 'gmm', freq_gmm)
        print_op_result('grouped_matmul', 'Cube', gmm_cores, r, m, fm, s2)
        results.append(('grouped_matmul', 'Cube', gmm_cores, r, m, fm, s2))

# 生成汇总表 (文件名带 SoC 前缀; 每指标 min/mean/max 三组)
summary_path = os.path.join(output_dir, f'summary_{soc_version}.csv')
with open(summary_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(['soc', '算子', '类型', '核数', 'shots', 'blocks', 'TaskDur稳态mean(us)',
                '段1kick-stars_min(us)', '段1kick-stars_mean(us)', '段1kick-stars_max(us)', '段1_n',
                '段2T0-kick_min(us)', '段2T0-kick_mean(us)', '段2T0-kick_max(us)', '段2_n',
                '段3T1-T0_min(us)', '段3T1-T0_mean(us)', '段3T1-T0_max(us)',
                '头开销T1-stars_min(us)', '头开销T1-stars_mean(us)', '头开销T1-stars_max(us)',
                'T2-stars_min(us)', 'T2-stars_mean(us)', 'T2-stars_max(us)'])
    for op, typ, cores, r, m, fm, s2 in results:
        seg1 = fm.get('seg1_dist') if fm else None
        row = [soc_version, op, typ, cores, m.get('shots', 1) if m else 1, m['n'] if m else '',
               f"{r['mean']:.3f}"]
        if seg1:
            row += [fmt(seg1[k]) for k in ('min', 'mean', 'max')] + [seg1['n']]
        else:
            row += ['N/A', 'N/A', 'N/A', '']
        if s2:
            row += [fmt(s2[k]) for k in ('min', 'mean', 'max')] + [s2['n']]
        else:
            row += ['N/A', 'N/A', 'N/A', '']
        if m:
            row += [fmt(m.get(k)) for k in ('t1_t0_min', 't1_t0_mean', 't1_t0_max')]
            row += [fmt(m.get(k)) for k in ('t1_stars_min', 't1_stars_mean', 't1_stars')]
            row += [fmt(m.get(k)) for k in ('t2_stars_min', 't2_stars_mean', 't2_stars')]
        else:
            row += ['N/A'] * 9
        w.writerow(row)

print(f"汇总 CSV: {summary_path}")
print()
print("============================================")
print("  完成!")
print("============================================")
PYEOF
