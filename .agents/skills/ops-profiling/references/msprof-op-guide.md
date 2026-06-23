# msprof op 采集与分析

> 标准上板采集流程：依赖 `$ASCEND_HOME/tools/msopprof/bin/msopprof`，产出 8 份独立 CSV + 逐核 `PipeUtilization.csv`，可走完 **构建 → 采集 → 归档 → 判定 → 优化 → 回归** 闭环。

---

## 前提

- 环境中存在 `$ASCEND_HOME/tools/msopprof/bin/msopprof`
- 推荐用法包含 `--warm-up` 以规避 DVFS

---

## Step 1：构建算子

**ops-transformer aclnn/eager 样例方式**：

```bash
bash build.sh --pkg --soc=ascend910b --ops={operator_name} --vendor_name=custom -j16 -O3
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
bash build.sh --run_example {operator_name} eager cust --vendor_name=custom --soc=ascend910b
```

**ops-transformer 局部 ST/native runner 方式**：

```bash
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0 \
OP_ST_BATCH_FILE=artifacts/{operator_name}/profiling/perf_cases.tsv \
msprof op --application=artifacts/{operator_name}/profiling/run_{operator_name}_st_native.sh \
  --output=artifacts/{operator_name}/profiling/msprof_op/round_NNN
```

该方式仅适用于已经建立 native runner wrapper 的算子。当前 `ops-transformer`
默认只优先支持 `lightning_indexer` 与 `recurrent_gated_delta_rule` 的本地 ST runner；
其他算子不能假设存在 `agent_ops` 或统一 native runner。wrapper 必须加载
`custom_transformer/op_api/lib/libcust_opapi.so`。

---

## Step 2：采集

```bash
# 基本用法
msprof op ./demo

# 推荐用法（含预热 + 指定输出）
msprof op --warm-up=10 --output=./msprof_output ./demo

# 多次运行取中位数或均值
msprof op --warm-up=10 --launch-count=5 --output=./msprof_output ./demo
```

在 `ops-transformer` 中如果 application 是 batch runner，`--launch-count` 必须覆盖真实
device task 数。batch case 数只在一 case 一 task 时成立；如果一个 case 触发多个
runtime task，`--launch-count` 用 case 数会只采集前缀并污染矩阵。

### 关键参数

| 参数 | 说明 | 何时使用 |
|------|------|---------|
| `--warm-up=N` | 预热 N 次后再采集 | **始终建议**，避免 DVFS（动态调频）影响首次运行 |
| `--launch-count=N` | 运行或采集 N 次用于统计 | 需要统计稳定性时 |
| `--output=<dir>` | 指定输出目录 | 避免结果散落 |
| 无需 `--soc-version` | 上板自动检测硬件 | — |

**输出**：在指定目录或当前目录下生成 `OPPROF_{timestamp}_XXX/` 文件夹。
不同 CANN 版本可能产出 `OpBasicInfo.csv` 或带后缀的 `OpBasicInfo*.csv`；解析脚本应使用
glob 匹配。

如果同一个 batch 内混合多个 kernel suffix 或 tiling key，不能按 CSV 文件名、
timestamp 或行顺序直接映射到 case。应解析 `msprof op` 控制台日志里的
`Start profiling ... kernel` 顺序，并在每个 kernel 内按 launch index 取对应
`OpBasicInfo*.csv` 行；否则会把不同 case 的耗时错配。

---

## Step 3：归档 + 统计摘要

```bash
# 找到最新 OPPROF 目录
OPPROF_DIR=$(ls -td <output_dir>/OPPROF_* | head -1)

# 归档 CSV + 生成摘要
python3 {skill_path}/scripts/perf_summary.py $OPPROF_DIR artifacts/{operator_name}/msprof_summary
```

脚本会：

1. 在目标摘要目录下创建 `round_NNN/` 归档目录（轮次自动递增）
2. 复制全部 8 个 CSV 原始文件到归档目录
3. 生成 `summary.txt`（各指标 min/avg/max，**不做达标判定**）

---

## Step 4：性能标准判定

对照下表与各流水占比，判定算子性能是否达标。**性能达标**指：主导流水与算子类型匹配（见 4.3）、表 4.2 中严重项未集中触发，且核间负载、带宽等未同时恶化。

### 4.1 总体判定流程

```
读取 OpBasicInfo.csv → Task Duration、Block Dim
    ↓
读取 PipeUtilization.csv → 各流水占比最高的单元（主导流水）
    ↓
对照 4.2 阈值与 4.3 算子类型预期
    ↓
综合结论
    ├── 指标整体健康、主导合理 → 达标或已接近硬件极限
    ├── 多项警告或主导与类型不符 → 有优化空间 → Step 5
    └── 多项严重项 → 严重瓶颈，须优化 → Step 5
```

### 4.2 各指标达标标准

| 指标 | 达标条件 | 警告条件 | 严重问题 |
|------|---------|---------|---------|
| **核间负载均衡** | 各核 `ai*_time(us)` 差异 <10% | 差异 10-30% | 差异 >30% |
| **Block Dim** | 等于可用核数（910B: 20~40 核） | 远小于可用核数 | Block Dim = 1 |
| **VEC ratio** | 与算子类型匹配（见 4.3） | VEC ratio >80% | VEC ratio >90% 且无优化空间 |
| **MTE2 ratio** | <30%（计算型算子） | 30-50% | >50%（搬运成为瓶颈） |
| **fixpipe_ratio** | <5% | 5-15% | >15%（地址未对齐） |
| **icache_miss_rate** | <5% | 5-15% | >15%（代码量过大） |
| **bank conflict 总占比** | `aiv_vec_total_cflt_ratio` <5% | 5-15% | >15% |
| **L2 Cache 总命中率** | >80% | 50-80% | <50% |
| **头开销** | <总耗时的 10% | 10-30% | >30% |
| **DoubleBuffer 效果** | MTE2/VEC 重叠 >30% | 重叠 10-30% | 重叠 <5% |
| **带宽利用率** | `bw_usage_rate` >60% | 30-60% | <30% |

### 4.3 不同算子类型的预期 ratio 分布

| 算子类型 | 主导流水 | 预期 ratio | 异常信号 |
|---------|---------|-----------|---------|
| **Elementwise**（Add/Mul/Relu） | VEC | vec_ratio 50-80% | MTE2 ratio > VEC ratio |
| **Reduction**（ReduceSum/Max） | VEC | vec_ratio 40-70% | scalar_ratio >20% |
| **Activation**（Softmax/Gelu） | VEC | vec_ratio 60-85% | 大量 cast 指令 |
| **MatMul** | CUBE | cube_ratio 40-70% | vec_ratio > cube_ratio |
| **纯搬运**（Transpose/Concat） | MTE2/MTE3 | mte2+mte3 合计 >50% | VEC ratio >30% |

---

## Step 5：瓶颈定位与优化

1. **先读 `summary.txt`** — 全局概览
2. **结合 `csv_fields_reference.md`** — 理解字段含义和阈值
3. **发现异常时再展开原始 CSV**
4. **再按下表与 [`optimization_quickref.md`](optimization_quickref.md)** — 确认瓶颈并查找优化方法

确认瓶颈类型后，以 `optimization_quickref.md` 中的具体方法为准。

**快速查找**：

| 瓶颈类型 | 判定条件 | 首选优化 |
|---------|---------|---------|
| **VEC Bound** | `aiv_vec_ratio` 最高 | UB 融合、减少 Cast、融合指令 |
| **MTE2 Bound** | `ai*_mte2_ratio` 最高 | 增大搬运粒度 ≥16KB、512B 对齐、L2 CacheMode |
| **CUBE Bound** | `aic_cube_ratio` 最高 | L0C 累加、L1 数据复用 |
| **SCALAR Bound** | `ai*_scalar_ratio` >30% | 缩小 TilingData、减少核数 |
| **核间不均衡** | 各核耗时差异 >10% | 调整 Tiling 切分策略 |
| **Bank Conflict** | `vec_bank_cflt_ratio` >5% | 调整 UB 地址、添加 padding |
| **头开销大** | 头开销占比 >30% | 减少核数、缩小 TilingData、TPipe 外置 |
| **DoubleBuffer 未生效** | MTE2/VEC 无重叠 | 检查 InitBuffer 是否设置 bufNum=2 |
| **流水线气泡** | 多单元均 30-50%，无主导 | 增加 workspace 份数、异步迭代 |

---

## Step 6：验证优化效果

每次优化后重新执行 Step 2–3，数据自动归档为 `round_NNN+1`。

```bash
# 对比两轮摘要
diff ops/{operator_name}/docs/perf/round_001/summary.txt ops/{operator_name}/docs/perf/round_002/summary.txt

# 或直接读两个 summary.txt 进行对比分析
```

**对比要点**：

1. Task Duration 是否下降
2. 瓶颈单元的 ratio 是否改善
3. 核间均衡是否改善（aiv_time min/max 差距）
4. 是否引入新的瓶颈

同 case 至少采集 3 次取中位数或均值。若某次样本明显偏离，保留原始 CSV 路径并解释是否属于
环境波动；不要只用最优单次样本作为优化结论。

性能优化的保留结论必须基于代表性矩阵，而不是单个目标 shape。目标大 shape 可以作为
优化信号，但最终矩阵应覆盖小 shape、边界附近 shape、常见中等 shape 和目标大 shape。
若目标 case 提升但通用 case 出现稳定回退，应判定为未通过性能回归，回退或继续优化。
当 baseline 与 candidate 采集间隔较长或结果接近噪声区间时，补跑同时间段 A/B
baseline 再计算提升比例。

提升比例按同一 case 的 `(baseline_us - candidate_us) / baseline_us * 100%` 计算。
如果一次 ACLNN 调用包含多个 device task，除目标 kernel 外还要比较 total device time
`sum(Task Duration(us))`。低于约 1% 的变化默认按噪声处理，除非重复 A/B 采集稳定且
代表性矩阵没有回退。

---

## 数据目录结构

### 临时输出（采集后、归档前）

```
OPPROF_{timestamp}_XXX/
├── dump/                       # 原始性能数据（无需关注）
├── OpBasicInfo.csv             # 算子基本信息（名称、核数、总耗时、频率）
├── PipeUtilization.csv         # 各流水线单元耗时和占比（最重要）
├── ArithmeticUtilization.csv   # Cube/Vector 指令 cycle 占比和计算量
├── Memory.csv                  # 内存读写带宽和数据搬运量
├── MemoryL0.csv                # L0A/L0B/L0C 读写带宽
├── MemoryUB.csv                # UB 读写带宽（Vector/Scalar）
├── L2Cache.csv                 # L2 Cache 命中率
├── ResourceConflictRatio.csv   # Bank conflict 和资源冲突占比
└── visualize_data.bin          # MindStudio Insight 可视化文件
```

### 持久归档

```
ops/{算子名}/docs/perf/
├── round_001/
│   ├── OpBasicInfo.csv
│   ├── PipeUtilization.csv
│   ├── Memory.csv
│   ├── ResourceConflictRatio.csv
│   ├── L2Cache.csv
│   ├── ArithmeticUtilization.csv
│   ├── MemoryUB.csv
│   ├── MemoryL0.csv
│   └── summary.txt            # 统计摘要（min/avg/max，不含判定）
├── round_002/
└── ...
```

字段含义详见 [`csv_fields_reference.md`](csv_fields_reference.md)。

---

## 上板 vs 仿真选择

| 维度 | 上板 (msprof op) | 仿真 (msprof op simulator) |
|------|-----------------|---------------------------|
| 需要 NPU | 是 | 否 |
| 时序精度 | 真实硬件时序 | 周期级模型估算 |
| 输出 | 8 个 CSV 文件 | CSV + trace.json |
| 指令级流水图 | 需加参数或单独仿真 | 默认输出 |
| **资源冲突数据** | **有**（ResourceConflictRatio.csv） | 无 |
| **L2 Cache** | **真实命中率** | 估算 |
| DVFS 影响 | 有（需 warm-up） | 无 |
| 适合阶段 | 性能验收、生产调优 | 早期开发、指令级调试 |

**建议**：开发阶段用仿真快速迭代，验收阶段用上板确认真实性能。

---

## 注意事项

1. **必须 warm-up**：首次运行受 DVFS 影响，耗时偏高。始终使用 `--warm-up=10`
2. **频率检查**：读取 `OpBasicInfo.csv` 的 `Current Freq` 和 `Rated Freq`，若 Current < Rated，说明芯片未满频运行
3. **MTE2/MTE3 带宽共享**：同时读写 GM 时总带宽共享，评估搬运段负载时宜按 MTE2、MTE3 合并字节量与平台带宽对照
4. **小数据量场景**：数据量很小时头开销占比会很高，这不一定是算子问题，而是数据量不足
5. **多核同地址访问**：多核同时读同一 512B 地址范围会被串行化，导致 MTE2 耗时异常

---

## 相关资源

| 文件 | 内容 |
|------|------|
| [`csv_fields_reference.md`](csv_fields_reference.md) | 8 个 CSV 文件的完整字段定义和阈值 |
| [`optimization_quickref.md`](optimization_quickref.md) | 各瓶颈类型的具体优化方法和案例 |
| `scripts/perf_summary.py` | 统计摘要生成 + CSV 归档（Step 3 调用） |
