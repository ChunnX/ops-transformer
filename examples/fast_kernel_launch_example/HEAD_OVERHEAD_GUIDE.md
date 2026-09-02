# Kernel Launch 头开销测试指南

## 概述

本指南说明如何在 Ascend 950PR 环境上复测 add 和 grouped_matmul 两个算子的 kernel launch 头开销数据（1核 + 满核两组）。

提供两种测量方法：

| 方法 | 原理 | 精度 | 适用场景 |
|------|------|------|---------|
| **A: Early Return + msprof** | 在计算主流程前插入 `return;`，用 msprof Task Duration 测量 | ~0.1us | 快速基线测量 |
| **B: GetSystemCycle 打点** | kernel 内 `GetSystemCycle()` 读取 SYS_CNT 寄存器，精确到周期级 | ~1 cycle | 需要细粒度分解（InitBuffer vs Tiling vs Dispatch） |

### 方法 A: Early Return + msprof

**原理**：在 kernel 计算主流程前插入 `return;`（early return），让 kernel 只跑初始化部分（tiling 解析 / buffer 初始化），跳过实际计算。用 msprof 采集的 Task Duration 即为"头开销"。

> **注意**：return 必须放在**函数作用域**（for/tail 块之外）。如果 return 在 for 循环体内，当 tileNum=0 时 return 是死代码，控制流会落到 tail 块执行实际计算，导致测量结果包含 DMA+计算开销而非纯头开销。详见 [tail-return 验证测试](#方法-b-getsystemcycle-打点)。

## 环境要求

| 项目 | 要求 |
|------|------|
| NPU | Ascend 950PR (dav-3510) |
| CANN | 9.1.0+ (需含 msprof) |
| torch | 2.7.1+ |
| torch_npu | 2.7.1.post4+ (需支持 Ascend950) |
| Python | 3.12+ |

> **注意**：torch_npu 2.7.1.post2 及更早版本不支持 Ascend950PR，会报 `Unsupported soc version`。需使用 torch_npu 2.7.1.post4 或更高版本。

## 一键执行

```bash
# 1. Clone 仓库（token 已内嵌）
git clone -b head-overhead-a5 https://oauth2:NpWjsPrRrAxRsyv5sYqKhCMi@gitcode.com/liulinxiang/ops-transformer.git
cd ops-transformer/examples/fast_kernel_launch_example

# 2. Source CANN 环境
source <CANN安装路径>/set_env.sh

# 3. 一键执行
bash run_head_overhead.sh
```

脚本会自动完成：
1. 检查 CANN 环境 + NPU 设备 + torch_npu
2. 通过 sed 修改源码核数，分别构建 1核 和 满核 两个 wheel
3. 用 msprof 采集 add / grouped_matmul 各 1核+满核 共四组数据（101 次循环/组）
4. 汇总输出结果表 + `summary.csv`

采集结果默认保存在 `head_overhead_results/` 目录下。

## 手动步骤（如需逐步执行）

### Step 1: Clone 仓库并切换分支

```bash
git clone -b head-overhead-a5 https://oauth2:NpWjsPrRrAxRsyv5sYqKhCMi@gitcode.com/liulinxiang/ops-transformer.git
cd ops-transformer
```

### Step 2: Source CANN 环境

```bash
source <CANN安装路径>/set_env.sh
# 例如:
# source /home/developer/Ascend/cann-9.1.0-beta.3/set_env.sh
```

### Step 3: 进入工程目录并安装依赖

```bash
cd examples/fast_kernel_launch_example
pip install -r requirements.txt
```

### Step 4: 修改核数 + 构建 Wheel

通过 sed 将核数写死为 1（1核）或实际核数（满核），分别构建：

```bash
# ---- 1核版本 ----
sed -i 's/int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) \/ MIN_ELEMS_PER_CORE);/int64_t numBlocks = 1;/' csrc/add/ascend950/add.cpp
sed -i 's/uint32_t blockDim = tilingData.gmmBaseParams.get_coreNum();/uint32_t blockDim = 1; tilingData.gmmBaseParams.coreNum = 1;/' csrc/grouped_matmul/ascend950/grouped_matmul.cpp

export NPU_SOC_VERSION=ascend950
python3 setup.py clean
python3 -m build --wheel --no-isolation
python3 -m pip install dist/*.whl --force-reinstall --no-deps

# ---- 满核版本（恢复源码后重新 sed + 构建）----
# 恢复源码: git checkout -- csrc/add/ascend950/add.cpp csrc/grouped_matmul/ascend950/grouped_matmul.cpp
# sed 改为实际核数（Vector=56, Cube=28），重复上述构建步骤
```

### Step 5: 用 msprof 采集头开销数据

```bash
# add（msprof 直接包裹）
msprof --output=/tmp/msprof_add_1core python3 tests/add/test_add.py
msprof --output=/tmp/msprof_add_full python3 tests/add/test_add.py

# grouped_matmul（用 msprof op 子命令，CANN 9.1.0 的 msprof 无法采集 Cube 核 op_summary）
msprof op --output=/tmp/msprof_gmm_1core \
  --application="python3 tests/grouped_matmul/test_groupedmatmul.py" \
  --launch-count=101 --warm-up=0 --replay-mode=application
msprof op --output=/tmp/msprof_gmm_full \
  --application="python3 tests/grouped_matmul/test_groupedmatmul.py" \
  --launch-count=101 --warm-up=0 --replay-mode=application
```

### Step 6: 查看采集结果

```bash
# add: op_summary CSV
find /tmp/msprof_add_1core -name "op_summary_*.csv"
find /tmp/msprof_add_full -name "op_summary_*.csv"

# grouped_matmul: OpBasicInfo CSV
find /tmp/msprof_gmm_1core -name "OpBasicInfo_*.csv"
find /tmp/msprof_gmm_full -name "OpBasicInfo_*.csv"
```

关键指标：`Task Duration(us)` = 头开销（含 kernel 启动 + 初始化）。跳过前 2 次（冷启动），取稳态数据。

## 参考数据（Ascend950PR, dav-3510, CANN 9.1.0-beta.3）

> 设备：28 Cube 核 / 56 Vector 核

| 算子 | 类型 | 核数 | 冷启动(us) | 稳态 mean(us) | 稳态 median(us) | 稳态 min(us) | 稳态 max(us) | n |
|------|------|------|-----------|--------------|----------------|-------------|-------------|---|
| **add** | Vector | 1核 | 5.747 | 2.722 | 2.027 | 1.636 | 4.866 | 99 |
| **add** | Vector | 56核(满) | 8.291 | 4.255 | 4.068 | 3.376 | 6.252 | 99 |
| **grouped_matmul** | Cube | 1核 | 4.230 | 1.840 | 1.822 | 1.746 | 2.235 | 99 |
| **grouped_matmul** | Cube | 28核(满) | 4.646 | 2.923 | 2.908 | 2.649 | 3.293 | 99 |

**关键发现**：
- add：1核→满核，头开销从 2.7us 涨到 4.3us（+59%），多核同步开销显著
- gmm：1核→满核，头开销从 1.8us 涨到 2.9us（+59%），同样有多核同步开销
- gmm 1核头开销（1.84us）比 add 1核（2.72us）更低，因为 Cube 核初始化路径更短

## 目录结构

```
examples/fast_kernel_launch_example/
├── csrc/                          # 算子源码
│   ├── add/ascend950/             # add 算子 (dav-3510)
│   ├── grouped_matmul/ascend950/  # grouped_matmul 算子 (dav-3510)
│   └── incre_flash_attention/     # IFA 算子（编译时自动排除）
├── tests/                         # 测试脚本 (101次循环, 无精度校验)
│   ├── add/test_add.py            # 基础功能测试
│   ├── add/cycle_test.py          # GetSystemCycle 精确打点测试
│   ├── add/compile_kernel.py      # 独立 .o 编译与体积分析
│   ├── add/test_tail_return.py    # tail-return 死代码验证
│   └── grouped_matmul/test_groupedmatmul.py
├── run_head_overhead.sh           # 一键测试脚本
├── HEAD_OVERHEAD_GUIDE.md         # 本文档
├── CMakeLists.txt
├── setup.py
└── build_and_test.sh              # 原始构建+测试脚本
```

### 方法 B: GetSystemCycle 打点

**原理**：在 kernel 内部关键位置调用 `AscendC::GetSystemCycle()` 读取 SYS_CNT 硬件寄存器，获取精确的周期级时间戳。结果写入 GM buffer，host 侧读取后计算各阶段耗时。

**打点位置**：
- `t0`：InitBuffer(×3) + SetGlobalBuffer(×3) 完成后
- `t1`：tiling 计算完成后（= prologue 结束）
- `t_end`：kernel 结束时

**使用方法**：

```bash
# 1. 将 cycle 版本复制为正式 add.cpp
cp csrc/add/ascend950/add_cycle.cpp csrc/add/ascend950/add.cpp

# 2. 运行测试
python3 tests/add/cycle_test.py
```

**输出示例**（A5/dav-3510, 1000×1000 fp16, 56 blocks）：
```
Run   InitBuf    Tiling   Prologue     Total
0       455         9       464      1813
1       441         9       450      1782
...

=== Frequency Calibration ===
Block0 mean total: 1800 cycles
msprof aiv_time: 2.984 us -> freq = 604 MHz  (实际约 283 MHz)

=== HEAD OVERHEAD (freq=283 MHz) ===
InitBuffer(×3)+SetGlobalBuffer(×3): 455 cyc = 1608 ns
Tiling calculation:                  9 cyc =    32 ns
Prologue total (per block):        464 cyc = 1.639 us
Wall-clock (kernel):              1800 cyc = 6.361 us
Dispatch+exit:                    1336 cyc = 4.721 us
```

**交叉验证**：
- GetSystemCycle wall-clock 与 msprof Task Duration 误差 < 0.5%
- prologue 周期数 / SYS_CNT 频率 ≈ msprof 注释版 aiv_time
- SYS_CNT 频率 ≈ 283 MHz（A5 系统调度时钟，非 AIV 计算时钟）

**tail-return 验证**：

```bash
# 验证 for 体内的 return 是否为死代码
python3 tests/add/test_tail_return.py
```

该测试对比两种场景：
- T1: for+return(死代码)+tail 完整执行 → ~2.93us (56 blocks)
- T2: for+return(死代码)+tail 内 return → ~1.40us (56 blocks)

差异 ~1.5us 即为 tail 块 CopyIn+Add+CopyOut 的实际开销，证明 for 体内的 return 是死代码（tileNum=0）。
