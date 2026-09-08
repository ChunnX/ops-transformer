# Kernel Launch 头开销测试指南

## 概述

本指南说明如何在 Ascend 950PR 环境上复测 add、grouped_matmul、incre_flash_attention 三个算子的 kernel launch 头开销数据，并给出打点前后（T0 之前 entry 段 / T2 之后 epilogue 段）的反汇编阶段分析。

测量方法：**GetSystemCycle 打点**——kernel 内 `GetSystemCycle()` 读取 SYS_CNT 寄存器，周期级精度（~1 cycle），支持头开销的细粒度分解（调度 / 硬件头 / 软件头 / kernel 主体 / 硬件尾）。

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
git clone -b head-overhead-clean https://oauth2:NpWjsPrRrAxRsyv5sYqKhCMi@gitcode.com/liulinxiang/ops-transformer.git
cd ops-transformer/examples/fast_kernel_launch_example

# 2. Source CANN 环境
source <CANN安装路径>/set_env.sh

# 3. 一键执行
bash run_head_overhead.sh
```

脚本会自动完成：
1. 检查 CANN 环境 + NPU 设备 + torch_npu，平台自动检测（A5/A3/A2）
2. sed 改核数构建满核 wheel（IFA 及其 meta 算子在 ascend950 纳入构建）
3. 采集 add / grouped_matmul（NPUGraph capture + replay 多拍）；ascend950 上追加采集 incre_flash_attention（250 次发射 + 打点回读）
4. 汇总输出结果表 + `summary_{soc}.csv`（三算子 min/mean/max 长表）

可选环境变量：`RUN_ADD` / `RUN_GMM`（默认 1）、`SKIP_BUILD=1`（跳过构建）、`REPLAY_SHOTS`（多拍次数，默认 5）、`ADD_CORES` / `GMM_CORES`（真核数实验）。

采集结果默认保存在 `head_overhead_results/` 目录下。

## 打点方法：GetSystemCycle

**原理**：在 kernel 内部关键位置调用 `AscendC::GetSystemCycle()` 读取 SYS_CNT 硬件寄存器（A5 实测 1000MHz，1 cycle = 1ns），获取周期级时间戳。结果写入持久 GM buffer（跨 launch 复用，capture/replay 兼容），host 侧读取后计算各阶段耗时。

**时间轴与打点位置**：

```
STARS dispatch → kick(核启动) → T0(kernel入口) → T1(prologue后) → [计算主体] → T2(kernel末) → DCCI → END → pmu_end
```

| 打点 | add / grouped_matmul | incre_flash_attention |
|------|---------------------|----------------------|
| T0 | kernel 入口第一行 | AIV 入口（`headStartCycle`） |
| T1 | TPipe Init 完成后、首个 MTE2 搬运前 | 首个 MTE2 搬运前（`beforeFirstMte2Cycle`） |
| T2 | kernel 末尾（MTE3/FIX 排空同步后） | AIV 侧 `op.Process()` 返回后（timing slot 7） |

**输出指标口径**（每指标 min=最快核 / mean=核平均 / max=最慢核）：

| 指标 | 含义 |
|------|------|
| 调度+硬件头 T0-stars | STARS dispatch → kernel 入口（含 ICache/参数预取/栈帧/参数装载） |
| 软件头 T1-T0 | TPipe/tiling 初始化 |
| 头开销 T1-stars | 前两项之和 |
| kernel主体 T2-T1 | 实际计算 |
| kernel末指令 T2-stars | dispatch → kernel 末打点 |
| 硬件尾 pmu_end-T2 | 写回 GM + SyncAll + DCCI + END（ffts PMU 权威时间戳） |
| 全链路 pmu_end-stars | dispatch → 最慢核结束（与 TaskDur 同语义闭环校验） |

add/gmm 采用 NPUGraph capture + replay 多拍（默认 5 拍），各拍独立计算全部指标后取 median，滤除单拍快照的 icache 冷热 / 核启动错峰干扰；IFA 采用 250 次发射（50 热身 + 200 稳态），56 AIV 核逐 block 回读打点。

## 参考数据（打点口径，Ascend950PR, CANN 9.1.0, SYS_CNT=1000MHz）

| 指标 (mean/max us) | add (Vector, 56核) | grouped_matmul (Cube, 28核) | incre_flash_attention (MIX, 28+56核) |
|------|-----------|-----------|-----------|
| 调度+硬件头 T0-stars | 0.16 / 0.17 | 0.79 / 1.07 | 1.84 / 4.25 |
| 软件头 T1-T0 | 0.28 / 0.36 | 0.68 / 1.06 | 1.15 / 3.97 |
| **头开销 T1-stars** | **0.44 / 0.51** | **1.47 / 1.68** | **3.00 / 4.98** |
| kernel主体 T2-T1 | 7.10 / 7.18 | 0.48 / 1.28 | 0.27（AIV 视角） |
| kernel末指令 T2-stars | 7.55 / 7.69 | 1.97 / 2.97 | 3.27 / 5.19 |
| 硬件尾 pmu_end-T2 | 0.34~1.12 | 0.53~0.76 | 0.64 |
| TaskDur 稳态 mean | 8.96 | 3.83 | 4.61 |

注：IFA 的 T2 只打 AIV 侧，AIC Cube 计算拖尾约 0.9us 计入 TaskDur 而非 T2，解读 IFA 数据时注意是 AIV 视角。

## 打点前后的反汇编阶段分析

### 操作步骤

```bash
# 1. 提取 device 二进制 (host .o 的 .aicore_binary section, 本身为 elf64-hiipu ELF)
objcopy --dump-section .aicore_binary=add_kernel.bin \
  build/csrc/add/ascend950/CMakeFiles/add_obj.dir/add.cpp.o

# 2. 反汇编 (需支持 hiipu64 的 llvm-objdump, A5 反汇编工具包)
llvm-objdump -d add_kernel.bin > add_full_aicore.txt

# 3. 定位打点: 搜索 SYS_CNT 读取 (T0/T1/T2 即 MOV Xn, SYS_CNT)
grep -n 'SYS_CNT' add_full_aicore.txt
```

Mapping Symbols：`$x`=MainScalar / `$k`=SIMD / `$t`=SIMT；MIX 混合核有 `_mix_aic` 与 `_mix_aiv` 两套函数。注意正常发布二进制是 strip 过的（mapping symbols 丢失会反汇编失败），调试需用未 strip 的构建产物。工程内已归档三算子反汇编产物：`disasm/` 目录。

### T0 之前的 entry 段（打点前）

三算子 entry 段结构一致：

```
① PARA_BASE 装载      MOV X0, PARA_BASE + MOVK        (2 条)
② 参数预取            DC_PRELOADI [X0], #offset        (N 条, 64B 粒度, 编译器自动插入)
③ 栈帧设置            X29 = SYS_VA_BASE + 0xFE000 + COREID×stride (~14 条, 每核独立栈)
④ 参数装载            LD_IO/LDP 从 PARA_BASE 读参数     (M 条, 含 SYS 总线往返等待)
⑤ T0 打点             MOV Xn, SYS_CNT
```

**实测分段对账**：

| 阶段 (mean us) | add (56核) | gmm (28核) | IFA (28AIC+56AIV) |
|------|-----------|-----------|-----------|
| T0-stars 调度+硬件头 | 0.16 | 0.79 | 1.86 |
| entry 指令数 | 30 | 82 | AIC 37 / AIV 55 |
| 参数预取量 | 48B (×1) | 2KB (×33) | 1.25KB (×20, 每核) |

> **kick 拆分仅 A5 有效**（A5 上 kick=ffts pmu_start，可拆为 stars→kick 调度 ~0.13-0.19us + kick→T0 硬件头：add 0.03 / gmm 0.65 / IFA 1.66us）。A3 的 ffts f[19] 非 kick 时刻（实测恒为 dispatch+20cyc），无拆分意义——故上表只列合并口径 T0-stars（两平台通用），A3 汇总输出也不拆 kick 段，`seg1_pairs_*.csv` 仅 A5 有效。

- **add/gmm**：指令级估算与实测吻合。gmm 慢 0.63us 根因 = 参数区体积（48B vs 2KB）：预取数据量 + 装载指令数双重放大
- **IFA**：指令数与参数量均小于 gmm，但 kick→T0 是其 2.5 倍，~1.4us 缺口归因：**230KB 大二进制 ICache 冷启动**（8 组模板实例化只用 1 组）+ **MIX 1:2 派发错峰**（max/mean≈2.4 倍佐证）+ 84 核重复预取同一参数区的总线竞争

### T2 之后的 epilogue 段（打点后）

```
T2 打点
【尾1: T2 写回 GM】   LD_IO cycleBuf地址 + ST_IO 写 T2        (~0.13-0.18us)
【尾2: SyncAll】      is_sync_all 分支; add/IFA: BAR.ALL 硬件栅栏
                       gmm: 软件自旋轮询 + BAR.F + BAR.ALL      (~0.02-0.15us)
【尾3: DCCI+END】     MOVI X15,#0; DCCI X15,#1; END            (~0.40-0.50us)
```

实测硬件尾（pmu_end-T2）与逐段估算合计吻合，**DCCI 占硬件尾 60~80%**。

**DCCI 实验方法**（测量 kernel 末自动 DCache 全刷的代价）：

```cmake
# csrc/add/ascend950/CMakeLists.txt: 关闭 kernel 末自动 DCCI 插入
# 注意: add_sources 宏只收单参数需拼接; 该选项为 LLVM 后端隐藏选项, 必须 -mllvm 传递
add_sources("--npu-arch=dav-3510 -mllvm -cce-aicore-dcci-before-kernel-end=false")
```

验证与结果：
1. 反汇编确认：kernel 实例化的 `DCCI X15,#1` 全部消失，epilogue 变为直接 `END`
2. TaskDur A/B（604 稳态样本）：**关闭后 8.49us vs 基线 8.88~9.03us → DCCI ≈ 0.4-0.5us**
3. 副作用：打点数据（cycleBuf）读不回——证明标量写停在 DCache 脏行，**可见性依赖 DCCI**；需要打点时应在源码对 cycleBuf 调 `DataCacheCleanAndInvalid<T, CacheLine>(GlobalTensor)` 单行刷（A5/3510 支持）代替全量刷

## 目录结构

```
examples/fast_kernel_launch_example/
├── csrc/                          # 算子源码
│   ├── add/ascend950/             # add 算子 (dav-3510)
│   ├── grouped_matmul/ascend950/  # grouped_matmul 算子 (dav-3510)
│   ├── incre_flash_attention/ascend950/       # IFA 算子 (MIX 1:2)
│   └── incre_flash_attention_meta/ascend950/  # IFA metadata (AICPU)
├── tests/                         # 测试脚本
│   ├── add/                       # test_add / cycle_test / compile_kernel / test_tail_return
│   ├── grouped_matmul/test_groupedmatmul.py
│   └── incre_flash_attention/     # test.py + run_head_overhead.sh + analyze_head_overhead.py
├── disasm/                        # 三算子 device 反汇编产物 (含 add 关 DCCI 对照版)
├── scan_replay.py                 # NPUGraph capture/replay + 逐 block 打点 dump
├── parse_stars_log.py             # STARS dispatch 事件解析
├── run_head_overhead.sh           # 一键测试脚本 (构建+三算子采集+汇总)
├── HEAD_OVERHEAD_GUIDE.md         # 本文档
├── CMakeLists.txt
├── setup.py
└── build_and_test.sh              # 原始构建+测试脚本
```
