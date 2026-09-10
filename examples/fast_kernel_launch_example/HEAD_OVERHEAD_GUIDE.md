# Kernel Launch 头开销测试指南

## 概述

本指南说明如何在 Ascend 950PR (A5) 与 Ascend 910C (A3) 环境上复测 add、grouped_matmul、incre_flash_attention 三个算子的 kernel launch 头开销数据，并给出打点前后（T0 之前 entry 段 / T2 之后 epilogue 段）的反汇编阶段分析。

测量方法：**GetSystemCycle 打点**——kernel 内 `GetSystemCycle()` 读取 SYS_CNT 寄存器，周期级精度（~1 cycle），支持头开销的细粒度分解（调度 / 硬件头 / 软件头 / kernel 主体 / 硬件尾）。

## 环境要求

| 项目 | A5: Ascend 950PR | A3: Ascend 910C |
|------|------|------|
| NPU | Ascend 950PR (dav-3510, 28 Cube + 56 Vector) | Ascend910_9392 (dav-2201, 24 Cube + 48 Vector) |
| CANN | 9.1.0T103 (需含 msprof) | 9.1.0T103 实测 (需含 msprof) |
| SYS_CNT | 1000MHz (1 cycle = 1ns) | 50MHz (1 cycle = 20ns) |
| torch | 2.7.1+ (2.12.0+cpu 实测通过) | 2.7.1+ |
| torch_npu | 2.7.1.post4+ (需支持 Ascend950；2.12.0.dev 实测通过) | 无 Ascend950 支持要求 |
| Python | 3.11+ (3.11 / 3.12 实测通过) | 3.12+ |

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
2. 查询设备核数（CANN platform_config），sed 改核数构建满核 wheel（IFA 及其 meta 算子在 ascend950 / ascend910_93 纳入构建；ascend910b 主线目录完整保留，构建按 NPU_SOC_VERSION 惰性跳过；临时排除部分 CANN 版本不兼容的无关算子，trap 退出自动恢复源码）
3. 采集 add / grouped_matmul（NPUGraph capture + replay 多拍）；ascend950 与 ascend910_93 上追加采集 incre_flash_attention（250 次发射 + 打点回读）
4. 汇总输出结果表 + `summary_{soc}.csv`（三算子 min/mean/max 长表）

可选环境变量：`RUN_ADD` / `RUN_GMM`（默认 1）、`SKIP_BUILD=1`（跳过构建）、`REPLAY_SHOTS`（多拍次数，默认 5）、`ADD_CORES` / `GMM_CORES`（真核数实验）。

采集结果默认保存在 `head_overhead_results/` 目录下。

## 打点方法：GetSystemCycle

**原理**：在 kernel 内部关键位置调用 `AscendC::GetSystemCycle()` 读取 SYS_CNT 硬件寄存器（A5 实测 1000MHz，1 cycle = 1ns；A3 实测 50MHz，1 cycle = 20ns，脚本按 SoC 内置默认，可用 `SYS_CNT_FREQ_MHZ` 覆盖），获取周期级时间戳。打点值写入持久 GM cycleData buffer（每核 3×int64，跨 launch 复用，capture/replay 兼容），由 host 侧读回解析；dispatch 时刻取自 STARS 日志的每核 sys cnt。

**时间轴与打点位置**：

```
STARS dispatch → kick(核启动) → T0(kernel入口) → T1(prologue后) → [计算主体] → T2(kernel末) → DCCI → END → pmu_end
```

| 打点 | add / grouped_matmul | incre_flash_attention |
|------|---------------------|----------------------|
| T0 | kernel 入口第一行 | AIV 入口（`headStartCycle`） |
| T1 | TPipe Init 完成后、首个 MTE2 搬运前 | 首个 MTE2 搬运前（`beforeFirstMte2Cycle`） |
| T2 | kernel 末尾（MTE3/FIX 排空同步后） | AIV 侧 `op.Process()` 返回后（timing slot 7） |

add kernel 以 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 声明任务类型。

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

add/gmm 采用 NPUGraph capture 101 次连发 + replay 多拍（默认 5 拍），各拍独立计算全部指标后取 median，滤除单拍快照的 icache 冷热 / 核启动错峰干扰；IFA 采用 250 次发射（50 热身 + 200 稳态），逐 block 回读打点（A5 56 AIV 核 / A3 48 AIV 核）。

## 测试输入与参考数据

测试输入 shape（scan_replay.py 定义，A5/A3 双平台同一 shape）：

- add：fp16 一维 **7,339,920** 元素（336×21845；A5 56 核 / A3 48 核均分整 tile，无尾块分支）
- grouped_matmul：bf16 **E=1, M=448, K=16, N=16**（x [448,16] × weight [16,16]；M=448 切满 A5 28 Cube 核 / A3 24 Cube 核，聚焦头开销取小 K/N）

### A5（Ascend950PR / dav-3510, CANN 9.1.0T103, 满核 56 AIV + 28 AIC, SYS_CNT=1000MHz）

数据版本 20260909 刷新：以 5 轮独立复测平均为准（每轮全量重建 + capture + 5-shot median，各轮 min/mean/max 分别跨轮取平均）。add/grouped_matmul 与 PR 描述 §3.2（B 环境独立复测，20260904，NPU 15，提交 c26563855）跨机器一致：add 头开销 0.33/0.79/1.04 精确相同，gmm 头开销差 ≤0.04us，TaskDur 差 ≤0.06us。

主表（B 环境复测，20260909，NPU 0，CANN 9.1.0T103，提交 6e6e2693f）：

| 指标 (min/mean/max us) | add (Vector, 56核) | grouped_matmul (Cube, 28核) | incre_flash_attention (MIX, AIV 56核打点) |
|------|-----------|-----------|-----------|
| 调度+硬件头 T0-stars | 0.15 / 0.16 / 0.17 | 0.28 / 0.79 / 1.19 | 0.41 / 2.33 / 5.37 |
| 软件头 T1-T0 | 0.18 / 0.63 / 0.88 | 0.58 / 0.99 / 1.57 | 0.45 / 1.57 / 4.94 |
| **头开销 T1-stars** | 0.33 / 0.79 / **1.04** | 1.08 / 1.77 / **2.10** | 1.37 / 3.90 / **6.52** |
| kernel主体 T2-T1 | 6.91 / 6.99 | 0.57 / 1.38 | 0.24（AIV 视角） |
| kernel末指令 T2-stars | 7.16 / 7.70 / 8.03 | 1.43 / 2.35 / 3.48 | 1.66 / 4.14 / 6.70 |
| 硬件尾 pmu_end-T2 | 0.52 ~ 0.86 | 0.49 ~ 0.63 | 0.43 ~ 0.62 |
| TaskDur 稳态 mean | 8.85 | 4.09 | 5.79 |

软件栈复验（torch 2.12.0+cpu + torch_npu 2.12.0.dev，CANN 9.1.0T103，同提交 6e6e2693f，5 轮）：grouped_matmul 与主表一致（头开销 1.08/1.79/2.11，max 精确相同，TaskDur 4.11）；IFA 头开销 1.35/3.70/5.73 略低（TaskDur 5.42）；add 轮间波动加大（软件头 max 0.87~1.54us、TaskDur 8.83~10.26us），头开销均值升至 0.90/1.22，但其最优轮（0.32/0.79/1.04）与主表精确一致。add 的 T0-stars（调度+硬件头）在全部轮次恒为 0.15/0.16/0.17，是最稳定的参照指标。

> 旧版参考表（头开销 add 0.44/0.51、gmm 1.47/1.68、IFA 3.00/4.98 mean/max）经多轮独立复测均无法复现——差距集中在其偏低的软件头（add 0.28/0.36 vs 实测 0.63/0.88）——已废弃，以本表为准。

注：IFA 的 T2 只打 AIV 侧，AIC Cube 计算拖尾计入 TaskDur 而非 T2，解读 IFA 数据时注意是 AIV 视角。

### A3（Ascend910_9392, CANN 9.1.0T103, SYS_CNT=50MHz，5 轮独立复测跨轮平均；**return 形态历史数据，IFA 主体未执行**）

| 指标 (mean/max us) | add (Vector, 48核) | grouped_matmul (Cube, 24核) | incre_flash_attention (Vector, 48核) |
|------|-----------|-----------|-----------|
| 调度+硬件头 T0-stars | 0.49 / 0.74 | 1.60 / 2.06 | 2.53 / 3.91 |
| 软件头 T1-T0 | 0.69 / 1.35 | 1.63 / 2.36 | 7.08 / 9.42 |
| **头开销 T1-stars** | **1.19 / 1.91** | **3.24 / 4.02** | **9.61 / 12.13** |
| kernel末指令 T2-stars | 8.62 / 8.97 | 4.40 / 6.22 | 9.63 / 12.15 |
| TaskDur 稳态 mean | 10.22 | 6.99 | 9.83 |

注：
- A3 上 ffts f[19] 非 kick 时刻（实测恒为 dispatch+20cyc），无 kick 拆分，硬件尾/全链路不输出分布口径（仅单点 max），故表中无对应行。
- A3 的 IFA 为纯 Vector 48 核形态（A5 为 MIX 28AIC+56AIV，AIC Cube 拖尾计入 TaskDur 而非 T2），两平台 IFA 口径不同，跨平台对比时注意。

## 参考数据（打点口径，Ascend910_93/A3, CANN 9.1.T103, SYS_CNT=50MHz）

**当前形态**（IFA_HEAD_TIMING_MODE 宏系统已删除，kernel 全量执行 + 独立 cycleBuf 回读）。上文 A3 小节的 5 轮复测为历史 return 形态（IFA 主体未执行，T2≈T1+ε、TaskDur 仅含头开销），两套数据不可比；IFA TaskDur 9.83→96.84us 即主体从未执行到真实执行的差异。

多拍 median 口径（add/gmm 各 5 拍，IFA 1 拍，稳态=跳过前 2 次；min=最快核 / max=最慢核）：

| 指标 (mean/max us) | add (Vector, 48核) | grouped_matmul (Cube, 24核) | incre_flash_attention (MIX 1:2, 24+48核) |
|------|-----------|-----------|-----------|
| 调度+硬件头 T0-stars | 0.49 / 0.72 | 1.67 / 2.02 | 13.32 / 16.28 |
| 软件头 T1-T0 | 0.69 / 1.36 | 1.61 / 2.24 | 4.33 / 8.32 |
| **头开销 T1-stars** | **1.18 / 2.10** | **3.31 / 4.08** | **17.65 / 19.62** |
| kernel主体 T2-T1 | 7.87 / 7.74 | 2.80 / 4.02 | 76.49 / 78.48 |
| kernel末指令 T2-stars | 8.55 / 9.10 | 4.47 / 6.26 | 94.14 / 98.02 |
| 硬件尾 pmu_end-T2 | 0.80 | 0.60 | 0.66 |
| TaskDur 稳态 mean | 10.16 | 7.13 | 96.84 |

### A3 IFA T0-stars 偏大的三层分解（16us）

1. **每核 baseline ~6us（软件，反汇编证实）**：T0（kernel 第一条 `MOV Xn, SYS_CNT`，AIV 函数偏移 0xC0/第 48 条指令）之前是 33 个 GM_ADDR 指针 + **按值传递的大 tiling struct** 的参数搬运（13 条标量 GM 读 + 8 条 ST 写栈）。标量 GM 读 ~100-200 core cycle/条 → ~6us。对比 add 仅 5-6 个参数（0.72us）、gmm（2.02us）。
2. **逐核 stagger 10.08us（硬件 kick 串行化）**：MIX kernel 24 组 × [1 AIC + 2 AIV] = 72 task，FFTS 逐组 kick ~0.15-0.2us/任务（实测 T0-stars 对 block_id 斜率 155.7ns/blk，B0=6.06 → B47=16.14）。前几组间隔 ~1us、后段压缩至 ~0.1us（kick 管道渐进加速特征）。
3. **调度基线 ~2-4us**：STARS dispatch 为完成驱动（shot 内 launch 间距 ≈ kernel 时长 102us）。

### return 版本（MODE=2，已删）测的头开销偏小的原因

头开销不是常数，依赖**前驱 kernel 的完成模式**：

- **长 kernel（~100us，真实形态）**：所有块几乎同时完成 → 资源一次性集中释放 → 72 task burst kick 串行化完全暴露 → T1-stars 13.6~19.6us
- **短 kernel（return 版 ~10us）**：块完成时间分散（启动 stagger + 短执行）→ FFTS **流水式 kick**（某组资源一释放即 kick 下一 kernel 对应组）→ kick 延迟被前 kernel 尾部重叠隐藏 → T1-stars 仅 5.5~11.5us

即 return 版本测到的是"短 kernel 自我复制"的乐观值，kick 开销被流水化隐藏 ~6-8us；真实长 kernel 接续场景以非 return 版本数据为准。

### A3 死锁教训（排查 3 天，两个独立根因）

1. **live pipeline events + early return**：Process 入口的 4 条 SetFlag（V_MTE2/MTE3_V 初始同步）若在提前 return 的路径上执行，kernel 携带 live pipeline events 退出，下一次 graph replay 必死锁（runtime 报 "three minutes timeout"，device 0 整体 hang 连普通 matmul 都超时，只能重启容器恢复）。源码原有注释 "Early return cannot leave pipeline events live across consecutive graph launches" 即此意。
2. **timing 缓冲越界写**：timing 基址曾用 `attentionOut+4MB`，而 attentionOut 输出仅 ~384KB —— 越界写破坏内存导致死锁/NaN。修复：host 侧持久分配独立 `g_cycle_tensor`（64 blocks × 8 slots int64，对齐 add 的 g_cycle_tensor 方案），kernel 经 `cycleBuf` 参数写入，新增 `torch.ops.ascend_ops.npu_fused_ifa_cycle_tensor()` 回读（无 tensor 入参的 op 需注册 CompositeExplicitAutograd）。
3. **构建宏陷阱**：IFA 的 CMakeLists 曾以 `-DIFA_HEAD_TIMING_MODE=2`（缓存默认值）传给编译器，**优先级高于源文件 `#ifndef` 默认值** —— 源码改默认 MODE=0 不生效，所有"MODE=0"轮次实际编译为 MODE=2（主体从未执行，T2 恒等于 T1+ε，输出 NaN）。该宏系统已整体删除。

## 打点前后的反汇编阶段分析

### 操作步骤

```bash
# 1. 提取 device 二进制 (host .o 的 .aicore_binary section, 本身为 elf64-hiipu ELF)
objcopy --dump-section .aicore_binary=add_kernel.bin \
  build/csrc/add/ascend950/CMakeFiles/add_obj.dir/add.cpp.o   # A3 把 ascend950 换成 ascend910_93

# 2. 反汇编 (需支持 hiipu64 的 llvm-objdump, A5 反汇编工具包)
llvm-objdump -d add_kernel.bin > add_full_aicore.txt

# 3. 定位打点: 搜索 SYS_CNT 读取 (T0/T1/T2 即 MOV Xn, SYS_CNT)
grep -n 'SYS_CNT' add_full_aicore.txt
```

A3 (dav-2201) 反汇编需用 ccec_compiler 包的 llvm-objdump 并指定 mcpu，否则 Ascend 专用指令显示为 `<unknown>`：

```bash
# llvm-objdump 来自 CANN-compiler 包 (如 CANN-compiler-7.12.*-linux.aarch64.run,
# bash xxx.run --full --install-path=<prefix> 安装后位于 <prefix>/<ver>/compiler/ccec_compiler/bin/)
llvm-objdump --mcpu=dav-c220-vec -d add_kernel.bin > add_full_aicore.txt
```

Mapping Symbols：`$x`=MainScalar / `$k`=SIMD / `$t`=SIMT；MIX 混合核有 `_mix_aic` 与 `_mix_aiv` 两套函数。注意正常发布二进制是 strip 过的（mapping symbols 丢失会反汇编失败），调试需用未 strip 的构建产物。工程内已归档三算子反汇编产物（A5 `*_ascend950_*` / A3 `*_ascend910_93_*`）：`disasm/` 目录。

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
| T0-stars 调度+硬件头 | 0.16 | 0.79 | 2.33 |
| entry 指令数 | 30 | 82 | AIC 37 / AIV 55 |
| 参数预取量 | 48B (×1) | 2KB (×33) | 1.25KB (×20, 每核) |

> **kick 拆分仅 A5 有效**（A5 上 kick=ffts pmu_start，可拆为 stars→kick 调度 ~0.14-0.19us + kick→T0 硬件头：add 0.03 / gmm 0.65 / IFA ~2.1us；20260909 B 环境 5 轮实测 stars→kick：add/gmm 0.135、IFA 0.188，轮间恒定）。A3 的 ffts f[19] 非 kick 时刻（实测恒为 dispatch+20cyc），无拆分意义——故上表只列合并口径 T0-stars（两平台通用），A3 汇总输出也不拆 kick 段，`seg1_pairs_*.csv` 仅 A5 有效。

- **add/gmm**：指令级估算与实测吻合。gmm 慢 0.63us 根因 = 参数区体积（48B vs 2KB）：预取数据量 + 装载指令数双重放大
- **IFA**：指令数与参数量均小于 gmm，但 kick→T0（~2.1us）是其 3.3 倍，~1.5us 缺口归因：**230KB 大二进制 ICache 冷启动**（8 组模板实例化只用 1 组）+ **MIX 1:2 派发错峰**（max/mean≈2.3 倍佐证）+ 84 核重复预取同一参数区的总线竞争

> A3 的 entry/epilogue 段可用归档产物 `disasm/*_ascend910_93_disasm.txt` 同法分析：entry 段要素与 A5 一致（PARA_BASE 装载 → 预取/栈帧 → 参数装载 → T0 打点），但 A3 的 add/gmm 二进制中**无 `DC_PRELOADI` 参数预取**（A5 上 add ×1 / gmm ×33），IFA 保留（60 处），跨平台对账时注意该差异。

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

## 编译适配（跨平台修复）

- grouped_matmul_utils.h 宏守卫添加 `__NPU_ARCH__ == 3510`（GetSplitValueFromGroupList 在 dav-3510 下可用）
- ascend950/ascend910_93 的 groupedmatmul_kernel 增加 `__cube__` 核属性（A5/A3 Cube 核函数声明）
- CMakeLists.txt ASC/AICPU 路径自动探测（兼容 CANN 8.5.T107 的 ASCConfig.cmake 与新版 asc-config.cmake）
- add.cpp (ascend910_93)：HardEvent/SetFlag/WaitFlag 补 `AscendC::` 前缀，AllocEventID 返回的 TEventID 显式 `static_cast<event_t>`（新版 CANN ccec 不再注入全局命名空间）
- grouped_matmul_kernel.h (ascend910_93)：GMM_CUBE_IMP 宏内 "Hard Event" typo 修正为 HardEvent
- 上述修复已在 A3（Ascend910_9392）实测编译通过并完成 5 轮复测

## 目录结构

```
examples/fast_kernel_launch_example/
├── csrc/                          # 算子源码 (ascend950=A5 / ascend910_93=A3 / ascend910b=A2)
│   ├── add/ascend950/             # add 算子 (dav-3510); ascend910_93/ascend910b 下为对应平台源码
│   ├── grouped_matmul/ascend950/  # grouped_matmul 算子 (dav-3510); 同上多平台
│   ├── incre_flash_attention/ascend950/       # IFA 算子 (A5: MIX 1:2; A3: 纯 Vector)
│   └── incre_flash_attention_meta/ascend950/  # IFA metadata (AICPU)
├── tests/                         # 测试脚本
│   ├── add/                       # test_add / cycle_test / compile_kernel / test_tail_return（for 体内 return 死代码与 tail 块开销的历史探索结论）
│   ├── grouped_matmul/test_groupedmatmul.py
│   └── incre_flash_attention/     # test.py + run_head_overhead.sh + analyze_head_overhead.py
├── disasm/                        # 三算子 device 反汇编产物 (A5/A3 两平台, 含 add 关 DCCI 对照版)
├── scan_replay.py                 # NPUGraph capture 101 次连发 + 多拍 replay + 逐 block 打点 dump
├── parse_stars_log.py             # STARS dispatch 事件解析
├── parse_ffts_pmu.py              # ffts_profile PMU 解析（核侧 pmu_start/pmu_end 时戳）
├── run_head_overhead.sh           # 一键测试脚本 (构建+三算子采集+汇总, 平台自动检测)
├── HEAD_OVERHEAD_GUIDE.md         # 本文档
├── CMakeLists.txt
├── setup.py
└── build_and_test.sh              # 原始构建+测试脚本
```
