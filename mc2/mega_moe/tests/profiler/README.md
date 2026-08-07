# MegaMoeLayered Device-Side Profiling（子包路径）

给 `op_kernel/arch35/mega_moe_layered.h`（URMA/MIX_AIC_1_2 CCU 通信算子）做 cycle 级打点。

> 本目录代码基于 `ops-kernel-profiler` skill 的**子包路径**（workspace profBuf + address-bridge）。
> 背景机制、坑位与排障详见 skill：
> `cannbot-skills/ops/ops-kernel-profiler/references/subpackage/subpackage-path.md`
> `cannbot-skills/ops/ops-kernel-profiler/references/subpackage/ccu-integration-lessons.md`

## 已完成的代码改动

| 文件 | 改动 |
|------|------|
| `op_kernel/arch35/profiler_core.h` | 从 skill 复制（不改）。GM ring buffer + DCCI flush 机制 |
| `op_kernel/arch35/profiler_config.h` | 静态配置。`PROF_N_CORES=84`（56 AIV + 28 AIC，Ascend 950 MIX_1_2）。**按你的芯片改** |
| `op_kernel/arch35/profiler.h` | 事件 ID 定义（KERNEL + 5 个主阶段对） |
| `op_kernel/arch35/mega_moe_tiling.h` | `MegaMoeTilingData` 尾部新增 `uint64_t profBufAddr[4]` |
| `op_kernel/arch35/mega_moe_layered.h` | 顶部打点开关；`PROF_LOCAL_DEFINE()`；`Init()` 里 `_prof_buf_gm = workspaceGM + profBufAddr[0]`；`Process()` 各阶段 `PROF_MARK`；结尾 `PROF_FINALIZE/PROF_FINALIZE_AIC` + address-bridge |
| `op_host/op_tiling/arch35/mega_moe_tiling.cpp` | 顶部 `MEGA_MOE_ENABLE_PROFILING` 开关；`SetWorkspace()` 在 workspace 尾部预留 `PROF_BUF_TOTAL` 并把偏移写入 `profBufAddr[0]` |
| `op_host/scripts/profiler/pair_map.json` | 事件配对 |
| `op_host/scripts/profiler/*.py` | 5 个解析/可视化脚本（从 skill 复制，不改） |
| `tests/profiler/test_aclnn_mega_moe_profiler.cpp` | 独立 aclnn C++ 测试程序（标定 + 打点两阶段） |
| `torch_extension/.../ops/csrc/mega_moe.cpp` | torch 扩展内接 dump：env 开关 `MEGA_MOE_PROF_DUMP` / `MEGA_MOE_PROF_CALIB` / `MEGA_MOE_PROF_DELTA` |

## 事件定义

| ID | 阶段 | 含义 |
|----|------|------|
| 0x0001 / 0x00FF | KERNEL_BEGIN/END | kernel 整体 |
| 0x0002 / 0x0003 | PREPARE | SendAndQuantBuffInit + SendMaskCal + ResetFlagList + ResetDispatchState + SharedExpertCopyInput |
| 0x0004 / 0x0005 | SHARED_GMM1 | 共享专家 GMM1 + SwiGLU（仅 sharedExpertNum_>0） |
| 0x0006 / 0x0007 | DISPATCH_GMM1 | 路由专家主循环：dispatch + SendCntCal + MetaInfoCalAndDispatch + GMM1 + SwigluQuant + ExpertTokenNumCopyOut |
| 0x0008 / 0x0009 | GMM2_COMBINE | GMM2 + Combine + 共享专家 GMM2 |
| 0x000A / 0x000B | UNPERMUTE | Unpermute |

## 前提

- CANN SDK 已 `source set_env.sh`；NPU 可访问（`npu-smi info` 正常）
- 芯片核数确认：编译运行
  `cannbot-skills/ops/ascendc-profiler/scripts/host_check_cores.cpp`
  用输出替换 `op_kernel/arch35/profiler_config.h` 里的 `PROF_N_CORES` / `PROF_AIV_CORES`。
  当前默认：Ascend 950 MIX_1_2 = `84` / `56`。

## 两种回读方式

打点在 kernel 内部，与 launch 方式无关（torch 扩展最终也走 `aclnnMegaMoe` → 同一个 kernel）。
区别只在 host 侧从哪读回 workspace 尾部的 profBuf：

| 方式 | 说明 | 适用 |
|------|------|------|
| **A. torch 扩展内接 dump（推荐）** | `NpuMegaMoe` 里 env 开关控制，手动接管 workspace 生命周期，同步后回读 profBuf。贴合真实推理流程（EP group/context 已就绪），需重新编译 torch 扩展 | 测真实 torch 调用 |
| **B. 独立 C++ 测试程序** | `tests/profiler/test_aclnn_mega_moe_profiler.cpp` 直调 aclnn，自己建 HCCL EP group + Mc2Context | 快速验证、无 torch 改动 |

## 工作流（7 步）

### Step 1-2：代码已就位（编译开关确认）

两个开关必须一致（默认均已开启）：

- `op_host/op_tiling/arch35/mega_moe_tiling.cpp` 顶部：`MEGA_MOE_ENABLE_PROFILING 1`
- `op_kernel/arch35/mega_moe_layered.h` 顶部：`MEGA_MOE_ENABLE_PROFILING 1`

### Step 3：编译算子包（标定版，bridge ON）

kernel 侧需同时定义 `ENABLE_PROFILING` 和 `PROF_ADDR_CALIB`。`build.sh --pkg` 通常无法传自定义宏，
因此用文件内开关：把 `mega_moe_layered.h` 顶部的 `MEGA_MOE_PROF_ADDR_CALIB` 改为 `1`，然后：

```bash
cd ops-transformer
source ~/env.sh
# [MODIFY] --ops / --soc 按你的构建
bash build.sh --pkg --ops=mega_moe --soc=ascend950 -j64
```

安装到自定义路径并加载环境：

```bash
ARCH_DIR=$(uname -m)
cd build_out
bash cann-ops-transformer-custom_linux-${ARCH_DIR}.run --quiet --install-path=$HOME/SUB
source $HOME/SUB/vendors/custom_transformer/bin/set_env.bash
# 清理旧 .o（可能缓存了无 profiler 的旧版本，<1MB 是旧的）
find $HOME/SUB/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend950/mega_moe_apt/ \
     -name "*.o" -size -1000k -delete
```

### Step 4：标定（二选一）

**方式 A（torch 内接）**：重新编译 torch 扩展后，跑一次真实推理，设 `MEGA_MOE_PROF_CALIB=1`：

```bash
MEGA_MOE_PROF_CALIB=1 python your_mega_moe_infer.py
# stdout 输出: [PROFILER] MegaMoe rank N calib: ... delta=XXXX (0xYY)
# 记录 delta
```

**方式 B（独立 C++）**：

```bash
cd mc2/mega_moe/tests/profiler
g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DENABLE_PROFILING -DPROF_ADDR_CALIB \
    test_aclnn_mega_moe_profiler.cpp -o prof_calib \
    -I $HOME/SUB/vendors/custom_transformer/op_impl/ai_core/tbe/include \
    -I $ASCEND_HOME_PATH/include -I $ASCEND_HOME_PATH/../.. \
    -L $ASCEND_HOME_PATH/lib64 \
    -L $HOME/SUB/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -m) \
    -lascendcl -lacl_op_compiler -lhcccl -lhcl -ltiling_api -lpthread
./prof_calib
```

输出 `[CALIB] rank N: ... delta=XXXX (0xYY)`。**记录每个 rank 的 delta**（同环境通常一致）。

### Step 5：重新编译算子包（打点版，bridge OFF）

把 `mega_moe_layered.h` 顶部的 `MEGA_MOE_PROF_ADDR_CALIB` 改回 `0`，重新 `build.sh --pkg` + 安装 + 清理旧 .o。

### Step 6：打点（二选一，与 Step 4 一致）

**方式 A（torch 内接）**：重新编译 torch 扩展，把 delta 填入 `MEGA_MOE_PROF_DELTA` 并设 `MEGA_MOE_PROF_DUMP=1` 跑推理：

```bash
MEGA_MOE_PROF_DELTA=16777216 MEGA_MOE_PROF_DUMP=1 python your_mega_moe_infer.py
# 生成 prof_rank{device}.bin
```

> ⚠️ 标定阶段的 delta 在正式打点前需要把算子包重新编译成 bridge OFF 版本（Step 5），
> 否则 address-bridge 仍会写 yOut[0..8] 干扰输出。

**方式 B（独立 C++）**：把标定 delta 填进 `test_aclnn_mega_moe_profiler.cpp` 的 `WS_ADDR_DELTA`，去掉 `-DPROF_ADDR_CALIB` 重新编译运行。

### Step 7：解析可视化

```bash
SCRIPTS=cannbot-skills/ops/ascendc-profiler/scripts
cd mc2/mega_moe
python3 $SCRIPTS/parse_prof.py \
  --event-names-from op_kernel/arch35/profiler.h \
  --pair-map op_host/scripts/profiler/pair_map.json \
  --aiv-cores 56 \
  prof_rank*.bin
python3 $SCRIPTS/to_chrome_trace.py prof_paired.csv   # → trace_rank*.json → MindStudio Insight
python3 $SCRIPTS/draw_gantt.py prof_paired.csv 0 gantt_rank0.svg "MegaMoeLayered" --freq 1650
```

## torch 扩展重新编译

方式 A 修改了 `torch_extension/cann_ops_transformer/ops/csrc/mega_moe.cpp`，需重新编译：

```bash
cd torch_extension
python setup.py build install   # 或 pip install .
```

> 打点只改 host 回读逻辑（env 开关），不改生产路径：`MEGA_MOE_PROF_DUMP`/`MEGA_MOE_PROF_CALIB` 未设置时
> 走原 `ACLNN_CMD` 分支，行为与未打点完全一致。

## 常见坑速查

| 症状 | 检查 → 修复 |
|------|------------|
| `prof_rank*.bin` 全零 | ① host 读错地址 → 确认 `MEGA_MOE_PROF_DELTA`/`WS_ADDR_DELTA` 已填标定值；② kernel 未写 → 确认 `MEGA_MOE_ENABLE_PROFILING=1` 已编译进 kernel（旧 .o 已清理） |
| delta 标定出来为 0 | 非 CCU 通信场景，workspace 无偏移 → 直接用 `hostBase + profBufOffset` 读（delta=0） |
| delta 标定出来巨大（>1GB 或回绕） | bridge 没写到输出：① 装的 kernel .o 是旧的/无 profiler（`kernelWsBase` 读到的是 token 数）→ 清理旧 .o 重编重装；② tiling 的 `profBufAddr[0]==0`（`kernelWsBase` 读到 0）→ tiling .so 未同步；③ 独立 C++ 标定读的是 `yDev` → 改读 `tokenNumDev`（kernel 现在写 `expertTokenNumsOut`） |
| dump 后事件乱、耗时/cycle 离谱 | host 回读偏移不能直接用 `workspaceSize - PROF_BUF_TOTAL`，必须用 kernel 相对偏移 `workspaceSize - PROF_BUF_TOTAL - delta - RESERVED(50MB)`（torch 扩展与独立测试程序均已按此修正） |
| AIC 核数据全 0 | 确认用了 `PROF_FINALIZE_AIC`（已在代码中） |
| aclrtSetDevice 507033 | 未 `source $HOME/SUB/vendors/custom_transformer/bin/set_env.bash` |
| 部分 rank 崩溃 507015 | `profBufAddr` 数组已按 rank 隔离；若仍复现检查 tiling 侧是否有 static counter |
| torch 方式没产出 .bin | 确认重新编译了 torch 扩展；确认 env 变量拼写（`MEGA_MOE_PROF_DUMP=1`） |
| workspace 不够 | `RESERVED_WORKSPACE_SIZE` 已含 50MB 余量；profBuf 追加在最后，通常不触发 |

## 精度/开销说明

- 打点只写 workspace 尾部独立区域，不触碰输出数据（正式打点阶段 bridge OFF，`yOut` 零干扰）
- `PROF_MARK` 直接 GM 写 + DCCI，每次 ~千级 cycle；当前每个核 ~11 个事件，开销可接受
- 若后续需要更细粒度（如 per-expert GMM1），按 `event-id-spec.md` 流程在 `Process()` 内加事件对
