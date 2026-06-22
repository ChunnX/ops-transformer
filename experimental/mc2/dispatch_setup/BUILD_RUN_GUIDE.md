# dispatch_min 编译运行指南

`dispatch_setup`（CCU 写状态位）+ `dispatch_teardown`（AIV aclnn 等状态位）· 单 die · 两条流 · 写/等状态位串行同步。

> 目标平台 **a6（Ascend 950PR）**；本机为 a5，仅用于编译验证。
> a5 已实测：teardown 编译打包通过（`EXIT=0`）、driver 骨架链接通过（`build=0`）。

---

## 1. 它做什么（最小机制）

```
dispatch_setup(CCU)                       dispatch_teardown(AIV/aclnn)
 streamSetup 上 LocalCopyNb  ──写 1──▶  statusGm[0]   ──忙等==1──▶  写 done[0]=1 收尾
```

- `dispatch_setup`（CCU）：在 `streamSetup` 上用 `LocalCopyNb` 把 `1` 写进本卡共享状态区 `statusGm`（int32, slot0）。
- `dispatch_teardown`（AIV/aclnn）：在 `streamTeardown` 上忙等 `statusGm[0] == 1`，等到后写 `done[0] = 1` 收尾。
- 两条流**无显式 event 依赖**——teardown 的忙等轮询本身就是跨流同步。
- 状态区契约（字节级）：`int32` 元素、slot0、就绪值 `1`、按 32B 对齐分配（AIV 一次 `DataCopy` 搬 32B，避免越界）。`LocalCopyNb` 是裸字节 DMA，状态字保持 4 字节，不受 CCU Variable 8 字节规则约束。

---

## 2. 工程结构

```text
experimental/mc2/
├── dispatch_setup/                     # CCU 算子 + 集成驱动（standalone cmake）
│   ├── dispatch_setup_ccu.h / .cpp     # CcuKernel 子类：段1建资源→段2 Load→段10 LocalCopyNb 写 1
│   │                                   #   （a6-only 部分 #ifdef DISPATCH_CCU_A6）
│   ├── dispatch_min_driver.cpp         # 拉起两算子 + 两条流 + 状态位同步 + 校验
│   ├── CMakeLists.txt                  # 建 driver exe；find_path/find_library 兜底；DISPATCH_CCU_A6 开关
│   ├── run.sh                          # 一键构建（TARGET=a5|a6）
│   ├── rank_table.json.tmpl            # 单 die HCCL ranktable 模板
│   ├── BUILD_RUN_GUIDE.md              # 本文档（编译运行指南，唯一文档，md-only）
└── dispatch_teardown/                  # AIV aclnn 注册算子（BUILD_OPEN_PROJECT 注册框架）
    ├── op_host/dispatch_teardown_def.cpp + CMakeLists.txt
    ├── op_host/op_tiling/dispatch_teardown_tiling.h / .cpp
    ├── op_kernel/dispatch_teardown.cpp     # AIV 忙等 status==1 → 写 done=1
    └── op_api/aclnn_dispatch_teardown.h / .cpp
```

---

## 3. 环境准备（每次会话）

```bash
ssh ops950                                      # = ssh root@123.60.114.41 -p 20028
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mc2_py310
source /home/0617/env/set_env.sh                # a6 换成 a6 的 CANN set_env.sh
```

---

## 4. 一键构建（推荐）

`run.sh` 用一个 `TARGET` 开关切换两种模式，自动定位散装 opapi 库、处理 CCU 围栏与装包：

```bash
cd /home/developer/ops-transformer/experimental/mc2/dispatch_setup

TARGET=a5 ./run.sh     # a5 骨架验证：CCU 关、不装包、直链散装 libcust_opapi.so（已实测 build OK）
TARGET=a6 ./run.sh     # a6 全功能：-DDISPATCH_CCU_A6=ON + 安装算子包（运行所需）
```

可用 env 覆盖：`SOC_VERSION` / `BUILD_SOC` / `ASCEND_HOME_PATH` / `OPS_TRANSFORMER_ROOT` / `CONDA_ENV` / `JOBS`。

---

## 5. 手动分步（等价于 run.sh，便于排错）

### 5.1 编译 teardown（AIV aclnn 注册算子）  ✅ a5 实测 EXIT=0

```bash
cd /home/developer/ops-transformer
bash build.sh --pkg --experimental --soc=ascend950 --ops=dispatch_teardown -j16
```

- 产物：`build_out/cann-ops-transformer-custom_linux-x86_64.run`（算子包）、`build/libcust_opapi.so`（散装库，导出 `aclnnDispatchTeardown`）。
- **注意**：`experimental/mc2` 下的算子**必须**带 `--experimental`，否则不会被扫描编译。
- **a6**：把 `--soc=ascend950` 换成 a6 白名单 `SUPPORT_COMPUTE_UNIT_SHORT` 里的 token（如 `ascend_910_96`）。`dispatch_teardown_def.cpp` 已含同源双 `AddConfig`（`ascend950` + a6 token）。**`--soc` / `AddConfig` / 白名单 token 三者必须一字不差且相同**（详见 §9.1）。

要**运行**还需装包（让 aclnn 运行时能加载并拉起 kernel）：

```bash
./build_out/cann-ops-transformer-custom_linux-x86_64.run --force   # 共享机谨慎：会写 CANN opp vendor
```

### 5.2 编译 driver（CCU 算子 + 集成驱动）

**a5 骨架验证（CCU 围栏关）** ✅ 实测 build=0、产出可执行：

```bash
cd /home/developer/ops-transformer/experimental/mc2/dispatch_setup
rm -rf build && mkdir build
cmake -B build \
  -DASCEND_CANN_PACKAGE_PATH=/home/0617/env/ascend-toolkit/latest \
  -DSOC_VERSION=ascend950pr_9599 \
  -DLIB_cust_opapi=/home/developer/ops-transformer/build/libcust_opapi.so
cmake --build build -j8
# 产物 build/dispatch_min_driver（CCU 关，仅编译/链接验证，不可有意义运行）
```

**a6 全功能（开 CCU 路径）** ⏳ 待 a6：

```bash
cmake -B build \
  -DASCEND_CANN_PACKAGE_PATH=<a6 CANN 路径> \
  -DSOC_VERSION=<a6 SoC 串> \
  -DDISPATCH_CCU_A6=ON          # ← 编入 CCU kernel + 启用拉起 + 段10 LocalCopyNb 真实写
cmake --build build -j8
# 装了算子包后 find_library 通常能自动找到 cust_opapi，可省 -DLIB_cust_opapi
```

---

## 6. 运行  ⏳ 仅 a6 且 TARGET=a6 有意义

> **a5 别跑**：CCU 围栏关 → 没人往状态区写 1 → teardown 的 AIV 会**死等**（§24）。运行是 a6 的事。

```bash
# (1) §8 清场（跑 NPU 前必做）
npu-smi info; pkill -9 python; sleep 3; npu-smi info

# (2) 单 die rank table：填本卡 NPU device_ip
cp rank_table.json.tmpl rank_table.json
hccn_tool -i 0 -ip -g                 # 取 device_ip，填进 rank_table.json 的 FILL_NPU_DEVICE_IP

# (3) 运行库路径（CANN lib64 + 散装 opapi 目录；装了包则只需 CANN lib64）
export LD_LIBRARY_PATH=<CANN>/x86_64-linux/lib64:/home/developer/ops-transformer/build:$LD_LIBRARY_PATH

# (4) 跑
./build/dispatch_min_driver ./rank_table.json
# 期望: [driver] PASS: teardown observed status==1 (done=1)
```

---

## 7. `DISPATCH_CCU_A6` 开关

唯一总开关，同时控制 CCU kernel 段10 的 `LocalCopyNb` 真实写 与 驱动里的 CCU 拉起段。

| 状态 | cmake | 效果 |
|---|---|---|
| **OFF**（默认） | 不传 | 不编 `dispatch_setup_ccu.cpp`、驱动 CCU 拉起桩掉 → a5 可编可链骨架 |
| **ON** | `-DDISPATCH_CCU_A6=ON` | 编入 CCU kernel + 启用拉起 + 段10 真实写 → 需 a6 libhcomm（含 CCU 符号） |

---

## 8. a5 已验证状态（机器证据，非自评）

| 环节 | 命令 | 状态 |
|---|---|---|
| teardown 编译打包 | `build.sh --pkg --experimental --soc=ascend950 --ops=dispatch_teardown` | ✅ EXIT=0（AIV kernel .o / proto / aclnnInner / .run） |
| driver + CCU 两源文件编译 | — | ✅ .o 均产出 |
| driver（CCU 关）链接 | a5 cmake + 散装 `cust_opapi` | ✅ build=0，产出 `dispatch_min_driver` |
| CCU 侧（kernel + 拉起）链接 | `-DDISPATCH_CCU_A6=ON` | ❌ a6-only（见 §10） |
| 上板运行 | — | ❌ 需 a6 硬件 |

---

## 9. a6 适配清单（两套机制分开看；代码里 `[Tx]` 标注）

> **关键**：`dispatch_teardown`（注册式 aclnn 算子）和 `dispatch_setup`（CCU + 驱动）是**两套不同机制**，改芯片的方式不一样，别互相套用。

### 9.1 teardown 算子 —— 靠 `--soc` + `AddConfig`（编译期）

- **`--soc=<a6 token>`** 与 `dispatch_teardown_def.cpp` 的 `AICore().AddConfig("<token>", ...)`，必须和 build.sh 白名单 `SUPPORT_COMPUTE_UNIT_SHORT` 里的 token **一字不差、三者相同**（如 `ascend_910_96` 还是 `ascend910_96`，以白名单为准）。
- 已改为**同源双配置**（`ascend950` + a6 token），a5/a6 都能编、不必每次手改。
- 其余（cmake 透传 `ASCEND_COMPUTE_UNIT`、kernel 架构、产物目录、build.sh 特判）由 build.sh/工具链**自动**，不碰。

### 9.2 CCU setup 侧 —— **不用 `--soc`/`AddConfig`**，靠 `DISPATCH_CCU_A6` + 运行期值

> grep 实测：CCU 头零 arch `#ifdef`、`HcclCommConfig` 无 soc 字段、comm-init API 无 soc 参、`DevType` 把 a5/a6 都归 `DEV_TYPE_950=6`。所以芯片差异**不在编译开关**里。

**编译期（影响能否编过 / 链上）：**

| 标记 | 项 | a6 要做 |
|---|---|---|
| — | 开关 | 加 `-DDISPATCH_CCU_A6=ON` + a6 CANN/hcomm 路径 |
| **T1** | `LocalCopyNb` 6 参 + `CreateBlockCcuBuf` | 对真 a6 头核签名；差了改 `dispatch_setup_ccu.cpp`（已 `#ifdef` 围住） |
| **T2** | `WaitEvent` 1 参 + `CompletedEvent::SetMask` | 同上核签名 |
| — | CCU 符号导出 | a6 libhcomm 须导出 `HcclCcuKernel*`/`CcuKernel`（a5 没导出才链不了），`nm` 核一下 |

**运行期（能编 ≠ 能跑对）：**

| 标记 | 项 | 说明 |
|---|---|---|
| **T8** | a6 `SOC_VERSION` 运行期串 | **a5/a6 只靠这个串区分**（`DevType` 都是 `DEV_TYPE_950`、分不了）；并确认 CANN 路径 |
| **T7** | `rank_table.json` a6 device_ip/id | 模板已给，填真值；单 die CCU 实例可建性（a5 不行）上板验 |
| **T10** | CCU 使能模式 | 真启 CCU 可能需 `config.hcclOpExpansionMode`=CCU_MS(4)/CCU_SCHED(5) 或 `HcclCommQueryCcuIns` 校验实例；低层 thread-acquire 路径是否需要待 a6 核（参考未显式设、**可能不需要**） |
| **T3** | 状态字跨流 cache 可见性 | AIV 读 GM 是否需 dcci/非 cache 才能看到 CCU 的写（上板才暴露） |
| **T9** | teardown 装包 opapi 库名 | 驱动链接用；a5 已用散装 `build/libcust_opapi.so` 解决 |

**明确不用动**（grep 坐实）：`SOC_VERSION` 宏 / arch `#ifdef` / `CATLASS_ARCH`（CCU 头里没有，别加）；`DevType` 代码分支（分不了 a5/a6，别靠它判芯片）；`CreateBlockCcuBuf` 在 `#ifdef` 外（无害，a5 头也有、已 a5 编过证实）。

### 9.3 a6 上 CCU setup 实操步骤（编过 + 链上；≠ 跑通）

> CCU setup 随【驱动 cmake】编（不是 build.sh），驱动要链 teardown 的 `libcust_opapi`，所以**先 build teardown 算子，再 build 驱动**。
> 行号会随文件变动漂移，下面一律按**调用名 grep 定位**，不写死行号。

**Step 1 — 核 3 个 a6-only CCU API 签名（唯一可能改码处，可能 0 改）**

`dispatch_setup_ccu.cpp` 段1/段10 现按 a6 形态写、用 `#ifdef DISPATCH_CCU_A6` 围住，4 个调用是：

- `CreateBlockCcuBuf(1U, &localCopyBuf_)`
- `event_.SetMask(1U)`
- `LocalCopyNb(readyDst_, readySrc_, lenBytesVar, &localCopyBuf_, 1U, event_)`（6 参）
- `WaitEvent(event_)`（1 参）

在 a6 上核真头：

```bash
find <a6 CANN> -name ccu_kernel.h
grep -n "LocalCopyNb\|WaitEvent\|CreateBlockCcuBuf" <上面的 ccu_kernel.h>
grep -rn "SetMask" <a6 .../hcomm/ccu/>      # CompletedEvent::SetMask，多半在 ccu_datatype_v1.h
```

一致 → 一行不改；不一致（参数个数/顺序/类型变了）→ 在 `dispatch_setup_ccu.cpp` 里 `grep` 出那几个调用、只改它们，其它别动。

**Step 2 — 开开关 + 给 a6 路径（build 命令，不改码）**

```bash
# (1) 先 build teardown（产 libcust_opapi 供驱动链；--soc 用白名单 token）
cd <a6>/ops-transformer
bash build.sh --pkg --experimental --soc=<a6 白名单 token> --ops=dispatch_teardown -j16
# (2) 再 build 驱动（CCU kernel 随它编入）
cd experimental/mc2/dispatch_setup
rm -rf build && mkdir build
cmake -B build \
  -DASCEND_CANN_PACKAGE_PATH=<a6 CANN 路径> \
  -DDISPATCH_CCU_A6=ON \
  -DLIB_cust_opapi=<a6>/ops-transformer/build/libcust_opapi.so
cmake --build build -j8
```

- `-DDISPATCH_CCU_A6=ON` 就是把 CCU kernel 编入 + 解开 `#ifdef` 的开关。
- `-DSOC_VERSION` 可不传（对驱动只是装饰性宏 `[T8]`；CCU 选芯片靠 `DISPATCH_CCU_A6` + a6 libhcomm）。
- cmake 若报找不到 a6 hcomm 头（a6 目录布局可能与 a5 不同）→ 手动指 `-DHCOMM_PKG_INC=... -DHCOMM_INT_INC=... -DCANN_PUB_INC=...`（报错会提示哪个没定位到）。

**Step 3 — nm 核 CCU 符号导出（验证，不改码）**

```bash
find <a6 CANN> -name libhcomm.so
nm -D <上面的 libhcomm.so> | grep -iE "HcclCcuKernelLaunch|HcclCcuKernelRegister|HcclThreadAcquireWithStream"
nm -D <上面的 libhcomm.so> | grep -i "CcuKernel"
```

非空 → 能链；空 → 该 a6 libhcomm 没编出 CCU 符号（换对的库 / 确认 CCU 特性已开），**不是改代码能解决**。

> 这 3 步只到「编过 + 链上」。要真跑还需 §9.2 运行期表（`[T8]` soc 串 / `[T7]` rank_table / `[T10]` CCU 使能 / `[T3]` cache）。

---

## 10. 排错 / a5↔a6 边界

### 编译期揪出并已修的 4 类（对抗核验看不见、只有编译器/链接器能逮）

| 现象 | 根因 | 修法 |
|---|---|---|
| `ACLNN_API / ACLNN_SUCCESS does not name a type / not declared` | 本 CANN 版本公共 aclnn 头未导出这俩宏（只在注册内部 `aclnn_util.h`），对外头不自洽 | 对外头/驱动 `#ifndef` 兜底定义 |
| `hccl_types.h: No such file or directory` | `ccu_kernel.h` 内部 plain `#include "hccl_types.h"`，缺 include 根 | driver include 加 `${HCOMM_INT_INC}/hccl` |
| `undefined reference to aclCreateTensor` | 该符号在 `libnnopbase`；CANN 公共头根实为 `x86_64-linux/include` | 链 `libnnopbase`；`find_path` 多后缀兜底 |
| `undefined reference to HcclCcuKernel* / hcomm::CcuKernel::CcuKernel` | **a5 libhcomm 未导出 CCU 拉起符号（nm 实测为空）= a6-only** | CCU 侧整体 `#ifdef DISPATCH_CCU_A6` 围栏；a6 上开 |

> **本质边界**：CCU kernel 继承 `hcomm::CcuKernel`、拉起走 `HcclCcuKernel*`——这些符号在 a5 libhcomm **不存在**，故 CCU 路径在 a5 **链接无解**（编译能过）。这是硬件/库代际边界，非代码缺陷。a6 上 `-DDISPATCH_CCU_A6=ON` + a6 libhcomm 即解。

### 运行期常见（参考项目 §8 / 卡死类）

- HCCL 建链失败（`ERR02005` / `store->get got 0 bytes`）→ 别人占卡，跑前 §8 清场。
- teardown 无返回/卡死 → 状态区没被写 1（CCU 没跑 / 状态地址不一致 / cache 未刷 `[T3]`）；PRINTF + DumpTensor 看 `statusGm` 实际值。
- aclnn 返回 `561002` → 算子包未装或环境不符（先 `.run --force` 装包）。

---

*本文档与 `run.sh` 的命令均来自 a5 实编验证（teardown EXIT=0 / driver build=0）。CCU 链接与上板运行待 a6。*
