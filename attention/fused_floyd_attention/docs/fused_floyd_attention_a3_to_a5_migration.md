# FusedFloydAttention 算子 A3 → A5 迁移分析

> 算子路径：`/mnt/workspace/gitCode/cann/ops-transformer/attention/fused_floyd_attention`
>
> 本文档仅归档**有证据支撑**的信息，每条均标注证据来源（文件路径:行号 或参考文档）。需验证项单独列出。

---

## 1. 架构映射

**证据**：`ops/npu-arch/references/npu-hardware-params.md` §0 产品映射表

| 项 | A3（源） | A5（目标） |
|---|---|---|
| 芯片型号 | Ascend910_93 | Ascend950PR / Ascend950DT |
| SocVersion | ASCEND910B（运行时映射） | ASCEND950 |
| NpuArch | DAV_2201 | DAV_3510 |
| `__NPU_ARCH__` | 2201 | 3510 |
| archXX 简写 | arch22 | arch35 |

> 注：`Ascend910_93` 的 SocVersion 字符串运行时映射到 `SocVersion::ASCEND910B`（非独立枚举值），NpuArch 同为 `DAV_2201`。A2（Ascend910B1~B4）与 A3 共用 `DAV_2201`。

---

## 2. Device 侧编译宏

**证据**：
- `ops/npu-arch/SKILL.md:16` — `__DAV_C310__` 等价于 `NpuArch::DAV_3510` / `arch35` / `__NPU_ARCH__=3510`
- 本算子 kernel：`op_kernel/fused_floyd_attention.cpp:22` 使用 `#ifdef __DAV_C220_CUBE__`
- CANN 仓库全局 grep 结果（见 §4）

| 架构 | Cube 侧编译宏 | Vector 侧编译宏 |
|---|---|---|
| A2/A3（DAV_2201） | `__DAV_C220_CUBE__` | `__DAV_C220_VEC__`（推测，未在本算子使用） |
| A5（DAV_3510） | `__DAV_C310_CUBE__` | `__DAV_C310_VEC__` |

---

## 3. 算子当前支持状态

### 3.1 README 标注 A5 不支持

**证据**：`README.md:5-12`

```
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
```

### 3.2 算子定义未注册 ascend950

**证据**：`op_host/fused_floyd_attention_def.cpp:87-88`

```cpp
this->AICore().AddConfig("ascend910b", aicore_config);
this->AICore().AddConfig("ascend910_93", aicore_config);
```

仅注册了 A2（ascend910b）和 A3（ascend910_93），**未注册 ascend950**。

### 3.3 CMake 已预置 A5 编译分支

**证据**：`op_host/CMakeLists.txt:18-25`

```cmake
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME FusedFloydAttention
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
        )
endif()
```

CMake 层已为 ascend950 配置计算单元 `Ascend950PR_9599` 并启用 `ENABLE_CV_COMM_VIA_SSBUF` 宏。

### 3.4 Kernel 仅含 A3 Cube 分支

**证据**：`op_kernel/fused_floyd_attention.cpp:22-113`

```cpp
#ifdef __DAV_C220_CUBE__ // CUBE 实现
    ...
#else // VECTOR 实现
    ...
#endif
```

仅判断 `__DAV_C220_CUBE__`，**未含 `__DAV_C310_CUBE__` 分支**。A5 编译时该 `#ifdef` 为假，会直接走 `#else` 的 VECTOR 实现。

---

## 4. `__DAV_C310_CUBE__` 宏存在性与使用模式

**证据**：CANN 仓库 `ops-transformer` 目录 grep 结果，共 16 处使用 `__DAV_C310_CUBE__`。

### 4.1 典型 A5 已适配算子

| 算子 | A3 路径 | A5 路径 |
|---|---|---|
| `flash_attention_score` | `#ifdef __DAV_C220_CUBE__`（`op_kernel/flash_attention_score.cpp:37`） | `#ifdef __DAV_C310_CUBE__`（`op_kernel/arch35/flash_attention_score_entry_regbase.h:23`） |
| `prompt_flash_attention` | arch22/ | `op_kernel/arch35/prompt_flash_attention_entry_regbase.h:66` |
| `incre_flash_attention` | arch22/ | `op_kernel/arch35/incre_flash_attention_entry_regbase.h:113` |
| `flash_attn` | — | `op_kernel/flash_attn.cpp:158,173` + `op_kernel/arch35/flash_attn_entry_regbase.h:103` |
| `sparse_flash_attention` | — | `op_kernel/sparse_flash_attention.cpp:27` |
| `kv_quant_sparse_flash_attention` | — | `op_kernel/kv_quant_sparse_flash_attention.cpp:27` |
| `fused_infer_attention_score` | — | `op_kernel/arch35/prompt_flash_attention_entry_regbase.h:52,106` |
| `mega_moe` | — | `op_kernel/arch35/mega_moe_workspace_info.h:19` |

### 4.2 典型 A5 适配模式

A5 已适配的算子普遍采用 **arch35 独立目录** + `__DAV_C310_CUBE__` 宏分支的模式（如 `flash_attention_score` 的 `arch35/flash_attention_score_entry_regbase.h`）。

**证据**：`attention/flash_attention_score/op_kernel/arch35/flash_attention_score_entry_regbase.h:23-38`

```cpp
#ifdef __DAV_C310_CUBE__ // CUBE 实现
#define INVOKE_FA_OP_IMPL_BASEAPI(templateClass, ...) ...
#else // VECTOR 实现
#define REGBASE_COPY_TILING_DATA(tiling) ...
#endif
```

### 4.3 多架构兼容写法

部分算子采用单分支兼容多架构的写法。

**证据**：`attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h:69`

```cpp
#if defined (__DAV_C220_CUBE__) || defined (__DAV_C310_CUBE__) || (defined __DAV_310R6_CUBE__)
```

**证据**：`experimental/attention/quant_sals_indexer/op_kernel/quant_sals_indexer.cpp:29`

```cpp
#if defined(ASCENDC_CPU_DEBUG) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
```

---

## 5. DAV_3510 相对 DAV_2201 的关键架构差异

### 5.1 Buffer 容量

**证据**：`ops/npu-arch/references/npu-hardware-params.md` §2.3（DAV_2201）、§2.5（DAV_3510）、§3 架构关键差异速查

| Buffer | DAV_2201（A2/A3） | DAV_3510（A5） |
|---|---|---|
| L1 | 512 KB | 512 KB |
| L0A | 64 KB | 64 KB |
| L0B | 64 KB | 64 KB |
| L0C | 128 KB | **256 KB** |
| UB | 192 KB | **248 KB** |
| BT | 1 KB | **4 KB** |
| SSBuffer | — | **256 KB**（新增） |

### 5.2 核心类型与核数

**证据**：`ops/npu-arch/references/npu-hardware-params.md` §2.3、§2.5、子型号变化参数

| 项 | DAV_2201 | DAV_3510 |
|---|---|---|
| 核心类型 | CubeCore, VectorCore | CubeCore, VectorCore |
| Cube:Vec 比例 | 1:2 | 1:2 |
| 典型 CubeCore 数 | 24（Ascend910B2） | 28（PCIE）/ 32（Server） |
| 典型 VectorCore 数 | 48 | 56（PCIE）/ 64（Server） |

### 5.3 4:2 稀疏

**证据**：`ops/npu-arch/references/npu-hardware-params.md` §2.3、§2.5、§3

| 架构 | 4:2 稀疏 |
|---|---|
| DAV_2201 | 支持（`sparsity=1`） |
| DAV_3510 | **不支持**（`sparsity=0`） |

> 注：本算子的 `SparseMode`/`SparseEnum` 指 attention mask 压缩模式（ALL/NONE/CAUSAL/BAND/PREFIX 等），见 `op_host/fused_floyd_attention_tiling_general.cpp:75-85`，**非 4:2 权重稀疏**，因此 A5 取消 4:2 不影响本算子功能。

### 5.4 数据通路改动

**证据**：`ops/npu-arch/references/npu-arch-guide.md` §DAV_3510 相比 DAV_2201 的关键数据通路改动（第 275-285 行）、§AI Core Buffer 层级（第 255-263 行）、§指令序列与 BufferID 同步（第 287-291 行）

DAV_3510 新增三条 CV 直通通路：
1. **L0C → UBuffer**：Cube 结果直达 Vector，FixPipe 输出到 UB（新增路径，区别于传统 L0C→GM）
2. **UBuffer ↔ L1**：Vector 与 Cube 双向直连（UB→L1 为新增方向）
3. **SSBuffer 消息通路**：CV 间细粒度同步信号

**删除的通路**：`L1→GM` 和 `GM→L0A/L0B` 的数据通路已被删除。

**保留的通路**：
- `L0C→GM`（经 FixPipe）— 标准 Cube 输出路径，A5 仍保留
  - 证据：`npu-arch-guide.md:262` BT/FP Buffer 仍用于 FixPipe 配置；`:289` A5 执行单元队列含 FixPipe；`:183` BT 从 1KB 扩到 4KB（FixPipe Bias 表扩容）
- `L0C→UB`（经 FixPipe）— A5 新增，让 Cube 结果直达 Vector

> 典型受益场景：FA / FIA 类融合算子可彻底消除 workspace A/B/C 的 GM 读写。

### 5.5 同步机制

**证据**：`ops/npu-arch/references/npu-arch-guide.md` §指令序列与 BufferID 同步（第 287-291 行）

DAV_3510 各执行单元拥有独立指令队列（Cube / FixPipe / MTE1 / MTE2 / MTE3 / SIMD VF / SIMT VF），引入 **BufferID 同步机制**，消除原 set/wait 强制配对需求。

### 5.6 Cube MMAD 新增数据类型

**证据**：`ops/npu-arch/references/npu-arch-guide.md` §Cube MMAD 支持的数据类型（第 337-358 行）

DAV_3510 新增 FP8 / MXFP8 / MXFP4 / HiF8 Cube MMAD 支持。本算子当前仅使用 FP16/BF16/FP32（见 `def.cpp:25`），不受影响。

---

## 6. 本算子与架构差异的关联点

### 6.1 Tiling 侧平台信息获取（已自适应）

**证据**：`op_host/fused_floyd_attention_tiling_general.cpp:397-420`

```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
aivNum = ascendcPlatform.GetCoreNumAiv();
aicNum = ascendcPlatform.GetCoreNumAic();
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
```

Tiling 侧通过 `PlatformAscendC` 接口运行时获取 UB/L1/L0C 容量与核数，**能自动适配 A5 的 256KB L0C / 248KB UB / 28~32 核**。

### 6.2 Kernel 侧 Buffer 分配（硬编码，未自适应）

**证据**：`op_kernel/fused_floyd_attention_s1s2_bn2gs1.h:485-509`

```cpp
uint64_t stage1Size = 8 * 1024;
uint64_t stage1AttenSize = 9 * 1024;
uint64_t stage1PongSize = 35 * 1024;
uint64_t stage2Size = 64 * 128;
uint64_t maskTBufPongSize = 16 * 1024;
```

UB Buffer 分配为硬编码值，未利用 A5 增大的 UB（248KB vs 192KB）。

### 6.3 Cube↔Vector 数据交互（走 GM workspace）

**证据**：`op_kernel/fused_floyd_attention_s1s2_bn2gs1.h:431-451`

```cpp
this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + ...));
this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + ...));
this->stage1Res[0].SetGlobalBuffer((__gm__ INPUT_T *)(workspace + ...));
this->mm2Res[0].SetGlobalBuffer((__gm__ T *)(workspace + ...));
this->vec2Res[0].SetGlobalBuffer((__gm__ T *)(workspace + ...));
```

Cube 与 Vector 间所有中间结果（`mm1Res`/`mm2Res`/`stage1Res`/`vec2Res`）均通过 GM workspace 中转，**未使用 A5 新增的 L0C→UB / UB→L1 直通通路**。

### 6.4 Matmul C 侧 TPosition

**证据**：`op_host/fused_floyd_attention_tiling_general.cpp:1172`（bmm1）、`1192`（bmm1k1）、`1213`（bmm2v2）、`1236`（bmm2）

```cpp
bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
```

bmm1/bmm1k1/bmm2/bmm2v2 的 C 侧（输出）均设为 `TPosition::GM`。

**证据**：`op_kernel/fused_floyd_attention_s1s2_bn2gs1.h:152`

```cpp
using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::NORMAL>;
```

kernel 侧 c1Type/c3Type 同样为 `TPosition::GM`。

> `TPosition::GM` 走 `L0C→FixPipe→GM` 标准路径，A5 保留该通路（见 §5.4），**无需改造**。
>
> 旁证：A5 已适配的 `flash_attention_score` arch22 模板中 30+ 处使用 `TPosition::GM` 作为 bmm1/bmm2 C 侧输出（`op_host/arch22/flash_attention_score_tiling_general.cpp:2802,2828,3089,3356,3634,3882` 等），且 arch35 目录无独立 tiling 实现（复用 arch22），证实 `TPosition::GM` 在 A5 上是有效路径。

### 6.5 同步机制（SetFlag/WaitFlag 强配对）

**证据**：`op_kernel/fused_floyd_attention_s1s2_bn2gs1.h` 与 `op_kernel/fused_floyd_attention_s1s2_bn2gs1special.h` 中 100+ 处 `SetFlag<HardEvent::XXX>` / `WaitFlag<HardEvent::XXX>` 调用（grep 结果）。

本算子大量使用 set/wait 强配对同步，A5 上向后兼容可运行，但未迁移到 BufferID 同步机制。

### 6.6 `ENABLE_CV_COMM_VIA_SSBUF` 宏

**证据**：`op_host/CMakeLists.txt:23` 为 ascend950 预置 `-DENABLE_CV_COMM_VIA_SSBUF=true`

**证据**：对本算子目录执行 grep `ENABLE_CV_COMM_VIA_SSBUF` / `USE_SSBUF` / `SSBuffer` / `GetSsbufBaseAddr`，仅在 CMakeLists.txt 出现 1 处（宏定义处），kernel 与 host 代码中**均无对该宏及相关 API 的引用**。

> 本算子未使用该宏，迁移时不需处理该宏的消费方问题。

---

## 7. 迁移修改点清单

### P0 阻塞项（不改则 A5 无法运行）

| # | 文件:行 | 修改内容 | 证据 |
|---|---|---|---|
| 1 | `op_host/fused_floyd_attention_def.cpp:88` | 新增 `this->AICore().AddConfig("ascend950", aicore_config);` | §3.2 |
| 2 | `op_kernel/fused_floyd_attention.cpp:22` | 扩展 Cube 宏分支为 `#if defined(__DAV_C220_CUBE__) \|\| defined(__DAV_C310_CUBE__)`，或参考 `flash_attention_score` 的 arch35 模式新增 `#elif defined(__DAV_C310_CUBE__)` 分支 | §3.4、§4.2 |
| 3 | `README.md:7` + `docs/aclnnFusedFloydAttention.md:8` | A5 支持状态由 × 改为 √ | §3.1 |

### P0 验证项（功能正确性需确认）

无。

> 原「Matmul `TPosition::GM` 输出通路」已澄清：走 `L0C→FixPipe→GM` 标准路径，A5 保留（见 §5.4），且 `flash_attention_score` arch22 模板大量使用 `TPosition::GM`（见 §6.4 旁证），无需验证。
>
> 原「`ENABLE_CV_COMM_VIA_SSBUF` 宏消费方」已澄清：本算子 kernel 与 host 代码均未引用该宏及相关 API（见 §6.6），无需处理。

### P1 性能优化项（非阻塞）

| # | 文件:行 | 修改内容 | 证据 |
|---|---|---|---|
| 1 | `op_kernel/fused_floyd_attention_s1s2_bn2gs1.h:485-509` | Buffer 分配改为基于 `aicoreParams_.ubSize` 动态计算，利用 A5 增大的 UB | §6.2 |

### P2 深度优化项（较大重构）

| # | 优化内容 | 证据 |
|---|---|---|
| 1 | SetFlag/WaitFlag → BufferID 同步迁移 | §5.5、§6.5 |
| 2 | 采用 L0C→UB / UB→L1 直通消除 GM workspace | §5.4、§6.3 |

### 测试更新

| # | 文件:行 | 修改内容 | 证据 |
|---|---|---|---|
| 1 | `tests/ut/op_host/test_fused_floyd_attention_tiling.cpp:36,56,66,86` | 硬编码 `Ascend910B`、`aivNum=48`、`ubSize=196608`(192KB) 等 A3 参数，需新增 A5 用例（`SocVersion::ASCEND950`、`aivNum=56/64`、`ubSize=253952`、`l0cSize=262144`） | §5.1、§5.2 |

---

## 8. 参考文档索引

| 文档 | 位置 | 用途 |
|---|---|---|
| NPU 架构知识 | `ops/npu-arch/SKILL.md` | 架构代际概述、概念定义 |
| 硬件参数真值 | `ops/npu-arch/references/npu-hardware-params.md` | 产品映射表、各架构 Buffer/核数参数 |
| 架构指南 | `ops/npu-arch/references/npu-arch-guide.md` | DAV_3510 微架构、数据通路、同步机制、数据格式 |
| SIMT 架构 | `ops/npu-arch/references/simt-arch-guide.md` | SIMT 线程层次、内存空间 |
| 算子设计介绍 | `docs/FFA算子设计介绍.md`（算子目录内） | CV tiling 分离策略、流水设计、IterateBatch L1 约束 |
