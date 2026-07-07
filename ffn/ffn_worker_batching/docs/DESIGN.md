# FfnWorkerBatching 详细设计文档（新增 Ascend950/A5 支持）

> 本设计只覆盖「为已有 A2/A3 算子新增 Ascend950(arch35/A5) 支持」。A5 对外支持范围严格对齐 A2；A2/A3 分支保留不动（`def.cpp` 现有 AddConfig、`config/ascend910b`、`config/ascend910_93`、现有 arch22 kernel/tiling 全部不改）。本阶段只做设计，不写实现代码。

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人 |
|-----|---------|---------|-------|
| v1.0 | 初版：A5(arch35) 全新代次路径设计，范围对齐 A2 | 2026-07-06 | CANNBot(Architect) |

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | FfnWorkerBatching（snake：`ffn_worker_batching`；OpType：`FfnWorkerBatching`） |
| 算子类别 | 融合搬移/重排类（Sort 排序 + IndexGather 索引 gather + Histogram/分组计数）；非计算类，无浮点算术 |
| 支持数据类型 | 3 profile（token_dtype）：P0=FP16 / P1=BF16 / P2=dynamic_quant_int8（对齐 A2，A5 不扩展 FP8/MXFP8/HiF8） |
| **目标芯片（本次）** | Ascend950DT / Ascend950PR（实测 Ascend950DT_9584） |
| **目标架构（本次）** | arch35 / **DAV_3510**（SocVersion 配置名 `ascend950`；CANN 9.1.0，NpuArch=3510） |
| 现有芯片（保留不动） | Ascend910B（A2）/ Ascend910_93（A3），arch22 / DAV_2201 |
| 芯片→架构映射依据 | `npu-arch` 技能：Ascend950DT/PR → DAV_3510；Ascend910B/910_93 → DAV_2201 |
| 编程框架 | 手写 Ascend C（arch35 代次）；路线为**经典 SIMD（MemBase）为主 + 局部 RegBase 可选**（见 §3.1 路线决策） |

### 1.2 算子功能

Attention 与 FFN 分离部署（PD 分离的 MoE 变体）场景下，在 FFN worker 上从 1024 字节调度上下文 `schedule_context` 的 FfnArea 逐字节解析 token 路由信息，按专家维度聚合并重排 token，产出各专家连续 token 数据块，供后续 GroupedMatmul 消费。三步融合：**排序生成 gather 索引 → 多核 gather 提取 token → 单核扫描生成 group_list**。

### 1.3 算法语义（三步，与 A2 一致；真值见 spec.yaml `math_semantics`）

1. **sort**：对所有 token 的 expert_id 稳定排序，被 mask 的无效 token 置极大值排到末尾，生成按专家聚合的 gather 索引序列 `gather_idx`。
2. **gather**：多核并行按 `gather_idx` 从 `token_data` 提取 hidden states → `y[out_i,:] = token_data[gather_idx[out_i],:]`（逐字节搬运，dtype 随 token_dtype）；token_dtype=2 时每 token 尾部 32B 的 FP32 scale 一并搬运 → `dynamic_scale[out_i]`；同步查表得 `session_ids/micro_batch_ids/token_ids/expert_offsets`。
3. **group_listing**：单核扫描排序后 expert_id 序列查专家跳变点，生成 `group_list[e]=[expert_id_e, token_num_e]`（未用专家 `[0,0]`），并统计 `actual_token_num`。

精度口径：所有输出为整数索引或原样字节 gather，与 CPU golden **二进制一致**（`np.array_equal`）。

### 1.4 本次范围与不做项

- **做**：新增 A5 全新代次路径（`_apt.cpp` 入口 + `op_kernel/arch35/` + `_tiling_arch35.cpp` + `config/ascend950/`），功能/精度对齐 A2；NORM(100) + RECV(101) 两条路径 × 3 profile 全覆盖。
- **不做**：① 不移植 A2 从不启用的保留分支 `TILING_KEY_RECV_GATHER=102`（tiling 从不 SetTilingKey(102)）；② 不扩展 arch35 独有窄类型（FP8/MXFP8/HiF8）；③ 不改 A2/A3 任何配置与逻辑；④ 不重新设计三步热点算法，仅做 arch35 指令/资源模型适配。

---

## 2. 架构设计

### 2.1 逻辑视图（模块职责）

| 模块 | 职责 | A5 相关文件 |
|------|------|---------|
| **op_api** | ACLNN 接口层（现有，A5 共用不改） | `aclnn_ffn_worker_batching.*`（如存在） |
| **op_graph** | 图模式 IR 原型（现有，A5 共用不改） | `ffn_worker_batching_proto.h` |
| **op_host** | 算子定义、Tiling、Shape 推导 | `ffn_worker_batching_def.cpp`（+A5 config）、**新增 `ffn_worker_batching_tiling_arch35.cpp`**、`ffn_worker_batching_tiling.h`（+A5 TilingData 注册）、`ffn_worker_batching_infershape.cpp`（共用不改） |
| **op_kernel** | Kernel 侧计算 | **新增入口 `ffn_worker_batching_apt.cpp`** + **新增 `op_kernel/arch35/` 全套头文件** |
| **config** | 二进制配置 | **新增 `op_host/config/ascend950/`** |

> **隔离原则**（`ascendc-registry-invoke-template` 代际隔离章 + moe 样板实证）：ACLNN 接口、IR 定义 A2/A5 **共用**；`op_host` tiling、`op_kernel` 实现、`config` 目录 A2(arch22)/A5(arch35) **隔离**。

### 2.2 开发视图（A5 新增后的目录）

```
ffn/ffn_worker_batching/
├── op_host/
│   ├── ffn_worker_batching_def.cpp              # 改：新增 AddConfig("ascend950", regbaseCfg)，A2/A3 AddConfig 不动
│   ├── ffn_worker_batching_infershape.cpp       # 不改（infershape 与代次无关）
│   ├── ffn_worker_batching_tiling.cpp           # 不改（arch22 monolithic tiling 原样保留）
│   ├── ffn_worker_batching_tiling.h             # 改：仅追加 A5 TilingData 注册（arch22 TilingData 不动）
│   ├── ffn_worker_batching_tiling_arch35.cpp    # 新增：A5 Regbase tiling（1000 优先级 + IsRegbaseSocVersion 守卫）
│   └── config/
│       ├── ascend910b/                          # 不改
│       ├── ascend910_93/                        # 不改
│       └── ascend950/                           # 新增：ffn_worker_batching.json + _simplified_key.ini
├── op_kernel/
│   ├── ffn_worker_batching.cpp                  # arch22 入口（不改）
│   ├── ffn_worker_batching_apt.cpp              # 新增：arch35 入口，同名 extern "C" 符号 ffn_worker_batching
│   ├── ffn_wb_*.h (13 个)                        # arch22 头（不改）
│   └── arch35/                                  # 新增：13 个头移植 + arch35 TilingData 平铺 struct
├── op_graph/ffn_worker_batching_proto.h         # 不改（IR 共用）
└── docs/ (本设计所在)
```

### 2.3 运行视图（数据流总览，NORM/RECV 两路径）

```
GM: schedule_context(1024B INT8)
  │ ScheduleContextInfoCompute<isRecv> 逐字节解析 FfnArea 指针/路由字段
  ▼
[RECV 专属] KernelScanTokenInfo 扫描 token_info → SyncAll
  │
  ▼ (step 1 sort → 生成 gather_idx，被 mask token 置极大值排末尾)
  排序：单核 KernelSortMaskOneCore / 多核 KernelSortMaskMultiCore
        （RECV 用 scan 变体：KernelScanSortMaskOneCore / MultiCore / MultiCoreBsK）
  │ ValidGatherIdxLengthCompute → actual_token_num、validGatherIdxLength
  ▼ (多核分工：部分核 gather，其余核 group_listing)
  step 2 gather：KernelFfnWBGatherOutAll<isRecv>  → y / session_ids / micro_batch_ids / token_ids / expert_offsets / dynamic_scale(仅 P2)
  step 3 group_listing：KernelFfnWBGroupListing(NORM) / KernelFfnWBScanGroupListingOneCore(RECV) → group_list
  │ SyncAll → (NORM 多核聚合时最后一核 ProcessExpertCount)
  ▼
GM: 8 路输出
```

### 2.4 用户视图

| 调用方式 | 支持 | A5 |
|---------|------|----|
| ACLNN 单算子（两段式 `aclnnFfnWorkerBatchingGetWorkspaceSize`/`aclnnFfnWorkerBatching`） | ✅ | 接口共用，A5 经 def AddConfig 自动纳入 |
| GE IR 构图（静态/动态 shape） | ✅ | A5 config 置 `DynamicShapeSupportFlag(true)`（对齐 moe 样板 regbaseCfg） |

---

## 3. 实现方案

### 3.1 路线决策（RegBase vs 经典 SIMD/MemBase）

**决策：arch35 主路线走经典 SIMD/MemBase（Sort/MrgSort/Concat/Extract/DataCopyPad/TQue/Duplicate/SetAtomicAdd 等 kernel-side API），RegBase 仅作局部可选优化，本次不强制引入。**

依据（读码实证）：

1. **样板算子 `moe_init_routing_v3/op_kernel/arch35/` 的同族原语（sort/gather/expert_tokens_count）在 DAV_3510 上主体仍是经典 SIMD API**——`moe_v3_sort_one_core.h`/`moe_v3_sort_multi_core.h` 用 `Sort<float,true>` + `Concat` + `Extract` + `ArithProgression` + `GetSortLen/GetSortOffset`；`moe_v3_mrgsort.h` 用 `MrgSort<float,true>`；`moe_v3_gather_out.h` 用 `TQueBind` + `DataCopyPad`；`moe_v3_expert_tokens_count.h` 用 `Duplicate` + `SetAtomicAdd`。这些与 A2 的 `ffn_wb_*` 完全同族。
2. **RegBase（`__VEC_SCOPE__`/`RegTensor`/`MicroAPI`）在样板中只出现在 quant/cast 密集的 gather 变体**（`moe_v3_gather_dynamic_quant.h`、`*_fp8_*`、`*_hif8_*`、`*_mxfp*` 等）——即需要在寄存器上做反量化/类型转换算术的路径。
3. **FfnWorkerBatching 的 gather 对所有 3 profile 都是"原样字节搬运，非重算"**（spec.yaml `math_semantics` / `numerical_tolerance` = bitwise_equal）：P0/P1 搬 fp16/bf16 原始字节，P2 搬 int8 数据 + 尾部 32B FP32 scale 原始字节——**无任何寄存器级算术**。因此不需要 RegBase 反量化管线，经典 `DataCopyPad` 字节 gather 即可满足二进制一致要求。

> 结论记录：主路线 = 经典 SIMD/MemBase（arch35 代次重编译 + 资源模型适配）。若后续性能验收（可选项）需要，可在 gather 内层对 int8/scale 搬运引入 RegBase，但**不改变对外语义**，且需另做 RegBase 专项设计（参 `ascendc-regbase-best-practice`）。**A5 不因引入 RegBase 而偏离 A2 的三步算法**。

### 3.2 Kernel 模板选择与 13 头文件移植映射

**入口结构**（对齐 moe_gating_top_k_apt.cpp / moe_init_routing_v3_apt.cpp 实证）：

`op_kernel/ffn_worker_batching_apt.cpp` —— 与 arch22 入口 `ffn_worker_batching.cpp` **同名同签名** `extern "C" __global__ __aicore__ void ffn_worker_batching(...)`，11 参数：`schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num, workspace, tiling`。`#include "arch35/*.h"`。用 `GET_TILING_DATA_WITH_STRUCT(FfnWorkerBatchingArch35TilingData, ...)` + `TILING_KEY_IS(100/101)` 分发到 NORM/RECV。

> 同名 extern "C" 符号不冲突的机制：`def.cpp` 通过 `ExtendCfgInfo("opFile.value","ffn_worker_batching_apt")` 让 **ascend950 编译单元只取 `_apt.cpp`**、A2/A3 只取 `ffn_worker_batching.cpp`——每个 soc 编译产物中该符号唯一（`ascendc-registry-invoke-template` 代际隔离章 + add_example 实证）。详见 §8。

**13 头文件移植映射表**（源 `op_kernel/*.h` → 目标 `op_kernel/arch35/*.h`；参考蓝本为 moe_init_routing_v3/op_kernel/arch35/ 同族原语）：

| # | 源文件（arch22） | 目标（arch35） | 步骤 | 主要 class/struct | arch35 参考蓝本 | 移植动作 |
|---|---|---|---|---|---|---|
| 1 | `ffn_wb_common.h` | `arch35/ffn_wb_common.h` | 公共 | `BufferInfo` / `ScheduleContextInfo` / `SortCustomTilingDataKernel` / `FfnWBGroupListingTileInfo` + 常量宏 | `moe_v3_common.h` | 常量/结构原样保留，UB 相关常量改为从 TilingData 运行时值推导（见 §3.6） |
| 2 | `ffn_wb_get_schedule_context.h` | `arch35/ffn_wb_get_schedule_context.h` | 公共 | `ScheduleContextInfoCompute<isRecv>`（GM 指针逐字节解析） | `moe_v3_common.h` 指针解析 | 核对 arch35 GM 指针/DCCI 语义（见 §5 风险） |
| 3 | `ffn_wb_scan_token_info.h` | `arch35/ffn_wb_scan_token_info.h` | RECV 前置 | `KernelScanTokenInfo` | `moe_v3_row_idx_gather.h` | DataCopyPad 搬运语义适配 |
| 4 | `ffn_wb_scan_get_valid_experts.h` | `arch35/ffn_wb_scan_get_valid_experts.h` | RECV(102 保留分支用) | `KernelScanGetValidExperts : SortMaskBase` | - | **仅随 SortMaskBase 一起移植保留，不产生 102 分发** |
| 5 | `ffn_wb_sort_base.h` | `arch35/ffn_wb_sort_base.h` | sort 基类 | `SortMaskBase` | `moe_v3_sort_base.h` | 排序 key/mask 基类，Sort API 重编译 |
| 6 | `ffn_wb_sort_one_core.h` | `arch35/ffn_wb_sort_one_core.h` | sort(NORM 单核) | `KernelSortMaskOneCore : SortMaskBase` | `moe_v3_sort_one_core.h` | `Sort<float,true>`+`Concat`+`Extract` 原语对齐 |
| 7 | `ffn_wb_sort_multi_core.h` | `arch35/ffn_wb_sort_multi_core.h` | sort(NORM 多核) | `KernelSortMaskMultiCore : SortMaskBase` | `moe_v3_sort_multi_core.h` | 多核切分 + MrgSort 归并 |
| 8 | `ffn_wb_sort_mrgsort.h` | `arch35/ffn_wb_sort_mrgsort.h` | sort 归并 | `SortCustomMrgsort` / `SortCustomMrgsortParam` | `moe_v3_mrgsort.h` | `MrgSort<float,true>` 4 路归并（禁 MrgSort4） |
| 9 | `ffn_wb_sort_mrgsort_out.h` | `arch35/ffn_wb_sort_mrgsort_out.h` | sort 归并输出 | `SortCustomMrgsortOut` | `moe_v3_mrgsort_out.h` | 归并结果落 workspace |
| 10 | `ffn_wb_scan_sort_one_core.h` | `arch35/ffn_wb_scan_sort_one_core.h` | sort(RECV 单核) | `KernelScanSortMaskOneCore : SortMaskBase` | `moe_v3_sort_one_core.h` | RECV 扫描态单核排序 |
| 11 | `ffn_wb_scan_sort_multi_core.h` | `arch35/ffn_wb_scan_sort_multi_core.h` | sort(RECV 多核) | `KernelScanSortMaskMultiCore : SortMaskBase` | `moe_v3_sort_multi_core.h` | RECV 扫描态多核排序 |
| 12 | `ffn_wb_scan_sort_multi_core_bsk.h` | `arch35/ffn_wb_scan_sort_multi_core_bsk.h` | sort(RECV 多核 BsK) | `KernelScanSortMaskMultiCoreBsK : SortMaskBase` | `moe_v3_sort_multi_core.h` | BsK 非规则切分变体 |
| 13 | `ffn_wb_gather_out_all.h` | `arch35/ffn_wb_gather_out_all.h` | gather | `KernelFfnWBGatherOutAll<isRecv>` | `moe_v3_gather_out.h` + `moe_v3_gather_dynamic_quant.h`（P2 scale） | TQueBind + DataCopyPad 字节 gather；P2 尾部 32B scale 连续搬运 |
| 14 | `ffn_wb_group_listing.h` | `arch35/ffn_wb_group_listing.h` | group_listing(NORM) | `KernelFfnWBGroupListing` | `moe_v3_expert_tokens_count.h` | 专家跳变扫描 + Duplicate/SetAtomicAdd 计数 |
| 15 | `ffn_wb_scan_group_listing_one_core.h` | `arch35/ffn_wb_scan_group_listing_one_core.h` | group_listing(RECV) | `KernelFfnWBScanGroupListingOneCore` | `moe_v3_expert_tokens_count.h` | RECV 单核 group_list 生成 |
| + | （新增，无 arch22 对应） | `arch35/ffn_worker_batching_arch35_tiling_def.h` | TilingData | `FfnWorkerBatchingArch35TilingData`（平铺 struct） | `moe_init_routing_v3_arch35_tiling_def.h` | kernel 侧 TilingData 平铺 struct，`GET_TILING_DATA_WITH_STRUCT` 读取 |

> 计数：题述"13 个 ffn_wb_*.h" = 上表 #1-#3、#5-#15 中除 #4 外的 13 个功能头（`ffn_wb_scan_get_valid_experts.h` 为 102 保留分支伴随件）。全部 15 个 `.h`（含 #4）+ 1 个新增 TilingData def 一并落到 `arch35/`。

### 3.3 数据类型支持方案（3 profile，token_dtype 0/1/2）

**与 spec.yaml `dtype_policy.supported_combinations` 一一对应，A5 全量对齐 A2，不增删。** token_dtype 在 kernel 内处理，**与 TilingKey 正交**（TilingKey 只由 need_schedule 决定）：

| profile | token_dtype | y dtype | dynamic_scale | kernel 内处理（`KernelFfnWBGatherOutAll`） |
|---------|:---:|:---:|:---:|---|
| P0 | 0 | FP16 | 空 tensor | `tokenDtypeSize_ = sizeof(half)`；纯 hidden states 字节 gather |
| P1 | 1 | BF16 | 空 tensor | `tokenDtypeSize_ = sizeof(half)`（bf16 同 2 字节）；纯字节 gather |
| P2 | 2 | INT8 | **有效** | `tokenDtypeSize_ = sizeof(int8_t)`；每 token 尾部 `BLOCK_BYTES=32` 承载 FP32 scale，`maxBlockSize_` 扣除 32B 后按 32B 对齐，scale 与 int8 数据连续搬运 → dynamic_scale |

- `y` dtype 由 infershape 依 token_dtype 动态设置（0→FP16 / 1→BF16 / 其它→INT8），**与代次无关，A5 共用现有 infershape 不改**。
- `schedule_context` 恒 INT8（唯一物理输入）；`token_payload` 是 spec 建模键（表达 y dtype 随 token_dtype），非独立 IR 输入。

### 3.4 API 映射方案（A2 指令/接口 → arch35 对应）

> 本算子为非计算类，核心 API 集中在 **排序（Sort/MrgSort/Concat/Extract）+ 索引生成（ArithProgression）+ 数据搬运（DataCopy/DataCopyPad）+ 计数（Duplicate/SetAtomicAdd）+ 同步（SyncAll/SetFlag/WaitFlag）**。arch35 主体沿用同族经典 SIMD API（样板 moe_init_routing_v3/arch35 实证在 DAV_3510 上可用）。

| 计算步骤 | Ascend C API | 参数签名（样板实证） | 平台验证 | 约束说明 |
|---------|-------------|---------|---------|---------|
| 索引初始化 | `ArithProgression<int32_t>` | `(dst, start, stride, count)` | ✅ DAV_3510（`moe_v3_sort_one_core.h:48`） | 前需等 MTE2→V |
| 合成 proposal | `Concat` | `(dst, srcValue, tmpBuffer, repeatTimes)` | ✅ DAV_3510（`moe_v3_sort_one_core.h:93`） | proposal=8B（value4B+index4B） |
| tile 内排序 | `Sort<float, true>` | `(dstProposal, srcValue, srcIndex, tmpBuffer, repeatTimes)` | ✅ DAV_3510（`moe_v3_sort_one_core.h:98`） | 稳定排序；repeatTimes=sortNum/32 |
| proposal 分离 | `Extract` | `(dstValue, dstIndex, srcProposal, repeatTimes)` | ✅ DAV_3510（`moe_v3_sort_one_core.h:103`） | - |
| 4 路归并 | `MrgSort<float, true>` | `(dst, MrgSortSrcList, elementCountList, sortedNums, validBitTail, exhaustion)` | ✅ DAV_3510（`moe_v3_mrgsort.h:133-141`） | **禁 `MrgSort4`**；4 路各自有序 |
| proposal 地址换算 | `GetSortLen<float>` / `GetSortOffset<float>` | `(n)` → n×2（float 数） | ✅ DAV_3510（`moe_v3_mrgsort.h:117,157`） | 用内置 API，禁自定义 |
| 字节 gather 搬入/搬出 | `DataCopyPad` + `DataCopyExtParams`/`DataCopyPadExtParams<T>` | `(dst, src, DataCopyExtParams{blockCount,blockLen(byte),srcStride,dstStride,rsv}, pad)` | ✅ DAV_3510（`moe_v3_gather_out.h:176-178`） | 非对齐必用 DataCopyPad；blockLen 单位 byte |
| 对齐搬运 | `DataCopy` | `(dst, src, len)` | ✅ DAV_3510（`moe_v3_sort_multi_core.h:147`） | 仅严格 32B 对齐时用 |
| 队列管理 | `TQue` / `TQueBind<VECIN,VECOUT,N>` | - | ✅ DAV_3510（`moe_v3_gather_out.h:47-49`） | gather CopyIn/CopyOut 双缓冲 |
| 清零/初始化 | `Duplicate` | `(dst, scalar, count)` | ✅ DAV_3510（`moe_v3_expert_tokens_count.h:190`） | group_list 初始化 [0,0] |
| 原子累加计数 | `SetAtomicAdd<int32_t>()` / `SetAtomicNone()` | 包裹 DataCopyPad | ✅ DAV_3510（`moe_v3_expert_tokens_count.h:231-233`） | 多核 group 计数聚合，保确定性 |
| 多核同步 | `SyncAll()` | - | ✅ DAV_3510（本算子 `ffn_worker_batching.cpp:103,124`） | step 间栅栏 |
| tmpBuffer 大小 | `GetSortTmpSize` / `GetConcatTmpSize` | `(platform, tileSize, 4)` | ⚠️ 需 arch35 平台参数复核 | UB 规划用 |

> **RegBase 可选优化路径（非本次必做）**：若 P2 int8 gather 需寄存器级优化，参 `moe_v3_gather_dynamic_quant.h` 的 `__VEC_SCOPE__`/`RegTensor`/`AscendC::Reg::LoadAlign/StoreAlign` 结构；但 FfnWorkerBatching gather 为原样字节搬运无算术，默认不启用。

### 3.5 数据流设计（sort → gather → group_listing）

**NORM 路径（TilingKey 100，`FfnWorkerBatchingProcess`）**：
```
1. ScheduleContextInfoCompute<false>: 解析 schedule_context → contextInfo（含 FfnArea 指针、A/BS/K/coreNum）
2. TilingSort(&tilingdataSort, &contextInfo): 判定单核 vs 多核排序
   ├─ tilingKey==TILINGKEY_ONECORE_SORT → KernelSortMaskOneCore.Process()
   └─ else                              → KernelSortMaskMultiCore.Process()
3. ValidGatherIdxLengthCompute: 求 validGatherIdxLength + 写 actual_token_num
4. 多核分工（usedCoreNum = validGatherIdxLength>GROUP_LISTING_MULTI_CORE_LENGTH ? GROUP_LISTING_MULTI_AIV_NUM : GROUP_LISTING_AIV_NUM）：
   ├─ GetBlockIdx() < coreNum-usedCoreNum → KernelFfnWBGatherOutAll<false>.Process()  → y/ids/dynamic_scale
   └─ else                                → KernelFfnWBGroupListing.Process(groupListingDealFlag) → group_list
5. SyncAll(); if(groupListingDealFlag && 末核) expertTokenOutMultiOp.ProcessExpertCount()
```

**RECV 路径（TilingKey 101，`FfnWorkerBatchingRecvProcess`）**：
```
1. ScheduleContextInfoCompute<true>
2. KernelScanTokenInfo.Process() → SyncAll()   （RECV 专属：先扫描接收 token_info）
3. TilingScanSort(&tilingdataSort, &contextInfo):
   ├─ TILINGKEY_ONECORE_SORT      → KernelScanSortMaskOneCore
   ├─ isNormalBsK != NORMAL_BSK   → KernelScanSortMaskMultiCoreBsK
   └─ else                        → KernelScanSortMaskMultiCore
4. ValidGatherIdxLengthCompute → actual_token_num
5. 多核分工（GROUP_LISTING_SCAN_AIV_NUM）：
   ├─ blockIdx < coreNum-GROUP_LISTING_SCAN_AIV_NUM → KernelFfnWBGatherOutAll<true>.Process()
   └─ else                                          → KernelFfnWBScanGroupListingOneCore.Process() → group_list
```

> 三步热点算法照搬 A2（spec 未改）；arch35 仅重编译原语 + 按运行时 UB/核数重算切分阈值。单核/多核排序选择、gather 与 group_listing 的核间分工逻辑保持 A2 语义。

### 3.6 Tiling 策略规划（arch35 双代隔离 + UB 动态取值切分）

#### 3.6.1 双代隔离方案（关键：与现有 arch22 tiling 隔离防符号冲突）

**读码校正（重要事实，必须记录）**：题述假设「与现有 2000 档隔离」——但**现有 ffn_worker_batching arch22 tiling 并非 `REGISTER_OPS_TILING_TEMPLATE` 模板注册，而是经典 monolithic `IMPL_OP_OPTILING(FfnWorkerBatching).Tiling(Tiling4FfnWorkerBatching)`**（`ffn_worker_batching_tiling.cpp:250`，直接实例化 `FfnWorkerBatchingTiling` 对象跑 `RunFfnWorkerBatchingTiling()`，不走 `TilingRegistry` 优先级遍历）。因此仓内**不存在**现成的 2000 档模板供并列。样板 `moe_gating_top_k` 用的是两层模板注册（`_tiling_base.cpp` 里 `IMPL_OP_OPTILING(...).Tiling(→DoTilingImpl)` 分派器 + 多个 `REGISTER_OPS_TILING_TEMPLATE(..., 2000/1000)` 候选类，各带 `IsCapable()` 守卫）。

**隔离方案（二选一，推荐方案 A，兼顾"A2/A3 逻辑不动"与"1000 档 Regbase + IsRegbaseSocVersion 守卫"意图）**：

- **方案 A（推荐，最小改动，arch22 逻辑零改）**：保持 arch22 `Tiling4FfnWorkerBatching` 单函数不动；在**分派器入口**加一层 SoC 守卫——
  ```cpp
  // ffn_worker_batching_tiling.cpp（仅在 Tiling4FfnWorkerBatching 顶部加分流；arch22 类体不动）
  static ge::graphStatus Tiling4FfnWorkerBatching(gert::TilingContext* context) {
      if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {   // ascend950 → 走模板注册表
          return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
      }
      FfnWorkerBatchingTiling tilingObject(context);                     // A2/A3 原路径，逐字不动
      return tilingObject.RunFfnWorkerBatchingTiling();
  }
  ```
  arch35 tiling 类在新文件 `ffn_worker_batching_tiling_arch35.cpp` 里用
  `REGISTER_OPS_TILING_TEMPLATE(FfnWorkerBatching, FfnWorkerBatchingRegbaseTiling, 1000);` 注册，类 `IsCapable()` 返回 `IsRegbaseSocVersion(context_)`（对齐 moe_init_routing_v3_tiling_arch35.cpp:181-184）。1000 档只在 ascend950 分支被 `DoTilingImpl` 遍历到，与 A2/A3 完全隔离。
- **方案 B（对齐 moe 完整两层模板）**：把 arch22 也改造成 `REGISTER_OPS_TILING_TEMPLATE(FfnWorkerBatching, FfnWorkerBatchingTilingBase, 2000)`（`IsCapable` 恒真作 fallback），分派器改为纯 `DoTilingImpl`。**代价**：需重构 arch22 tiling 类继承 `TilingBaseClass`，违反"A2/A3 逻辑保留不动"。**故不采用，除非评审要求全仓统一模板风格。**

> 结论：采用**方案 A**。1000 档 Regbase 模板 + `IsRegbaseSocVersion` 守卫，经分派器 SoC 分流与 arch22 monolithic 路径隔离；A2/A3 tiling 类体零改（仅分派器函数加 3 行守卫）。此为对题述"1000 档 + IsRegbaseSocVersion 守卫 vs 现有档"意图的忠实落地，并如实修正"现有为 2000 档模板"的前提偏差。

#### 3.6.2 TilingData 隔离（防 REGISTER_TILING_DATA_CLASS 符号冲突）

arch22 现有 `REGISTER_TILING_DATA_CLASS(FfnWorkerBatching, FfnWorkerBatchingTilingData)`（`tiling.h:41`）保留不动。arch35 新增 TilingData 用**带后缀 op-type 名**注册以隔离（对齐 moe_gating_top_k `REGISTER_TILING_DATA_CLASS(MoeGatingTopK_10000, MoeGatingTopKRegbaseTilingData)`）：
```cpp
// tiling.h 追加（arch22 段不动）
BEGIN_TILING_DATA_DEF(FfnWorkerBatchingRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, Y);          TILING_DATA_FIELD_DEF(int64_t, H);
TILING_DATA_FIELD_DEF(int64_t, tokenDtype); TILING_DATA_FIELD_DEF(int64_t, expertNum);
TILING_DATA_FIELD_DEF(int64_t, coreNum);    TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, sortLoopMaxElement); TILING_DATA_FIELD_DEF(int64_t, sortNumWorkSpace);
// arch35 视需要可加：groupListingCoreNum / scanBsK 标记等（运行时推导，不含 arch 常量）
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FfnWorkerBatching_1000, FfnWorkerBatchingRegbaseTilingData)
```
kernel 侧对应 `arch35/ffn_worker_batching_arch35_tiling_def.h` 平铺 struct（`GET_TILING_DATA_WITH_STRUCT` 读取，对齐 moe_init_routing_v3_arch35_tiling_def.h）。

#### 3.6.3 多核切分 + UB 动态取值切分（禁写死任何 arch UB 字节常量）

**平台参数运行时获取**（arch35 tiling 的 `GetPlatformInfo()`，与 A2 同一取值逻辑，仅返回值随平台）：
```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();                        // 核数运行时取（禁写死）
uint64_t ubSize = 0;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);// UB 容量运行时取（禁写死）
tilingData_.set_coreNum(aivNum);  tilingData_.set_ubSize(ubSize);  context_->SetBlockDim(aivNum);
```
> DAV_3510 实测 UB=248KB、DAV_2201=192KB——**这些数值只用于设计推演，代码中不得出现任何 arch 相关 UB 字节数/核数常量**，一律经上述接口返回值推导。

**UB 派生切分阈值**（沿用 A2 公式结构，输入换成 arch35 运行时 ubSize，自动重算）：

| 阈值 | A2 公式（`ffn_worker_batching_tiling.cpp`） | arch35 处理 |
|------|------|------|
| `sortLoopMaxElement` | `ubSize / (sizeof(int32)*2*4) / 32 * 32`（32=ONE_REPEAT_SORT_NUM 对齐） | **同公式**，ubSize=运行时 248KB → 阈值自动增大；不写死 |
| `sortNumWorkSpace` | `= Y_`（Y=A*BS*K） | 同 |
| group_listing 单/多核阈值 `GROUP_LISTING_MULTI_CORE_LENGTH` / AIV 数 | kernel 侧常量（片上暂存约束） | 随 arch35 UB/核数复核；若为纯并行度阈值可保留，若含 UB 字节假设须改运行时推导 |
| RECV 限核阈值 `TH_RECV_CORE_NUM=32` / `TH_RECV_Y_NUM=200000` | `needSchedule==1 && Y_<=200000 → useCoreNum=min(coreNum,32)` | A5 可复用或按 arch35 核数重标定（设计期决策：默认复用，穿刺阶段实测调优） |

**硬校验阈值（对外规格，与代次无关，arch35 原样保留）**：`EXPERT_IDX_MAX=8192`、`MAX_SESSION_NUM=1024`（A≤1024）、`MAX_K_NUM=64`（topK+1≤64）——这些是对外 shape 支持范围，来自 spec `shape_constraints`，A5 全对齐 A2，不随 UB 变。

**TilingKey 设置**（`SetBatchingTilingKey`，A5 沿用 A2 语义，只由 need_schedule 决定）：`need_schedule==0 → 100(NORM)`；`need_schedule==1 → 101(RECV)`；**从不 SetTilingKey(102)**（102=RECV_GATHER 保留分支不移植/不分发）。token_dtype 正交，kernel 内处理。

### 3.7 内存管理策略

#### 3.7.1 Workspace 公式（arch35 同 A2 公式，sysWorkspaceSize 取平台值）

`SetBatchingWorkspaceSize`（`ffn_worker_batching_tiling.cpp:46-61`，A5 原样沿用）：
```
workspace[0] = MAX_RESERVE_WK_NUM(128) * sizeof(int32)          // 有效专家数 + recv 遍历 micro_batch_id
             + Y * sizeof(int32)                                 // 每核每 loop 排序后有效专家数
             + Y * sizeof(int32) * NUM_TWO(2) * NUM_FOUR(4)      // 排序占用 + 排序后 expert_ids/expert_idx
             + expert_num * sizeof(int32)
             + sysWorkspaceSize;                                 // = ascendcPlatform.GetLibApiWorkSpaceSize() 平台值
```
> A5 与 A2 同公式；`sysWorkspaceSize` 经 `GetLibApiWorkSpaceSize()` 取 arch35 平台值，非写死。

#### 3.7.2 UB 分配（gather 为主，双缓冲；全部由运行时 ubSize/tiling 值推导）

以 gather 步骤 `KernelFfnWBGatherOutAll`（UB 占用主体）为例，Buffer 规划（对齐 A2 `ffn_wb_gather_out_all.h`，arch35 重编译）：

| Buffer | 用途 | 大小公式（运行时值推导，无 arch 常量） | 说明 |
|--------|------|--------|------|
| `outQueALL_`（VECIN/OUT 双缓冲） | ids 四元组 + token 数据块 | `BUFFER_NUM * (PER_LOOP_ROWS*sizeof(int32)*VAR_NUM + PER_LOOP_ROWS*BLOCK_SIZE)` | VAR_NUM=id 路数 |
| `expertIdxQue_` | 排序后 expert_idx | `BUFFER_NUM * PER_LOOP_ROWS*sizeof(int32)` | |
| `tmpBuffer_`（RECV 时） | session 号对齐暂存 | `sessionNumBlockAlign_*sizeof(int32)*BUFFER_NUM`，`sessionNumBlockAlign_=Align(A, sizeof(int32))` | A≤1024 |
| token 数据块 | 每 token hidden states（+P2 尾部 32B scale） | `maxBlockSize_*tokenDtypeSize_ + (P2?BLOCK_BYTES(32):0)`；`maxBlockSize_=(maxTokenSize-(P2?32:0))/32*32/tokenDtypeSize_` | maxTokenSize 由单次可用 UB 推导 |

- **单次处理行数 `PER_LOOP_ROWS`、`maxBlockSize_`** 等均由 tiling 传入的运行时 `ubSize` 派生（arch35=248KB 时自动放大，减少 loop 次数），**代码不出现 192/248 等 arch 字节常量**。
- 排序步骤 UB：Sort/Concat 的 tmpBuffer 经 `GetSortTmpSize`/`GetConcatTmpSize`(平台参数) 求，proposal 8B/元素；`sortLoopMaxElement` 已由 §3.6.3 UB 派生。
- **UB 容量验证公式（DAV_3510）**：`sum(全部 InitBuffer) ≤ GetCoreMemSize(UB) 返回值（实测 248KB）`；开发期以运行时返回值断言，禁用编译期常量。

---

## 4. spec.yaml 一致性映射（强制章节）

> 以下字段以 `docs/spec.yaml` 为**唯一真值源**，禁止从 REQUIREMENTS 正文重新解释。逐字段列出 spec 取值 → 本设计落点。题述"最易误用 5 字段"用 ★ 标注。

| spec.yaml 字段 | spec 取值（原文） | 本设计承接落点 |
|---|---|---|
| ★ `dtype_policy.supported_combinations` | P0 `{schedule_context:int8, token_payload:float16}→{y:float16,...}`；P1 `token_payload:bfloat16→y:bfloat16`；P2 `token_payload:int8→y:int8`（其余输出恒定：group_list int64 / session/micro_batch/token/expert_offsets int32 / dynamic_scale float32 / actual_token_num int64） | §3.3 三 profile 表逐行对齐；kernel `KernelFfnWBGatherOutAll` 依 tokenDtype 决定 tokenDtypeSize_；y dtype 由现有 infershape 依 token_dtype 设置（共用不改）。A5 不增删组合 |
| ★ `outputs[].shape_rule` | `y=(A*BS*(topK+1), H)`（max_out_shape 展开）；`group_list=(expert_num,2)`；id 四类=`(A*BS*(topK+1),)`；`dynamic_scale=(Y,)`；`actual_token_num=(1,)` | §1.3/§2.3 数据流 + §3.7 workspace（Y=A*BS*K）；shape 由现有 infershape 静态推导，A5 共用不改；硬校验阈值 §3.6.3（A≤1024/topK+1≤64） |
| ★ `numerical_tolerance.per_dtype` | float16/bfloat16/int8/int32/int64/float32 全部 `{rtol:0, atol:0, metric:bitwise_equal}` | §1.3 精度口径"二进制一致"；§3.1 决策"原样字节 gather 无算术"直接满足 bitwise_equal；§3.5 三步无浮点运算 |
| ★ `boundary_conditions[]` | k=0(全 mask)、k=N(A=1024/topK+1=64 上界)、same_bin(expert_num=8192)、expert_num 越界(>8192)报错、topK+1>64 报错、A>1024 报错、max_out_shape 长度≠4 报错、dtype≠int8 报错、非 1 维报错、need_schedule=0(100)、need_schedule=1(101) | §3.6.3 硬校验阈值（EXPERT_IDX_MAX/MAX_SESSION_NUM/MAX_K_NUM + CheckInputParam dtype/维度）；§3.5 两路径 TilingKey 100/101；边界行为 A5 全对齐 A2（校验逻辑与代次无关） |
| ★ `extreme_inputs[]` | all_same_index(expert_num=8192 全同专家)、token_dtype=2(尾部 32B FP32 scale 连续 gather)、normal_sparse(~50% mask，need_schedule=1) | §3.3 P2 profile（BLOCK_BYTES=32 scale 连续搬运）；§3.5 sort 将 mask token 置极大值排末尾；穿刺用例见 PLAN 迭代一/三 |
| `op.category` / `op.paradigms` | `index_gather` / `[IndexGather, SortSelect, Histogram]` | §1.1 算子类别；§3.5 三步对应 Sort(SortSelect)→gather(IndexGather)→group_listing(Histogram) |
| `op.platform_constraints.supported_chips` | `[Ascend910B, Ascend950PR, Ascend950DT]` | §1.1 目标芯片；A5=Ascend950DT/PR，A2=Ascend910B（A3=Ascend910_93 本次范围外保留） |
| `math_semantics.formula` (textual_only) | 三步文字语义（sort→gather→group_listing） | §1.3 算法语义逐条对齐 |
| `determinism` | `required:true, accumulation_order:sequential, bitwise_reproducible:true`；排序 key=专家 ID+稳定序，group_list 整数精确累加，多核固定聚合序 | §3.4 SetAtomicAdd 整数精确累加；§3.5 稳定排序 + 固定核间分工；A5 沿用同一确定性策略 |
| `broadcast` | `kind:none` | 非算术算子无广播；§3.1/§3.5 |
| `numerical_stability` | `required:false, techniques:[]` | 非计算类无稳定性技术；§3.1 |

**冲突检查结论**：本设计逐字段核对 spec.yaml 与 REQUIREMENTS.md，**未发现任何 spec-owned 字段冲突**。REQUIREMENTS §4.4/§4.5/§4.6 与 spec `dtype_policy`/`boundary_conditions`/`shape_rule` 完全一致（SPEC_REVIEW.md 已判 ✅通过）。设计未从 REQUIREMENTS 重新解释覆盖任何 spec 字段。

---

## 5. arch22/arch35 符号隔离方案（强制章节）

综合 §3.2 / §3.6，四层隔离，确保 A2/A3(arch22) 与 A5(arch35) 共存无符号冲突、A2/A3 逻辑零改：

| 隔离层 | arch22（保留不动） | arch35（新增） | 隔离机制 | 冲突为何不发生 |
|--------|------|------|---------|---------|
| **① Kernel 入口符号** | `ffn_worker_batching.cpp` 的 `extern "C" ffn_worker_batching(11 参数)` | `ffn_worker_batching_apt.cpp` 的**同名同签名** `extern "C" ffn_worker_batching(11 参数)` | `def.cpp` `ExtendCfgInfo("opFile.value","ffn_worker_batching_apt")` 只对 ascend950 生效；A2/A3 默认取 `ffn_worker_batching.cpp` | 每个 soc 编译产物只含一个 opFile → 同名符号在各自 .o 中唯一，不同时链接 |
| **② def AddConfig** | `AddConfig("ascend910_93").AddConfig("ascend910b")` | 追加 `OpAICoreConfig regbaseCfg; regbaseCfg.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true).ExtendCfgInfo("opFile.value","ffn_worker_batching_apt"); AddConfig("ascend950", regbaseCfg);` | 三条 AddConfig 独立 soc 键 | 不同 soc 键，A2/A3 两行不动 |
| **③ Host tiling 注册** | monolithic `IMPL_OP_OPTILING(FfnWorkerBatching).Tiling(Tiling4FfnWorkerBatching)`（类体不动） | 新文件 `_tiling_arch35.cpp`：`REGISTER_OPS_TILING_TEMPLATE(FfnWorkerBatching, FfnWorkerBatchingRegbaseTiling, 1000)` + `IsCapable()=IsRegbaseSocVersion` | 分派器入口按 `IsRegbaseSocVersion` 分流（§3.6.1 方案 A）：ascend950→模板注册表(1000 档)，A2/A3→原 monolithic 路径 | SoC 运行时分流，1000 档只在 950 被遍历；arch22 函数符号不变 |
| **④ TilingData 注册** | `REGISTER_TILING_DATA_CLASS(FfnWorkerBatching, FfnWorkerBatchingTilingData)`（不动） | `REGISTER_TILING_DATA_CLASS(FfnWorkerBatching_1000, FfnWorkerBatchingRegbaseTilingData)` | op-type 名加 `_1000` 后缀区分 | 注册键不同，两 TilingData 类共存 |

> ⚠️ 唯一需触碰 arch22 文件的点：`Tiling4FfnWorkerBatching` 分派器函数顶部加 3 行 `IsRegbaseSocVersion` 守卫（§3.6.1）。**arch22 tiling 类 `FfnWorkerBatchingTiling` 本体、kernel、config 全部零改**。若评审要求 arch22 文件绝对零触碰，可改用方案 B 的完整模板化（代价见 §3.6.1），需另评审确认。

---

## 6. API 验证记录

> 本阶段 `ASC_DEVKIT_DIR` 未设置、在线文档检索后端不可用（见 LOG 环境检查）。API 可用性证据基于**已部署在 ascend950/arch35 上、可编译可运行的样板算子** `moe_init_routing_v3/op_kernel/arch35/`（同族 sort/gather/count 原语）——这是 DAV_3510 平台可用性的直接实证。开发阶段 devkit 就绪后须按 `ascendc-docs-search` 补充逐 API 文档变体核验。

| API | 平台验证（DAV_3510） | 样板实证位置 | dtype | 参数签名/约束 | 状态 |
|-----|------|------|------|------|------|
| `Sort<float, true>` | ✅ | `moe_v3_sort_one_core.h:98` | float(proposal) | `(dstProposal, srcValue, srcIndex, tmpBuffer, repeatTimes=sortNum/32)`；稳定排序 | ✅ 样板实证 |
| `Concat` | ✅ | `moe_v3_sort_one_core.h:93` | float | `(dst, srcValue, tmpBuffer, repeatTimes)`；proposal 8B | ✅ 样板实证 |
| `Extract` | ✅ | `moe_v3_sort_one_core.h:103` | float→value/index | `(dstValue, dstIndex, srcProposal, repeatTimes)` | ✅ 样板实证 |
| `MrgSort<float, true>` | ✅ | `moe_v3_mrgsort.h:133-141` | float | `(dst, MrgSortSrcList, elementCountList, sortedNums, validBitTail, exhaustion=1)`；禁 MrgSort4 | ✅ 样板实证 |
| `ArithProgression<int32_t>` | ✅ | `moe_v3_sort_one_core.h:48` | int32 | `(dst, start, stride, count)` | ✅ 样板实证 |
| `GetSortLen<float>`/`GetSortOffset<float>` | ✅ | `moe_v3_mrgsort.h:117,157` | - | `(n)→n×2`（float 数） | ✅ 样板实证 |
| `DataCopyPad`+`DataCopyExtParams`/`DataCopyPadExtParams<T>` | ✅ | `moe_v3_gather_out.h:176-178` | int8/int32/half | `(dst,src,ExtParams{blockCount,blockLen(byte),srcStride,dstStride,rsv},pad)`；非对齐必用 | ✅ 样板实证 |
| `DataCopy` | ✅ | `moe_v3_sort_multi_core.h:147` | float | `(dst,src,len)`；仅 32B 对齐 | ✅ 样板实证 |
| `TQue`/`TQueBind<VECIN,VECOUT,N>` | ✅ | `moe_v3_gather_out.h:47-49` | - | 双缓冲队列 | ✅ 样板实证 |
| `Duplicate` | ✅ | `moe_v3_expert_tokens_count.h:190,268` | int32 | `(dst,scalar,count)`；group_list 清零 | ✅ 样板实证 |
| `SetAtomicAdd<int32_t>()`/`SetAtomicNone()` | ✅ | `moe_v3_expert_tokens_count.h:231-233` | int32 | 包裹 DataCopyPad；整数精确累加保确定性 | ✅ 样板实证 |
| `SyncAll()` | ✅ | 本算子 arch22 + 样板通用 | - | 多核栅栏 | ✅ 通用 |
| `GetSortTmpSize`/`GetConcatTmpSize` | ⚠️ | A2 用法 | - | `(platform, tileSize, 4)` | ⚠️ 需 arch35 平台参数复核（开发期 devkit 验证） |
| `GetCoreNumAiv`/`GetCoreMemSize(UB)`/`GetLibApiWorkSpaceSize`/`IsRegbaseSocVersion` | ✅ | moe tiling_base + tiling_arch35 | - | host tiling 平台接口 | ✅ 样板实证 |

**验证检查清单**：
- [x] 已确认核心 sort/gather/count/搬运 API 在 DAV_3510 上可用（样板 moe_init_routing_v3/arch35 已部署运行）
- [x] 已确认支持所需 dtype（int8/fp16/bf16/fp32/int32/int64 字节搬运 + float proposal 排序）
- [x] 参数签名与样板实证一致
- [x] 已确认禁用 API：`MrgSort4`（用 `MrgSort<float,true>`）、`SetValue/GetValue`（用 DataCopyPad）
- [ ] `GetSortTmpSize/GetConcatTmpSize` 的 arch35 平台参数、DCCI 语义 → 开发期 devkit 就绪后逐 API 文档变体复核（§7 风险）
- [x] 无 API 不可用导致的方案阻塞（主路线全部经典 SIMD，样板已证 DAV_3510 可用）

---

## 7. 性能优化与风险评估

### 7.1 性能优化策略

- **动态获取核数/UB**：`GetBlockDim()`/`GetCoreNumAiv()` 动态取核数（禁写死）；arch35 UB=248KB > A2 192KB，`sortLoopMaxElement`/`PER_LOOP_ROWS`/`maxBlockSize_` 自动放大 → 单次排序/gather 批量更大、loop 更少（REQUIREMENTS §1 设计演进趋势）。
- **多核并行 + 双缓冲**：sort 多核切分 + MrgSort 归并；gather 与 group_listing 核间分工（部分核 gather、其余核 group_listing）；`TQueBind` 双缓冲流水 MTE2/V/MTE3。
- **本次性能为可选验收**（REQUIREMENTS §7：无量化性能指标）：以功能 + 精度对齐 A2 为主，性能达标为可选。

### 7.2 风险评估

| 风险 | 说明 | 严重度 | 缓解 |
|------|------|:---:|------|
| **ScheduleContext GM 解析 + DCCI 语义** | 1024B 结构体在 arch35 GM 指针/cache 模型下逐字节解析，`DataCacheCleanAndInvalid` 语义可能不同 | 中 | 参 `moe_v3_common.h` 指针解析；开发期核对 arch35 DCCI；迭代一穿刺 NORM P0 首验 |
| **Sort/MrgSort 元素上限 + tmpBuffer** | arch35 UB 不同、`GetSortTmpSize` 平台参数需复核；`MGR_SORT_MAX_ELEMENT` 类上限 | 中 | 阈值全经运行时 UB 推导（§3.6.3）；`GetSortTmpSize` 开发期 devkit 验证；禁写死 |
| **P2 int8 + 32B scale 连续 gather** | token_dtype=2 尾部 32B FP32 scale 搬运在 arch35 语义 | 中 | 参 `moe_v3_gather_dynamic_quant.h`；迭代三穿刺 P2 profile |
| **tiling 分派器守卫改动** | §3.6.1 方案 A 需在 arch22 分派器加 3 行 | 低 | 仅入口分流，不改 arch22 类体；UT 覆盖 A2/A3 回归确保零行为变化 |
| **内部切分阈值重算与 A2 功能等价** | UB 差异致多核/单核边界变化 | 低 | UT 覆盖边界 shape（A=1024/topK+1=64/expert_num=8192/空）；ST 对拍 A2 二进制一致 |
| **RegBase 误用** | 若后续引入 RegBase 优化 gather | 低 | 本次主路线不启用 RegBase；若启用须另做 RegBase 专项设计（`ascendc-regbase-best-practice`），保持字节语义不变 |

> 总体：技术路径经两个同域样板（moe_init_routing_v3 / moe_gating_top_k）在 ascend950 实证可行，无阻塞性风险；风险点均可在开发/穿刺阶段闭环。

---

## 8. 交付件清单

**必需（本次新增/改动）**：
- `op_host/ffn_worker_batching_def.cpp`（改：+ascend950 regbaseCfg AddConfig）
- `op_host/ffn_worker_batching_tiling.cpp`（改：分派器加 IsRegbaseSocVersion 守卫 3 行）
- `op_host/ffn_worker_batching_tiling.h`（改：追加 arch35 TilingData 注册）
- `op_host/ffn_worker_batching_tiling_arch35.cpp`（新增：1000 档 Regbase tiling）
- `op_host/config/ascend950/ffn_worker_batching.json` + `_simplified_key.ini`（新增）
- `op_kernel/ffn_worker_batching_apt.cpp`（新增：arch35 入口）
- `op_kernel/arch35/*.h`（新增：15 头移植 + 1 TilingData def，见 §3.2）
- `tests/`（UT/ST，测试设计阶段产出）

**不动**：`op_host/ffn_worker_batching_infershape.cpp`、arch22 `ffn_worker_batching.cpp` + 13 头、`config/ascend910b`、`config/ascend910_93`、`op_graph/`、`op_api/`（若有）。

**可选**：README 产品支持情况更新 Ascend950 为 √（文档阶段）。

---

## 9. 迭代规划

详见 `PLAN.md`。三迭代概览：

| 迭代 | 目标 | 代码 | 穿刺 |
|------|------|------|------|
| 迭代一 | 骨架搭建 | 单 TilingKey(100 NORM) + P0 fp16 骨架（apt 入口 + arch35 sort/gather/group_listing 主链 + 1000 档 tiling） | NPU 实跑穿刺 RECV(101)/多核 sort/P2 int8/P1 bf16 分支 |
| 迭代二 | 策略整合 | NORM+RECV 双 TilingKey 整合 + 单/多核 sort 全路径 | 穿刺迭代三难点（P2 32B scale、边界 shape、DCCI） |
| 迭代三 | 全量覆盖 | 3 profile × 2 路径全功能 + 边界 | 全 UT/ST 覆盖，NPU 实跑对拍 A2 二进制一致 |





