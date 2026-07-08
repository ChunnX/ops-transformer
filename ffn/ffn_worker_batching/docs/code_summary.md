# 代码概要

算子: **FfnWorkerBatching** (Ascend950 / arch35 / DAV_3510 全量移植) | 功能: MoE worker-batching 调度算子——按专家分组排序 token，gather token 数据到 y 并生成 group_list（专家→token 计数）及每 token 元数据 | 侧别: **混合（Kernel + Tiling）**

> 本次为多文件检视。file_input 覆盖：arch35 kernel 头 16 个 + arch35 kernel 入口 `ffn_worker_batching_apt.cpp`（Kernel 侧）、`ffn_worker_batching_tiling_arch35.cpp`（arch35 Tiling 侧）、`ffn_worker_batching_tiling.cpp`（仅 IsRegbaseSocVersion 分派守卫在检视范围内，A2/A3 monolithic tiling 类不在范围）。
> arch35 是 A2 的忠实全量移植：多数 kernel 头与 A2 逐字节一致，仅 `ffn_wb_group_listing.h`（k=0 守卫）与 tiling def / 入口做了 arch35 适配。

---

## 侧别识别（Step 2 结论）

| 文件 | 侧别 | 判定依据 |
|------|------|---------|
| `op_kernel/ffn_worker_batching_apt.cpp` | Kernel（入口） | `__global__ __aicore__`、`GetUserWorkspace`、`GET_TILING_DATA_WITH_STRUCT`、`TILING_KEY_IS` |
| `op_kernel/arch35/*.h`（15 个计算头 + 1 个 tiling def） | Kernel | `__aicore__`、`AscendC::`、`pipe->InitBuffer`、`TQue`/`TBuf` |
| `op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h` | ABI（host/kernel 共用平铺 struct） | 纯 struct 定义，无逻辑 |
| `op_host/ffn_worker_batching_tiling_arch35.cpp` | Tiling | `gert::TilingContext`、`TilingBaseClass`、`GetTilingData<...>()`、`REGISTER_OPS_TILING_TEMPLATE` |
| `op_host/ffn_worker_batching_tiling.cpp`（仅分派守卫） | Tiling（路由） | `Tiling4FfnWorkerBatching` + `IsRegbaseSocVersion` 分派 |

---

## 代码脉络

**入口**: `ffn_worker_batching`（apt.cpp:173，`__global__`）→ 由 runtime 按 tiling 拉起 → `TILING_KEY_IS` 分派：
- `TILING_KEY_NORM`(100) → `FfnWorkerBatchingProcess`（apt.cpp:49）
- `TILING_KEY_RECV`(101) → `FfnWorkerBatchingRecvProcess`（apt.cpp:113）
- 102(RECV_GATHER，A2 保留分支)在 arch35 **不分发**（tiling 从不 SetTilingKey(102)）。
- AIC 核 / workspace 为空 直接 return（apt.cpp:179-188）。

**数据流**:
`schedule_context`(GM，1024B struct) → `ScheduleContextInfoCompute` 搬入 UB 解析出 A/M/BS/K/HS + buffer 指针 → 三步：
1. **sort**：GM `expert_ids`(或 token_info 内嵌) → UB 排序 → workspace `sortedexpertIds` / `sortedRowIds`
2. **ValidGatherIdxLengthCompute**：读 workspace[0] 有效长度 → 写 `actual_token_num`
3. **gather / group_listing**：workspace `sortedRowIds` → `y`(token 数据) + `session_ids/micro_batch_ids/token_ids/expert_offsets/dynamic_scale` + `group_list`

**计算核心**: 三步算法（NORM 与 RECV 同构，RECV 前置 scan）：

| 步骤 | NORM(100) | RECV(101) |
|------|-----------|-----------|
| 前置 | 无 | `KernelScanTokenInfo`（block0 忙等当前 micro batch ready，推进 pollIdx）→ SyncAll |
| step1 sort | `TilingSort` → 单核 `KernelSortMaskOneCore` / 多核 `KernelSortMaskMultiCore`(+MrgSort) | `TilingScanSort` → 单核 `KernelScanSortMaskOneCore` / 多核BsK `KernelScanSortMaskMultiCoreBsK`(isNormalBsK!=1) / 多核 `KernelScanSortMaskMultiCore` |
| step2 gather | `KernelFfnWBGatherOutAll<false>` | `KernelFfnWBGatherOutAll<true>` |
| step3 group_listing | `KernelFfnWBGroupListing`（多核 or 单核，按 validGatherIdxLength>512 分流）| `KernelFfnWBScanGroupListingOneCore`（保留 1 核单核）|

**分支覆盖**:
| 分支条件 | 位置 | 触发场景 | 处理逻辑 | 涉及 API |
|---------|------|---------|---------|----------|
| `TILING_KEY_IS(100/101)` | apt.cpp:194-200 | need_schedule=0/1 | NORM vs RECV 两条 Process | GET_TILING_DATA_WITH_STRUCT |
| `tilingdataSort.tilingKey==ONECORE_SORT` | apt.cpp:62,134 | totalLength ≤ sortLoopMaxElement | 单核排序 | — |
| `tilingdataSort.isNormalBsK != NORMAL_BSK` | apt.cpp:139 | BS*K+pad > UB（单个 session 放不下） | BsK 切分多核排序 | — |
| `validGatherIdxLength > GROUP_LISTING_MULTI_CORE_LENGTH(512)` | apt.cpp:83 | 大 shape | group_listing 转 4 核并行 + ProcessExpertCount | SyncAll |
| `GetBlockIdx() < coreNum - usedCoreNum` | apt.cpp:89,156 | 核间分工 | 前段核 gather，后段核 group_listing | — |
| **k=0 守卫** `validGatherIdxLength > 0 && ...` | **group_listing.h:162** | validGatherIdxLength==0（无有效 token） | 跳过 lastExpertId flush（避免写出脏专家条目），仅由 [0,0] 终止符产出 group_list[0]=[0,0] | SetValue/DataCopyPad |
| `tokenDtype == 2`(TOKEN_KIND_TWO) | gather_out_all.h:97,219,241 | int8 动态量化 | 额外搬运 dynamic_scale（int8 数据+scale 连续排布） | Gather/ArithProgression |

**核心 API**: `DataCopyPad` `Sort<float,true>` `MrgSort<float,true>` `Concat` `Extract` `GatherMask` `CompareScalar` `Cast` `Muls` `Duplicate` `ReduceSum` `ArithProgression` `InitGlobalMemory` `SyncAll` `SetAtomicAdd/None` `DataCacheCleanAndInvalid` `SetFlag/WaitFlag`

**输出**: `y`(int8/half token 数据)、`session_ids/micro_batch_ids/token_ids/expert_offsets`(int32)、`dynamic_scale`(float, 仅 dtype2)、`group_list`(int64 [id,count])、`actual_token_num`(int64)。同步：EnQue/DeQue（MTE↔V）、SetFlag/WaitFlag（手动流水），跨阶段 SyncAll。

---

## 算子业务语义（Kernel 侧）

**数学运算**: 非计算类调度算子。核心是 argsort：`sorted, rowIdx = argsort(expert_ids)`（按专家 id 升序，同时输出稳定 gather 索引），后接 gather：`y[i] = token_data[route(rowIdx[i])]`，及 run-length 计数：`group_list[e] = count(expert_ids == e)`。
**输入输出**: 1 主结构输入(schedule_context) + attr → 6~7 输出（y / 4 类 idx / dynamic_scale / group_list / actual_token_num）。
**计算模式**: 混合——sort 主体为 **Multi-Step Vector Decomposition**（Cast→Muls→CompareScalar→GatherMask→Concat→Sort→Extract 链式）+ 多核归并为 **N-way MrgSort 树**（每级 4 路，跨核 SyncAll 同步）；gather 为 **Double Buffer Vector Pipeline**（BUFFER_NUM=2）。
**同步契约**:
- `EnQue`/`DeQue` → MTE↔Vector 阶段交接（各 sort/gather 内部）
- `SetFlag/WaitFlag`（HardEvent，经 `SetWaitFlag` 封装）→ 手动流水重叠（MTE2_S / S_MTE3 / V_MTE3 / MTE2_MTE3 等）
- `SyncAll` → 跨核屏障（sort 分级归并、gather/group_listing 分工后）
- `SetAtomicAdd` → 多核 group_listing 专家计数原子累加、多核 rsvdCnt 汇总

### 分支业务含义
| 分支 | 位置 | 业务含义 |
|------|------|---------|
| NORM vs RECV | apt.cpp:194 | need_schedule=0 直接调度 / =1 需先扫描接收 token ready（推理服务 micro-batch 流水） |
| isNormalBsK | apt.cpp:139 / common.h:314 | 单个 session 的 BS*K+pad 是否超 UB：超则按 BS*K 再切块排序（BsK 变体） |
| k=0 守卫 | group_listing.h:162 | 当前 micro batch 无有效 token（validGatherIdxLength==0）时，`ComputeOneCore` 对 0 字节搬运后的 UB 脏值 `GetValue(0)` 会得到伪 expertId；守卫跳过 flush，只输出 [0,0] 终止符，对齐 spec k_zero 契约 |
| tokenDtype 0/1/2 | 全局 | 0=FP16, 1=BF16, 2=int8 动态量化（token 数据+dynamic_scale 连续）；仅 dtype2 产出 dynamic_scale 且 tokenDtypeSize=1B |

### 模板参数语义
| 参数 | 取值 | 业务含义 |
|------|------|---------|
| `isScanMode`（ScheduleContextInfoCompute）| false/true | NORM 解析 session/microBatch/expertIds buf + outNum；RECV 解析 tokenInfoBuf + pollIdx + 算 BsKPaddingCount |
| `isScanFlag`（KernelFfnWBGatherOutAll）| false/true | false 用 session_ids/micro_batch_ids 查表；true 直接用 aIndices/curMicroBatchID |

---

## Tiling 业务语义（Tiling 侧）

**切分策略**:
- **sort（NORM）**: 按元素数切分，needCoreNum 向上取到 4 的幂（便于 4 路 MrgSort），不超物理核数（common.h:TilingMultiCoreSort）。
- **sort（RECV）**: 按 A（session）切分（common.h:TilingScanMultiCoreSort）。
- **gather / group_listing**: 按 validGatherIdxLength 均分到核。
**Buffer 策略**: gather 双缓冲（BUFFER_NUM=2）；sort 单缓冲（bufferNum=1）。UB 容量运行时经 `GetCoreMemSize(UB)` 取值，sortLoopMaxElement 由 ubSize 派生。
**TilingKey 轴**: 仅 need_schedule（0→100 NORM / 1→101 RECV）。token_dtype 与 TilingKey **正交**，kernel 内运行时分支处理（tiling_arch35.cpp:241 注释）。

### 校验策略（arch35 tiling，GetShapeAttrsInfo → CheckInputParam + GetAttrsInfo）
| 校验条件 | 位置 | 数学不变量 |
|---------|------|-----------|
| x dtype == INT8 | tiling_arch35.cpp:124 | 输入必须 int8 |
| x shape dim == 1 | tiling_arch35.cpp:131 | 输入 1D |
| 0 < expert_num ≤ 8192 | tiling_arch35.cpp:143 | 专家数上界 EXPERT_IDX_MAX |
| max_out_shape.size == 4 | tiling_arch35.cpp:150 | [A,BS,K,H] 四维 |
| 0 < A ≤ 1024 | tiling_arch35.cpp:158 | session 数 ≤ MAX_SESSION_NUM |
| BS > 0；0 < K ≤ 64；H > 0 | tiling_arch35.cpp:161-169 | K ≤ MAX_K_NUM |
| token_dtype ∈ [0,2] | tiling_arch35.cpp:175 | 三种 dtype |
| need_schedule ∈ [0,1]；0 ≤ layer_num ≤ expert_num | tiling_arch35.cpp:183,191 | — |
| aivNum≠0；ubSize≠0 | tiling_arch35.cpp:106,111 | GetPlatformInfo 硬件保证 |

### 切分变量语义
| 变量 | 公式（tiling_arch35.cpp） | 业务含义 |
|------|------|---------|
| `Y` | A*BS*K（:171） | 总 token 工作量 |
| `coreNum` | aivNum；RECV 且 Y≤200000 时 min(aiv,32)（:213-217） | 实际使用核数（RECV 小 shape 限核） |
| `sortLoopMaxElement` | ubSize/(4·2·4)/32·32（:220） | 单次 UB 可排序 int32 数（32 对齐向下取整） |
| `sortNumWorkSpace` | Y（:230） | 记录每次排序后计数所需 workspace 元素数 |

### TilingKey 语义
| 轴 | 取值 | 业务含义 |
|----|------|---------|
| need_schedule | 0→100 / 1→101 | 是否需要 recv 扫描调度（tiling_arch35.cpp:239-243 GetTilingKey） |

### Workspace 数学来源（tiling_arch35.cpp:248 GetWorkspaceSize，与 A2 同公式）
`workspaceSize = 128·4 + Y·4 + Y·4·2·4 + expertNum·4 + sysWorkspaceSize`
- 前 128 int32：总有效专家数 + recv micro_batch_id 轮询
- Y int32：每核每 loop 排序后有效专家个数
- Y·8 int32：排序 scratch（sortedExpertIds + sortedRowIds + 两块 MrgSort ping-pong 各 2 倍 key-value）
- expertNum int32：group_list 原子加临时计数
- sysWorkspaceSize：`GetLibApiWorkSpaceSize()`（arch35 平台值）

---

## 变量溯源（TilingData → Kernel）

| 变量 | 声明 | 初始化（Host） | 校验 | 来源类型 |
|------|------|--------------|------|---------|
| Y | tiling_def.h:28 | tiling_arch35.cpp:171,223 = A*BS*K | 各维度 OP_CHECK_IF(:158-165) | TilingData |
| H | tiling_def.h:29 | tiling_arch35.cpp:157,224 = max_out_shape[3] | :167 H>0 | TilingData |
| tokenDtype | tiling_def.h:30 | tiling_arch35.cpp:178,225 | :175 ∈[0,2] | TilingData |
| expertNum | tiling_def.h:31 | tiling_arch35.cpp:146,226 | :143 (0,8192] | TilingData |
| coreNum | tiling_def.h:32 | tiling_arch35.cpp:213-217,227 | GetPlatformInfo aivNum≠0 | 硬件配置 |
| ubSize | tiling_def.h:33 | tiling_arch35.cpp:228 = GetCoreMemSize(UB) | :111 ≠0 | 硬件配置 |
| sortLoopMaxElement | tiling_def.h:34 | tiling_arch35.cpp:220,229 | 派生自 ubSize | 编译期派生 |
| sortNumWorkSpace | tiling_def.h:35 | tiling_arch35.cpp:230 = Y | — | TilingData |
| A/M/BS/K/HS | common.h:76-79 | get_schedule_context.h:50-69（从 schedule_context GM 解析） | ASSERT_MSG A≤1024/M≤64/K≤64/curMicroBatchID<M | 外部输入（GM struct）|
| curMicroBatchID | common.h:82 | get_schedule_context.h:116（pollIdx GM 读） | ASSERT_MSG <M（:117） | 外部输入 |
| BsKPaddingCount | common.h:92 | get_schedule_context.h:120 = Align(BS*K)-BS*K | — | 编译期派生 |
| validGatherIdxLength | common.h:91 | common.h:331（读 workspace[0]） | — | 运行时（sort 产出） |

---

## 函数清单（按文件，供检视定位）

**apt.cpp（入口）**
| 函数 | 行范围 | 角色 |
|------|--------|------|
| FfnWorkerBatchingProcess | 49-110 | NORM 入口 |
| FfnWorkerBatchingRecvProcess | 113-171 | RECV 入口 |
| ffn_worker_batching | 173-201 | __global__ 分派 |

**ffn_wb_common.h（共享工具/结构）**: `ScheduleContextInfo`/`SortCustomTilingDataKernel`/`FfnWBGroupListingTileInfo` struct；`Ceil`(141) `Align`(149) `Min`(157) `Max`(163) `CeilDiv`(169) `PowerOfFourCeil`(180) `SetWaitFlag`(133)；`TilingOneCoreSort`(193) `TilingMultiCoreSort`(206) `TilingScanMultiCoreSort`(258) `TilingSort`(287) `TilingScanSort`(301) `ValidGatherIdxLengthCompute`(320) `Tiling4SrcToDstCompute`(340)

**ffn_wb_get_schedule_context.h**: `ScheduleContextInfoCompute<isScanMode>`(34) — 解析 schedule_context

**ffn_wb_scan_token_info.h**: `KernelScanTokenInfo::Init`(68)/`Process`(94，block0 忙等 micro batch ready)/`CopyInFlag`(57)

**ffn_wb_sort_base.h**: `SortMaskBase` 基类（成员 + 常量，expertStart_=1000000 无效值门限）

**ffn_wb_sort_one_core.h**: `KernelSortMaskOneCore` Init(150)/Process(189)/CopyIn(51)/SortCompute(64)/CopyOut(136)

**ffn_wb_sort_multi_core.h**: `KernelSortMaskMultiCore` Init(349)/Process(411)/VBSProcess(265)/VMSProcess(285)/SortOutProcess(321)/UBSortCompute(100)/CopyOutValidCount(336) 等

**ffn_wb_sort_mrgsort.h / _out.h**: `SortCustomMrgsort`(36)/`SortCustomMrgsortOut`(30) — 4 路归并核心（CopyIn/MrgsortCompute/UpdateSortInfo/CopyOut，_out 额外 Extract 反算 int32）

**ffn_wb_scan_sort_one_core.h**: `KernelScanSortMaskOneCore` Init(179)/Process(212)/CopyInAndClear(51，搬入并清 token_info flag)/SortCompute(88)

**ffn_wb_scan_sort_multi_core.h**: `KernelScanSortMaskMultiCore` Init(404)/Process(494)/VBSProcess/VMSProcess/SortOutProcess/ClearTokenInfoFlag(466) — 按 A 切分

**ffn_wb_scan_sort_multi_core_bsk.h**: `KernelScanSortMaskMultiCoreBsK` Init(450)/Process(555) — 单 session 的 BS*K 超 UB 时按块切分（splitNum_/perCorePad_/lastCorePad_）

**ffn_wb_gather_out_all.h**: `KernelFfnWBGatherOutAll<isScanFlag>` Init(38)/Process(137)/CopyXIn(232)/CopyXOut(251)/CopyAllLocalOut(209) — gather token 数据 + 元数据

**ffn_wb_group_listing.h**: `KernelFfnWBGroupListing` Init(88)/Process(142，含 **k=0 守卫:162**)/ProcessExpertCount(184)/ComputeOneCore(299)/expertCountCompute(255)

**ffn_wb_scan_group_listing_one_core.h**: `KernelFfnWBScanGroupListingOneCore` Init(34)/Process(58)/Compute(104)

**tiling_arch35.cpp**: `FfnWorkerBatchingRegbaseTiling`（TilingBaseClass 子类）GetPlatformInfo(97)/GetShapeAttrsInfo(199)/DoOpTiling(208)/GetTilingKey(239)/GetWorkspaceSize(245)/PostTiling(253)/CheckInputParam(119)/GetAttrsInfo(137)；`REGISTER_OPS_TILING_TEMPLATE(...,1000)`(271)

**tiling.cpp（仅分派守卫）**: `Tiling4FfnWorkerBatching`(237) — IsRegbaseSocVersion→1000 模板 / 否则 monolithic

---

## API 调用索引（关键定位）

| API | 位置 | 上下文 |
|-----|------|--------|
| Sort<float,true> | sort_one_core.h:118, sort_multi_core.h:150, scan_sort_one_core.h:146, scan_sort_multi_core.h:187, scan_sort_multi_core_bsk.h:206 | UB 内 32 对齐块排序（升序） |
| MrgSort<float,true> | mrgsort.h:142/146/150, mrgsort_out.h:158/162/166 | 4 路归并 |
| Concat / Extract | sort_one_core.h:111/124, mrgsort_out.h:196 | Sort 前打包 / 后拆包 |
| GatherMask | sort_*.h（掩码压缩无效专家 id）, scan_group_listing_one_core.h:139, scan_get_valid_experts.h:259/294 | 按 mask 压缩 |
| CompareScalar | sort_*.h（CMPMODE::GT vs -expertStart_）, scan_group_listing_one_core.h:127（EQ） | 生成有效性/相等掩码 |
| Cast / Muls | sort_*.h | int32↔float 转换 + 取负（升降序反转） |
| DataCopyPad | 全体 | GM↔UB 非对齐搬运 |
| DataCopyPadExtParams{true,0,pad,INT_MAX} | scan_sort_*.h:59/108/116 | RECV 场景 BsK 尾部补 INT_MAX（无效专家）|
| InitGlobalMemory | sort_one_core.h:171, sort_multi_core.h:378/382, scan_token_info.h:82 | 清零 workspace/groupListTmp |
| SetAtomicAdd/None | group_listing.h:237/239, sort_multi_core.h:343/345, scan_sort_*.h CopyOutValidCount | 多核计数汇总 |
| SyncAll | 各 Process 末尾 + apt.cpp:106,128 | 跨核屏障 |
| DataCacheCleanAndInvalid | common.h:329, sort_one_core.h:147, mrgsort.h:202, scan_token_info.h:126 | GM 读回前刷 cache |
| ReduceSum | scan_token_info.h:107 | 统计 ready session 数 |
| ReduceMin/Max | scan_get_valid_experts.h:263/264 | （仅未接入的 valid_experts 路径）|

---

## 常量清单（关键）

| 常量 | 值 | 位置 | 用途 |
|------|-----|------|------|
| TILING_KEY_NORM / RECV | 100 / 101 | apt.cpp:42-43, tiling_arch35.cpp:44-45 | 双 TilingKey |
| expertStart_ | 1000000 | sort_base.h:49 | 无效专家 id 门限（≥ 视为无效/pad） |
| ONE_REPEAT_SORT_NUM | 32 | common.h:44 | Sort 单 repeat 元素数（32 对齐） |
| ONE_REPEAT_COMPARE_NUM | 64 | common.h:52 | CompareScalar 单 repeat 数 |
| MRG_LIST_NUM / MAX_MRGSORT_LIST | 4 | common.h:35 / 各 sort 类 | 归并路数 |
| NORMAL_BSK | 1 | common.h:40 | BS*K+pad ≤ UB 标志 |
| GROUP_LISTING_MULTI_CORE_LENGTH | 512 | common.h:58 | group_listing 转多核阈值 |
| GROUP_LISTING_MULTI_AIV_NUM / AIV_NUM / SCAN_AIV_NUM | 4 / 1 / 1 | common.h:59-61 | group_listing 用核数 |
| OFFSET_SORTED_EXPERT_IDS | 128 | common.h:63 | workspace 前 128 int32 保留 |
| SCAN_BATCHID_GM_OFFSET | 32 | common.h:64 | scan 阶段 rsvdCnt/batchid 区 |
| MIN_FP32 | -3.4e38 | common.h:43 | 排序 pad 填充值 |
| TH_RECV_CORE_NUM / TH_RECV_Y_NUM | 32 / 200000 | tiling_arch35.cpp:52-53 | RECV 限核阈值 |
| EXPERT_IDX_MAX / MAX_SESSION_NUM / MAX_K_NUM | 8192 / 1024 / 64 | tiling_arch35.cpp:49-51 | 校验上界 |
| STRUCT_LENGTH | 1024 | get_schedule_context.h:32 | schedule_context 解析长度(B) |
| BLOCK_BYTES / BLOCK_SIZE | 32 | common.h:53 等 | 32B 对齐粒度 |

---

## 跨文件防御摘要

| 关联文件 | 关键发现 | 位置 | 影响范围 |
|---------|---------|------|---------|
| ffn_wb_group_listing.h | **k=0 守卫**：`validGatherIdxLength > 0 &&` 前置于 lastExpertId flush（arch35 新增，A2 无此条件） | :162 | validGatherIdxLength==0 时避免写脏 expertId；ComputeOneCore(:302) 仍会读脏值但不落盘 |
| get_schedule_context.h | A≤1024 / M≤64 / K≤64 / curMicroBatchID<M 的 ASSERT_MSG（运行期，非 host 校验） | :51,56,65,117 | 约束 sort/gather 所有偏移计算 |
| sort_base.h | expertStart_=1000000 全局无效门限；INT_MAX/BLOCK_SIZE 常量 | :49,58,59 | 所有 sort/scan 类继承，掩码逻辑依赖 |
| ffn_wb_common.h | TilingSort/TilingScanSort 在 kernel 侧运行时算 needCoreNum/tilingKey（非 host tiling） | :287,301 | 决定 apt.cpp 的单/多核分支 |
| ffn_worker_batching_arch35_tiling_def.h | host/kernel 共用平铺 struct（非 TILING_DATA_FIELD_DEF 宏），字段与 A2 FfnWorkerBatchingTilingData 一一对应 | 全文件 | ABI 契约；host 经 GetTilingData<...>() 直写，kernel 经 GET_TILING_DATA_WITH_STRUCT 读 |
| tiling.cpp | **IsRegbaseSocVersion 分派守卫**：arch35→TilingRegistry(1000)；A2/A3→monolithic RunFfnWorkerBatchingTiling | :241-245 | 决定 arch35 走 tiling_arch35.cpp；A2/A3 零回归 |

---

## 芯片架构参数（arch35 运行时取值，禁写死）

| 参数 | 值 | 来源 | 影响范围 |
|------|-----|------|---------|
| aivNum | 运行时 | GetCoreNumAiv()（tiling_arch35.cpp:105） | coreNum、多核切分 |
| ubSize | 运行时 | GetCoreMemSize(UB)（tiling_arch35.cpp:110） | sortLoopMaxElement、所有 InitBuffer 分配上限 |
| sysWorkspaceSize | 运行时 | GetLibApiWorkSpaceSize()（tiling_arch35.cpp:115） | workspace 总大小 |
| 对齐 | 32B / 32 元素 | BLOCK_BYTES / ONE_REPEAT_SORT_NUM | DataCopyPad 边界、Sort repeat |

> 符合 memory 约束「UB 容量/核数运行时接口查询，禁写死 arch 常量」——tiling_arch35.cpp 全程用 platform API，无 arch35 数值常量硬编码。

---

## 代码关联 / 跨文件关系

| 关系类型 | 源 | 目标 | 内容 | 位置 |
|---------|----|----|------|------|
| include | apt.cpp | 12 个 arch35 头 | 全量 kernel 头 | apt.cpp:29-40 |
| include | tiling_arch35.cpp | tiling_def.h（跨 op_kernel） | 共用平铺 struct | tiling_arch35.cpp:25 |
| 数据流 | tiling_arch35.cpp | apt.cpp/所有 kernel | FfnWorkerBatchingArch35TilingData（8 字段） | DoOpTiling:223-230 → apt.cpp:191 |
| 数据流 | sort 类 | gather/group_listing | workspace sortedExpertIds/sortedRowIds + validGatherIdxLength | apt.cpp:77,91 |
| 共享 | ffn_wb_common.h | 全体 kernel | 常量/工具/结构 namespace FfnWbBatching | common.h |
| 函数调用 | sort_multi_core.h | mrgsort.h/_out.h | SortCustomMrgsort(Out) 4 路归并 | :63,328 |

---

## ⚠️ 检视重点提示（供后续检视阶段）

1. **k=0 守卫（group_listing.h:162）**：arch35 相对 A2 唯一的逻辑改动。`ComputeOneCore`（:299-318）在 inputNum=0 时循环体不执行、`lastExpertId` 仍在 :303 被赋为 0 字节搬运的 UB 脏值。守卫保证不落盘，但需确认：(a) `firstExpertId`(:304) 同被赋脏值是否有其他消费者；(b) 该守卫仅覆盖单核 group_listing 路径（groupListingDealFlag==0），多核路径（Process:143-150 CopyIn/Compute/CopyOut）与 scan 单核路径（scan_group_listing_one_core.h）的 k=0 行为是否一致。
2. **`ffn_wb_scan_get_valid_experts.h`（KernelScanGetValidExperts）为死代码**：该 396 行类在 arch35 目录内定义，但 **apt.cpp 未 include、任何被 include 的头也未引用**（grep 确认仅在自身文件出现）。系 A2 忠实移植遗留的备选 sort 路径，arch35 RECV 实际走 One/MultiCore/MultiCoreBsK 三分支。检视时可标注「未接入，不影响运行时」。
3. **双 TilingKey 与 token_dtype 正交**：TilingKey 仅由 need_schedule 决定；dtype 0/1/2 在 kernel 内运行时分支（gather_out_all.h:47/59/97）。检视 dtype2(int8 动态量化)的 dynamic_scale 连续排布搬运（:241-245，`xSrcOffset + copySize` 读 scale）。
4. **IsRegbaseSocVersion 分派（tiling.cpp:241）**：确认 arch35 命中 Regbase 分支走 1000 模板，A2/A3 保持 monolithic 不回归。
5. **RECV 忙等（scan_token_info.h:100 while(true)）**：block0 轮询直到 readyNum==A 才 break，无超时保护——需确认上游契约保证 micro batch 必然 ready。

---

## 返回摘要

- 侧别: **混合（Kernel + Tiling）**
- 算子名: FfnWorkerBatching（arch35 / Ascend950 移植）
- 功能概述: MoE worker-batching 调度——按专家排序 token、gather token 数据与元数据、生成 group_list，支持 NORM/RECV 双路径与 token_dtype 0/1/2
- 入口: `ffn_worker_batching`(apt.cpp:173) → FfnWorkerBatchingProcess / FfnWorkerBatchingRecvProcess；Tiling 入口 `FfnWorkerBatchingRegbaseTiling`(1000 模板)
- 核心 API: Sort / MrgSort / GatherMask / CompareScalar / Concat / Extract / DataCopyPad / SetAtomicAdd / SyncAll
