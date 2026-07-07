# FfnWorkerBatching 需求文档（新增 Ascend950/A5 支持）

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人(gitId) |
|-----|---------|---------|-------|
| v1.0 | 初始版本：为已支持 A2/A3 的 FfnWorkerBatching 新增 Ascend950(arch35/A5) 支持，支持范围对齐 A2 | 2026-07-07 | CANNBot |

## 1. 需求背景

- **需求来源**：为 ops-transformer 仓已有算子 `FfnWorkerBatching`（MoE/FFN 分离部署域）新增 Ascend950(arch35/A5) 支持。该算子当前已支持 Atlas A2(Ascend910B)/A3(Ascend910_93)，本次在 A2/A3 分支保留不动的前提下，新增 A5 全新代次路径，A5 支持范围对齐现有 A2 实现。
- **基线对齐**：
  - [x] 用户给定实现（对标现有 A2/A3 生产实现，dtype/shape/输入输出/功能路径与 A2 一致）
  - [ ] 框架 API
  - [ ] 论文公式

### 模型结构分析

**适用场景**：融合算子场景（Sort + Gather + GroupList 三步融合，MoE/FFN 分离部署）

- **模型结构分析**：Attention 与 FFN 分离部署（disaggregated prefill/decode，PD 分离的 MoE 变体）。Attention worker 按专家路由将 token 发送到对应 FFN worker 的预分配数据区，`FfnWorkerBatching` 在 FFN worker 上从调度上下文中扫描/读取 token，按专家维度聚合并重排，产出各专家连续 token 数据块，供后续 GroupedMatmul（FFN 前向）消费。
- **设计演进趋势**：融合排序 / gather / group_list 三步为单算子，减少 kernel launch 与中间 IO；A5 代次借助 arch35 更大 UB（容量运行时经 `GetCoreMemSize(UB)` 取，不写死）与新指令模型进一步降低搬运开销。

## 2. 运行环境

| 项目 | A2（现有） | A3（现有） | A5（本次新增） |
|-----|-----------|-----------|--------------|
| 服务器型号 | Atlas A2 训练/推理系列 | Atlas A3 训练/推理系列 | Atlas A5 训练/推理系列 |
| 芯片号 | Ascend910B | Ascend910_93 | Ascend950（DT/PR）；实测 Ascend950DT_9584 |
| 编译宏架构 | DAV_2201（arch22） | DAV_2201（arch22） | DAV_3510（arch35）|
| SocVersion 配置名 | ascend910b | ascend910_93 | ascend950 |

- **CANN 版本**：9.1.0（含 Ascend950 platform_config，NpuArch=3510）
- **驱动版本**：7.0.t9.0.B827
- **实机验证**：Ascend950DT_9584 设备可用（runtime `GetCurNpuArch()=DAV_3510`），满足 A5 真实 NPU 精度验收前提
- **芯片→架构映射依据**：`npu-arch` 技能——Ascend910B/Ascend910_93 → DAV_2201；Ascend950DT/PR → DAV_3510

> 运行时禁止硬编码核数/UB 容量，须经 `PlatformAscendC` 接口（`GetCoreNumAiv()` / `GetCoreMemSize(UB)`）动态获取，A5 与 A2/A3 保持同一取值逻辑。

## 3. 调用方式

| 调用方式 | 是否支持 | 说明 |
|---------|---------|------|
| GE IR 构图（图模式） | ✅ | 现有默认支持，A5 对齐 |
| ACLNN 单算子调用 | ✅ | 现有默认支持，A5 对齐 |
| torch_npu 单算子 | ❌ | 不涉及 |
| torch.compile 入图 | ❌ | 不涉及 |
| GE 图模式-静态 shape | ✅ | infershape 由 `max_out_shape` 属性静态推导，支持静态 shape |
| GE 图模式-动态 shape | ✅ | 输入 `UnknownShapeFormat=ND`，A5 config 需置 `DynamicShapeSupportFlag(true)` |

> 来源：README「约束说明」——"该接口支持图模式（GEIR）和单算子模式（aclnn）"；`op_graph/ffn_worker_batching_proto.h` REG_OP 定义。

## 4. 算子规格

### 4.1 基本信息

- **算子名称**：FfnWorkerBatching（snake：`ffn_worker_batching`；OpType：`FfnWorkerBatching`）
- **算子类别**：融合类（Sort 排序 + IndexGather 索引 gather + Scan/分组计数），无算术数值计算，属搬移/重排类
- **仓规划**：`ffn/ffn_worker_batching/`（MoE/FFN 分离部署域）
- **算法语义（三步计算流程）**：
  1. **排序生成 gather 索引**：对所有 token 的专家 ID（`expert_ids`）排序（被 mask 的无效 token 初始化为大值 `MIN_FP32` 对应的极值以排到末尾），生成按专家聚合的 gather 索引序列。
  2. **多核 gather 提取 token**：多核并行按 gather 索引从 `token_data` 提取 token 的 hidden states（`y`）与 dynamic scale（`dynamic_scale`，仅 token_dtype=2），同时查表得到每个输出 token 对应的 `session_id`、`micro_batch_id`、`token_id`、`expert_offset`。
  3. **单核扫描生成 group_list**：单核扫描排序后的专家 ID 序列，查找专家跳变点，生成 `group_list`（每个专家的 `[expert_id, expert_token_num]`），并统计 `actual_token_num`（有效 token 总数）。
- **数学/形状语义**（无浮点算术，全部为整数索引与字节搬运）：
  - 输出 token 数上限 $Y = A \times BS \times (topK+1)$，$A$=Attention worker 数，$BS$=micro batch size，$topK+1$=topK 加共享专家数。
  - `y[out_i, :] = token_data[gather_idx[out_i], :]`（按排序后专家聚合顺序重排的 hidden states）。
  - `group_list[e] = [expert_id_e, token_num_e]`，未使用专家填 `[0, 0]`。

### 4.2 输入输出规格

来源：`op_host/ffn_worker_batching_def.cpp`、`op_host/ffn_worker_batching_infershape.cpp`、`op_graph/ffn_worker_batching_proto.h`、README 参数表。

#### 输入

| 名称 | dtype | format | shape | 含义 |
|------|-------|--------|-------|------|
| schedule_context | INT8 | ND | `[1024]`（1 维，1024 字节调度上下文结构体） | 调度上下文，含 CommonArea/ControlArea/AttentionArea/FfnArea。算子从 FfnArea 读取 `token_info_buf`/`token_data_buf` 及 layer_id/session_id/micro_batch_id/expert_ids 等路由信息。 |

> tiling `CheckInputParam()` 强制：输入 dtype 必须 INT8、shape 必须 1 维。

#### 输出

| 名称 | dtype | format | shape | 含义 |
|------|-------|--------|-------|------|
| y | FP16 / BF16 / INT8（随 token_dtype） | ND | `[Y, H]` | 按专家排序重排后的 token hidden states |
| group_list | INT64 | ND | `[expert_num, 2]` | 每专家 `[expert_id, expert_token_num]`，未用专家填 `[0,0]` |
| session_ids | INT32 | ND | `[Y]` | 每个输出 token 对应的 Attention session ID |
| micro_batch_ids | INT32 | ND | `[Y]` | 每个输出 token 对应的 micro batch ID |
| token_ids | INT32 | ND | `[Y]` | 每个输出 token 在原始输入中的位置索引 |
| expert_offsets | INT32 | ND | `[Y]` | 每个输出 token 在其所属专家分组内的偏移 |
| dynamic_scale | FP32 | ND | `[Y]` | 动态量化 scale，仅 token_dtype=2 有效；token_dtype=0/1 时为空 tensor |
| actual_token_num | INT64 | ND | `[1]`（标量） | 所有专家有效 token 数之和 |

> infershape 输出：`y={Y,H}`、`group_list={expert_num,2}`、`session/micro_batch/token/expert_offsets/dynamic_scale={Y}`、`actual_token_num={1}`；其中 `Y=max_out_shape[0]*max_out_shape[1]*max_out_shape[2]`、`H=max_out_shape[3]`。

### 4.3 属性表

来源：`ffn_worker_batching_def.cpp` L69-73、`ffn_worker_batching_tiling.cpp` `GetAttrsInfo()`、README 参数表。

| 名称 | 类型 | 必选/可选 | 默认值 | 取值含义 / 约束 |
|------|------|----------|--------|----------------|
| expert_num | Int | REQUIRED | - | 本卡专家总数（= 每层本卡专家数 × layer_num）。取值范围 `(0, 8192]`；决定 group_list 大小 |
| max_out_shape | ListInt | REQUIRED | - | 输出 shape 上限，格式 `{A, BS, topK+1, H}`，长度必须为 4。用于推导 `Y=A*BS*(topK+1)` 与 `H` |
| token_dtype | Int | OPTIONAL | 0 | 0=FP16；1=BF16；2=dynamic_quant_int8（INT8 数据与 FP32 dynamic scale 连续排布）。取值范围 `[0,2]`；=2 时输出 dynamic_scale |
| need_schedule | Int | OPTIONAL | 0 | 0=仅 batching 不扫描数据；1=先扫描接收数据再 batching。取值范围 `[0,1]` |
| layer_num | Int | OPTIONAL | 0 | 层数，每层专家独立索引。取值范围 `[0, expert_num]` |

> `max_out_shape` 各分量约束（tiling `GetAttrsInfo()`）：`A=max_out_shape[0]∈(0,1024]`、`BS=max_out_shape[1]>0`、`topK+1=max_out_shape[2]∈(0,64]`、`H=max_out_shape[3]>0`。代码变量 `K_` 对应 `max_out_shape[2]`，即 topK+1（含共享专家）。

### 4.4 dtype 矩阵（逐 profile）

来源：`def.cpp` 三 profile DataType 列表 + `config/ascend910b|ascend910_93/ffn_worker_batching.json` 三 bin（fp16/bf16/int8）+ infershape `InferDataType`。A5 三 profile 全量对齐 A2/A3。

| profile | token_dtype | schedule_context | y | group_list | session_ids | micro_batch_ids | token_ids | expert_offsets | dynamic_scale | actual_token_num |
|---------|-------------|------------------|-----|-----------|-------------|-----------------|-----------|----------------|---------------|------------------|
| P0 (fp16) | 0 | INT8 | **FP16** | INT64 | INT32 | INT32 | INT32 | INT32 | FP32(空) | INT64 |
| P1 (bf16) | 1 | INT8 | **BF16** | INT64 | INT32 | INT32 | INT32 | INT32 | FP32(空) | INT64 |
| P2 (int8) | 2 | INT8 | **INT8** | INT64 | INT32 | INT32 | INT32 | INT32 | **FP32(有效)** | INT64 |

- 除 `y` 与 `dynamic_scale` 有效性外，其余输出 dtype 三 profile 恒定。
- `y` dtype 由 infershape 依 token_dtype 动态设置：0→FP16、1→BF16、其它→INT8。
- **A5 dtype 矩阵与 A2 全对齐**：三 profile 一致，无新增/删减 dtype（A5 不引入 arch35 独有的 FP8/MXFP8/HiF8 等窄类型，本次不扩展支持范围）。

### 4.5 功能路径矩阵（need_schedule × token_dtype）

来源：kernel `ffn_worker_batching.cpp`（TILING_KEY 分发）+ tiling `SetBatchingTilingKey()`。

| need_schedule | token_dtype | TilingKey | 有效路径 | 说明 |
|:---:|:---:|:---:|:---:|------|
| 0 | 0 (fp16) | 100 (NORM) | ✅ | 仅 batching：排序→gather→group_listing（`FfnWorkerBatchingProcess`） |
| 0 | 1 (bf16) | 100 (NORM) | ✅ | 同上，y=BF16 |
| 0 | 2 (int8) | 100 (NORM) | ✅ | 同上，y=INT8 + 输出 dynamic_scale |
| 1 | 0 (fp16) | 101 (RECV) | ✅ | 接收+batching：扫描 token_info→排序→gather→group_listing（`FfnWorkerBatchingRecvProcess`） |
| 1 | 1 (bf16) | 101 (RECV) | ✅ | 同上，y=BF16 |
| 1 | 2 (int8) | 101 (RECV) | ✅ | 同上，y=INT8 + 输出 dynamic_scale |

- **TilingKey 只由 `need_schedule` 决定**（0→100，1→101），token_dtype 在 kernel 内部由 `contextInfo.tokenDtype` 分支处理（`ffn_wb_gather_out_all.h` 按 tokenDtype 决定 `tokenDtypeSize_`、是否搬运 dynamic_scale）。
- **6 条组合全部有效**，A5 需全部对齐覆盖。
- ⚠️ **保留分支说明**：kernel 中定义了 `TILING_KEY_RECV_GATHER=102`（`FfnWorkerBatchingRecvSpecProcess`），但当前 `ffn_worker_batching_tiling.cpp` 只 `SetTilingKey(100/101)`，**从未设置 102**——该分支为已实现但未启用的保留路径。A5 移植时保持一致语义（可保留代码但不产生 102 的 tiling 分发），不作为对齐必需项。

### 4.6 shape 约束与关系式

来源：README「约束说明」+ tiling `GetAttrsInfo()`/`ScheduleContext` 结构 + `ffn_wb_common.h`。

- **核心关系式**：$Y = A \times BS \times (topK+1)$；`y.shape=[Y,H]`，`group_list.shape=[expert_num,2]`，其余 id 类输出 `.shape=[Y]`，`actual_token_num.shape=[1]`。
- **参数上限**（README + tiling 硬校验）：
  - $A$（Attention worker 数）≤ 1024（tiling `MAX_SESSION_NUM=1024`，硬校验 `A∈(0,1024]`）。
  - $M$（micro batch 数）≤ 64（README 约束；运行时取自 `ScheduleContext.common.micro_batch_num`）。
  - $topK+1$（`max_out_shape[2]`）≤ 64（tiling `MAX_K_NUM=64`，硬校验 `∈(0,64]`）。
  - $BS$（micro batch size）、$Y$：泛化无硬上限（受内存限制）。
  - $H$（hidden size）：泛化无硬上限。
  - `expert_num` ≤ 8192（tiling `EXPERT_IDX_MAX=8192`，硬校验 `∈(0,8192]`）；kernel 侧 `MAX_EXPERT_NUM=5120` 为片上暂存约束。
- **对齐/量化约束**：token_dtype=2 时，输入 INT8 数据与 FP32 scale 在 `token_data` 中连续排布（`ffn_wb_gather_out_all.h`：每 token 尾部额外 `BLOCK_BYTES=32` 字节承载 scale）。
- **A5 shape 约束与 A2 全对齐**：上述所有上限/关系式由 host infershape + tiling 决定，与代次无关，A5 完全对齐（阈值型内部切分参数因 UB 容量不同会重算，但对外可见 shape 支持范围不变，见 §4.8）。

## 5. ACLNN API 接口定义

### 5.1 接口声明（两段式）

```cpp
// 第一段：计算 workspace 大小
aclnnStatus aclnnFfnWorkerBatchingGetWorkspaceSize(
    const aclTensor *scheduleContext,
    int64_t          expertNum,
    const aclIntArray *maxOutShape,
    int64_t          tokenDtype,
    int64_t          needSchedule,
    int64_t          layerNum,
    aclTensor       *y,
    aclTensor       *groupList,
    aclTensor       *sessionIds,
    aclTensor       *microBatchIds,
    aclTensor       *tokenIds,
    aclTensor       *expertOffsets,
    aclTensor       *dynamicScale,
    aclTensor       *actualTokenNum,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);

// 第二段：执行计算
aclnnStatus aclnnFfnWorkerBatching(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

> 参数顺序遵循「INPUT → ATTR → OUTPUT → workspaceSize/executor」的 aclnn 生成规约，与 `def.cpp` 输入/属性/输出声明顺序一致。属性 token_dtype/need_schedule/layer_num 为 OPTIONAL，带默认值 0。

### 5.2 参数说明

| 参数名 | 类型 | 输入/输出 | 说明 |
|-------|------|----------|------|
| scheduleContext | const aclTensor* | 输入 | 1024 字节调度上下文，INT8，1 维 |
| expertNum | int64_t | 输入(属性) | 本卡专家总数，(0,8192] |
| maxOutShape | const aclIntArray* | 输入(属性) | `{A,BS,topK+1,H}`，长度=4 |
| tokenDtype | int64_t | 输入(属性) | 0=FP16/1=BF16/2=int8 动态量化，默认 0 |
| needSchedule | int64_t | 输入(属性) | 0=仅 batching/1=接收+batching，默认 0 |
| layerNum | int64_t | 输入(属性) | 层数，[0,expertNum]，默认 0 |
| y | aclTensor* | 输出 | 重排 token hidden states `[Y,H]`，dtype 随 tokenDtype |
| groupList | aclTensor* | 输出 | `[expertNum,2]`，INT64 |
| sessionIds/microBatchIds/tokenIds/expertOffsets | aclTensor* | 输出 | `[Y]`，INT32 |
| dynamicScale | aclTensor* | 输出 | `[Y]`，FP32，仅 tokenDtype=2 有效 |
| actualTokenNum | aclTensor* | 输出 | `[1]`，INT64 |
| workspaceSize | uint64_t* | 输出 | Device workspace 大小 |
| executor | aclOpExecutor** | 输出 | 算子执行器 |

### 5.3 约束与限制

- schedule_context 必须 INT8、1 维、1024 字节。
- `max_out_shape` 长度必须为 4，各分量满足 §4.6 上限。
- token_dtype=2 时输出 dynamic_scale 有效，否则为空 tensor。
- 无 broadcast 语义（非算术算子）。
- 输出 shape 完全由属性静态推导，与 schedule_context 运行时内容一致性由调用方保证。

## 6. 图模式 IR 定义

来源：`op_graph/ffn_worker_batching_proto.h`。

- **IR 算子名称**：`FfnWorkerBatching`
- **输入**：`schedule_context: TensorType({DT_INT8})`
- **输出**：`y({INT8,FP16,BF16})`、`group_list({INT64})`、`session_ids/micro_batch_ids/token_ids/expert_offsets({INT32})`、`dynamic_scale({FLOAT32})`、`actual_token_num({INT64})`
- **属性**：`expert_num(Int, REQUIRED)`、`max_out_shape(ListInt, REQUIRED)`、`token_dtype(Int,0)`、`need_schedule(Int,0)`、`layer_num(Int,0)`
- **动态 shape 支持**：是（输入 UnknownShapeFormat=ND；A5 config `DynamicShapeSupportFlag(true)`）
- **Restrictions**：proto 标注 `THIS FUNCTION IS EXPERIMENTAL`（生命周期：experimental）。A5 沿用该标注。

## 7. 性能要求

- **本次无量化性能指标**（LOG.md 未给出利用率/带宽/延迟硬指标）。
- **验收基线**：A5 功能 + 精度对齐 A2；性能达标验收为可选（LOG 6 阶段 3.2 标注"视需求是否含性能指标"）。
- 优化方向（非验收项）：利用 arch35 更大 UB（容量运行时查询，不写死）提升单次排序/gather 批量，减少 loop 次数。

## 8. 约束与要求

### 8.1 计算约束

- 无浮点算术，输出为整数索引 + 字节搬运，结果须与 CPU golden **二进制一致**。
- 排序需保证被 mask 的无效 token 排到末尾（初始化为极大值），有效 token 按专家 ID 升序稳定聚合。

### 8.2 资源约束

- workspace 由 tiling `SetBatchingWorkspaceSize()` 计算：`128*sizeof(int32) + Y*sizeof(int32) + Y*sizeof(int32)*8 + expert_num*sizeof(int32) + sysWorkspaceSize`（A5 同公式，`sysWorkspaceSize` 经 `GetLibApiWorkSpaceSize()` 取平台值）。
- 核数、UB 容量运行时经 `PlatformAscendC` 动态获取（`GetCoreNumAiv()` / `GetCoreMemSize(CoreMemType::UB, ubSize)`），**禁止硬编码任何 arch 相关的 UB 字节数或核数常量**；A5 与 A2/A3 共用同一取值逻辑，仅返回值随平台不同。
- need_schedule=1 且 Y ≤ 200000 时限制使用核数 ≤ 32（tiling `TH_RECV_CORE_NUM=32`/`TH_RECV_Y_NUM=200000`）——A5 阈值可复用或按 arch35 重标定（设计阶段决策）。

### 8.3 确定性计算

**适用场景**：含排序（Sort/MrgSort）+ 分组扫描。

- **默认支持**：是。相同 schedule_context 输入必须产出完全相同的 y / group_list / ids（二进制一致）。
- **累加顺序**：group_list 计数为整数精确累加，多核切分后需保证聚合顺序确定。
- **数值精度**：无浮点误差（整数 + 原样字节搬运）。
- **性能影响**：确定性由算法结构天然保证，无额外开销。
- **实现策略**：排序 key 使用专家 ID + 稳定序；A5 沿用同一确定性策略。

### 8.4 特殊约束

- token_dtype=2：INT8 token 数据与 FP32 dynamic scale 在 `token_data` 中连续（每 token 尾部 32B 承载 scale），gather 时一并搬运。
- A2/A3 分支保留不动：`def.cpp` 现有 `AddConfig("ascend910_93")/("ascend910b")` 不删改；`config/ascend910b`、`config/ascend910_93` 不改；现有 arch22 kernel/tiling 逻辑不改。A5 走全新 `_apt.cpp` 入口 + `op_kernel/arch35/` + `_tiling_arch35.cpp`，与现有隔离防符号冲突。

### 4.4-精度要求（各 dtype 容差）

本算子为**非计算类（搬移/重排，无数值计算）**，精度标准取自 `ops-precision-standard` → 非计算类标准：

| 输出 | dtype | 精度判定 |
|------|-------|---------|
| y | FP16 / BF16 / INT8 | 二进制一致（Bitwise Match，`np.array_equal`）|
| group_list | INT64 | 二进制一致 |
| session_ids / micro_batch_ids / token_ids / expert_offsets | INT32 | 二进制一致 |
| dynamic_scale | FP32 | 二进制一致（原样字节 gather，非重算）|
| actual_token_num | INT64 | 二进制一致 |

> 全部输出为整数索引或原样字节搬运，**无浮点算术误差**，验收标准为 CPU golden 二进制一致；A5 与 A2 采用同一精度口径。

## 9. A5 与 A2 支持范围对齐性说明

逐项标注「A5 是否等同 A2」。**结论：对外支持范围全对齐；差异仅在内部实现代次（arch35 全新 kernel/tiling），不影响任何对外可见的支持范围。**

| 维度 | A2（Ascend910B, arch22） | A5（Ascend950, arch35） | 是否对齐 |
|------|--------------------------|--------------------------|:--------:|
| 芯片配置 | AddConfig(ascend910b) | 新增 AddConfig(ascend950, regbaseCfg) | ✅ 新增，A2 不动 |
| 输入 schedule_context | INT8, ND, 1 维, 1024B | 同 | ✅ 全对齐 |
| 输出集合（8 个） | y/group_list/session_ids/micro_batch_ids/token_ids/expert_offsets/dynamic_scale/actual_token_num | 同 | ✅ 全对齐 |
| dtype 矩阵（3 profile） | P0 fp16 / P1 bf16 / P2 int8 | 同 3 profile | ✅ 全对齐 |
| 属性表（5 个） | expert_num/max_out_shape/token_dtype/need_schedule/layer_num | 同 | ✅ 全对齐 |
| 功能路径（need_schedule×token_dtype） | 6 条有效组合（TilingKey 100/101） | 同 6 条 | ✅ 全对齐 |
| shape 约束（A≤1024/K≤64/M≤64/expert_num≤8192/BS,Y,H 泛化） | 见 §4.6 | 同 | ✅ 全对齐 |
| 关系式 Y=A×BS×(topK+1) | ✅ | ✅ | ✅ 全对齐 |
| 精度口径 | 二进制一致 | 二进制一致 | ✅ 全对齐 |
| 调用方式（GEIR + aclnn + 动/静态 shape） | ✅ | ✅ | ✅ 全对齐 |
| 生命周期 | experimental | experimental | ✅ 全对齐 |
| **内部实现代次** | arch22 kernel/tiling | **arch35 全新 kernel/tiling** | ⚠️ 差异（实现细节，非支持范围）|
| **内部切分阈值** | UB 容量（运行时查询）推导 sortLoopMaxElement 等 | UB 容量（运行时查询）重算（对外 shape 范围不变）| ⚠️ 差异（内部参数随平台取值，非支持范围）|

**差异归纳**（均为实现层，不影响对外支持范围）：
1. A5 kernel/tiling 为 arch35 全新代次路径（不能复用 arch22 指令/资源模型），经独立 `_apt.cpp` 入口 + `op_kernel/arch35/` + `_tiling_arch35.cpp` 实现。
2. UB 容量在 A5 与 A2 上不同（tiling 期经 `GetCoreMemSize(UB)` 查询，代码中不出现写死的 arch 相关字节数）导致 `sortLoopMaxElement`、`GROUP_LISTING_*` 多核阈值等内部切分参数在 A5 上重算——这是 tiling 依平台动态取值的正常结果，**不改变对外可见的 shape/dtype/功能路径支持范围**。
3. 本次 A5 **不扩展** 到 arch35 独有的窄数据类型（FP8/MXFP8/HiF8 等），严格对齐 A2 的 3 profile。

## 10. arch35 可行性评估

基于两个已支持 ascend950 的样板算子 `moe/moe_init_routing_v3`、`moe/moe_gating_top_k`（同 ops-transformer 仓 moe 域），评估技术路径可行性。

### 10.1 技术路径（arch35 移植机制，已由样板验证）

| 移植面 | 样板做法（moe_init_routing_v3 实证） | FfnWorkerBatching A5 对应动作 |
|--------|--------------------------------------|-------------------------------|
| def 芯片配置 | `def.cpp` 保留 A2/A3 AddConfig，新增 `OpAICoreConfig regbaseCfg` + `.ExtendCfgInfo("opFile.value","moe_init_routing_v3_apt")` + `AddConfig("ascend950", regbaseCfg)` | 新增 `ExtendCfgInfo("opFile.value","ffn_worker_batching_apt")` + `AddConfig("ascend950", regbaseCfg)`，A2/A3 不动 |
| kernel 入口 | 独立 `moe_init_routing_v3_apt.cpp`，`#include "arch35/*.h"` | 新增 `ffn_worker_batching_apt.cpp` + `op_kernel/arch35/` 子目录 |
| host tiling | `moe_init_routing_v3_tiling_arch35.cpp`，`REGISTER_OPS_TILING_TEMPLATE(..., 1000)` + `IsCapable()` 返回 `IsRegbaseSocVersion(context_)` 守卫 | 新增 `ffn_worker_batching_tiling_arch35.cpp`，同 1000 优先级 + `IsRegbaseSocVersion` 守卫，与现有 tiling 隔离防符号冲突 |
| config 目录 | 新增 `config/ascend950/`（binary.json + simplified_key.ini） | 新增 `config/ascend950/` |

### 10.2 算法原语可复用性（关键可行性依据）

`moe_init_routing_v3/op_kernel/arch35/` 已在 ascend950 上实现 FfnWorkerBatching 所需的**同族原语**：
- 排序：`moe_v3_sort_one_core.h` / `moe_v3_sort_multi_core.h` / `moe_v3_mrgsort.h` / `moe_v3_mrgsort_out.h` / `moe_v3_sort_base.h`
- gather：`moe_v3_gather_out.h` + 量化变体（`moe_v3_gather_dynamic_quant.h` 等）
- 分组计数：`moe_v3_expert_tokens_count.h`

这与 FfnWorkerBatching 的 sort（`ffn_wb_sort_*`/`ffn_wb_scan_sort_*`）→ gather（`ffn_wb_gather_out_all`）→ group_listing（`ffn_wb_group_listing`/`ffn_wb_scan_group_listing_one_core`）三步流程**结构同构**，arch35 原语可作为 A5 kernel 实现的直接参考蓝本。**技术路径可行。**

### 10.3 风险点

| 风险 | 说明 | 缓解 |
|------|------|------|
| 符号冲突 | A5 tiling 与现有 arch22 tiling 共存于同一 host 编译单元 | `REGISTER_OPS_TILING_TEMPLATE(...,1000)` + `IsRegbaseSocVersion` 守卫隔离（样板已验证） |
| ScheduleContext 逐字节解析 | 1024B 结构体在 arch35 内存/指针模型下的地址与 DCCI 缓存刷新（`DataCacheCleanAndInvalid`）语义 | 设计阶段核对 arch35 GM 指针/cache line 语义，穿刺用例验证 |
| Sort/MrgSort 元素上限与 tmpBuffer | arch22 `MGR_SORT_MAX_ELEMENT=1024`、`sortLoopMaxElement` 依 UB 推导；arch35 UB 容量不同、Sort API 约束需重新查证 | 设计阶段查 arch35 Sort/MrgSort API 文档，阈值一律由 tiling 期 `GetCoreMemSize(CoreMemType::UB)` 返回值推导，**禁止写死任何 arch 相关 UB 常量** |
| int8 动态量化连续排布 gather | token_dtype=2 每 token 尾部 32B scale 的 gather 在 RegBase/arch35 搬运语义 | 参考 `moe_v3_gather_dynamic_quant.h` 实现，穿刺 P2 profile |
| 内部切分阈值重算 | UB 差异导致多核/单核切分边界变化，需保证与 A2 相同的功能覆盖 | tiling 全部经 `PlatformAscendC` 动态取值，UT 覆盖边界 shape |

**总体结论**：A5 支持范围可完全对齐 A2；技术路径经两个同域样板算子（moe_init_routing_v3 / moe_gating_top_k）验证可行，无阻塞性风险，风险点均可在设计/开发阶段闭环。

## 11. 需求追溯

| 需求项 | 来源 | 本文档承接位置 |
|--------|------|---------------|
| 新增 A5 支持、A2/A3 不动 | LOG.md §四 | §2、§8.4、§9 |
| 支持范围对齐 A2 | LOG.md §四 | §4.4、§4.5、§9（全对齐结论）|
| 输入/输出/属性规格 | def.cpp / infershape / proto | §4.2、§4.3、§6 |
| dtype 矩阵 3 profile | def.cpp / config JSON | §4.4 |
| 功能路径 need_schedule×token_dtype | kernel / tiling | §4.5 |
| shape 约束 + Y=A×BS×(topK+1) | README / tiling | §4.6 |
| arch35 移植机制 | 样板 moe_init_routing_v3 / moe_gating_top_k | §10 |
| 精度口径 | ops-precision-standard 非计算类 | §4.4-精度、§8.1 |

