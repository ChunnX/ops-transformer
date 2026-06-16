# MoeDistributeDispatchV2 A5 分支算子功能分析

## 1. 分析范围

本文重新分析 `mc2/moe_distribute_dispatch_v2` 下的 `MoeDistributeDispatchV2` 算子，重点关注 Ascend David V100/V120（代码中主要体现为 `NpuArch::DAV_3510`、`__NPU_ARCH__ == 3510`，产品文档中对应 Ascend 950DT）的 A5 分支。

主要代码依据：

- 算子定义：`mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp`
- InferShape：`mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_infershape.cpp`
- ACLNN 接口与 A5 路径选择：`mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp`
- A5 Host Tiling：`mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_v2_tiling_arch35.cpp`
- 公共 Tiling 校验：`mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_helper.cpp`
- A5 CCU Device：`mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_arch35.h`
- A5 MTE/FullMesh Device：`mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp`、`op_kernel/moe_distribute_dispatch_v2.h`、`op_kernel/moe_distribute_dispatch_v2_full_mesh.h`
- Tiling 数据结构：`mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h`
- 官方接口说明：`mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV2.md`

本文重点说明：

1. 算子功能。
2. 算子输入/输出的含义、维度、来源和目的。
3. Host Tiling 的切分与校验原则。
4. Device 侧 A5 分支的执行流程。

## 2. 算子功能

`MoeDistributeDispatchV2` 是 MoE 网络中 Dispatch 阶段的通信与数据重排算子。它根据每个 token 的 topK 专家选择结果，把本 rank 的 token 分发给 EP（Expert Parallelism）通信域中持有目标专家的 rank，并在目标 rank 上生成按专家聚合后的 `expandXOut`。该输出会与 CombineV2 系列算子配套，用于后续专家计算后的反向合并。

从功能上看，算子完成四类工作：

1. **按专家路由 token**

   输入 `x[BS, H]` 表示本 rank 的 token 特征，`expertIds[BS, K]` 表示每个 token 选中的 K 个专家。Dispatch 会把 token 按专家归属复制/展开，并送到拥有对应专家的 EP rank。

2. **可选量化**

   A5 支持 `quantMode` 0 到 4：

   | quantMode | 含义 | A5 主要输出类型 |
   | --- | --- | --- |
   | 0 | 非量化 | `BF16`、`FLOAT16`、`HIFLOAT8`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2`、部分 `FLOAT4` |
   | 1 | 静态量化 | `INT8`、`HIFLOAT8` |
   | 2 | per-token 动态量化 | `INT8`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2` |
   | 3 | per-group 动态量化 | `FLOAT8_E4M3FN`、`FLOAT8_E5M2` |
   | 4 | MX 量化 | `FLOAT8_E4M3FN`、`FLOAT8_E5M2`、`FLOAT4_E2M1`、`FLOAT4_E1M2` |

3. **EP 域可变长通信**

   不同目标专家收到的 token 数不同，因此 A5 分支按可变长通信组织发送 buffer 和接收 buffer。A5 CCU 路径使用 `HalfAllToAllV`/`AlltoAllvWrite` 形态；MTE/fullmesh 路径走 `MoeDistributeDispatchV2` 或 `MoeDistributeDispatchV2FullMesh` 模板。

4. **生成 Combine 辅助信息**

   V2 接口输出 `assistInfoForCombineOut`，用于后续 CombineV2 系列算子恢复 dispatch 前后的对应关系。代码中部分变量仍叫 `expandIdx`，但对外语义是 `assistInfoForCombineOut`。文档也明确提示业务逻辑不应依赖该辅助 tensor 的内部元素布局。

## 3. A5 分支入口与通信路径

### 3.1 Host/API 路径选择

`op_api/moe_distribute_dispatch_v2_base.cpp` 通过当前平台判断 A5：

- `GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510` 表示 A5/Ascend 950DT。
- A5 上 `performanceInfoOptional` 被置空，说明 V2 API 入口在 A5 上不使用该性能信息输入。
- 当 `commAlg == "ccu"` 时，仍调用 `aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize`，并设置 HCCL server type 为 `NNOPBASE_HCCL_SERVER_TYPE_CCU`。
- 当 A5 且不是 `ccu` 时，通过 `Mc2Context::GetMc2ContextTensor` 获取 MC2 context，调用 `aclnnInnerMoeDistributeDispatchV3GetWorkspaceSize`；执行阶段若 user handle 标记为 AIV，则调用 `aclnnInnerMoeDistributeDispatchV3`，即 A5 MTE 新模板路径。

因此，A5 不是单一路径，而是有两条主要路线：

| 路径 | 触发条件 | Host Tiling | Device 模板 |
| --- | --- | --- | --- |
| CCU | `commAlg == "ccu"` | `MoeDistributeDispatchTilingImpl(..., OP_VERSION_2)` | `MoeDistributeDispatchA5`，定义在 `arch35/moe_distribute_dispatch_arch35.h` |
| MTE/fullmesh | `commAlg` 为空或 fullmesh 类配置 | `MoeDistributeDispatchA3TilingFuncImplPublic` 的公共实现，由 A5 封装调用 | `MoeDistributeDispatchV2` 或 `MoeDistributeDispatchV2FullMesh` |

### 3.2 Device 编译分支

`op_kernel/moe_distribute_dispatch_v2.cpp` 中通过：

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
```

选择 A5 相关 include 和 namespace。A5 分支按模板参数进一步区分：

- `ArchTag == TILINGKEY_TPL_A5`
- `CommMode == TILINGKEY_TPL_CCU` 或 `TILINGKEY_TPL_MTE`
- `FullMesh == TILINGKEY_ENABLE_FULLMESH` 或 `TILINGKEY_NO_FULLMESH`
- `QuantMode`、`ScaleMode`、输入输出 dtype 组合

CCU 路径实例化 `MoeDistributeDispatchA5`；MTE 非 fullmesh 路径实例化 `MoeDistributeDispatchV2`；MTE fullmesh 路径实例化 `MoeDistributeDispatchV2FullMesh`。

## 4. 输入、属性和输出

### 4.1 关键符号

| 符号 | 含义 |
| --- | --- |
| `BS` | 本 rank 输入 token 数，即 `x.shape[0]` |
| `H` | hidden size，即 `x.shape[1]` |
| `K` | 每个 token 选择的专家数，即 `expertIds.shape[1]` |
| `epWorldSize` | EP 通信域 rank 数 |
| `tpWorldSize` | TP 通信域 rank 数；A5 CCU 当前 Host Tiling 要求为 1 |
| `moeExpertNum` | MoE 专家总数 |
| `sharedExpertNum` | 共享专家数量 |
| `sharedExpertRankNum` | 放置共享专家的 rank 数 |
| `moeRankNum` | `epWorldSize - sharedExpertRankNum` |
| `localMoeExpertNum` | `moeExpertNum / moeRankNum` |
| `globalBS` | EP 域全局 BS；为 0 时按 `BS * epWorldSize` 推导 |
| `A` | 本 rank 输出侧最大 token 容量，用于 `expandXOut` 和辅助信息 shape |

### 4.2 A 的计算

InferShape 和 Tiling 中都使用 `A` 作为输出容量核心参数。

对于共享专家 rank：

```text
maxBs = globalBSReal / epWorldSize
rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum
maxSharedGroupNum = ceil(epWorldSize / rankNumPerSharedExpert)
A = maxBs * maxSharedGroupNum
```

对于普通 MoE 专家 rank：

```text
localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)
A = globalBSReal * min(localMoeExpertNum, K)
```

其中：

```text
globalBSReal = globalBS == 0 ? BS * epWorldSize : globalBS
```

如果存在 `elasticInfo`，InferShape 会把共享专家和普通 MoE 专家的容量取更保守的上界，避免动态容量不足。

### 4.3 输入 Tensor

| 输入 | 类型 | 维度 | 来源 | 目的 |
| --- | --- | --- | --- | --- |
| `x` | `FLOAT16`、`BFLOAT16`、`FLOAT8_E5M2`、`FLOAT8_E4M3FN`、`HIFLOAT8`、`FLOAT4_E2M1`、`FLOAT4_E1M2` | `(BS, H)` | MoE router 前一层输出或上游隐藏状态 | 待 dispatch 的 token 特征 |
| `expertIds` / `expert_ids` | `INT32` | `(BS, K)` | router/topK 选择结果 | 指定每个 token 要发送到哪些专家 |
| `scalesOptional` / `scales` | `FLOAT32` 或 `FLOAT8_E8M0` | 与量化模式相关：静态量化可为 1D/2D，动态量化通常按专家和 H 对齐 | 量化平滑参数或低精度输入附带 scale | 参与静态量化、动态量化或低精度非量化 scale 携带 |
| `xActiveMaskOptional` / `x_active_mask` | `BOOL` | A5 CCU V2 路径要求 1D `(BS,)` | 上游动态 BS 或 token 有效性标记 | 控制前缀 token 是否参与通信 |
| `expertScalesOptional` / `expert_scales` | `FLOAT32` | `(BS, K)` | router 输出的专家权重 | 供分层/Combine 类路径携带 token 权重；A5 当前 dispatch V2 输出 `expandScalesOut` 文档标注不支持 |
| `elasticInfoOptional` / `elastic_info` | `INT32` | 可选 | 弹性通信/容量信息 | InferShape 中用于扩大容量上界 |
| `performanceInfoOptional` / `performance_info` | `INT64` | 可选 | 性能统计 buffer | A5 V2 API 基路径会置空，不作为 A5 主分析重点 |

### 4.4 属性

| 属性 | 含义 | A5 关键约束 |
| --- | --- | --- |
| `group_ep` / `groupEp` | EP 通信域名称 | 必填，长度在 `[1, 128)` |
| `ep_world_size` / `epWorldSize` | EP 通信域大小 | CCU V2 tiling 中范围 `[2, 1024]` |
| `ep_rank_id` / `epRankId` | 当前 rank 在 EP 域内的编号 | `[0, epWorldSize)` |
| `moe_expert_num` / `moeExpertNum` | MoE 专家数量 | V2 范围 `(0, 1024]`，且可被 `epWorldSize - sharedExpertRankNum` 整除 |
| `group_tp` / `groupTp` | TP 通信域名称 | A5 CCU Host Tiling 当前要求 `tpWorldSize == 1`，TP 输出不支持 |
| `tp_world_size` / `tpWorldSize` | TP 域大小 | CCU 路径检查范围 `[0, 2]`，后续属性检查要求为 1 |
| `tp_rank_id` / `tpRankId` | TP rank id | `tpWorldSize <= 1` 时必须为 0 |
| `expert_shard_type` | 专家切分类型 | A5 CCU 要求为 0 |
| `shared_expert_num` | 共享专家数 | V2 范围 `[0, 4]`，需与 `sharedExpertRankNum` 同为 0 或同非 0 |
| `shared_expert_rank_num` | 共享专家 rank 数 | `[0, epWorldSize)`；非 0 时需能被 `sharedExpertNum` 整除 |
| `quant_mode` | 量化模式 | A5 支持 0、1、2、3、4 |
| `global_bs` | EP 域全局 BS | 0 或 `maxBS * epWorldSize`；使用 token mask 时不允许不同 rank BS |
| `expert_token_nums_type` | `expertTokenNumsOut` 语义 | 0 表示前缀和，1 表示计数 |
| `comm_alg` | 通信算法 | A5 Tiling 封装允许空、`fullmesh_v1`、`fullmesh_v2`、`ccu`；CCU 走 A5 专用实现 |
| `zero/copy/const_expert_num` | V4/扩展专家相关属性 | V2 A5 主路径中不是核心逻辑 |
| `y_dtype` | 输出 dtype | API 层由 `expandXOut->GetDataType()` 传入 inner op |

### 4.5 输出 Tensor

| 输出 | 类型 | 维度 | 目的 | 下游 |
| --- | --- | --- | --- | --- |
| `expandXOut` / `expand_x` | 与 `quantMode`/`y_dtype` 匹配 | `(max(tpWorldSize, 1) * A, H)`；A5 CCU 实际不支持 TP，通常为 `(A, H)` | 按目标专家重排后的 token 特征 | 专家 FFN 或 Combine 前处理 |
| `dynamicScalesOut` / `dynamic_scales` | `FLOAT32` 或 `FLOAT8_E8M0` | A5：per-token 为 `(A,)`；per-group 为 `(A, ceil(H/128))`；MX 为 `(A, even(ceil(H/32)))`；非量化低精度带 scales 时可为 2D | 保存动态量化或低精度输入 scale | Combine/反量化 |
| `assistInfoForCombineOut` / `assist_info_for_combine` | `INT32` | `(A * 128,)` | 给 CombineV2 系列算子使用的路由辅助信息 | `MoeDistributeCombineV2` 等 |
| `expertTokenNumsOut` / `expert_token_nums` | `INT64` | 共享专家 rank 为 `(1,)`；普通 MoE rank 为 `(localMoeExpertNum,)` | 当前 rank 本地专家收到的 token 数或前缀和 | 专家计算和 Combine |
| `epRecvCountsOut` / `ep_recv_count` | `INT32` | A5 文档约束为 `(epWorldSize * localExpertNum,)`；共享专家按本地专家数 1 处理 | 每个源 EP rank 到本地专家的接收 token 数 | CombineV2 的 EP send/recv 统计 |
| `tpRecvCountsOut` / `tp_recv_count` | `INT32` | `(tpWorldSize,)`，但 A5 当前不支持 TP 输出 | TP 域接收计数 | TP Combine；A5 CCU 当前不使用 |
| `expandScalesOut` / `expand_scales` | `FLOAT32` | 文档标注 A5 当前不支持；InferShape 在 `expertScales` 非空时可设为 `(A,)` | 输出 token 权重 | Combine 可选输入 |

## 5. Host Tiling 切分原则

### 5.1 A5 Tiling 总流程

A5 Tiling 入口在 `op_host/op_tiling/arch35/moe_distribute_dispatch_v2_tiling_arch35.cpp`。主流程是：

1. `MoeDistributeDispatchV2TilingFuncA5::MoeDistributeDispatchTilingFuncImpl` 读取 `commAlg`。
2. 如果 `commAlg == "ccu"`，调用 `MoeDistributeDispatchTilingImpl(context, OP_VERSION_2)`，进入 A5 CCU 专用 tiling。
3. 如果 `commAlg` 为空或 fullmesh 类配置，则构造 `DispatchV2Config`，调用 `MoeDistributeDispatchA3TilingFuncImplPublic`，即 A3/A5 公共 MTE/fullmesh tiling。

下面重点描述 CCU 专用 tiling，因为它是 `arch35` 中最完整的 A5 分支。

### 5.2 属性校验与基础 tiling 数据

`GetContextAttrs` 从 attrs 中读取并写入 `MoeDistributeDispatchV2Info`：

- `epWorldSize`、`epRankId`
- `tpWorldSize`、`tpRankId`
- `expertShardType`
- `sharedExpertNum`、`sharedExpertRankNum`
- `moeExpertNum`
- `quantMode`
- `expertTokenNumsType`

关键检查：

- `epWorldSize` 在 V2 中为 `[2, 1024]`。
- `tpWorldSize` 先允许 `[0, 2]`，但 `CheckAttrs` 当前要求 `tpWorldSize == 1`。
- `moeExpertNum` 必须可被 `epWorldSize - sharedExpertRankNum` 整除。
- `sharedExpertNum` 与 `sharedExpertRankNum` 必须同为 0 或同非 0；非 0 时 `sharedExpertRankNum % sharedExpertNum == 0`。
- `localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)`，并要求大于 0。

### 5.3 dtype、quantMode 与 scales 匹配

A5 使用 `QUANT_MODE_MAP` 将 `quantMode + expandX dtype` 映射为内部 `RealModeA5`：

- `NON_QUANT` 支持 `BF16/FLOAT16/HIFLOAT8/FP8/FP4` 等非量化搬运。
- `STATIC_QUANT` 支持 `INT8` 和 `HIFLOAT8`。
- `PERTOKEN_DYNAMIC_QUANT` 支持 `INT8/FP8_E4M3/FP8_E5M2`。
- `PERGROUP_DYNAMIC_QUANT` 支持 `FP8_E4M3/FP8_E5M2`。
- `MX_QUANT` 支持 `FP8_E4M3/FP8_E5M2/FP4_E2M1/FP4_E1M2`。

`CheckQuantModeAndScales` 的核心规则：

- `MX_QUANT` 时 `scales` 必须为空。
- `NON_QUANT` 且 `x` 为 `BF16/FLOAT16` 时 `scales` 必须为空。
- `NON_QUANT` 且 `x` 为 `HIFLOAT8/FP8` 等低精度类型时 `scales` 必须非空。
- `STATIC_QUANT` 时 `scales` 必须非空。

公共 helper 还检查：

- `expertIds` 必须是 `INT32`。
- `expertTokenNumsOut` 必须是 `INT64`。
- `assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut` 必须是 `INT32`。
- 不允许 `FRACTAL_NZ` 格式。

### 5.4 shape 校验原则

输入 shape：

- `x` 必须为 2D，`(BS, H)`。
- `expertIds` 必须为 2D，`(BS, K)`，且 `expertIds.shape[0] == x.shape[0]`。
- `K` 范围为 `(0, min(16, moeExpertNum)]`。
- A5 V2 的 `H` 范围为 `[1024, 8192]`。
- `BS` 范围为 `(0, 512]`。

`scales` shape：

- 静态量化允许 1D 或 2D。
- HIFLOAT8 静态量化要求 1D。
- 非静态量化若传 `scales`，通常要求 2D。
- 2D scales 的 dim0 需要与 `moeExpertNum` 或 `moeExpertNum + sharedExpertNum` 对齐，dim1 需要等于 `H`。

输出 shape：

- `expandXOut.dim0 >= A * tpWorldSize`，`expandXOut.dim1 == H`。
- `dynamicScalesOut.dim0 >= A * tpWorldSize`；per-token 为 1D，per-group/MX 为 2D。
- `assistInfoForCombineOut.dim0 >= BS * K` 的底层检查来自旧变量名 `expandIdx`；InferShape 对外设为 `A * 128`。
- `expertTokenNumsOut`：共享专家 rank 为 1，普通 MoE rank 为 `localMoeExpertNum`。
- `epRecvCountsOut.dim0 >= epWorldSize * localExpertNum`；如果 `tpWorldSize == 2`，乘以 `tpWorldSize`，但 A5 CCU 当前又要求 TP 为 1。
- `tpRecvCountsOut.dim0 == tpWorldSize`。

### 5.5 globalBS 与 token mask

`globalBS` 的原则：

- `globalBS == 0` 时，tiling 内部设置为 `BS * epWorldSize`。
- 非 0 时，必须是 `maxBS * epWorldSize`，且不小于本 rank `BS * epWorldSize`。
- 如果传入 `xActiveMask`，则不允许各 rank BS 不同，因为 mask 只描述本 rank 前缀有效 token。

`xActiveMask` 在 A5 V2 中要求：

- 可选。
- 如果存在，Host helper 要求是 1D。
- dtype 必须为 `BOOL`。
- Device 侧通过求和得到 `activeBs_`，后续只处理 `activeBs_ * K` 个 expert 路由项。

### 5.6 workspace 与通信窗口切分

A5 CCU 的 workspace 由 `SetWorkSpace` 计算：

```text
workspace =
  SYSTEM_NEED_WORKSPACE
  + epWorldSize * sizeof(uint64_t) * BUFFER_NUM * BUFFER_NUM
  + epWorldSize * (
      COUNT_OFFSET
      + maxBS * Align512(alignedH * sizeof(x_dtype)) * localMoeExpertNum
    )
```

其中：

- `SYSTEM_NEED_WORKSPACE = 16 MB`
- `COUNT_OFFSET = 512`
- `maxBS = globalBS / epWorldSize`
- `alignedH` 根据量化模式变化：
  - 默认按 32 字节对齐。
  - MX 按 256 对齐。
  - per-group 按 128 对齐。

`CheckCommAttrs` 会检查通信窗口大小是否足够。A5 实际物理窗口按 `HCCL_BUFFSIZE * 2` 计算：

```text
actualSize =
  aivNum * 512
  + 2 * epWorldSize * (
      maxBS * Align512(alignedH * 2) * localMoeExpertNum
      + COUNT_OFFSET
    )
```

如果 `actualSize > maxWindowSize`，tiling 失败并提示需要增大 `HCCL_BUFFSIZE`。

### 5.7 通信 tiling

A5 CCU 路径 `SetCommTiling` 固定设置：

- op type：`HCCL_CMD_HALFALLTOALLV`
- 算法字符串：`AlltoAll=level0:fullmesh;level1:pairwise`
- 通信引擎：`A5_CCU_ENGINE`

代码只为 EP 域设置有效通信，注释明确指出 A5 不支持 TP 通信；非 HalfAllToAllV 的 op type 会出错。

### 5.8 AIV 切分原则

Host 侧通过 `PlatformAscendC` 获取：

- `aivNum = GetCoreNumAiv()`
- UB 大小 `totalUbSize`
- `blockDim = CalcTschBlockDim(aivNum, 0, aivNum)`

Device 侧按 AIV id 切分 token 或 rank：

- `SplitToCore(curSendCnt, curUseAivNum, start, end, count, isFront)` 把 token/rank 平均分给 AIV，多余元素分给前若干 AIV。
- 若存在共享专家：
  - `sharedUsedAivNum = max(1, aivNum / (K + 1))`
  - `moeUsedAivNum = aivNum - sharedUsedAivNum`
  - 前半或后半 AIV 分别处理 MoE 专家和共享专家 scatter。
- 若没有共享专家，绝大多数 AIV 用于 MoE scatter/gather。

### 5.9 Tiling key

A5 CCU tiling key 由：

```text
GET_TPL_TILING_KEY(
  tp=false,
  quantMode,
  scaleMode=isScales,
  fullMesh=TILINGKEY_NO_FULLMESH,
  commMode=TILINGKEY_TPL_CCU,
  arch=TILINGKEY_TPL_A5
)
```

MTE/fullmesh A5 路径则使用 `commMode=TILINGKEY_TPL_MTE`，并根据 fullmesh 或 hierarchy 设置 `templateDispatch`。

## 6. Device 侧功能详细介绍

### 6.1 Kernel 模板选择

A5 kernel 入口 `moe_distribute_dispatch_v2` 在 `op_kernel/moe_distribute_dispatch_v2.cpp`。当 `__NPU_ARCH__ == 3510` 时：

- CCU 路径实例化 `MoeDistributeDispatchA5<...>`。
- MTE fullmesh 路径实例化 `MoeDistributeDispatchV2FullMesh<...>`。
- MTE 非 fullmesh 路径实例化 `MoeDistributeDispatchV2<...>`。

CCU 路径是 A5 专有实现，下面重点展开。

### 6.2 Init：全局属性、buffer、通信窗口

`MoeDistributeDispatchA5::Init` 做以下初始化：

1. 保存 `epRankId_` 和 `aivId_`。
2. `InitGlobalAttrs` 从 tiling 中读取：
   - `BS/H/K`
   - `epWorldSize`
   - `moeExpertNum`
   - `sharedExpertRankNum`
   - `sharedExpertNum`
   - `aivNum`
   - `expertTokenNumsType`
   - `scalesCount`
   - `globalBS`
3. 计算：
   - `moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_`
   - `localExpertNum_ = moeExpertNum_ / moeExpertRankNum_`
   - `bskNum_ = BS * K`
   - `axisMaxBs_ = globalBS / epWorldSize`
   - `maxBsKNum_ = axisMaxBs_ * K`
4. 绑定 GM tensor：
   - `xGT_`
   - `expertIdsGT_`
   - `expandXOutGT_`
   - `expandIdxGT_`，实际对应 `assistInfoForCombineOut`
   - `dynamicScaleGT_`
   - `expertTokenNumsGT_`
   - `sendCountsGT_`，对应 EP recv/send 统计输出
   - `scalesGT_`
   - `xActiveMaskGT_`
5. `InitBuf` 初始化排序、频次统计、cumSum、临时 buffer。
6. 根据 dtype 和量化模式计算 token 打包大小：
   - `perTokenInSize_ = H * sizeof(XType)`
   - `perTokenOutSize_ = H * sizeof(ExpandXOutType)`
   - `perTokenMergeSize_` 默认 `Align32(H) * sizeof(out_dtype)`，量化模式会追加 scale。
   - `perTokenCommSize_ = Align512(perTokenMergeSize_)`
   - `perRankDataSize_ = COUNT_OFFSET + perTokenCommSize_ * axisMaxBs_ * localExpertNum_`
7. `InitCommAndStatus` 初始化 CCU HCCL 对象和双 buffer 通信窗口。

### 6.3 QuantInit：量化模式对打包布局的影响

`QuantInit` 决定 token 在通信 buffer 中的布局：

- 非量化无 scales：只拷贝 token 数据。
- 非量化但带 scales：在 token 后追加 scale 数据。
- 静态量化：输出量化 token，不额外输出 dynamic scale。
- per-token 动态量化：token 后追加一个 `float` scale。
- per-group 动态量化：按 `Align128(H)` 存 token，并追加 `ceil(H/128)` 个 `float` scale。
- MX 量化：按 `Align256(H)` 存 token，并追加 `Align2(ceil(H/32))` 个 `fp8_e8m0` scale。

所有模式最终按 `Align512(perTokenMergeSize_)` 作为通信粒度。

### 6.4 token mask 处理

如果 `isTokenMaskFlag_` 为 true，`Process` 首先调用 `CalcTokenActiveMask`：

1. 从 GM 读取 `xActiveMask[BS]` 到 UB。
2. 转为 half 后求和。
3. 得到 `activeBs_`。
4. 更新 `activeBsKNum_ = activeBs_ * K`。
5. 后续 scatter 只处理有效 token 的 topK 路由。

该设计要求 mask 中 true 是前缀连续的，否则 Device 侧只按 true 的数量截断前缀，无法表达任意稀疏 token。

### 6.5 TokenScatter：排序、统计与 send buffer 填充

`TokenScatter` 是 CCU 路径的核心发送准备阶段。

当 `activeBs_ != 0`：

1. 把 `expertIds[activeBs, K]` 拷贝到 UB。
2. 转为 float 并乘以 -1，用 `Sort<float, true>` 排序。
3. 再乘以 -1 得到按 expert id 排序的 `sortedOutI32LT_`。
4. `sortedIndex1LT_` 保留排序前 token/expert 路由项索引。
5. `GetExpertFreq` 统计每个 expert 收到的 token 数。
6. `GetExpertCumSum` 计算专家前缀和。

随后按 AIV id 分流：

- `aivId_ < moeUsedAivNum_` 执行 `MoeScatter`。
- 其他 AIV 执行 `ShareScatter`。

### 6.6 普通 MoE Scatter

`MoeScatterCopyTokens` 处理普通 MoE 专家：

1. 将 `activeBsKNum_` 个路由项按 `moeUsedAivNum_` 切分给 AIV。
2. 对每个路由项：
   - 读取排序后的 `expertId`。
   - 计算目标专家 rank：`toExpertRankId = expertId / localExpertNum_`。
   - 目标 EP rank：`toRankId = toExpertRankId + sharedExpertRankNum_`。
   - 计算该 token 在目标 rank 的专家内偏移 `preCount`。
   - 根据 `sendBufGM_ + COUNT_OFFSET * (toRankId + 1) + ...` 定位 send buffer。
   - 调用 `ProcessToken` 把 token 量化/打包到目标位置。

`MoeScatter` 还会为每个目标 rank 写入：

- `sendSizeGM_`
- `sendOffsetGM_`
- send buffer 开头的每专家 token 计数

这些信息供后续 `AlltoAllvWrite` 读取。

### 6.7 共享专家 Scatter

`ShareScatter` 处理共享专家 rank：

1. 如果 `sharedExpertRankNum_ == 0`，直接返回。
2. `ShareScatterCopyTokens` 按 `activeBs_ * sharedExpertNum_` 计算共享专家发送量。
3. `CalcToSharedRankId` 根据共享专家编号和当前 rank 在共享专家复制组内的位置，计算目标共享专家 rank。
4. 将 token 打包到共享专家 rank 的 send buffer。
5. `SetShareExpertSendSize` 为共享专家 rank 设置发送 size、offset 和 count。

共享专家的本质是：同一个 token 可能发送到共享专家复制组中的对应 rank，不按普通 MoE 专家的 `expertId / localExpertNum` 规则路由。

### 6.8 ProcessToken：量化与 token 打包

`ProcessToken` 按模板参数静态选择量化路径：

- `QuantMode == 0`：直接从 `xGT_` 拷贝 token 到 UB，再拷贝到 send buffer；如果 `IsSmoothScaleExist`，把输入 scales 追加到 token 尾部。
- `STATIC_QUANT`：`CastToFp32 -> Mul(scales) -> Cast(out_dtype)`。
- `PERTOKEN_DYNAMIC_QUANT`：计算 token 绝对值最大值，得到动态 scale，量化后把 `1 / dynamicScale` 写到 token 尾部。
- `PERGROUP_DYNAMIC_QUANT`：调用 `quant::ComputePerTileDynamic`，按 group 生成 scale。
- `MX_QUANT`：计算 max exponent、scale 和 fp8/fp4 数据，scale 写到 token 尾部。

这样通信发送的 payload 已经包含目标专家计算所需的 token 数据和必要 scale。

### 6.9 Communication：CCU AllToAllV

`Communication` 的流程：

1. `SyncAll<true>()` 等待所有 AIV 完成 send buffer 准备。
2. `aivId_ == 0` 计算本 rank 实际发送数据大小：
   - 普通 MoE rank：累加本 rank 持有的本地专家频次。
   - 共享专家 rank：`activeBs_ * perTokenCommSize_`。
3. AIV0 调用：

   ```cpp
   hccl_.AlltoAllvWrite<true>(sendBufGM_, sendOffsetGM_, sendSizeGM_, recvOffset_, localDataSize)
   ```

4. 其他 AIV 同时执行 `SaveExpandIdx`，生成 Combine 辅助信息。
5. 初始化 gather 阶段需要的接收计数 buffer。
6. AIV0 `hccl_.Wait(hcclHandleId_)` 等待通信完成。
7. `SyncAll<true>()` 确保所有 AIV 看到通信结果。

CCU 路径使用通信窗口双 buffer：`InitCommAndStatus` 通过 CCL window 中的 status 位在前半和后半窗口之间切换，避免读写冲突。

### 6.10 SaveExpandIdx / assistInfoForCombineOut

`SaveExpandIdx` 生成 `assistInfoForCombineOut` 的基础索引信息：

1. 基于 `sortedIndex1LT_` 再做一次排序，恢复 token 在排序前后的对应关系。
2. 对每个路由项计算其在目标专家输出序列中的 offset。
3. 写入 `expandIdxGT_`，该 GM tensor 对应对外输出 `assistInfoForCombineOut`。

虽然代码变量名为 `expandIdx`，但 V2 对外输出 shape 是 `(A * 128,)`，语义是 CombineV2 的辅助信息。

### 6.11 TokenGather：接收 buffer 到输出 tensor

通信完成后，`TokenGather` 把 CCL window 中收到的数据整理为最终输出：

1. 初始化 `tokenGatherQue_`，每个元素大小为 `perTokenCommSize_`。
2. 如果当前 rank 是共享专家 rank，把它视为 `localExpertNum = 1` 的特殊 MoE rank。
3. `TokenGatherProcess` 按 EP rank 切分给 AIV。
4. `MoeGatherPrepare` 从 `recvBufGM_` 中读每个源 rank 的 count，计算：
   - 每个本地专家总 token 数。
   - 每个源 rank 在当前专家输出中的前缀偏移。
5. 对每个本地专家、每个源 rank：
   - 根据 count 和 prefix 从接收 buffer 读取 token。
   - 拷贝 token 数据到 `expandXOutGT_[outPreCount * H]`。
   - `CopyScalesToOut` 把 token 尾部 scale 拷贝到 `dynamicScalesOut`。
6. `UpdateMoeRankOutput` 写：
   - `epRecvCountsOut`
   - `expertTokenNumsOut`

`expertTokenNumsType == 0` 时，`expertTokenNumsOut` 写前缀和；否则写每个专家的计数。

### 6.12 A5 MTE / fullmesh Device 简述

当不是 `commAlg == "ccu"` 时，A5 走 MTE/AIV 模板：

- `MoeDistributeDispatchV2`：普通非 fullmesh dispatch。
- `MoeDistributeDispatchV2FullMesh`：fullmesh dispatch。

这两类模板使用 MC2/HCCL context window、自定义状态位、cumsum、mask、window copy 等机制完成 dispatch。相比 CCU 路径：

- 通信不是 `MoeDistributeDispatchA5` 中的 `Hccl<HCCL_SERVER_TYPE_CCU>`。
- Host Tiling 走 A3/A5 公共实现。
- `fullmesh_v2` 会选择 fullmesh 模板，使用窗口状态和专家/Rank 粒度的收发计数组织通信。

本次重点分析 A5 专用 CCU 分支；MTE/fullmesh 可视为 A5 上复用 V2 通用 MC2 模板的一条路线。

## 7. Host 与 Device 的数据流总结

```text
输入:
  x[BS,H], expertIds[BS,K], optional scales/mask

Host Tiling:
  1. 判断 A5/DAV_3510 和 commAlg
  2. 校验 ep/tp/expert/quant/globalBS
  3. 计算 localMoeExpertNum、A、H、K、maxBS
  4. 校验输出容量和 dtype
  5. 计算 per-token 通信布局、workspace、window size
  6. 生成 tiling key 和 HCCL/MC2 tiling

Device CCU:
  1. Init: 绑定 GM、计算 perTokenCommSize/perRankDataSize
  2. 可选 xActiveMask 求 activeBs
  3. TokenScatter: expertIds 排序、统计、token 打包
  4. AlltoAllvWrite: EP 域可变长通信
  5. SaveExpandIdx: 生成 Combine 辅助索引信息
  6. TokenGather: 从接收窗口拷贝到 expandXOut/dynamicScalesOut
  7. 输出 expertTokenNumsOut、epRecvCountsOut
```

## 8. A5 分支关键限制与注意事项

1. **A5 CCU 当前不支持 TP 通信**

   Host Tiling 中 `CheckAttrs` 要求 `tpWorldSize == 1`，`SetCommTiling` 也注明 A5 不支持 TP 通信，只设置 HalfAllToAllV。

2. **`commAlg` 决定 A5 走哪条实现**

   - `ccu`：走 A5 CCU 专用 Host Tiling 和 `MoeDistributeDispatchA5`。
   - 空/fullmesh：走 A5 MTE/V3 扩展路径和通用 MC2 模板。

3. **`xActiveMask` 只表达前缀有效 token**

   Device 侧只是对 bool mask 求和得到 `activeBs_`，后续处理前 `activeBs_` 个 token。因此 mask 必须是 true 前缀，不适合任意稀疏 token。

4. **通信窗口大小是 tiling 成败关键**

   A5 会用 `HCCL_BUFFSIZE * 2` 作为实际物理窗口大小，并按 `maxBS`、`localMoeExpertNum`、`H`、对齐后的 token 大小估算通信窗口需求。大 BS、大 H、多本地专家或 FP8/FP4 scale 附加信息都可能导致窗口不足。

5. **`assistInfoForCombineOut` 不建议业务解析**

   文档明确说明不同产品、算法或版本中辅助输出元素值可能不同，应直接传给 CombineV2 系列算子，不应在业务逻辑里依赖内部布局。

6. **共享专家会改变 `A` 和输出计数语义**

   共享专家 rank 的 `localExpertNum` 视作 1，`A` 按共享专家复制组容量计算；普通 MoE rank 则按 `globalBS * min(localMoeExpertNum, K)` 计算。

## 9. 结论

`MoeDistributeDispatchV2` 在 A5 上承担 MoE Dispatch 的核心路由、可选量化和 EP 域通信职责。Host 侧主要负责把通信拓扑、专家分布、量化模式、输出容量和通信窗口大小全部固化到 tiling 中；Device 侧根据 tiling 进行 token 排序、统计、量化打包、AllToAllV 通信和接收重排。

对 A5 分支而言，最重要的实现差异是：

- `DAV_3510` 编译分支启用 A5 专用代码。
- `commAlg == "ccu"` 时走 `MoeDistributeDispatchA5`，使用 CCU HCCL server 和 HalfAllToAllV。
- 非 CCU 时走 A5 MTE/V3 扩展路径，复用 V2 通用 MC2/fullmesh 模板。
- A5 支持更丰富的量化输出类型和 scale 布局，但也带来更严格的 dtype、shape 和 window size 约束。

