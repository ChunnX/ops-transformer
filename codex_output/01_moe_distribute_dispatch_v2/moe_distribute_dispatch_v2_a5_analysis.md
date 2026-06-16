# MoeDistributeDispatchV2 A5 分支算子分析

## 目录

- [1. 分析范围](#1-分析范围)
- [2. 算子功能](#2-算子功能)
  - [2.1 A5 通信路径](#21-a5-通信路径)
- [3. 算子输入/输出](#3-算子输入输出)
  - [3.1 关键符号](#31-关键符号)
  - [3.2 A 的计算](#32-a-的计算)
  - [3.3 输入](#33-输入)
  - [3.4 属性](#34-属性)
  - [3.5 输出](#35-输出)
- [4. Host Tiling 切分原则](#4-host-tiling-切分原则)
  - [4.1 分支选择](#41-分支选择)
  - [4.2 参数合法性检查](#42-参数合法性检查)
  - [4.3 dtype 与 quantMode 匹配](#43-dtype-与-quantmode-匹配)
  - [4.4 scales shape 检查与 tiling 信息](#44-scales-shape-检查与-tiling-信息)
  - [4.5 输出 shape 校验](#45-输出-shape-校验)
  - [4.6 workspace 与通信窗口](#46-workspace-与通信窗口)
  - [4.7 AIV 切分原则](#47-aiv-切分原则)
  - [4.8 Tiling key](#48-tiling-key)
- [5. Device 侧功能详细介绍](#5-device-侧功能详细介绍)
  - [5.1 Kernel 模板选择](#51-kernel-模板选择)
  - [5.2 Init 阶段](#52-init-阶段)
  - [5.3 token mask 处理](#53-token-mask-处理)
  - [5.4 TokenScatter：排序、统计与发送 buffer 填充](#54-tokenscatter排序统计与发送-buffer-填充)
  - [5.5 普通 MoE scatter](#55-普通-moe-scatter)
  - [5.6 共享专家 scatter](#56-共享专家-scatter)
  - [5.7 ProcessToken：量化与 token 打包](#57-processtoken量化与-token-打包)
  - [5.8 Communication：CCU AllToAllV](#58-communicationccu-alltoallv)
  - [5.9 SaveExpandIdx / assistInfoForCombineOut](#59-saveexpandidx--assistinfoforcombineout)
  - [5.10 TokenGather：接收 buffer 重排到输出](#510-tokengather接收-buffer-重排到输出)
  - [5.11 A5 MTE/fullmesh device 侧简述](#511-a5-mtefullmesh-device-侧简述)
- [6. A5 分支关键限制与风险点](#6-a5-分支关键限制与风险点)
- [7. 总结](#7-总结)

## 1. 分析范围

本文分析 `mc2/moe_distribute_dispatch_v2` 目录下 `MoeDistributeDispatchV2` 算子，重点关注 Ascend David V100/V120（A5）相关分支。代码中 A5 通过 `NpuArch::DAV_3510`、`__NPU_ARCH__ == 3510`、`ascend950`/`Ascend950` 等条件识别，对外文档中对应产品为 Ascend 950DT。

主要代码入口：

- ACLNN 接口层：`op_api/aclnn_moe_distribute_dispatch_v2.cpp`、`op_api/moe_distribute_dispatch_v2_base.cpp`
- 算子定义与 shape 推导：`op_host/moe_distribute_dispatch_v2_def.cpp`、`op_host/moe_distribute_dispatch_v2_infershape.cpp`
- Host tiling：`op_host/op_tiling/arch35/moe_distribute_dispatch_v2_tiling_arch35.cpp`
- A5 CCU device kernel：`op_kernel/arch35/moe_distribute_dispatch_arch35.h`
- A5 MTE/fullmesh device kernel：`op_kernel/moe_distribute_dispatch_v2.h`、`op_kernel/moe_distribute_dispatch_v2_full_mesh.h`
- Tiling key：`op_kernel/moe_distribute_dispatch_v2_tiling_key.h`
- 任务生成与融合：`op_graph/moe_distribute_dispatch_v2_gen_task.cpp`、`op_graph/fusion_pass/moe_distribute_dispatch_v2_fusion_pass.cpp`

## 2. 算子功能

`MoeDistributeDispatchV2` 是 MoE 模型中 Dispatch 阶段的通信与重排算子。它根据每个 token 的 `expertIds`，把本卡 token 分发给 EP（Expert Parallelism）通信域中持有目标专家的 rank，并输出目标专家侧连续排布的 token。输出会作为后续 `MoeDistributeCombineV2` 或 `MoeDistributeCombineAddRmsNorm` 等 Combine 系列算子的输入。

核心功能如下：

1. 可选量化

   算子可以在 dispatch 前对 token 做量化，支持非量化、静态量化、per-token 动态量化、per-group 动态量化和 MX 量化。A5 分支支持更多输出 dtype，包括 `INT8`、`HIFLOAT8`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2`，以及部分 `FLOAT4` 类型。

2. 按专家路由 token

   输入 `x` 的每个 token 对应 `K` 个专家。算子依据 `expertIds[BS, K]` 将 token 复制/展开到对应专家侧，形成 `expandXOut[A, H]`。其中 `A` 是本 rank 作为接收方时最多可能接收并输出的 token 数。

3. EP 域 AllToAllV 通信

   对于 A5，实际主路径为 EP 域内 `AllToAllV`/`HalfAllToAllV` 类通信。不同 rank 发送 token 数不同，因此使用 V 形态的可变长通信。

4. 共享专家处理

   `sharedExpertRankNum` 个 rank 可以放在 EP 域前部作为共享专家 rank。共享专家 token 发送目标与普通 MoE 专家不同，A5 CCU kernel 中单独分配一部分 AIV 处理共享专家 scatter。

5. 输出 Combine 辅助信息

   相比旧版 `MoeDistributeDispatch`，V2 不再直接输出 `(BS * K,)` 的 `expandIdx`，而输出 `assistInfoForCombineOut[A * 128]`。代码中该输出仍复用部分变量名 `expandIdx`，但语义是给 Combine 系列算子使用的辅助信息。用户侧不应依赖其内部布局。

6. 可选 token mask

   A5 CCU 分支支持 1D `xActiveMask`。mask 为 true 的前缀 token 参与通信，false 之后 token 不参与。Host tiling 对 A5 CCU 的 mask 维度约束为 1D。

### 2.1 A5 通信路径

A5 分支不是单一路径，实际由 `commAlg` 决定：

| `commAlg` | 接口/图优化路径 | Host tiling | Device kernel | 说明 |
| --- | --- | --- | --- | --- |
| `ccu` | V2 内部接口 | A5 专用 `MoeDistributeDispatchTilingImpl(..., OP_VERSION_2)` | `MoeDistributeDispatchA5` | 使用 CCU HCCL server，本文 Device 侧主分析对象 |
| `nullptr`/`""` | A5 下默认转 V3 内部接口，图融合会把 V2 融合到带 `Mc2Context` 的 V3 | 复用 A3/MTE tiling 公共实现，默认 fullmesh_v1 | `MoeDistributeDispatchV2` 或 `MoeDistributeDispatchV2FullMesh` | MTE/UB memory 通信路径 |
| `fullmesh_v1` | V3/MTE 路径 | A5 wrapper 调 A3 公共 tiling | 普通 fullmesh | 不走 A5 CCU kernel |
| `fullmesh_v2` | V3/MTE 路径 | A5 wrapper 调 A3 公共 tiling，打开 fullmesh 模板 | `MoeDistributeDispatchV2FullMesh` | 仅 tpWorldSize=1 场景 |

任务生成中也有相同判断：A5 + `ccu` 使用 `Mc2Arch35GenTaskCallBack`；A5 + `mte/fullmesh_v1/fullmesh_v2` 使用 `Mc2MoeGenTaskCallbackV2`。

## 3. 算子输入/输出

### 3.1 关键符号

| 符号 | 含义 |
| --- | --- |
| `BS` | 本 rank 输入 token 数，`x.shape[0]` |
| `H` | hidden size，`x.shape[1]` |
| `K` | 每个 token 选择的专家数，`expertIds.shape[1]` |
| `epWorldSize` | EP 通信域 rank 数 |
| `epRankId` | 本 rank 在 EP 域中的 id |
| `sharedExpertRankNum` | EP 域前部共享专家 rank 数 |
| `sharedExpertNum` | 共享专家数量 |
| `moeExpertNum` | MoE 专家总数，不含共享专家 |
| `moeRankNum` | `epWorldSize - sharedExpertRankNum` |
| `localMoeExpertNum` | 普通 MoE rank 上每 rank 专家数，`moeExpertNum / moeRankNum` |
| `localExpertNum` | 本 rank 输出专家数；共享专家 rank 为 1，普通 MoE rank 为 `localMoeExpertNum` |
| `globalBS` | EP 域全局 BS。为 0 时按 `BS * epWorldSize`；非 0 时通常为 `maxBS * epWorldSize` |
| `A` | 本 rank 输出 token 容量上界 |

### 3.2 A 的计算

Host infer shape 与 tiling 都围绕 `A` 推导输出容量。

普通 MoE rank：

```text
localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)
A = globalBSReal * min(localMoeExpertNum, K)
```

共享专家 rank（A5 V2 tiling 代码中 OP_VERSION_2 逻辑）：

```text
maxBS = globalBS / epWorldSize
rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum
maxSharedGroupNum = ceil(epWorldSize / rankNumPerSharedExpert)
A = maxBS * maxSharedGroupNum
```

当 `sharedExpertNum=1` 且 `sharedExpertRankNum=0` 时，tiling 会把它视为无共享专家场景。

### 3.3 输入

| 参数 | 含义 | 维度/shape | dtype/format | 来源 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `x` | 本 rank 待发送 token 特征 | `(BS, H)` | A5 支持 `FLOAT16`、`BFLOAT16`、`HIFLOAT8`、`FLOAT8_E5M2`、`FLOAT8_E4M3FN`、`FLOAT4_E2M1`、`FLOAT4_E1M2`；ND | MoE router 前的 hidden states | 被按 `expertIds` 分发给目标专家；量化模式下作为量化输入 |
| `expertIds` | 每个 token 的 topK 专家 id | `(BS, K)` | `INT32`，ND | Router/topK 选择结果 | 决定 token 发往哪个专家、哪个 EP rank |
| `scalesOptional`/`scales` | 平滑/量化 scale | 依量化模式不同 | `FLOAT32` 或 `FLOAT8_E8M0`，ND | 量化参数或低精度输入的 scale | 静态量化、动态量化平滑、非量化 FP8/FP4 携带 scale |
| `xActiveMaskOptional` | token 是否参与通信 | A5 CCU host tiling 要求 1D `(BS,)`；MTE 文档中还描述 2D 场景 | `BOOL`，ND | 可变有效 token 场景 | 只处理 mask 为 true 的前缀 token；A5 CCU kernel 通过 sum 得到 `activeBs` |
| `expertScalesOptional` | token 的专家权重 | `(BS, K)` | `FLOAT32`，ND | Router 权重 | V2 A5 CCU kernel 未实际使用；hierarchy/MTE 类路径可用于输出 `expandScalesOut` |
| `elasticInfoOptional` | EP 动态缩容信息 | 文档未固定 shape | `INT32`，ND | 通信域弹性缩容信息 | MTE/fullmesh 路径使用；A5 CCU kernel 未使用 |
| `performanceInfoOptional` | 通信耗时统计 | 文档未固定 shape | `INT64`，ND | 性能统计 tensor | A5 接口层会将该参数置空，CCU kernel 未使用 |

### 3.4 属性

| 属性 | 含义 | A5 主要约束 |
| --- | --- | --- |
| `groupEp` | EP 通信域名称 | 非空，长度 `(0, 128)` |
| `epWorldSize` | EP 域大小 | V2 A5 CCU tiling 要求 `[2, 768]` |
| `epRankId` | EP rank id | `[0, epWorldSize)` |
| `moeExpertNum` | MoE 专家总数 | `(0, 1024]`，且能被 `epWorldSize - sharedExpertRankNum` 整除 |
| `groupTp` | TP 通信域名称 | A5 CCU 当前实际不支持 TP，`tpWorldSize` 必须为 1 |
| `tpWorldSize` | TP 域大小 | CCU tiling 中检查必须为 1；MTE 文档中 A5 也标注不支持 TP |
| `tpRankId` | TP rank id | 无 TP 时为 0 |
| `expertShardType` | 专家分片类型 | 当前仅支持 0，表示共享专家 rank 排在 MoE rank 前 |
| `sharedExpertNum` | 共享专家数量 | V2 A5 CCU 支持 `[0, 4]`；与 `sharedExpertRankNum` 同为 0 或同非 0 |
| `sharedExpertRankNum` | 共享专家 rank 数 | `[0, epWorldSize)`；非 0 时需被 `sharedExpertNum` 整除 |
| `quantMode` | 量化模式 | A5 支持 0/1/2/3/4 |
| `globalBS` | EP 域全局 BS | 0 或 `maxBS * epWorldSize`；非 0 时不能小于本 rank `BS * epWorldSize`，且需被 `epWorldSize` 整除 |
| `expertTokenNumsType` | `expertTokenNumsOut` 语义 | 0 表示前缀和，1 表示各专家 token count |
| `commAlg` | 通信算法 | A5 代码支持 `ccu`、`fullmesh_v1`、`fullmesh_v2`、空值；`ccu` 走 A5 CCU 专用路径 |

### 3.5 输出

| 参数 | 含义 | 维度/shape | dtype/format | 目的 |
| --- | --- | --- | --- | --- |
| `expandXOut`/`expand_x` | 按专家重排后的 token 特征 | `(A * max(tpWorldSize, 1), H)`；A5 CCU 中实际 `tpWorldSize=1` | 非量化通常同 `x`；量化可为 `INT8`、`HIFLOAT8`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2`、`FLOAT4_E2M1`、`FLOAT4_E1M2`；ND | 作为本 rank 本地专家计算输入 |
| `dynamicScalesOut`/`dynamic_scales` | 动态量化 scale 输出，或低精度非量化 scale 携带输出 | A5 infer shape：per-token 为 `(A,)`；per-group 为 `(A, ceil(H/128))`；MX 为 `(A, even(ceil(H/32)))`；非量化且有 scales 时为 `(A, scales.shape[1])` | `FLOAT32` 或 `FLOAT8_E8M0` | Combine/后续反量化或低精度 scale 对齐使用 |
| `assistInfoForCombineOut`/`assist_info_for_combine` | 给 Combine 使用的辅助信息 | `(A * 128,)` | `INT32`，ND | CombineV2 系列恢复原 token/专家路由关系。内部布局随产品和算法变化，业务不应解析 |
| `expertTokenNumsOut`/`expert_token_nums` | 本 rank 每个本地专家收到的 token 数 | 共享专家 rank 为 `(1,)`；普通 MoE rank 为 `(localMoeExpertNum,)` | `INT64`，ND | 专家计算和 Combine 确定每个专家的 token 分段 |
| `epRecvCountsOut`/`ep_recv_count` | EP 域各发送 rank 对本 rank 本地专家的 token 数 | A5 CCU：共享专家 rank 近似为 `epWorldSize`；普通 rank 为 `epWorldSize * localMoeExpertNum`；输出 shape 至少满足该大小 | `INT32`，ND | Combine 反向通信或同步 token 分布 |
| `tpRecvCountsOut`/`tp_recv_count` | TP 域接收计数 | A5 当前不支持 TP，shape 通常为 `(1,)` | `INT32`，ND | TP allgather 场景使用；A5 CCU 无实际 TP |
| `expandScalesOut`/`expand_scales` | 展开后的专家权重 | A5 CCU 不支持，infer shape 置 0；hierarchy 路径可为 `(A,)` | `FLOAT32`，ND | 带 expert scales 的 Combine/融合场景 |

## 4. Host Tiling 切分原则

### 4.1 分支选择

A5 Host tiling 的入口是 `MoeDistributeDispatchV2TilingFuncA5::MoeDistributeDispatchTilingFuncImpl`。

1. 读取 `commAlg`。
2. 如果 `commAlg == "ccu"`，调用 A5 专用 `MoeDistributeDispatchTilingImpl(context, OP_VERSION_2)`。
3. 如果 `commAlg` 为空、`fullmesh_v1` 或 `fullmesh_v2`，走 MTE 路径，调用 `MoeDistributeDispatchA3TilingFuncImplPublic` 复用公共 tiling。
4. 其他值报错。

A5 non-ccu 图优化会把 V2 节点融合为带 `Mc2MoeContext` 的 V3 节点；`ccu` 节点不融合。

### 4.2 参数合法性检查

A5 CCU tiling 做以下核心检查：

- `groupEp` 非空，`groupTp` 在 TP 开启时非空。
- `epWorldSize` 范围 `[2, 768]`。
- `tpWorldSize` 最终要求为 1，当前 CCU 路径不支持 TP 通信。
- `moeExpertNum` 范围 `(0, 1024]`，且 `moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0`。
- `sharedExpertNum` 与 `sharedExpertRankNum` 要么都为 0，要么都非 0；非 0 时 `sharedExpertRankNum % sharedExpertNum == 0`。
- `BS` 范围 `(0, 512]`。
- `H` 范围 `[1024, 8192]`。
- `K` 范围 `(0, min(16, moeExpertNum)]`。
- `globalBS` 为 0 或合法的 `maxBS * epWorldSize`。
- 如果存在 token mask 且 `globalBS > BS * epWorldSize`，报错，因为不同 rank BS 时不能再用 active mask。

### 4.3 dtype 与 quantMode 匹配

A5 先用 `(quantMode, expandX dtype)` 映射到真实模式 `RealModeA5`：

| `quantMode` | 真实模式 | 支持的 `expandX` dtype |
| --- | --- | --- |
| 0 | `NO_SCALES` | `BF16`、`FLOAT16`、`HIFLOAT8`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2`、`FLOAT4_E2M1`、`FLOAT4_E1M2` |
| 1 | `STATIC_SCALES`/`HIF8_SCALES` | `INT8`、`HIFLOAT8` |
| 2 | `PERTOKEN_SCALES` | `INT8`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2` |
| 3 | `PERGROUP_SCALES` | `FLOAT8_E4M3FN`、`FLOAT8_E5M2` |
| 4 | `MX_SCALES` | `FLOAT8_E4M3FN`、`FLOAT8_E5M2`、`FLOAT4_E2M1`、`FLOAT4_E1M2` |

`scales` 与量化模式的匹配关系：

- `quantMode=4` 时 `scales` 必须为空。
- `quantMode=0` 且 `x` 为 `BF16/FLOAT16` 时 `scales` 必须为空。
- `quantMode=0` 且 `x` 为 `HIFLOAT8/FLOAT8/FLOAT4` 时 `scales` 必须非空。
- `quantMode=1` 静态量化时 `scales` 必须非空。
- `quantMode=2/3` 可选 `scales`，存在时作为 smooth scale。

### 4.4 scales shape 检查与 tiling 信息

Host tiling 会记录：

```text
scalesRow
scalesCol
scalesTypeSize
scalesCount
```

典型约束：

- 静态 INT8 量化的一维 `scales` 可为 `(1,)` 或 `(H,)`。
- 静态 HIF8 量化的一维 `scales` 必须为 `(1,)`。
- 二维 `scales` 在无共享专家时 dim0 为 `moeExpertNum`；有共享专家时 dim0 为 `moeExpertNum + sharedExpertNum`；dim1 必须等于 `H`。
- 非量化低精度携带 scale 时，代码允许按 token 输入 scale，dim0 需等于 `BS`。

### 4.5 输出 shape 校验

Host tiling 按计算出的 `A` 校验输出：

- `expandX.shape[0] >= A * tpWorldSize`
- `expandX.shape[1] == H`
- 动态量化模式下 `dynamicScales.shape[0] >= A * tpWorldSize`
- per-group：`dynamicScales.shape[1] == ceil(H / 128)`
- MX：`dynamicScales.shape[1] == even(ceil(H / 32))`
- `assistInfoForCombineOut.shape[0] >= BS * K` 的基础维度检查来自公共 helper；infer shape 中设置为 `A * 128`
- `expertTokenNumsOut.shape[0] == localExpertNum`
- `epRecvCountsOut.shape[0] >= epWorldSize * localExpertNum`
- `tpRecvCountsOut.shape[0] == tpWorldSize`

### 4.6 workspace 与通信窗口

A5 CCU tiling 的 workspace 由三部分组成：

```text
workspace =
    16MB system workspace
  + epWorldSize * sizeof(uint64_t) * 2 * 2
  + epWorldSize * (
        512
      + maxBS * Align512(alignedH * sizeof(x dtype)) * localMoeExpertNum
    )
```

其中：

- `maxBS = globalBS / epWorldSize`
- `alignedH` 默认按 32 对齐；MX 按 256 对齐；per-group 按 128 对齐。
- `512` 是每个 rank 数据块前部 count 区偏移。

通信窗口大小检查使用 A5 实际物理分配为 `HCCL_BUFFSIZE * 2` 的逻辑：

```text
actualSize =
  aivNum * 512
  + 2 * epWorldSize *
      (maxBS * Align512(alignedH * 2) * localMoeExpertNum + 512)
```

若 `actualSize > maxWindowSize * 2`，tiling 报错。

### 4.7 AIV 切分原则

A5 CCU device kernel 内的切分也由 tiling 参数决定：

- Host 通过平台信息获取 `aivNum`，设置 `blockDim = CalcTschBlockDim(aivNum, 0, aivNum)`。
- Device 侧以 `aivId = GetBlockIdx()` 区分 AIV。
- 存在共享专家时：
  - `sharedUsedAivNum = max(1, aivNum / (K + 1))`
  - `moeUsedAivNum = aivNum - sharedUsedAivNum`
  - 前部 AIV 处理普通 MoE scatter，后部 AIV 处理共享专家 scatter。
- 无共享专家时，几乎全部 AIV 用于 MoE scatter。
- `SplitToCore` 按 `curSendCnt / curUseAivNum` 加余数前置的方式切分 token 或 rank，前若干 AIV 多处理 1 个元素。

### 4.8 Tiling key

A5 CCU 专用 tiling key 由以下维度编码：

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

MTE/fullmesh 路径使用 `commMode=TILINGKEY_TPL_MTE`，并通过 `fullMesh` 字段区分普通 fullmesh、fullmesh_v2 或 hierarchy。

## 5. Device 侧功能详细介绍

本节以 A5 CCU kernel `MoeDistributeDispatchA5` 为主。

### 5.1 Kernel 模板选择

`op_kernel/moe_distribute_dispatch_v2.cpp` 中，A5 且 `CommMode == TILINGKEY_TPL_CCU` 时实例化：

```cpp
MoeDistributeDispatchA5<XType, ExpandXOutType, QuantMode, ScaleMode, false>
```

非 CCU 的 A5 MTE 分支则实例化：

- `MoeDistributeDispatchV2`
- `MoeDistributeDispatchV2FullMesh`

两者都完成 dispatch，但 CCU 分支使用 A5 专用 HCCL CCU server 与专用 send/recv buffer 布局。

### 5.2 Init 阶段

`Init` 完成以下动作：

1. 设置 float overflow 控制寄存器。
2. 读取 tiling 中的全局属性：
   - `BS/H/K`
   - `epWorldSize/epRankId`
   - `moeExpertNum/sharedExpertRankNum/sharedExpertNum`
   - `aivNum/expertTokenNumsType`
   - `scalesCount`
3. 计算：
   - `moeExpertRankNum = epWorldSize - sharedExpertRankNum`
   - `localExpertNum = moeExpertNum / moeExpertRankNum`
   - `bskNum = BS * K`
   - `axisMaxBs = globalBS / epWorldSize`
   - `activeBs = BS`，若有 mask 后续重新计算
4. 绑定 GM tensor：
   - `xGT_`
   - `expertIdsGT_`
   - `expandXOutGT_`
   - `expandIdxGT_`，实际对应 `assistInfoForCombineOut`
   - `dynamicScaleGT_`
   - `expertTokenNumsGT_`
   - `sendCountsGT_`，实际对应 `epRecvCountsOut`
   - 可选 `scalesGT_`、`xActiveMaskGT_`
5. 初始化 UB buffer：
   - 专家 id、排序索引、频次直方图、累计和 buffer
   - token 输入/输出 queue
   - mask 计算 buffer
6. 计算每 token 的通信打包大小：
   - `perTokenInSize = H * sizeof(XType)`
   - `perTokenOutSize = H * sizeof(ExpandXOutType)`
   - `perTokenMergeSize` 根据量化模式追加 dynamic scale 区
   - `perTokenCommSize = Align512(perTokenMergeSize)`
   - `perRankDataSize = 512 + perTokenCommSize * axisMaxBs * localExpertNum`
7. 初始化 CCU HCCL 上下文和双缓冲接收窗口：
   - `sendBufGM_ = workspaceGM`
   - `sendSizeGM_`、`sendOffsetGM_` 紧跟 send buffer
   - `recvBufGM_` 指向 HCCL window，按 status 在前/后半窗口间 ping-pong

### 5.3 token mask 处理

当 `isTokenMaskFlag_` 为 true：

1. 将 `xActiveMask[BS]` 从 GM 拷到 UB。
2. bool 转 half。
3. 调 `Sum` 得到 active token 数 `activeBs`。
4. 更新：

```text
activeBsKNum = activeBs * K
sortNum = Align(activeBsKNum, 32)
sortRepeat = ceil(activeBsKNum / 32)
```

后续 scatter 只处理前 `activeBs` 个 token 对应的 `activeBsKNum` 个 `(token, topK)` 项。

### 5.4 TokenScatter：排序、统计与发送 buffer 填充

`TokenScatter` 是 dispatch 前半段核心。

1. 读取 `expertIds[activeBs * K]` 到 UB。
2. 用 `Sort` 对 expert id 排序：
   - 代码先将 expert id 转 float 并乘以 -1，再做升序 sort，之后乘回 -1。
   - 这样可以得到按 expert id 聚集的 token/topK 索引。
3. `GetExpertFreq` 统计每个专家的 token 数。
4. `GetExpertCumSum` 计算每个专家的 token 累计数。
5. 根据 AIV 分工进入：
   - `MoeScatter`：普通 MoE 专家 token。
   - `ShareScatter`：共享专家 token。

### 5.5 普通 MoE scatter

`MoeScatterCopyTokens` 遍历排序后的 `(token, expert)`：

```text
expertId = sortedOutI32LT[tokenId]
toExpertRankId = expertId / localExpertNum
toRankId = toExpertRankId + sharedExpertRankNum
rankStartExpertId = toExpertRankId * localExpertNum
preCount = tokenId - expertCumSum[rankStartExpertId - 1]
tokenIndex = sortedIndex1[tokenId] / K
```

目标地址：

```text
sendBuf
  + 512 * (toRankId + 1)
  + (toRankId * axisMaxBs * localExpertNum + preCount) * perTokenCommSize
```

也就是说，每个目标 rank 对应一段 `perRankDataSize`，开头 512B 存 count，后面按 token 粒度连续存放打包后的 token。

随后 `MoeScatter` 写每个目标 rank 的：

- 发送 size，拆成两段 half size，写入 `sendSizeGM_`
- 发送 offset，写入 `sendOffsetGM_`
- 各本地专家 token count，写入 send buffer 的 count 区

### 5.6 共享专家 scatter

共享专家发送目标不是由 `expertId / localExpertNum` 决定，而由共享专家分组决定：

```text
toSharedExpertIdx = sendCnt / activeBs
sharedExpertRankIdInGroup = epRankId % rankNumPerSharedExpert
toSharedRankId = rankNumPerSharedExpert * toSharedExpertIdx + sharedExpertRankIdInGroup
```

`ShareScatterCopyTokens` 按 `activeBs * sharedExpertNum` 生成共享专家 token，写入对应共享专家 rank 的 send buffer。`SetShareExpertSendSize` 为共享专家 rank 写入发送 size/offset/count。

### 5.7 ProcessToken：量化与 token 打包

`ProcessToken` 对单个 token 执行：

1. 从 `x[tokenIndex, :]` 拷到 UB。
2. 如果 `QuantMode > 0`，执行 `QuantProcess`。
3. 如果 `QuantMode == 0`：
   - 直接拷贝 token。
   - 若 `IsSmoothScaleExist`，在 token 后追加 scale 字节。
4. 将打包结果写入目标 rank 的 send buffer。

各量化模式：

#### 非量化 `quantMode=0`

- `BF16/FLOAT16`：直接拷贝。
- 低精度 `HIFLOAT8/FLOAT8/FLOAT4` 且 `scales` 存在：token 后拼接 scale，通信后再写到 `dynamicScalesOut`。

#### 静态量化 `quantMode=1`

`QuantStatic`：

- `x` cast 到 fp32。
- 根据 `scalesCount` 选择 scalar、按 H、或按 expert 的 scale。
- fp32 乘 scale。
- 输出 `INT8` 或 `HIFLOAT8`。

#### per-token 动态量化 `quantMode=2`

`QuantDynamicPerToken`：

- `x` cast 到 fp32。
- 可选乘 smooth scale。
- 计算 `max(abs(x))`。
- `dynamicScale = dtypeMax / maxAbs`。
- token 乘 `dynamicScale` 后 cast 到输出 dtype。
- 在 token 尾部写 `1 / dynamicScale`，通信后写入 `dynamicScalesOut`。

#### per-group 动态量化 `quantMode=3`

`QuantDynamicPerTile`：

- block size 为 128。
- 可选 smooth scale。
- 调 `quant::ComputePerTileDynamic` 计算每组 scale 和 FP8 数据。
- token 尾部追加 `ceil(H / 128)` 个 float scale。

#### MX 量化 `quantMode=4`

`QuantDynamicMxFp8`：

- block size 为 32。
- 计算每组最大指数 `maxExp`。
- 生成 E8M0 scale。
- 计算 FP8 数据。
- token 尾部追加 `even(ceil(H / 32))` 个 E8M0 scale。

### 5.8 Communication：CCU AllToAllV

`Communication` 流程：

1. `SyncAll` 确保所有 AIV 完成 send buffer 构造。
2. `aivId == 0` 计算本 rank 实际发送数据大小：
   - 普通 MoE rank：累加本 rank 本地专家对应的 token 数。
   - 共享专家 rank：`activeBs * perTokenCommSize`。
3. 调用：

```cpp
hccl_.AlltoAllvWrite<true>(
    sendBufGM_,
    sendOffsetGM_,
    sendSizeGM_,
    recvOffset_,
    localDataSize)
```

4. 同时调用 `SaveExpandIdx` 生成 Combine 辅助索引。
5. 初始化 gather 所需 recv count buffer。
6. `aivId == 0` 等待 HCCL 完成。
7. `SyncAll` 确保所有 AIV 可读 recv buffer。

### 5.9 SaveExpandIdx / assistInfoForCombineOut

CCU kernel 中输出参数名仍叫 `expandIdxGT_`，实际绑定的是 `assistInfoForCombineOut`。`SaveExpandIdx` 会根据排序索引和专家累计计数，生成 token 在专家连续输出中的位置辅助信息。

注意：对外文档明确说明 `assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut`、`expandScalesOut` 的内部元素值会随产品、算法、版本变化，业务逻辑不应解析，只需要原样传给 Combine 系列算子。

### 5.10 TokenGather：接收 buffer 重排到输出

`TokenGather` 做 dispatch 后半段：

1. 初始化 `tokenGatherQue_`。
2. 共享专家 rank 视为 `localExpertNum=1` 的特殊 MoE rank。
3. `MoeGatherPrepare` 从 `recvBufGM_` 读取各 rank count：
   - `recvCntLT[localExpert, epRank]`
   - reduce 得到每个本地专家总 token 数 `moeExpertTokenNumsLT`
   - reduce 得到 rank 前缀和 `moeExpertRankTokenCumSumLT`
4. 对每个本地专家、每个发送 rank：
   - 从 `recvBufGM + rank * perRankDataSize + 512 + rankInPreCnt * perTokenCommSize` 读取 token。
   - 写到 `expandXOut[outPreCount, :]`。
   - 若 token 尾部带 scale，调用 `CopyScalesToOut` 写入 `dynamicScalesOut`。
5. `UpdateMoeRankOutput` 写：
   - `epRecvCountsOut[expertIndex, rank]`
   - `expertTokenNumsOut[expertIndex]`

`expertTokenNumsType` 影响 `expertTokenNumsOut`：

- 0：写本地专家 token 数的前缀和。
- 1：写每个专家自己的 token 数。

### 5.11 A5 MTE/fullmesh device 侧简述

当 A5 不走 `ccu` 时，device 侧使用公共 MTE/fullmesh kernel：

```text
AlltoAllDispatch()
SetStatus()
WaitDispatch()
LocalWindowCopy()
SetExpertTokenNumsA5()
```

若模板启用 TP allgather，则后续还有：

```text
AllGatherSetStatusAndWait()
AllgatherProcessOut()
AllGatherSetExpertTokenNumsAndTpRecvCount()
```

但 A5 对外约束当前不支持 TP 通信，实际常见路径是 EP AllToAllV 后本地窗口拷贝和 expert token num 输出。

## 6. A5 分支关键限制与风险点

1. `commAlg` 文档与代码存在版本差异

   V2 文档的 Ascend 950DT 小节写法偏保守，部分位置说 `commAlg` 当前不支持、传空指针即可；但当前代码已支持 `ccu`、`fullmesh_v1`、`fullmesh_v2` 和空值。分析实现时应以代码为准。

2. `ccu` 路径不走 V3 `Mc2Context` 新方案

   `op_api/moe_distribute_dispatch_v2_base.cpp` 中注释说明 `ccu` 暂时不支持新方案；因此 `ccu` 会走 V2 内部接口和 A5 专用 tiling/kernel。

3. A5 CCU 不支持 TP

   Host tiling 中直接要求 `tpWorldSize == 1`。虽然通用 kernel 存在 TP allgather 模板参数，但 A5 CCU 这条主线不使用。

4. `expertScalesOptional`、`elasticInfoOptional`、`performanceInfoOptional` 在 A5 CCU 中不是主功能输入

   A5 CCU kernel `MoeDistributeDispatchA5::Init` 不接收这些参数；接口层还会在 A5 将 `performanceInfoOptional` 置空。它们主要服务其他产品或 MTE/hierarchy/fullmesh 路径。

5. `assistInfoForCombineOut` 内部语义不要外部依赖

   Device 代码中变量仍叫 `expandIdx`，但 V2 对外输出是 `assistInfoForCombineOut[A * 128]`。其布局是 Combine 私有协议。

## 7. 总结

A5 分支的 `MoeDistributeDispatchV2` 本质是一个 MoE dispatch 通信算子：先依据 router 输出的 `expertIds` 将 token 按专家排序、统计和打包，可选执行量化，再通过 EP 域 AllToAllV 把 token 发送给目标专家所在 rank，最后在接收端按本地专家连续写出 `expandXOut`、`dynamicScalesOut` 和 Combine 所需辅助信息。

在 A5 上，`commAlg == "ccu"` 是专用 CCU 实现，host tiling 和 device kernel 都有 A5 独立代码；非 `ccu` 路径则更多复用 A3/MTE/fullmesh 公共逻辑，并可能通过图融合转为带 `Mc2Context` 的 V3 内部接口。
