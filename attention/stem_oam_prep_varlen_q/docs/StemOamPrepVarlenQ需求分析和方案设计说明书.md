# stem_oam_prep_varlen_q 需求分析和方案设计说明书

## 1 需求分析

### 1.1 需求背景

Stem OAM (Output-Aware Metric) 是大模型推理中的动态稀疏注意力机制，用于在 block-sparse attention 场景下实现高效的 token 级稀疏选择。Stem Prep Varlen Q 算子负责从变长 Q tensor 中提取 Q 数据，经 Scale Fusion (qScale) → Weighted Group Sum → Natural-order Flatten → Cast BF16 输出 qFlat，供后续 OAM score 计算。

- **需求来源**: Stem 论文 ：https://arxiv.org/pdf/2603.06274

| 项目 | 内容 |
|-----|------|
| 算子名称 | stem_oam_prep_varlen_q |
| 算子类别 | Reduction + Conversion (复合型) |
| 支持数据类型 | FP8_e4m3fn (输入) → BF16 (输出) |
| **目标芯片** | Ascend950DT |
| **目标架构** | arch35 (DAV_3510) |

**适用场景**: 融合算子场景（Scale Fusion + De-page Varlen + Weighted Group Sum + Flatten + Cast 多步组合，仅 FP8 输入路径）

**模型结构分析**: Stem OAM attention 中 Q 侧预处理将 varlen Q 从分页存储格式转化为按 stem block（128 token）分组的 qFlat 向量，用于与 kFlat/vBias 计算稀疏 attention score。与 K 侧预处理的关键区别在于 Q 侧不翻转 stride 维度（自然顺序）。


### 1.2 基准分析

**基准接口**: test_stem.py (Q Prep path)
```python
def _prep_q_reference(
    q_fp8: torch.Tensor,
    qscale: torch.Tensor,  # [total_tokens, H_q]
    q_seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    stem_block_size: int = 128,
    stem_stride: int = 16,
):
    """Pure PyTorch reference matching CUDA stem_oam_prep_varlen_q kernel.

    Q prep is identical between legacy and new quant_type paths — Q-scale is
    per-token-per-head in both cases and is independent of K/V scale layout.
    """
    device = q_fp8.device
    num_head_q = q_fp8.shape[1]
    dim_qk = q_fp8.shape[2]
    num_batch = q_seq_lens.shape[0]
    R = stem_block_size // stem_stride

    q_padded_lens = (
        (q_seq_lens.to(torch.int64) + stem_block_size - 1)
        // stem_block_size * stem_block_size
    ).to(torch.int32)
    max_q_padded = q_padded_lens.max().item()
    max_Qb = max_q_padded // stem_block_size

    qflat_out = torch.zeros(
        num_batch, num_head_q, max_Qb, stem_stride * dim_qk,
        dtype=torch.bfloat16, device=device,
    )

    for b in range(num_batch):
        q_len = q_seq_lens[b].item()
        q_padded = q_padded_lens[b].item()
        num_q_blocks = q_padded // stem_block_size
        cu_off = cu_seqlens_q[b].item()

        q_dense = torch.zeros(
            q_padded, num_head_q, dim_qk, dtype=torch.float32, device=device,
        )
        q_dense[:q_len] = q_fp8[cu_off:cu_off + q_len].float()

        for h in range(num_head_q):
            q_h = q_dense[:num_q_blocks * stem_block_size, h, :]
            q_blocks = q_h.reshape(num_q_blocks, R, stem_stride, dim_qk)
            positions = torch.arange(
                num_q_blocks * stem_block_size, dtype=torch.long, device=device,
            ).reshape(num_q_blocks, R, stem_stride)
            scales = qscale[cu_off + positions, h]  # per-token scale
            q_group_sum = (q_blocks * scales.unsqueeze(-1)).sum(dim=1)
            qflat_out[b, h, :num_q_blocks] = q_group_sum.reshape(
                num_q_blocks, stem_stride * dim_qk,
            ).to(torch.bfloat16)

    return qflat_out

```

**功能逻辑**: varlen slice → scale fusion (qScale) → weighted/unweighted group sum → flatten (natural order) → cast BF16/FP16

**数学公式**:

**阶段 0: 输入参数**

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| q | `[total_tokens, H_q, D=128]` | FP8_e4m3fn | 变长 Q tensor |
| qSeqLens | `[batch]` | INT32 | 每 batch Q 序列长度 |
| cuSeqLensQ | `[batch+1]` | INT32 | Q 偏移前缀和 |
| qScale | `[total_tokens, H_q]` | FP32 | Q per-token scale  |
| stemBlockSize (B) | scalar | INT | %32==0, ≤256 |
| stemStride (S) | scalar | INT | %16==0, ≤64, S≤B |

**典型case**:  
H_q = 8  
total_tokens = 1024k  
batch = 8 (最大值 1024)
H_kv = 2   
stemBlockSize = 128  
stemStride = 16  
max_seqlen = 256k  
kv_block_size = 128  
total_blocks = 9649

**阶段 1: Scale Fusion + De-page Varlen + Cast FP32**

$$
q\_scale[pos, h] = qScale[cu\_off + pos, h]
$$
q\_scale[pos, h] = qScale[cu\_off + pos, h]
$$

$$
q\_len[b] = qSeqLens[b],\quad cu\_off[b] = cuSeqLensQ[b],\quad Q\_seg[b] = q[cu\_off[b]:cu\_off[b]+q\_len[b],:,:]
$$

$$
q\_padded[b] = \lceil q\_len[b] / B \rceil \times B, \quad num\_Qb[b] = q\_padded[b] / B, \quad R = B / S
$$

$$
Q\_dense[b] = zeros(q\_padded,\ H\_q,\ D,\ FP32);\quad Q\_dense[:q\_len,:,:] = Cast(Q\_seg,\ FP32) \;\text{(zero-pad beyond } q\_len\text{)}
$$

**阶段 2: Weighted Group Sum (自然顺序, NO flip)**

$$
Q\_blocks[b,h] = reshape(Q\_dense[:num\_Qb \times B, h, :],\ [num\_Qb,\ R,\ S,\ D])
$$

$$
Q\_weighted[b,h,qb,r,g,:] = Q\_blocks[b,h,qb,r,g,:] \times q\_scale[cu\_off + qb \times B + r \times S + g, h]
$$

$$
Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_weighted[b,h,qb,r,g,:]
$$

**阶段 3: Flatten + Cast BF16**

$$
qFlat[b,h,qb,:] = reshape(Q\_group\_sum[b,h,qb,:,:],\ [kflat\_dim = S \times D]) \xrightarrow{Cast} BF16
$$

展开: $qFlat[b,h,qb,\ g \times D : (g+1) \times D] = Q\_group\_sum[b,h,qb,g,:]$, $g \in [0, S-1]$ (自然顺序)

**阶段 4: 输出**

| 输出 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| qFlat | `[batch, H_q, max_Qb, S×D]` | BF16 | Q weighted group sum (自然顺序) |

### 1.3 竞争分析

#### 1.3.1 接口

不涉及

#### 1.3.2 实现分析

不涉及

#### 1.3.3 优劣势

不涉及

### 1.4 需求分析总结

#### 1.4.1 aclnn API定义

每个算子分为两段式接口，必须先调用"aclnnStemOamPrepVarlenQGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnStemOamPrepVarlenQ"接口执行计算。

**函数原型**:

```cpp
aclnnStatus aclnnStemOamPrepVarlenQGetWorkspaceSize(
  const aclTensor   *q,
  const aclIntArray *qSeqLens,
  const aclIntArray *cuSeqLensQ,
  const aclTensor   *qScaleOptional,
  int64_t           stemBlockSize,
  int64_t           stemStride,
  aclTensor         *qFlat,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnStemOamPrepVarlenQ(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

**参数说明**:

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) |
|-------|---------|------|---------|-----------|
| q | 输入 | 变长 Q tensor，所有 batch 的 token 拼接存储 | FLOAT8_E4M3FN | [total_tokens, H_q, D] |
| qSeqLens | 输入 | 每个 batch 的 Q 序列长度 | INT32 | [batch] |
| cuSeqLensQ | 输入 | Q 的累积序列长度偏移量 | INT32 | [batch+1] |
| qScale | 输入(OPTIONAL) | Q 的 per-token scale factor，FP8 必填 | FLOAT | [total_tokens, H_q] |
| stemBlockSize | 输入 | stem block 大小，推荐默认 128 | INT64 | scalar |
| stemStride | 输入 | stem stride 大小，推荐默认 16 | INT64 | scalar |
| qFlat | 输出 | flattened Q 输出 | BFLOAT16 | [batch, H_q, max_Qb, kflat_dim] |

**返回值错误码**:

| 返回值 | 错误码 | 描述 |
|-------|------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | q、qFlat 存在空指针。qScale 在 FP8 输入时不允许空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | q dtype 不在支持范围；qScale dtype 不为 FLOAT；cuSeqLensQ 长度/单调性错误；stemBlockSize/stemStride 不满足约束；batch > 1024 |
| ACLNN_ERR_INNER | 361001 | 内部错误，如 Tiling 计算失败或 Kernel 查找失败 |

**约束说明**:
- 确定性说明: 默认确定性实现
- 仅支持 arch35 架构（Ascend 950PR/Ascend 950DT、Atlas A3 系列），不支持 arch22
- stemBlockSize % 32 == 0 且 stemBlockSize ≤ 256；stemStride % 16 == 0 且 stemStride ≤ 64 且 stemStride ≤ stemBlockSize。推荐默认值: stemBlockSize=128, stemStride=16
- qFlat 输出维度 kflat_dim = stemStride × D（默认 16×128=2048）
- Q 侧 stride 维度为自然顺序（g ∈ [0, S-1]），不翻转（与 K 侧处理不同）
- qSeqLens 中某 batch 值为 0 时，该 batch 对应的 qFlat 输出填充为零
- batch 最大值为 1024

#### 1.4.2 图模式IR

| 类别 | 名称 | 类型 | 类型范围 | 默认值 | Shape | 说明 |
|------|------|------|---------|--------|-------|------|
| INPUT | q | Tensor | FP8_E4M3FN | NA | [total_tokens, H_q, D] | 变长 Q tensor |
| INPUT | qSeqLens | IntArray | INT32 | NA | [batch] | Q 序列长度 |
| INPUT | cuSeqLensQ | IntArray | INT32 | NA | [batch+1] | Q 累积序列长度 |
| OPTIONAL_INPUT | qScale | Tensor | FP32 | NA | [total_tokens, H_q] | Q per-token scale, FP8必填 |
| ATTR | stemBlockSize | INT64 | {32,...,256} | 128 | NA | Stem block 大小，当前仅支持128 |
| ATTR | stemStride | INT64 | {16,...,64} | 16 | NA | Stride 大小，当前仅支持16 |
| OUTPUT | qFlat | Tensor | BF16/FP16 | NA | [batch, H_q, max_Qb, kflat_dim] | Q group sum 结果, FP16输入→FP16输出 |

### 1.5 约束与要求

当前仅支持：
head_dim=128；
stemBlockSize=128；
stemStride=16；

**精度要求**:
- FP8 → FP32 → BF16: rtol=1e-3, atol=1e-3

**边界情况处理**:
- qSeqLens 中存在 0 值 batch 时，该 batch 对应的 qFlat 填零
- 空 tensor (total_tokens=0) 时输出零 tensor
- batch 最大值为 1024，超出此限制返回错误

**确定性计算**: 默认支持。每个 (batch, head) pair 由单一 AI Core 处理，FP32 累加在单核内完成，无并行累加歧义。batch 最大值为 1024。

## 2 方案设计

### 2.1 API实现设计

**OpDef 定义** (`op_host/stem_oam_prep_varlen_q_def.cpp`)：

```cpp
// stem_oam_prep_varlen_q_def.cpp
#include "register/op_def_registry.h"

namespace ops {
class StemOamPrepVarlenQ : public OpDef {
 public:
  explicit StemOamPrepVarlenQ(const char* name) : OpDef(name) {
    this->Input("q")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("qSeqLens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND});
    this->Input("cuSeqLensQ")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND});
    // * optional inputs
    this->Input("qScale")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->Output("qFlat")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BFLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("stemBlockSize")
        .AttrType(OPTIONAL)
        .Int(128);
    this->Attr("stemStride")
        .AttrType(OPTIONAL)
        .Int(16);
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("opFile.value", "stem_oam_prep_varlen_q_apt");
    this->AICore().AddConfig("ascend950", aicore_config);
  }
};
OP_ADD(StemOamPrepVarlenQ);
}  // namespace ops
```

**模块职责表**：

| 模块 | 聣责 | 核心文件 |
|------|------|---------|
| **op_host** | 算子定义(OpDef) + Tiling 计算 + Shape 推导 | `_def.cpp`, `_infershape.cpp`, `arch35/_tiling.cpp`, `arch35/_tiling.h` |
| **op_kernel** | Kernel 实现 + TilingKey + TilingData | `arch35/.h`, `arch35/_tiling_key.h`, `arch35/_tiling_data.h`, `_arch35.cpp` |
| **op_api** | aclnn 接口封装 (GetWorkspaceSize + Execute) | `aclnn_stem_oam_prep_varlen_q.h`, `aclnn_stem_oam_prep_varlen_q.cpp`, `stem_oam_prep_varlen_q.h`, `stem_oam_prep_varlen_q.cpp` |

**模块依赖**：
```
op_api (ACLNN接口层)
├── aclnnStemOamPrepVarlenQGetWorkspaceSize()
│     ├──▶ op_host/_def.cpp: 获取算子定义
│     ├──▶ op_host/arch35/_tiling.cpp: 计算 TilingData (dtype→tilingKey, batch×H_q→total_tasks)
│     └──▶ 返回 workspaceSize
└── aclnnStemOamPrepVarlenQ()
      ├──▶ 加载 Kernel 二进制 (根据 tilingKey 选择模板)
      ├──▶ 传递 TilingData 到 Device
      └──▶ NPU Device 执行
```

**执行流程**：
```
aclnnStemOamPrepVarlenQGetWorkspaceSize()
     │
     ├──▶ op_host/_def.cpp: 获取算子定义
     ├──▶ op_host/_tiling.cpp: 计算 TilingData (dtype→tilingKey, batch×H_q→total_tasks)
     │
     ▼
aclnnStemOamPrepVarlenQ()
     │
     ├──▶ 加载 Kernel 二进制
     ├──▶ 传递 TilingData 到 Device
     │
     ▼
NPU Device 执行 (arch35 only)
     ├──▶ REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT
     ├──▶ GetBlockIdx() → 分配 (batch, head) pair
     ├──▶ GM → UB: q segment + scales (FP8 only)
     ├──▶ Cast → FP32 + Weighted/Unweighted Group Sum (R=stemBlockSize/stemStride 层累加)
     ├──▶ Flatten (自然顺序) + Cast FP32→BF16
     └──▶ UB → GM: qflat
```

### 2.2 图模式设计

**图模式算子原型定义**：

OpProto 定义文件：`op_graph/stem_oam_prep_varlen_q_proto.h`

```cpp
// stem_oam_prep_varlen_q_proto.h
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_VARLEN_Q_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_VARLEN_Q_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Stem OAM Varlen Q Prep: 从变长 Q tensor 提取 qFlat (weighted group sum).
 * @par Inputs:
 * @li q: [total_tokens, H_q, D] FP8_e4m3fn varlen Q tensor
 * @li qSeqLens: [batch] INT32, Q sequence lengths
 * @li cuSeqLensQ: [batch+1] INT32, cumulative Q sequence lengths
 * @par Optional Inputs (*):
 * @li qScale: [total_tokens, H_q] FP32, per-token scale (FP8 required)
 * @par Attributes:
 * @li stemBlockSize: INT, stem block size (default 128, must be multiple of 32, max 256)
 * @li stemStride: INT, stride size (default 16, must be multiple of 16, max 64, <= stemBlockSize)
 * @par Outputs:
 * @li qFlat: [batch, H_q, max_Qb, kflat_dim] BF16, Q group sum result
 * @par Constraints:
 * @li qScale is required for FP8 input
 * @li stemBlockSize must be multiple of 32, stemStride must be multiple of 16
 */
REG_OP(StemOamPrepVarlenQ)
    .INPUT(q, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(qSeqLens, TensorType({DT_INT32}))
    .INPUT(cuSeqLensQ, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(qScale, TensorType({DT_FLOAT}))
    .ATTR(stemBlockSize, Int, 128)
    .ATTR(stemStride, Int, 16)
    .OUTPUT(qFlat, TensorType({DT_BFLOAT16}))
    .END();
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_VARLEN_Q_PROTO_H_
```

**InferShape 逻辑**：  
需要设置qSeqLens为值依赖，然后根据qSeqLens来计算输出的shape
```
max_q_len = max(qSeqLens)
max_q_padded = ceil(max_q_len / stemBlockSize) × stemBlockSize
max_Qb = max_q_padded / stemBlockSize
kflat_dim = stemStride × dim_qk (D=128)

qFlat shape = [batch, H_q, max_Qb, kflat_dim]
```

**InferDtype 逻辑**：
```
qFlat dtype = BF16 (FP8/BF16 input) or FP16 (FP16 input)
```

### 2.3 总体规格设计
 
#### 2.3.1 模板划分表:

| 模板 | tilingKey | 触发条件 | 适用场景 |
|-----|----------|---------|---------|
| FP8 路径 | 0 | q.dtype == FP8_e4m3fn | FP8 反量化路径: 需要 qScale scale fusion → weighted group sum → BF16 输出 |

**分支差异对比** (仅 FP8 路径):

| 操作 | FP8 路径 |
|------|---------|
| Cast Q→FP32 | Cast\<fp8_e4m3fn_t, float\> |
| Scale Fusion | qScale (per-token 反量化) |
| Group Sum | Weighted: Q × q_scale, Muls + Add |
| Cast 输出 | Cast\<float, bfloat16_t\> |

#### 2.3.2 分核策略: 数据依赖驱动的分核

**数据依赖分析**:

| 操作 | 最小独立粒度 | 跨粒度依赖 |
|------|-------------|-----------|
| CopyInQBlock | (b,h,qb,row) | **无** — 每 Q 行独立从 GM 读取 |
| CastBlockToFP32 | (b,h,qb) | **无** — 单 qb 内 Cast |
| CopyInScalesBulk (FP8) | (b,h,qb) | **无** — B 个 scale 从 GM 独立读取 |
| AccumulateBlock | (b,h,qb) | **无** — group sum 仅累加当前 qb 内 R×S 行 |
| FlattenAndCast | (b,h,qb) | **无** — qFlat[b,h,qb,:] 完全独立 |
| CopyOutBlock | (b,h,qb) | **无** — GM 写入独立偏移 |

**结论**: Q Prep 的每个 (b,h,qb) 完全独立，**无跨 qb 数据依赖**。这与 KV Prep 的全局 Normalize 依赖（V Pass 需所有 kb 的 Σ(log)/Σ(log²)）形成鲜明对比。因此 Q Prep 模式 1/2 **无需 SyncAll**。  


**模式 0 (batch×H_q ≥ AICore数)**: 1 核/(batch,head)，无跨核同步

- task_id = GetBlockIdx()
- batch_idx = task_id / num_q_heads, head_idx = task_id % num_q_heads
- total_tasks = batch_size × num_q_heads
- 每个 (b,h) 核处理完整 qb 0..num_Qb-1 序列
- 超出 total_tasks 的核直接 return（空核）

**模式 1 (batch×H_q < AICore数, 模式 1)**: 按 (batch,head,qb_range) 分核，零 SyncAll

- 利用 Q Prep 无跨 qb 依赖的特性，将长序列的 qb 循环拆分到多个核
- 每个 (b,h) 分配 `cores_per_bh_base` 或 `cores_per_bh_base+1` 个核
- 每个 (b,h) 内的 qb 范围均匀分配到各核: core_in_bh 处理 [qb_start, qb_end)
- cu_off / q_len 由各核独立从 GM 读取（只读），无需同步

**模式 2 (flat block 分布)**: 按 block 总数均匀分核，零 SyncAll

- 计算总 block 数: total_blocks = H_q × Σ_b ceil(qSeqLens[b] / stemBlockSize)
- 将 total_blocks 均匀分配到 N 个核: blocks_per_core = total_blocks / N
- 每核获得 (batchIndex, headNumIndex, blockIndex, blockNum) 四元组
- 核从 batchIndex/headNumIndex 的 blockIndex 开始，处理 blockNum 个 block
- 若 blockNum 跨越 (b,h) 边界，核自动切换到下一个 (b,h) 并重读 varlen info
- 适用于 "seqLen 很大但 Batch 很小" 场景，比模式 1 负载更均衡
- 不同 batch 的 seqLen 不同时，加权分配自然产生：长 batch 的 block 占更多核


##### 2.3.2.1 模式 0  

**分核示意:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                    模式 0: AI Core 分核示意图                        │
│                                                                     │
│  task_id = 0        task_id = 1        ...   task_id = total_tasks-1│
│  ┌──────────┐      ┌──────────┐             ┌──────────┐           │
│  │ (b=0,h=0)│      │ (b=0,h=1)│             │(b=H_q-1) │           │
│  │ qb=0..N_Q│      │ qb=0..N_Q│             │ qb=0..N_Q│           │
│  │ cu_off=0 │      │ cu_off=0 │             │ cu_off=Σ │           │
│  └──────────┘      └──────────┘             └──────────┘           │
│                                                                     │
│  batch_idx = task_id / num_q_heads                                 │
│  head_idx  = task_id % num_q_heads                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Q 序列处理 (模式 0 单核内)**:

```
for qb = 0 .. num_Qb-1:
  CopyInQBlock → CastBlockToFP32 → [CopyInScalesBulk] → AccumulateBlock → FlattenAndCast → CopyOutBlock
```
##### 2.3.2.1 模式 1:

**Task 映射公式:**

```
P = batch × H_q  (b,h 总对数)
N = AICore 总数
cores_per_bh_base = N / P  (整数除法)
remainder_bh      = N % P  (前 remainder_bh 个 bh 对获得 +1 核)
big               = cores_per_bh_base + 1

task_id → (bh_idx, core_in_bh):
  if task_id < remainder_bh × big:
    bh_idx      = task_id / big
    core_in_bh  = task_id % big
  else:
    adj         = task_id - remainder_bh × big
    bh_idx      = remainder_bh + adj / cores_per_bh_base
    core_in_bh  = adj % cores_per_bh_base

  b = bh_idx / H_q
  h = bh_idx % H_q

qb 范围分配:
  cores_per_bh_this = (bh_idx < remainder_bh) ? big : cores_per_bh_base
  qpc  = num_Qb[b] / cores_per_bh_this
  qrem = num_Qb[b] % cores_per_bh_this

  if core_in_bh < qrem:
    qb_start = core_in_bh × (qpc + 1)
    qb_end   = qb_start + qpc + 1
  else:
    qb_start = qrem × (qpc + 1) + (core_in_bh - qrem) × qpc
    qb_end   = qb_start + qpc
```

**分核示意 (模式 1, batch=1, H_q=8, AICore=40)**:

```
P=8, N=40, cores_per_bh_base=5, remainder_bh=0, big=5

  task_id  0- 4: (b=0,h=0) core_in_bh=0..4  → qb [0..1638), [1638..3276), ... [6554..8192)
  task_id  5- 9: (b=0,h=1) core_in_bh=0..4  → 同上均分
  task_id 10-14: (b=0,h=2) core_in_bh=0..4
  ...
  task_id 35-39: (b=0,h=7) core_in_bh=0..4

  每个 (b,h) 5 核并行，各处理 ~1638 qb
  无 SyncAll，各核独立读 cu_off/q_len，独立写 qFlat 不同偏移
```

**分核示意 (模式 1, batch=1, H_q=32, AICore=40)**:

```
P=32, N=40, cores_per_bh_base=1, remainder_bh=8, big=2

  task_id  0-15: bh_idx 0..7 (big=2 cores each)
  task_id 16-39: bh_idx 8..31 (base=1 core each)

  8 个 (b,h) 获得 2 核 → 每核处理 ~4096 qb
  24 个 (b,h) 获得 1 核 → 处理全部 8192 qb (瓶颈)
  
  负载不平衡: 2:1 (cores_per_bh_base=1 时固有)
  随着 N/P 增大，不平衡缓解
```
**Q 序列处理 (模式 1 多核内)**:

```
for qb = qb_start .. qb_end-1:           ← 仅处理分配的 qb 范围
  CopyInQBlock → CastBlockToFP32 → [CopyInScalesBulk] → AccumulateBlock → FlattenAndCast → CopyOutBlock
```

##### 2.3.2.1 模式 2:

**Task 映射**:

```
Host (Tiling 函数):
  1. 读取 qSeqLens[b] → 计算 num_Qb_per_batch[b]
  2. total_blocks = H_q × Σ_b num_Qb_per_batch[b]
  3. blocks_per_core_base = total_blocks / N
  4. blocks_remainder = total_blocks % N
  5. 构建 batch_block_prefix: prefix[b] = Σ_{b'<b} H_q × num_Qb[b']
  6. 对每个核 core (0..N-1):
     global_start = core × blocks_per_core_base + min(core, blocks_remainder)
     block_num_core = blocks_per_core_base + (core < blocks_remainder ? 1 : 0)
     
     二分查找 batch_block_prefix → 找到 b
     local_in_batch = global_start - batch_block_prefix[b]
     h = local_in_batch / num_Qb_per_batch[b]
     qb = local_in_batch % num_Qb_per_batch[b]
     
     core_batch_idx[core] = b
     core_head_idx[core]  = h
     core_block_idx[core] = qb
     core_block_num[core] = block_num_core

Kernel:
  task_id → 读取 core_batch_idx/batch_idx/core_block_idx/core_block_num
  初始: batch_idx_, head_idx_, qb_start_, block_num_
  Process: while (remaining > 0) 循环
    avail = num_Qb_ - cur_qb  ← 当前(b,h)剩余 block
    step = min(remaining, avail)
    for i in 0..step-1: ProcessOneBlock(cur_qb+i)
    remaining -= step
    if remaining > 0: 切换到下一个 (b,h), ReadVarlenInfo(new_b)
```

**模式 2 分核示意 (batch=1, H_q=8, seqLen=1M, AICore=40)**:

```
total_blocks = 8 × 8192 = 65536
N=40, blocks_per_core_base=1638, blocks_remainder=16

  核 0:  (b=0, h=0, qb=0,    blockNum=1639)  ← 1639 blocks in (b=0,h=0)
  核 1:  (b=0, h=0, qb=1639, blockNum=1639)
  ...
  核 4:  (b=0, h=0, qb=6552, blockNum=1640)  ← 跨到 (b=0,h=1) 的 qb=0 开始
  核 5:  (b=0, h=1, qb=1640, blockNum=1638)  ← 完全在 (b=0,h=1) 内
  ...
  核 39: (b=0, h=4, qb=xxx,  blockNum=1638)

  每个 (b,h) 的 block 被多个核并行处理
  跨 (b,h) 边界的核自动切换 + 重读 varlen info
  无 SyncAll，各核独立
```

**模式 2 (batch=2, H_q=32, 不同 seqLen)**:

```
batch 0: seqLen=1M → num_Qb[0]=8192, blocks=32×8192=262144
batch 1: seqLen=100K → num_Qb[1]=781, blocks=32×781=24992
total_blocks = 287136, N=40, blocks_per_core ≈ 7178

  核 0..36: 全部在 batch 0 的 (b=0,h=0..32) 内处理
  核 37..39: 跨入 batch 1 的 block 空间

  自然加权: batch 0 占 91% 的 block → 占 91% 的核
```




#### 2.3.3 TilingData 字段与分核的关系:

| TilingData 字段 | 分核用途 | 计算方式 |
|---|---|---|
| split_mode | 选择分核粒度 | P≥N→0, P<N→1/2 |
| cores_per_bh_base | 模式1: 每(b,h)基准核数 | N/P 整数除法 |
| remainder_bh | 模式1: 获+1核的(b,h)数 | N%P |
| total_blocks | 模式2: 总 block 数 | H_q × Σ_b num_Qb[b] |
| blocks_per_core_base | 模式2: 每核基准 block 数 | total_blocks/N |
| blocks_remainder | 模式2: 前 remainder 核获+1 block | total_blocks%N |
| num_Qb_per_batch | 模式2: 每 batch 的 qb 数 | ceil(qSeqLens[b]/B) |
| core_batch_idx | 模式2: 每核起始 batch | prefix sum 二分查找 |
| core_head_idx | 模式2: 每核起始 head | (local_in_batch / num_Qb[b]) |
| core_block_idx | 模式2: 每核起始 qb | (local_in_batch % num_Qb[b]) |
| core_block_num | 模式2: 每核 block 数量 | blocks_per_core_base + (core<rem?1:0) |
| total_tasks | 决定核数 | 模式0=P, 模式1=N |
| used_core_num | 实际使用核数 | 模式0=min(P,N), 模式1=N |
| batch_size, num_q_heads | bh_idx→(b,h)映射 | bh_idx/H_q, bh_idx%H_q |
| max_q_padded | qScale stride | 已废弃，qScale shape 改为 [total_tokens, H_q] |
| max_Qb | qFlat stride | max_q_padded/B |
| total_tokens | Q GM 总 token 数 | q.size(0) |
| stem_block_size, stem_stride | Stem Block 结构参数 | aclnn API 传入 |
| kflat_dim | 输出每块宽度 | S × D |

#### 2.3.4 **TilingData 结构体**:

```cpp
struct StemPrepQTilingData {
    uint32_t batch_size = 0;
    uint32_t num_q_heads = 0;
    uint32_t dim_qk = 128;
    uint32_t stem_block_size = 128;
    uint32_t stem_stride = 16;
    uint32_t R_val = 8;
    uint32_t kflat_dim = 2048;
    uint32_t max_q_padded = 0;
    uint32_t max_Qb = 0;
    uint32_t total_tokens = 0;
    uint32_t total_tasks = 0;
    uint32_t used_core_num = 0;
    uint32_t split_mode = 0;           // 0=(b,h), 1=(b,h,qb_range), 2=flat block
    uint32_t cores_per_bh_base = 1;    // mode1: base cores per (b,h)
    uint32_t remainder_bh = 0;         // mode1: first remainder_bh bh-pairs get +1 core
    uint32_t total_blocks = 0;         // mode2: total block count
    uint32_t blocks_per_core_base = 0; // mode2: base blocks per core
    uint32_t blocks_remainder = 0;     // mode2: first remainder cores get +1 block
    uint32_t num_Qb_per_batch[256] = {}; // mode2: num_Qb for each batch
    uint32_t core_batch_idx[64] = {};  // mode2: per-core starting batch
    uint32_t core_head_idx[64] = {};   // mode2: per-core starting head
    uint32_t core_block_idx[64] = {};  // mode2: per-core starting qb
    uint32_t core_block_num[64] = {};  // mode2: per-core block count
};
```

### 2.4 FP8 路径设计

**触发条件**: q.dtype == FP8_e4m3fn, tilingKey = 0

**Host 侧 Tiling**:

```cpp
#include "stem_oam_prep_varlen_q_tiling_key.h"

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto qDesc = context->GetInputDesc(0);
    ge::DataType dtype_q = qDesc->GetDataType();
    uint32_t INPUT_TYPE = 0;
    if (dtype_q == ge::DT_BFLOAT16) INPUT_TYPE = 1;
    else if (dtype_q == ge::DT_FLOAT16) INPUT_TYPE = 2;
    ASCENDC_TPL_SEL_PARAM(context, INPUT_TYPE);

    auto tiling = context->GetTilingData<StemPrepQTilingData>();
    tiling->stem_block_size = stemBlockSize;  // from Attr
    tiling->stem_stride = stemStride;          // from Attr
    tiling->R_val = tiling->stem_block_size / tiling->stem_stride;
    tiling->kflat_dim = tiling->stem_stride * tiling->dim_qk;
    tiling->batch_size = context->GetInputShape(1)->GetStorageShape().GetShape()[0];
    tiling->num_q_heads = context->GetInputShape(0)->GetStorageShape().GetShape()[1];
    tiling->dim_qk = 128;
    tiling->max_q_padded = ...; // ceil(max_q_len / stemBlockSize) * stemBlockSize
    tiling->max_Qb = tiling->max_q_padded / tiling->stem_block_size;
    tiling->total_tokens = context->GetInputShape(0)->GetStorageShape().GetShape()[0];
    tiling->total_tasks = tiling->batch_size * tiling->num_q_heads;
    tiling->used_core_num = std::min(tiling->total_tasks, GetAICoreNum(context));
```

**Kernel 侧模板实例化**:

```cpp
#include "stem_oam_prep_varlen_q_tiling_key.h"

template<int INPUT_TYPE>
__global__ __aicore__ void stem_oam_prep_varlen_q_template(
    GM_ADDR q, GM_ADDR qscale, GM_ADDR q_seq_lens, GM_ADDR cu_seqlens_q,
    GM_ADDR qflat, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StemPrepQTilingData);
    GET_TILING_DATA_WITH_STRUCT(StemPrepQTilingData, tilingData, tiling);

    KernelStemPrepQ<INPUT_TYPE> op;
    op.Init(q, qscale, q_seq_lens, cu_seqlens_q, qflat, tilingData);
    op.Process();
}
```

**API 映射**:

| 计算步骤 | Ascend C API | 参数签名 | 平台验证 | 约束说明 | 替代方案 |
|---------|-------------|---------|---------|---------|---------|
| Q Block bulk 拷入 | DataCopyPad\<fp8_e4m3fn_t\> | B次逐行DataCopyPad, 每行D=128元素 | ✅ DAV_3510 | GM→UB: 拷入B行到qBlockBuf; 32B对齐dst | 2D DataCopyPad (GM stride) |
| FP8→FP32 bulk Cast | Cast\<fp8_e4m3fn_t, float\> | (dst, src, valid_rows×D, CAST_ROUND) | ✅ DAV_3510 | bulk Cast B×D 元素 (1次操作); Duplicate(0)预填FP32 | 逐行Cast (旧方案) |
| Scale bulk 拷入 | DataCopyPad\<float\> | (dst, src, DataCopyExtParams) | ✅ DAV_3510 | 一次拷入 B=stemBlockSize 个 scale; 256×4B=1KB | 多次拷入 S 个 (旧方案) |
| Scale Fusion | Muls\<float\> | (dst, src, scalar, count) | ✅ DAV_3510 | q_scale = scalesBuf[r*S+g] (per-token, read from qScale) | Mul(dst, src1, src2) 逐元素 |
| Group Sum 累加 | Add\<float\> | (dst, src1, src2, count) | ✅ DAV_3510 | FP32累加; 从qBlockFP32Buf[row_idx*D]读取 | ReduceSum (需 tmpBuf) |
| FP32→BF16 Cast | Cast\<float, bfloat16_t\> | (dst, src, count, CAST_ROUND) | ✅ DAV_3510 | CAST_ROUND 模式; kflat_dim 元素 | - |
| 结果写回 | DataCopyPad\<bfloat16_t\> | (dst, src, DataCopyExtParams) | ✅ DAV_3510 | UB→GM; kflat_dim×2B | - |

**API 验证记录**:

| API 名称 | 官方文档路径 | 通配符搜索结果 | 验证状态 | 备注 |
|---------|-------------|---------------|---------|------|
| DataCopyPad | docs/api/context/DataCopyPad*.md | DataCopyPad.md, DataCopyPadExtParams.md | ✅ 已验证 | arch35 支持 FP8/BF16/FP16/FP32 全类型 |
| Cast | docs/api/context/Cast*.md | Cast.md | ✅ 已验证 | fp8_e4m3fn→float DAV_3510 原生支持 |
| Mul | docs/api/context/Mul*.md | Mul.md | ✅ 已验证 | float 类型支持 |
| Muls | docs/api/context/Muls*.md | Muls.md | ✅ 已验证 | scalar 广播乘法 |
| Add | docs/api/context/Add*.md | Add.md | ✅ 已验证 | float 类型累加 |

**验证检查清单**:
- [x] 已用通配符搜索 API 所有变体文件
- [x] 已确认 API 在目标芯片架构（arch35）上可用
- [x] 已确认 API 支持所需的数据类型（FP8/BF16/FP16/FP32）
- [x] 已确认参数签名与官方文档一致
- [x] 已确认 tmpBuffer/对齐等约束条件
- [x] 如 API 不可用，已确定替代方案

**数据搬运详解**:

**GM 布局描述**:

| Tensor | Shape | 说明 |
|---|---|---|
| q | [total_tokens, H_q, D=128] | 变长存储，所有 batch 的 Q 拼在一起，通过 cuSeqLensQ 索引 |
| qScale | [total_tokens, H_q] | per-token scale，与 q 的 token 维度对齐 |
| qFlat 输出 | [batch, H_q, max_Qb, kflat_dim] | 每 (batch,head) 一条 max_Qb × kflat_dim 连续区域 |

**Varlen Slice 拷入详解**:

- 从 cuSeqLensQ GM 读取 cu_off = cuSeqLensQ[batch_idx]，获得该 batch 在 GM 中的起始 token 偏移
- 从 qSeqLens GM 读取 q_len = qSeqLens[batch_idx]
- 每个 qb block 的 GM 偏移: 从 cu_off + qb × B 开始，连续 B 行
- Q GM 偏移 per row: (cu_off + qb×B + i) × H_q × D + head_idx × D
- CopyInQBlock: B次 DataCopyPad 拷入 B 行到 qBlockBuf (InputT, 每行 D=128 元素)
- CastBlockToFP32: Duplicate(0) 预填整个 B×D FP32 buffer, 然后 bulk Cast valid_rows×D InputT→FP32

**FP8 Weighted Group Sum 数据搬运图**:

```
  ┌─────────────────────────────── GM ────────────────────────────────┐
  │ q[cu_off + qb×B + 0..B-1, head_idx, :] (B×D InputT elements)     │
  │ qScale[cu_off + qb×B + 0..B-1, head_idx] (B FP32 elements)       │
  └─────────────────── DataCopyPad ×B rows ──────────────────────────┤
  │  Q offset per row = global_pos × H_q × D + head_idx × D           │
  │  Scale offset = (cu_off + qb×B + row) × H_q + head_idx            │
  ▼                                                                   │
  ┌─────────────────────────────── UB ────────────────────────────────┐
  │ Step 1: CopyInQBlock(qb)                                            │
  │   qBlockBuf: B×D InputT                ← B次 DataCopyPad from GM   │
  │     ↓ Cast<InputT, float> (bulk B×D elements)                       │
  │                                                                     │
  │ Step 2: CastBlockToFP32(qb)                                         │
  │   qBlockFP32Buf: B×D FP32              ← Duplicate(0) + Cast valid │
  │     (超出 q_len_ 的行保持 0.0f, 不参与累加)                         │
  │                                                                     │
  │ Step 3: CopyInScalesBulk(qb) (FP8 only)                             │
  │   scalesBuf: B FP32                    ← 1次 DataCopyPad from GM    │
  │     q_scale[r*S+g] = scalesBuf[r*S+g]                               │
  │                                                                     │
  │ Step 4: AccumulateBlock(qb) — FP32 累加                             │
  │   groupSumBuf: S×D FP32 (kflat_dim)    ← Duplicate(0) 初始化       │
  │                                                                     │
  │   循环 R×S 次:                                                       │
  │     for r=0..R-1:                                                   │
  │       for g=0..S-1:                                                 │
  │         row_idx = r*S + g                                           │
  │         q_scale = scalesBuf[row_idx]                                │
  │         Muls(qWeightedBuf, qBlockFP32[row_idx*D], q_scale, D)        │
  │         Add(groupSum[g*D..], qWeightedBuf, groupSum[g*D..], D)      │
  │                                                                     │
  │ Step 5: Flatten + Cast → outQueueQflat                              │
  │   Cast(groupSumBuf, BF16) → outQueueQflat                              │
  │     ↓ DataCopyPad                                                   │
  └─────────────────────────────────────────────────────────────────────┘
  │                                                                     │
  ▼                                                                    │
  ┌─────────────────────────────── GM ────────────────────────────────┐
  │ qFlat[batch_idx, head_idx, qb, :]  (kflat_dim BF16 elements) │
  └───────────────────────────────────────────────────────────────────┘
```

**Stem Block 处理时序图**:

```
  时间线 (单核内, for qb=0..num_Qb-1):
  
  qb=0:
    ┌── CopyInQBlock(qb) ──────────────┐
    │ B次 DataCopyPad: B行 Q → qBlockBuf │
    │ (每行 D=128 InputT 元素)            │
    └───────────────────────────────────┘
    
    ┌── CastBlockToFP32(qb) ───────────┐
    │ Duplicate(qBlockFP32, 0, B*D)     │  ← zero-fill handles out-of-range rows
    │ Cast(valid_rows*D elements)       │  ← bulk Cast, 1次操作
    └───────────────────────────────────┘
    
    ┌── CopyInScalesBulk(qb) ──────────┐ (FP8 only)
    │ DataCopyPad B scales              │
    │ from qScale[cu_off+qb*B:cu_off+qb*B+B, head_idx] │
    └───────────────────────────────────┘
    
    ┌── AccumulateBlock(qb) ───────────┐
    │ Duplicate(groupSumBuf, 0)         │  ← FP32 累加器初始化
    │ for r,g:                          │
    │   Muls + Add (weighted)           │
    │   读取 qBlockFP32[row_idx*D]      │  ← 直接从 UB FP32 buffer 读
    └───────────────────────────────────┘
    
    ┌── Flatten + Cast + Write ──┐
    │ Cast(groupSum, BF16)  │
    │ DataCopyPad qFlat[b,h,0,:] │
    └────────────────────────────┘
  
  qb=1:
    ┌── CopyIn + Cast + Scales + Accumulate + Flatten + Write ──┐
    │ (同上流程)                               │
    └──────────────────────────────────────────┘
  
  ... (num_Qb 次)
```

**UB Buffer 生命周期表**:

| Buffer | 用途 | 生命周期 |
|---|---|---|
| qBlockBuf | B×D FP8 bulk拷入 | 每qb: CopyInQBlock写入, CastBlockToFP32读取 |
| qBlockFP32Buf | B×D FP32 Cast结果 + 0-fill | 每qb: Duplicate(0)+Cast写入, AccumulateBlock读取 |
| scalesBuf | B个FP32 scale bulk拷入 | 每qb一次拷入, 整个qb期间驻留UB |
| qWeightedBuf | Q × q_scale Muls中间 | 每次Muls写入, Add后可释放 |
| groupSumBuf | 累加器 (kflat_dim FP32) | 每qb: Duplicate(0)初始化, 累加, 末Cast→qflat |
| outQueueQflat | Cast→BF16输出 | 每qb Cast, 立即写GM |
| varlenInfoBuf | qSeqLens + cuSeqLensQ per-batch read | Init阶段读取一次 |

**内存管理**:

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| qBlockBuf | B×D FP8 bulk拷入 | MAX_STEM_BLOCK_SIZE×DIM_QK×sizeof(fp8_e4m3fn_t) = 256×128×1 = 32KB | VECCALC |
| qBlockFP32Buf | B×D FP32 Cast结果 | MAX_STEM_BLOCK_SIZE×DIM_QK×4B = 256×128×4 = 128KB | VECCALC |
| scalesBuf | B个FP32 scale bulk拷入 | MAX_STEM_BLOCK_SIZE×4B = 256×4 = 1KB; FP8 only | VECCALC |
| qWeightedBuf | Q × q_scale Muls中间 | DIM_QK×4B = 512B; FP8 only | VECCALC |
| groupSumBuf | 累加器 (kflat_dim) | MAX_QFLAT_DIM×4B = 64×128×4 = 32KB | VECCALC |
| outQueueQflat | BF16/FP16 输出 | MAX_QFLAT_DIM×2B = 64×128×2 = 16KB | VECOUT |
| varlenInfoBuf | qSeqLens + cuSeqLensQ per-batch read | VARLEN_READ_ALIGN×4B = 32B | VECCALC |

**总 UB 使用量 (FP8)**: 32KB + 128KB + 1KB + 512B + 32KB + 16KB + 32B ≈ 209.5KB ✅ (≤ 248KB)

**UB 容量验证**:

| 平台 | 总 UB | 向量可用 | 系统保留 | 验证公式 |
|------|-------|---------|---------|---------|
| DAV_3510 (Ascend950DT) | 248 KB | 248 KB | 0 KB | `sum(buffers) ≤ 248 * 1024` |

**Buffer 总和验证**:
- qChunkBuf[2] = 4KB
- qChunkFP32 = 8KB
- scalesBuf = 64B
- qGroupSumAcc = 8KB
- qWeightedTmp = 8KB (可复用 qChunkFP32)
- qflatOutBuf = 4KB
- reduceTmpBuf = 32KB (可复用，仅在 ReduceSum 路径使用)

复用优化后实际占用: qChunkBuf(~4KB) + qChunkFP32(~8KB) + scalesBuf(~64B) + qGroupSumAcc(~8KB) + qflatOutBuf(~4KB) = ~24KB + 64B ≈ 24KB ✅ (≤ 248KB)

**UB Buffer Sizing Strategy**: stemBlockSize and stemStride are runtime parameters, not constexpr constants. UB buffers use MAX constants (MAX_STEM_BLOCK_SIZE=256, MAX_STEM_STRIDE=64) for allocation, runtime tiling values for computation. All loop counts (R = stemBlockSize / stemStride), DataCopyPad extents, and buffer indices use runtime tiling values from TilingData. kflat_dim is derived as `stemStride × dim_qk` and stored in TilingData. qScale shape is [total_tokens, H_q], indexed by (cu_off + token_pos, head_idx).

#### 2.4.2 性能优化

**并行策略**:

**多核切分**: 双模式分核 (见 2.3 分核策略)

**流水线设计 — Double Buffer 流水线 (chunk-level pipelining)**:

```
时间线:
  chunk 0: [搬入Q_r0 + Cast] → [搬入scale_r0] → [Mul+Add累加]
  chunk 1:                                [搬入Q_r1 + Cast] → [搬入scale_r1] → [Mul+Add累加]
  ...
  chunk r: [搬入Q_r + Cast] 与 [计算chunk r-1] 并行 (Double Buffer)
```

**Double Buffer 机制**:
- `qChunkBuf[0]` 和 `qChunkBuf[1]` 交替使用
- 搬入 chunk r+1 数据与计算 chunk r 的 Weighted Mul+Add 并行执行
- 通过 pipe_barrier 或 event_t 同步保证数据一致性

**UB 切分策略**:

| 项目 | 说明 |
|-----|------|
| UB 容量 | 248KB (DAV_3510) |
| 单次处理数据量 | stemStride tokens × dim_qk × 4B; default 16×128×4 = 8KB (FP32 chunk) |
| 是否需要分 chunk | 是，按 R=stemBlockSize/stemStride stride group 分 R 个 chunk |
| chunk 大小计算公式 | chunk_size = stemStride × dim_qk × sizeof(float) |

#### 2.4.5 风险评估

**API 风险**:

- **Varlen Slice offset 计算**: DataCopyPad 的 srcOffset 需精确计算 cuSeqLensQ[b] × stride，offset 越界会导致数据错误
- **FP8 Cast 精度**: fp8_e4m3fn → float Cast 在 DAV_3510 上为原生支持，但精度范围有限 (FP8 动态范围比 FP32 小)
- **Scale scatter read**: per-position q_scale 可能导致非连续 GM 读取，影响搬运效率；替代方案为 bulk row load 后在 UB 内做位置映射

**精度风险**:

- **FP32→BF16/FP16 Cast**: qFlat 输出从 FP32 Cast 为 BF16(FP8/BF16输入)或FP16(FP16输入); BF16路径精度损失 rtol 约 1e-2; FP16路径精度损失 rtol 约 5e-4
- **R 层累加精度**: FP32 累加 R=stemBlockSize/stemStride 层加权结果，数值稳定性依赖 Q 和 scale 的值域范围
- **Zero-pad 影响**: 对齐到 stemBlockSize 的 zero-pad 区域不参与计算，但 DataCopyPad 需正确设置 padVal=0
- **batch 大小**: batch 最大值为 1024，超出此限制会导致 TilingData 数组越界

**应对措施**:

- Varlen offset 计算通过 cuSeqLensQ GM 拷入 UB 后逐元素读取，避免 Host 侧预计算复杂度
- Scale scatter read 改用 bulk DataCopyPad 拷入整行后在 UB 内用偏移索引定位
- FP32→BF16 使用 CAST_ROUND 模式减少截断误差; FP16 路径同理
- Zero-pad 通过 DataCopyPad 的 padVal 参数设为 0.0f 保证正确性
- batch 最大值限制为 1024，避免 TilingData 数组越界

### 2.5 PTA接口设计

**PTA (PyTorch Ascend) 接口设计**：

本算子通过 aclnn 框架接入 PyTorch，采用标准两段式调用：

```python
import torch_npu

qFlat = torch.ops.npu.stem_oam_prep_varlen_q(
    q,              # [total_tokens, H_q, 128] NPU Tensor
    qSeqLens,       # [batch] INT32
    cuSeqLensQ,     # [batch+1] INT32
    qScale=qScale,  # [batch, H_q, max_seqlen_pad_stemBlockSize] FP32
    stemBlockSize=128,
    stemStride=16
)
# Returns: qFlat [batch, H_q, max_Qb, kflat_dim] BF16
```

**C++ 桥接实现**：
```cpp
// stem_oam_prep_varlen_q_torch.cpp
at::Tensor stem_oam_prep_varlen_q_torch(
    const at::Tensor& q,
    const at::Tensor& qSeqLens,
    const at::Tensor& cuSeqLensQ,
    const c10::optional<at::Tensor>& qScaleOpt,
    int64_t stemBlockSize, int64_t stemStride)
{
    // 1. 输入校验 (dtype, device, shape)
    // 2. 计算输出 shape (max_Qb, kflat_dim)
    // 3. 创建 aclTensor (at::Tensor → aclTensor)
    // 4. aclnnStemOamPrepVarlenQGetWorkspaceSize()
    // 5. 分配 workspace
    // 6. aclnnStemOamPrepVarlenQ()
    // 7. 返回 qFlat
}
```

**register.cpp**：
```cpp
TORCH_LIBRARY(npu, m) {
    m.def("stem_oam_prep_varlen_q(Tensor q, Tensor qSeqLens, Tensor cuSeqLensQ, *, Tensor? qScale=None, int stemBlockSize=128, int stemStride=16) -> Tensor");
    m.impl("stem_oam_prep_varlen_q", &ascend_kernel::stem_oam_prep_varlen_q_torch);
}
```

### 2.6 代码结构

```
stem_oam_prep_varlen_q/
├── op_host/
│   ├── stem_oam_prep_varlen_q_def.cpp          # 算子定义 (OpDef 注册)
│   ├── stem_oam_prep_varlen_q_infershape.cpp   # Shape 推导 (InferShape/InferDtype)
│   └── arch35/
│       ├── stem_oam_prep_varlen_q_tiling.cpp    # Tiling 计算 (TilingFunc)
│       ├── stem_oam_prep_varlen_q_tiling.h      # Tiling 常量 + TilingData 结构体
├── op_kernel/
│   ├── stem_oam_prep_varlen_q_arch35.cpp        # Kernel 入口 (template instantiation + REGISTER_TILING)
│   └── arch35/
│       ├── stem_oam_prep_varlen_q.h             # Kernel 类定义
│       ├── stem_oam_prep_varlen_q_tiling_key.h  # TilingKey (0=FP8, 1=BF16, 2=FP16)
│       ├── stem_oam_prep_varlen_q_tiling_data.h # TilingData 结构体
├── op_api/
│   ├── aclnn_stem_oam_prep_varlen_q.h           # aclnn API 头文件
│   ├── aclnn_stem_oam_prep_varlen_q.cpp         # aclnn API 实现
│   ├── stem_oam_prep_varlen_q.h                 # 算子注册头文件
│   └── stem_oam_prep_varlen_q.cpp               # 算子注册实现
├── op_extension/
│   ├── stem_oam_prep_varlen_q_torch.cpp         # PyTorch 桥接 (aclnn调用)
│   ├── register.cpp                             # TORCH_LIBRARY 注册
│   ├── ops.h                                    # 函数声明
├── scripts/
│   ├── gen_data.py, golden.py, verify_result.py, test_torch.py
├── CMakeLists.txt                                # 标准自定义算子构建
└── tests/
    ├── ut/
    └── st/
└── docs/
    ├── aclnnStemOamPrepVarlenQ.md
    ├── REQUIREMENTS.md
    ├── DESIGN.md
    ├── PLAN.md
```

**交付件清单**：

**必需**:
- `op_host/stem_oam_prep_varlen_q_def.cpp` — 算子定义 (OpDef 注册)
- `op_host/stem_oam_prep_varlen_q_infershape.cpp` — Shape 推导 (InferShape/InferDtype)
- `op_host/arch35/stem_oam_prep_varlen_q_tiling.cpp` — Tiling 计算 (TilingFunc)
- `op_host/arch35/stem_oam_prep_varlen_q_tiling.h` — Tiling 常量 + TilingData 结构体
- `op_kernel/stem_oam_prep_varlen_q_arch35.cpp` — Kernel 入口 (REGISTER_TILING)
- `op_kernel/arch35/stem_oam_prep_varlen_q.h` — Kernel 类定义 + 计算逻辑
- `op_kernel/arch35/stem_oam_prep_varlen_q_tiling_key.h` — TilingKey 定义
- `op_kernel/arch35/stem_oam_prep_varlen_q_tiling_data.h` — TilingData 结构体
- `op_api/aclnn_stem_oam_prep_varlen_q.h` — aclnn API 头文件
- `op_api/aclnn_stem_oam_prep_varlen_q.cpp` — aclnn API 实现
- `op_api/stem_oam_prep_varlen_q.h` — 算子注册头文件
- `op_api/stem_oam_prep_varlen_q.cpp` — 算子注册实现
- `op_extension/stem_oam_prep_varlen_q_torch.cpp` — Torch 桥接层
- `op_extension/register.cpp` — TORCH_LIBRARY 注册
- `op_extension/ops.h` — 函数声明
- `scripts/gen_data.py` — 数据生成
- `scripts/golden.py` — Golden 参考实现
- `scripts/verify_result.py` — 结果校验
- `scripts/test_torch.py` — Torch 调用测试
- `CMakeLists.txt` — 标准自定义算子构建
- `tests/ut/` — 单元测试
- `tests/st/` — 系统测试

**可选**:
- `op_graph/` — 图模式适配

---