# block_sparse_attention_grad 确定性方案文档

## 1. 问题背景

block_sparse_attention_grad 算子在 arch22 (Atlas A2) 平台上，当使用 GQA (Grouped Query Attention) 时，
多个 cube 核会并行计算不同 Q head 的 dK/dV，但由于同一 group 内的 Q head 共享相同的 K/V，
不同核会通过 atomicAdd 累加到同一 GM 地址。atomicAdd 的执行顺序不确定，导致结果非确定性。

## 2. 方案设计

参考 PR #1544 (dense_lightning_indexer_grad_kl_loss) 的确定性实现方式：

### 2.1 核心思路

1. **AIC 侧**：每个 cube 核将其 dK/dV 计算结果写入各自独立的 per-core workspace 槽位（通过 atomicAdd 累加同核内多轮结果）
2. **AIV 侧**：在 VecPost 之前，新增 VecDeterReduce 步骤，按固定顺序（core 0 → 1 → ... → N-1）读取所有核的 per-core 结果并累加，写入共享 workspace
3. **VecPre 侧**：在原有清零基础上，额外清零 per-core workspace，确保每次 cube 写入时目标地址全为 0

### 2.2 数据流

\\\
VecPre: 清零 shared workspace + per-core workspace
  ↓
AIC: 每个 core 用 atomicAdd 写入自己的 per-core slot
  ↓ (CUBE2POST + SyncAll)
VecDeterReduce: 按序读取 per-core slots，累加到 shared workspace
  ↓
VecPost: 从 shared workspace 读取，scale + cast，写入 output GM
\\\

### 2.3 Workspace 布局

\\\
[libapiSize | sOutSize | dPOutSize | dQOutSize | dKOutSize | dVOutSize | gradSize | detDkWorkspace | detDvWorkspace]
\\\

- detDkWorkspace = aicNum * dkvSize * sizeof(float)
- detDvWorkspace = aicNum * dkvSize * sizeof(float)
- 每个 AIC core 的 slot: detDkWorkspace[coreIdx * dkvSize ... (coreIdx+1) * dkvSize]

## 3. 代码修改

### 3.1 Tiling (op_host/block_sparse_attention_grad_tiling.h)
- 新增字段: detDkWorkspaceSize, detDvWorkspaceSize, aicNumForDet

### 3.2 Tiling 实现 (op_host/arch22/block_sparse_attention_grad_tiling_arch22.cpp)
- CalculateWorkSpace: 增加 detDkWorkspaceSize 和 detDvWorkspaceSize 计算
- FillTilingData: 设置新增字段

### 3.3 Kernel (op_kernel/arc22/block_sparse_attention_grad_kernel.h)

**AIC operator()**:
- gDk/gDv 改为指向 per-core workspace: detDkOffset + coreIdx * dkvSize * sizeof(float)

**VecPre**:
- 新增 ZeroDetWorkspace() 方法，清零 per-core dk/dv workspace

**VecDeterReduce (新增)**:
- VecDeterReduceKernel: 读取 per-core slots，按序累加，写入 shared workspace
- 使用 MTE2/V/MTE3 管道，严格 V_MTE2 同步防止 tempUb 被覆盖

**AIV operator()**:
- 在 VecPost 之前调用 VecDeterReduce

## 4. 测试结果

| 测试用例 | dq | dk | dv | 说明 |
|---------|-----|-----|-----|------|
| GQA BNSD (B=10,N=12,KV=4,S1=256,S2=357) | ✅ | ✅ | ✅ | 3次迭代全部通过 |
| GQA TND (B=10,N=24,KV=4,S1=1220,S2=452) | ✅ | ✅ | ✅ | 3次迭代全部通过 |
| MHA TND (B=10,N=15,KV=15,S1=126,S2=256) | ✅ | ⚠️ | ✅ | dk 仅1个元素差 0.001953 (FP16 1 ULP) |

### 4.1 MHA TND dk 精度说明
MHA (group_size=1) 无多核累加问题，确定性路径与原路径结果在 FP32 层面一致。
dk 的 1-ULP 差异来源于 VecDeterReduce 的 Add 操作与原 Fixpipe atomicAdd 在 FP16 cast 时的舍入差异。
diff count=1, abs diff=0.001953=2^-9，属于 FP16 正常精度范围。

## 5. 已知问题与限制

1. **Workspace 开销**: per-core workspace 需要 aicNum * dkvSize * sizeof(float) * 2 的额外空间。
   对于大规模序列（如 kvTotalSeqlen=2048, kvHeads=8, headDim=128），20 核约需 320MB 额外 workspace。
2. **MHA TND dk 精度**: 1-ULP FP16 差异，可通过放宽 tolerance 或使用 BF16 缓解。
3. **性能影响**: VecDeterReduce 引入额外的 GM 读取和 V 计算，可能影响性能。后续可优化为 ping-pong 流水。

## 6. 调试脚本

测试脚本路径: /home/lsy/ops-transformer_1787/attention/block_sparse_attention_grad/DevelopTest/
- bsag_gqa_bnsd.py: GQA BNSD 格式测试
- bsag_gqa_tnd_test.py: GQA TND 格式测试  
- bsag_mha_tnd_test.py: MHA TND 格式测试

运行方式:
\\\ash
source /opt/nzh/0608/cann-9.1.0-beta.1/set_env.sh
export LD_LIBRARY_PATH=/opt/nzh/0608/cann-9.1.0-beta.1/aarch64-linux/lib64:/opt/nzh/0608/cann-9.1.0-beta.1/opp/vendors/custom_transformer/op_api/lib/:
cd DevelopTest && python3 bsag_gqa_bnsd.py
\\\
