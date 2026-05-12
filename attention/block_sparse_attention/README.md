# BlockSparseAttention算子

## 概述

BlockSparseAttention是一个基于CATLASS模板库实现的高性能稀疏注意力算子，支持灵活的块级稀疏模式。

## 功能特性

- **灵活的稀疏块模式**: 支持自定义的x*y块级稀疏模式,通过blockShape参数指定
- **多种KV Cache布局**: 支持TND [T, N, D]和BNSD [B, N, S, D]的的KV cache布局
- **多种Q输入布局**: 支持TND [T, N, D]和BNSD [B, N, S, D]两种Query输入布局
- **高性能计算**: 基于CATLASS模板库,充分利用昇腾A2硬件特性
- **多精度支持**: 支持FP16和BF16两种精度
- **非对齐序列**: 支持序列长度不能被块大小整除的场景,自动处理边界情况

## 接口定义

### aclnnBlockSparseAttentionGetWorkspaceSize

```cpp
aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query,                         // Query输入
    const aclTensor *key,                           // Key输入
    const aclTensor *value,                         // Value输入
    const aclTensor   *blockSparseMaskOptional,     // 稀疏Mask
    const aclTensor   *attenMaskOptional,           // Attention mask (当前不支持)
    const aclIntArray *blockShapeOptional,          // 稀疏块形状数组
    const aclIntArray *actualSeqLengthsOptional,    // 实际Q序列长度 (TND格式时必选)
    const aclIntArray *actualSeqLengthsKvOptional,  // 实际KV序列长度 (TND格式时必选)
    const aclTensor   *blockTableOptional,          // Block表 (用于PagedAttention，当前不支持)
    char *qInputLayout,                             // Query的数据排布格式 (TND/BNSD)
    char *kvInputLayout,                            // Key/Value的数据排布格式 (TND/BNSD)
    int64_t numKeyValueHeads,                       // KV头数
    int64_t maskType,                               // Mask类型
    double scaleValue,                              // 缩放因子
    int64_t innerPrecise,                           // Softmax计算采取的精度级别
    int64_t blockSize,                              // Block大小 (用于PagedAttention，当前不支持)
    int64_t preTokens,                              // 滑窗参数 (当前不支持)
    int64_t nextTokens,                             // 滑窗参数 (当前不支持)
    int64_t softmaxLseFlag,                         // 是否输出LSE
    aclTensor *attentionOut,                        // 输出tensor
    aclTensor *softmaxLseOptional,                  // Softmax LSE输出 (可选)
    uint64_t *workspaceSize,                        // 返回workspace大小
    aclOpExecutor **executor);                      // 返回executor
```

### aclnnBlockSparseAttention

```cpp
aclnnStatus aclnnBlockSparseAttention(
    void *workspace,                           // workspace地址
    uint64_t workspaceSize,                    // workspace大小
    aclOpExecutor *executor,                   // executor
    aclrtStream stream);                       // ACL stream
```

## 参数说明

### 输入参数

- **query**: Query tensor,支持以下布局
  - TND: [total_tokens, num_heads, head_dim]
  - BNSD: [batch, num_heads, seqlen, head_dim]

- **key**: Key tensor,支持以下布局
  - TND: [total_kv_tokens, num_kv_heads, head_dim]
  - BNSD: [batch, num_kv_heads, kv_seqlen, head_dim]

- **value**: Value tensor,支持以下布局
  - TND: [total_kv_tokens, num_kv_heads, head_dim]
  - BNSD: [batch, num_kv_heads, kv_seqlen, head_dim]

- **blockSparseMaskOptional**: 
  - 稀疏Mask: [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]

- **blockShapeOptional**: 
  - blockShapeX: Q方向块大小
  - blockShapeY: KV方向块大小

### 输出参数

- **attentionOut**: 注意力输出,shape与query相同
- **softmaxLse**: Softmax log-sum-exp输出 (可选)

## 使用示例

```cpp
// 1. 准备输入数据
int64_t batch = 1;
int64_t qSeqlen = 256;
int64_t kvSeqlen = 512;
int64_t numHeads = 8;
int64_t numKvHeads = 8;
int64_t headDim = 128;
int64_t blockShapeX = 128;
int64_t blockShapeY = 128;

// 2. 创建tensor (略)

// 3. 设置blockShape
int64_t blockShapeData[2] = {blockShapeX, blockShapeY};
aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);

// 4. 调用算子
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnBlockSparseAttentionGetWorkspaceSize(
    query, key, value, blockSparseMask, nullptr, blockShape, nullptr,
    nullptr, nullptr, "BNSD", "BNSD", numKvHeads, 0, scaleValue, 0, 128,
    2147483647, 2147483647, 0, attentionOut, nullptr, &workspaceSize, &executor);

if (ret == ACLNN_SUCCESS) {
    void *workspace = nullptr;
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    ret = aclnnBlockSparseAttention(workspace, workspaceSize, executor, stream);
    
    aclrtFree(workspace);
}
```

## 稀疏模式说明

### BlockSparseMask稀疏pattern

BlockSparseAttention使用blockSparseMask稀疏pattern,不需要selectIdx索引格式和传统的sBlockIdx计算:

- **blockSparseMask**: 形状为`[batch, numHeads, qBlockNum, kvBlockNum]`的稀疏Mask
- **blockShapeX**: Q方向基本块大小
- **blockShapeY**: KV方向基本块大小

### 示例

对于qSeqlen=512, kvSeqlen=1024, blockShapeX=128, blockShapeY=128:

```
Q方向: ceil(512/128)=4块 [0, 1, 2, 3]
KV方向: ceil(1024/128)=8块 [0, 1, 2, 3, 4, 5, 6, 7]

每个Q块可以选择任意KV块组合：
- Q块0选择的KV块: [0, 2, 5]
- Q块1选择的KV块: [1, 3, 6]
- Q块2选择的KV块: [0, 4, 7]
- Q块3选择的KV块: [2, 5, 6]
```

## 性能特点

- **内存效率**: 相比密集注意力,显著减少内存使用
- **计算效率**: 只计算指定的稀疏块,减少计算量
- **灵活配置**: 支持任意blockShape的稀疏块配置
- **硬件优化**: 充分利用昇腾A2的AI Core和Vector Core

## 注意事项

1. qInputLayout当前仅支持"TND"和"BNSD"。
2. kvInputLayout当前仅支持"TND"和"BNSD"。
3. blockShapeOptional如果传入，则必须包含至少两个元素[blockShapeX, blockShapeY]，且值必须大于0，blockShapeY必须为128的倍数。
4. qSeqlen和kvSeqlen不需要被blockShape整除，支持非对齐场景，实际分块数通过向上取整计算。
5. blockSparseMaskOptional当前必须传入，且shape必须为[batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)]。
6. attentionMaskOptional当前只支持传入nullptr。
7. maskType当前只支持输入0，表示不加mask。
8. actualSeqLengthsOptional在qInputLayout为“TND”时必选；actualSeqLengthsKvOptional在kvInputLayout为“TND”时必选。
9. 当前不支持PagedAttention，blockTableOptional当前只支持传入nullptr。
10. query输入为BFLOAT16时，innerPrecise只能配置为0。

## 编译说明

算子编译集成在ops-transformer的统一编译流程中:

```bash
bash build.sh --make_clean -j96 --pkg --soc=ascend910b --ops=block_sparse_attention
./build/cann-ops-transformer-custom_linux-aarch64.run
```

**注意**: 此算子位于 `attention/` 目录(非 `experimental/`),编译时**请勿使用**
`--experimental` 选项 — 该选项会过滤掉此算子。

## 测试与性能基准 (Python pytest + benchmark)

仓库提供了基于 PyTorch / `torch_npu` 的 Python 测试套件和性能基准,完整自包含
于本目录,可作为独立 cherry-pick 特性发布。

### 目录结构

| 子目录 | 内容 |
|---|---|
| `benchmark/` | 性能基准。`benchmark.py` 在 `[B=1, H=3, S=118806, D=128]` (BF16, BNSD) 工作负载下扫描 sparsity 0..0.9,对比 `npu_fusion_attention` 密集参考的延迟。共享工具集中在同目录的 `util.py`。详见 [`benchmark/README.md`](benchmark/README.md)。 |
| `tests/pytest/` | pytest 正确性测试套件 (3 个文件,共数百个测试用例)。 |

> Python 入口由 `torch_extension/npu_ops_transformer/` 提供 — 通过
> `npu_ops_transformer.ops.block_sparse_attention(...)` 调用。算子的 C++ 包装器
> `ops/csrc/block_sparse_attention.cpp` 在首次 `import npu_ops_transformer` 时
> 自动通过 ninja JIT 编译。详见
> [`../../torch_extension/README.md`](../../torch_extension/README.md)。

### 测试套件

```bash
# 先确保算子已编译安装 (见上方编译说明) 并且 torch_extension/npu_ops_transformer 已安装
cd attention/block_sparse_attention/tests/pytest
ASCEND_RT_VISIBLE_DEVICES=1 pytest test_attn.py test_lse.py test_joint.py -v
```

| 文件 | 覆盖内容 | 参考 |
|---|---|---|
| `test_attn.py` | 注意力输出 (attention_out) 在 `H ∈ {1,2,3,4}, S=24000, D=128, BF16, BNSD`, 2 种边框 (无边框/`FrameWidths(2,1,3,1)`),10 个 sparsity 级别 (0.0..0.9) 下的正确性。共 80 个测试用例。 | `npu_fusion_attention` (PFA),`sparse_mode=0` (sparsity=0) 或 `sparse_mode=1` + 每 token 的 `pfa_atten_mask` (sparsity>0)。 |
| `test_lse.py` | `softmax_lse` 输出: 密集 LSE vs FIAS, 稀疏 H=1 LSE vs FIAS (含定义差异), 多头稀疏 LSE vs Python float32 参考, 零输入 LSE = log(S), shape/dtype 检查 ([B,H,S,1] 4-D float32), 关闭 LSE 时返回空张量。涵盖 bfloat16 + float16。 | `npu_fused_infer_attention_score` (FIAS) + Python float32 实现 (`_lse_reference_sparse`)。 |
| `test_joint.py` | 同一次 `softmax_lse_flag=True` 调用同时返回的两个输出 (attention_out + softmax_lse) 的联合验证: 密集 (BF16 + FP16) 对 FIAS, 稀疏 H=1 BF16 (attention_out 对 PFA, LSE 对 Python 参考)。 | FIAS / PFA / Python float32 参考。 |

#### 容差与已知差异

- **attention_out**: 在所有测试 sparsity 下严格满足 `rtol=0.01, atol=0.001`
  (与 BSA 测试套件一致)。
- **softmax_lse 密集**: 严格满足 `rtol=1e-3, atol=1e-4`。
- **softmax_lse 稀疏 vs FIAS**: 存在 ~0.06–0.16 的偏差 (sparsity 0.5/0.8/0.9),
  属定义差异 (FIAS `sparse_mode=1` 与本算子在求和位置上不一致)。`test_lse.py` 中
  以 `SPARSE_FIAS_LSE_ATOL=0.25` 记录此偏差区间。
- **softmax_lse 稀疏 vs Python 参考**: 严格满足 `atol=5e-2` (BF16 矩阵乘累加误差预算)。
  通过此项即可证明算子 LSE 在自身定义下正确。

### 性能基准

```bash
cd attention/block_sparse_attention/benchmark
# 默认: 128×128 块粒度
ASCEND_RT_VISIBLE_DEVICES=1 python benchmark.py
# 或: 128×512 块粒度 + token 对齐的边框
ASCEND_RT_VISIBLE_DEVICES=1 BSA_BLOCK_SIZE_KV=512 BSA_FRAME=8,1,29,1 python benchmark.py
```

完整的基准方法、表格、对照数据见 [`benchmark/README.md`](benchmark/README.md)。

## 版本历史

- v1.0.0: 初始版本
