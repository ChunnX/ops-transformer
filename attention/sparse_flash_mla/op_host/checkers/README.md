# Sparse Flash MLA 系列算子拦截说明

## 1. 文档范围

本文档说明以下三个算子新增的 Host 侧参数拦截代码：

- `sparse_flash_mla`
- `mixed_quant_sparse_flash_mla`
- `quant_sparse_flash_mla`

拦截规则参考：

- `torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md`
- `torch_extension/cann_ops_transformer/docs/zh/mixed_quant_sparse_flash_mla.md`
- `torch_extension/cann_ops_transformer/docs/zh/quant_sparse_flash_mla.md`

代码采用与 `flash_attn/op_host/checkers` 相同的分层形式，将检查过程划分为：

1. 单参数检查（`CheckSinglePara`）
2. 参数存在性检查（`CheckParaExistence`）
3. 特性交叉检查（`CheckFeature`）
4. 多参数一致性检查（`CheckMultiPara`）

三个算子的公共检查代码统一放在本目录；混合量化和全量化算子的目录只保存各自的适配入口和差异规则。

## 2. 代码结构

```text
attention/
├── sparse_flash_mla/op_host/checkers/
│   ├── base_checker.{h,cpp}                 # Checker 基类及公共检查工具
│   ├── checker_context.h                    # 三算子统一检查上下文
│   ├── checker_adapter.h                    # TilingInfo 到统一上下文的适配
│   ├── checker_runner.{h,cpp}               # 四阶段检查编排
│   ├── common_checker.{h,cpp}               # Q/KV/输出、布局和公共属性
│   ├── seq_len_checker.{h,cpp}              # 序列长度类 Tensor
│   ├── sparse_compression_checker.{h,cpp}   # 稀疏索引、TopK 和压缩参数
│   ├── mask_checker.{h,cpp}                 # Mask 与窗口联动
│   ├── paged_attention_checker.{h,cpp}      # Paged Attention
│   ├── sinks_checker.{h,cpp}                # Sinks
│   ├── metadata_checker.{h,cpp}             # Metadata
│   ├── softmax_lse_checker.{h,cpp}          # Softmax LSE 输出
│   ├── sparse_variant_checker.{h,cpp}       # sparse_flash_mla 特有规则
│   ├── sparse_flash_mla_checker.{h,cpp}     # sparse_flash_mla 入口
│   └── checker_sources.cmake                # 公共源码一次性注册
├── mixed_quant_sparse_flash_mla/op_host/checkers/
│   ├── mixed_quant_variant_checker.{h,cpp}  # 混合量化特有规则
│   └── mixed_quant_sparse_flash_mla_checker.{h,cpp}
└── quant_sparse_flash_mla/op_host/checkers/
    ├── quant_variant_checker.{h,cpp}        # 全量化及 descale 规则
    └── quant_sparse_flash_mla_checker.{h,cpp}
```

`RegisterCommonCheckers` 按以下顺序注册公共 Checker：

```text
Common
  → SeqLen
  → SparseCompression
  → Mask
  → PagedAttention
  → Sinks
  → Metadata
  → SoftmaxLse
  → 算子差异 Checker
```

任一阶段、任一 Checker 返回失败后立即停止，Tiling 返回 `GRAPH_FAILED`。

## 3. 三算子公共拦截

### 3.1 Tensor 通用要求

| 检查项 | 拦截规则 |
| --- | --- |
| 必选输入/输出 | `q`、`ori_kv`、`attention_out`、`sinks`、`metadata`、`softmax_lse` 必须存在；`cmp_kv`可选 |
| 数据格式 | 所有被检查的 Tensor 仅支持 `ND` |
| 空 Tensor | Q、KV、输出、索引、长度、Block Table、Sinks 等任一维度小于等于0时拦截 |
| Q 布局 | 仅支持 `BSND`、`TND` |
| KV 布局 | 仅支持 `BSND`、`TND`、`PA_BBND` |
| 布局组合 | 非 `PA_BBND` 时，`layout_q`与`layout_kv`必须相同；PA 时 Q 可为 `BSND`或`TND` |
| Q/输出一致性 | `attention_out`的 rank 和 shape 必须与`q`完全相同 |
| Q 轴范围 | `q_n`在 `[1, 128]`内，Q head dim 固定为512，batch/sequence/token维均大于0 |
| KV 轴范围 | `kv_n`固定为1，序列、token、block 数和 block size 均必须有效 |
| Softmax scale | `softmax_scale`必须为有限值，拒绝 NaN 和 Inf |

### 3.2 三算子数据类型和 Head Dim 差异

| 算子 | q | ori_kv/cmp_kv | attention_out | KV head dim |
| --- | --- | --- | --- | --- |
| `sparse_flash_mla` | FP16/BF16 | FP16/BF16 | FP16/BF16 | 512 |
| `mixed_quant_sparse_flash_mla` | BF16 | FP8_E4M3FN | BF16 | `quant_mode=1`时608；`quant_mode=2`时584 |
| `quant_sparse_flash_mla` | HIFLOAT8 | HIFLOAT8 | BF16 | 512 |

`sparse_flash_mla`还要求`q`、所有非空 KV 和`attention_out`的数据类型完全一致。

在 Atlas A2/A3 上，`sparse_flash_mla`的`q_n`仅支持：

```text
1, 2, 4, 8, 16, 32, 64, 128
```

Ascend 950 上只检查`1 <= q_n <= 128`。

### 3.3 序列长度 Tensor

以下 Tensor 均要求 `int32`、`ND`、一维且非空：

- `cu_seqlens_q`
- `cu_seqlens_ori_kv`
- `cu_seqlens_cmp_kv`
- `seqused_q`
- `seqused_ori_kv`
- `seqused_cmp_kv`
- `cmp_residual_kv`

存在性和 shape 规则如下：

| 参数 | 存在性 | shape |
| --- | --- | --- |
| `cu_seqlens_q` | Q 为 TND 时必传；其他布局禁止传入 | `(b+1,)` |
| `cu_seqlens_ori_kv` | KV 为 TND 时必传；其他布局禁止传入 | `(b+1,)` |
| `cu_seqlens_cmp_kv` | KV 为 TND且`cmp_kv`存在时必传；无`cmp_kv`时禁止传入 | `(b+1,)` |
| `seqused_q` | 可选 | `(b,)` |
| `seqused_ori_kv` | PA 场景有效检查要求必传 | `(b,)` |
| `seqused_cmp_kv` | PA 且`cmp_kv`存在时有效检查要求必传 | `(b,)` |
| `cmp_residual_kv` | 由压缩模式和 Mask 联动决定 | `(b,)` |

Tensor 内部数值（例如累积长度单调性、首尾值及 residual 范围）在 Tiling 阶段不可读取，由调用方保证。

### 3.4 稀疏索引、TopK Length 与压缩属性

公共规则：

- `cmp_ratio > 0`。
- SWA（未传`cmp_kv`）要求`cmp_ratio=1`。
- `topk_value_mode`当前只支持1。
- `cmp_sparse_indices`、`cmp_topk_length`、`cmp_residual_kv`以及 cmp 侧长度/Block Table 均依赖`cmp_kv`。
- 稀疏索引仅支持 `int32`、`ND`：
  - BSND：`(b, q_s, kv_n, topk)`
  - TND：`(q_t, kv_n, topk)`
- TopK Length 仅支持 `int32`、`ND`：
  - BSND：`(b, q_s, kv_n)`
  - TND：`(q_t, kv_n)`

`sparse_flash_mla`当前不允许传入：

- `ori_sparse_indices`
- `ori_topk_length`
- `cmp_topk_length`

混合量化和全量化算子采用以下成对规则：

- `ori_mask_mode=0`且传入`ori_sparse_indices`时，必须传`ori_topk_length`；其他情况禁止传`ori_topk_length`。
- `cmp_mask_mode=0`且传入`cmp_sparse_indices`时，必须传`cmp_topk_length`；其他情况禁止传`cmp_topk_length`。

### 3.5 Mask 和窗口

| 参数 | 支持范围 |
| --- | --- |
| `ori_mask_mode` | 0、3、4 |
| `cmp_mask_mode` | 0、3 |
| `ori_win_left`、`ori_win_right` | -1或非负数 |

交叉规则：

- 非 Sliding Window（`ori_mask_mode != 4`）要求`ori_win_left=ori_win_right=-1`。
- `ori_mask_mode=4`时才允许使用非负窗口值。
- SWA 要求`cmp_mask_mode=0`。
- `sparse_flash_mla`的 HCA/CSA 要求`cmp_mask_mode=3`。
- 混合量化算子的 Causal/Sliding Window 模式（`ori_mask_mode=3/4`）在存在`cmp_kv`时要求`cmp_mask_mode=3`。
- 混合量化和全量化算子传入`ori_sparse_indices`时，要求`ori_mask_mode=cmp_mask_mode=0`。
- Atlas A2/A3 上的`sparse_flash_mla`固定要求`ori_win_left=127`、`ori_win_right=0`。

### 3.6 Paged Attention

| 检查项 | 拦截规则 |
| --- | --- |
| Block Table dtype/format | `int32`、`ND` |
| Block Table rank | 2 |
| Block Table shape | 第一维必须等于 batch，第二维必须大于0 |
| 非 PA 布局 | 禁止传入`ori_block_table`和`cmp_block_table` |
| PA 布局 | 必须传`ori_block_table`；`cmp_block_table`与`cmp_kv`同时存在或同时不存在 |
| Seqused 联动 | `ori_block_table`要求`seqused_ori_kv`；`cmp_block_table`要求`seqused_cmp_kv` |
| KV block size | 范围 `[1, 1024]` |
| A2/A3 sparse block size | 额外要求16对齐 |

Block Table 内的具体 block id 值由调用方保证。

### 3.7 Sinks、Metadata 和 Softmax LSE

| 参数 | 拦截规则 |
| --- | --- |
| `sinks` | 当前版本必传；`float32`、`ND`、shape为`(q_n,)` |
| `metadata` | 当前版本必传；`int32`、`ND`、shape为`(1024,)` |
| `softmax_lse` | 输出必须存在；`float32`、`ND` |

`softmax_lse` shape：

| 条件 | shape |
| --- | --- |
| `return_softmax_lse=false` | `(1,)` |
| `return_softmax_lse=true`且 Q 为 BSND | `(b, kv_n, q_s, q_n/kv_n)` |
| `return_softmax_lse=true`且 Q 为 TND | `(kv_n, q_t, q_n/kv_n)` |

同时检查`q_n`可被`kv_n`整除。

## 4. 算子特有拦截

### 4.1 sparse_flash_mla

按输入组合识别计算模式：

| 模式 | 输入组合 | 公共约束 | Atlas A2/A3附加约束 |
| --- | --- | --- | --- |
| SWA | 仅`ori_kv` | `cmp_ratio=1`、`cmp_mask_mode=0` | `cmp_ratio=1` |
| HCA | `ori_kv + cmp_kv`，无`cmp_sparse_indices` | `cmp_mask_mode=3`、必须有`cmp_residual_kv` | `cmp_ratio=128` |
| CSA | `ori_kv + cmp_kv + cmp_sparse_indices` | `cmp_mask_mode=3`、必须有`cmp_residual_kv` | `cmp_ratio=4`，TopK仅支持512或1024 |

Ascend 950 的 HCA/CSA 仅要求`cmp_ratio > 0`，CSA TopK 要求大于0。

### 4.2 mixed_quant_sparse_flash_mla

特有属性：

- `quant_mode`仅支持1或2。
- `rope_head_dim`仅支持64。
- Q/输出仅支持 BF16。
- KV 仅支持 FP8_E4M3FN。
- `quant_mode=1`时 KV head dim 为608。
- `quant_mode=2`时 KV head dim 为584。
- 当`cmp_mask_mode=3`且`cmp_ratio != 1`时，必须传入`cmp_residual_kv`。

### 4.3 quant_sparse_flash_mla

特有属性和 Tensor：

- `quant_mode`仅支持1。
- Q、ori KV、cmp KV 仅支持 HIFLOAT8。
- 输出仅支持 BF16。
- `q_descale`当前版本必传。
- `ori_kv_descale`当前版本必传。
- `cmp_kv_descale`与`cmp_kv`同时存在或同时不存在。
- 三个 descale Tensor 均要求`float32`、`ND`、shape为`(1,)`。
- 当`cmp_mask_mode=3`且`cmp_ratio != 1`时，必须传入`cmp_residual_kv`。

## 5. 新旧拦截链路关系

三个 Tiling 入口均已切换到新 Checker：

| 算子 | 新入口类 |
| --- | --- |
| `sparse_flash_mla` | `SparseFlashMlaChecker` |
| `mixed_quant_sparse_flash_mla` | `MixedQuantSparseFlashMlaChecker` |
| `quant_sparse_flash_mla` | `QuantSparseFlashMlaChecker` |

混合量化和全量化算子的原`*_check*.cpp`不再加入 Tiling 编译源；`sparse_flash_mla`原内嵌`SMLATilingCheck`不再由 Tiling 入口实例化。新规则不依赖三个算子原有 Checker 的检查结果。

解析器仍负责从 Tiling Context 获取 Tensor、属性、布局、shape 和硬件信息，并生成统一检查上下文；解析完成后只执行本文档描述的新 Checker 链路。

## 6. 检查边界

Host Tiling 阶段只检查可获得的描述信息和属性值，包括：

- Tensor 是否存在
- dtype、format、rank、shape
- 布局和硬件平台
- 标量属性范围
- 参数之间的存在性和形状联动

以下内容无法在当前阶段完整读取或验证，由调用方保证：

- `cu_seqlens_*`的首元素、末元素及单调性
- `seqused_*`的逐元素范围
- `cmp_residual_kv`每个元素是否位于`[0, cmp_ratio)`
- 稀疏索引是否为-1或合法 token 索引
- TopK Length 的逐元素范围
- Block Table 内的 block id 是否有效
- `metadata`是否由完全一致的前置参数生成
