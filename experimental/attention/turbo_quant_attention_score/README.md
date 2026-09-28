# TurboQuantAttentionScore

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

`TurboQuantAttentionScore` 在 Paged KV Cache 上直接由 TurboQuant 编码的 Key
计算 softmax 前的 attention score，无需将 Key 完整反量化回原空间：

```text
score[i] = x_norm[i] * <Hq, dequant(idx[i])>
         + (sqrt(pi/2) / qjl_dim) * gamma[i] * <S(Hq), qjl_signs[i]>
```

其中 `qjl_dim = head_dim`（当前测试均为 128）。无效 KV 位置写 `-inf`。

## 参数说明

| 参数名 | 输入/输出/属性 | shape | 数据类型 | 格式 |
|---|---|---|---|---|
| query_rotated | 输入 | `[T,H,D]` | BFLOAT16 | ND |
| query_qjl | 输入 | `[T,H,D]` | BFLOAT16 | ND |
| key_cache_idx | 输入 | `[num_blocks,128,KvH,idxBytes]` | UINT8 | ND |
| key_cache_qjl | 输入 | `[num_blocks,128,KvH,qjlBytes]` | UINT8 | ND |
| key_cache_norm | 输入 | `[num_blocks,128,KvH]` | BFLOAT16 / FLOAT16 | ND |
| key_cache_gamma | 输入 | `[num_blocks,128,KvH]` | BFLOAT16 / FLOAT16 | ND |
| block_table | 输入 | `[B,maxBlocks]` | INT32 | ND |
| seq_lens | 输入 | `[B]` | INT32 | ND |
| mse_bits | 属性 | 标量，默认 `3`，取值 `{2,3,4}` | INT | - |
| attn_scores | 输出 | `[T,H,maxKvLen]` | FLOAT32 | ND |

`idxBytes = ceil(D * mse_bits / 8)`，`qjlBytes = ceil(D / 8)`。

## 约束说明

- `block_size` 固定为 128。
- 仅支持 MHA/GQA（`H % KvH == 0`），不支持 MLA。
- `batch = block_table.shape[0]`；`batch == 1` 时允许 prefill（`T > 1`）；
  `batch > 1` 时要求 `T == batch`（decode）。
- 对 token `t`：`batchIdx = (batch == 1 ? 0 : t)`。
- InferShape 将 `maxKvLen` 设为 `maxBlocks * 128`；实际有效长度由 `seq_lens` 决定。
- 当前注册 `ascend910b` 与 `ascend910_93`。

## ops-transformer 集成

本目录是 ops-transformer 仓库 `experimental/attention` 下的单算子模块。

```text
├── CMakeLists.txt
├── docs/aclnnTurboQuantAttentionScore.md
├── examples/test_aclnn_turbo_quant_attention_score.cpp
├── op_host
├── op_kernel
└── tests/ut
```

调用方式见 [`docs/aclnnTurboQuantAttentionScore.md`](docs/aclnnTurboQuantAttentionScore.md)。
算法与 tiling 细节见 [`docs/design.md`](docs/design.md)。

## 贡献说明

本算子按 CANN 社区开源算子贡献规范提交，主要约定如下：

- 目录仅保留 `op_host` / `op_kernel` / `docs` / `examples` / `tests`，不包含 `op_graph`、`op_host/config`。
- Host 拆分为 `*_def.cpp`、`*_tiling.cpp`、`*_infershape.cpp`；Kernel 声明与实现分离，TilingData 使用标准 C++ POD。
- `ACLNTYPE` 设置为 `aclnn_exclude`；UB 对齐字节通过 `Ops::Base::GetUbBlockSize` 获取，不硬编码。
- 许可证、空指针检查、失败日志、命名语义等遵循社区 Code Review 清单。
- UT 覆盖 `op_host`、`op_kernel`、`op_api` 三部分。

欢迎通过 Issue / MR 反馈问题与改进建议；提交前请对照社区开源贡献算子检查项完成自检。
