# KeyPool

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Ascend 950PR/Ascend 950DT（A5） | √ |

## 功能说明

KeyPool 将当前输入 token 投影为 K 和 Gate，按 `cmp_ratio` 个 token
进行加权池化，并通过分页 `state_cache` 保留未完成压缩块，供后续调用继续
完成池化。

对每个 token，计算：

```text
K = hidden_states @ wk.T
G = hidden_states @ gate_weight.T
K_norm = LayerNorm(K, dim=-1)       # 仅在提供 norm_weight/norm_bias 时执行
```

随后：

1. 将 `K_norm` 和原始 `G` 写入 `state_cache`。`ape` 不写入 cache。
2. 根据 `cache_block_table` 将逻辑 token 位置映射到物理 cache block。
3. 对每个压缩组计算 `softmax(G + ape)`。
4. 沿压缩轴计算加权和，得到 `pooled_key`。

LayerNorm 只沿每个 token 的最后一维 `D` 计算。RoPE 的参数入口已保留，
但当前版本不实现 RoPE；传入非空 `cos`/`sin` 会被拒绝。

## 输入输出

输入顺序以 PyTorch 接口为例：

```python
torch_npu.key_pool(
    hidden_states, wk, gate_weight, ape, state_cache, cache_block_table,
    start_pos, norm_weight=None, norm_bias=None, cos=None, sin=None,
    cu_seqlens=None, seqused=None, cmp_ratio=4, norm_eps=1e-6,
    rotary_mode=1,
)
```

| 参数 | 类型和 Shape | 说明 |
| --- | --- | --- |
| `hidden_states` | BF16/FP16，`[B,S,H]` 或 `[T,H]` | BSH 或 TH 输入 |
| `wk` | 与 `hidden_states` 同 dtype，`[D,H]` | K 投影权重 |
| `gate_weight` | 与 `hidden_states` 同 dtype，`[D,H]` | Gate 投影权重 |
| `ape` | FP32，`[R,D]` | 位置偏置，`R=cmp_ratio` |
| `state_cache` | FP32，`[N,BS,2D]` | 前 `D` 列保存 K，后 `D` 列保存 Gate；原地更新 |
| `cache_block_table` | INT32，`[B,L]` | 逻辑 block 到物理 block 的映射 |
| `start_pos` | INT32，`[B]` | 当前输入在逻辑序列中的起始位置 |
| `norm_weight` | FP32，`[D]` 或 `None` | 与 `norm_bias` 成对出现 |
| `norm_bias` | FP32，`[D]` 或 `None` | 与 `norm_weight` 成对出现 |
| `cos`/`sin` | 保留接口 | 当前版本必须同时为空 |
| `cu_seqlens` | TH 时 INT32，`[B+1]` | TH 的累计序列边界；BSH 时必须为空 |
| `seqused` | 保留接口 | 当前版本必须为空 |

其中 `R` 表示压缩率，`N` 表示物理 block 数，`BS` 表示 block size，
`L` 表示每个 batch 的最大逻辑 block 数。

输出只有一个：

```text
pooled_key: 与 hidden_states 同 dtype，固定容量 BSH [B, Pcap, D]
Pcap = ceil(L * BS / cmp_ratio)
```

每个 batch 的有效池化结果写在前部，不足一个完整压缩组的尾部以及其余容量
保持为零。TH 输入同样返回 BSH 输出。

## 属性约束

| 属性 | 类型 | 默认值 | 支持范围 |
| --- | --- | --- | --- |
| `cmp_ratio` | INT64 | 4 | `2/4/8/16/32/64/128` |
| `norm_eps` | FLOAT | `1e-6` | 大于 0 |
| `rotary_mode` | INT64 | 1 | `0/1`；RoPE 当前未实现 |
| `state_cache_stride_dim0` | INT64 | 0 | 由调用方传入实际 stride |

## 约束说明

- 支持 BSH 和 TH 两种场景：
  - BSH 场景 `cu_seqlens` 必须为空；
  - TH 场景必须提供合法的 `cu_seqlens` 前缀和数组，首元素为 0，
    末元素为 `T`，且单调不减。
- `cmp_ratio` 仅支持 `2/4/8/16/32/64/128`；`norm_eps` 必须大于 0；
  `rotary_mode` 仅支持 `0/1`。
- `norm_weight` 和 `norm_bias` 必须同时提供或同时为空。
- `cos` 和 `sin` 必须同时为空，当前不支持 RoPE；`seqused` 必须为空，
  当前不支持通过 `attention_mask` 选择 token。
- 支持 BSH 场景的 `B=0` 或 `S=0`，以及 TH 场景的 `T=0`。

## 调用方式

- aclnn：参见 [aclnnKeyPool](docs/aclnnKeyPool.md)。
- PyTorch：使用安装后的 `torch_npu.key_pool`。

测试入口位于 `tests/pytest`，包括 CPU Golden、BSH/TH、LayerNorm、历史
cache、压缩率、跨 block 和长序列用例。
