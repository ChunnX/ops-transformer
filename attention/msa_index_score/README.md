# MsaIndexScore

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：计算 MSA（MiniMax Sparse Attention）模块 Index Branch 中的 block score。对每个 query token 与每个 KV sparse block，取该 block 内所有因果可见 token 的 $Q_{idx}$ 和 $K_{idx}$（可选 int8 反量化）的"matmul+maxpool"运算，得到逐 block 的重要性分数 `score`，用作 Index Branch 中后续 TopK 的输入。Prefill 与 Decode 由同一接口承载。

- **计算公式**：

    - 非量化场景：

    $$
    score = Maxpool[ Q_{idx}@K_{idx}^{T} ]
    $$

    - int8 量化场景：

    $$
    score = Maxpool[ scale \cdot Q_{idx}@K_{idx}^{T} ]
    $$

    完整公式：

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    其中 Maxpool 按 sparse block（长度为 $block\_size$）在 KV token 维上取最大值。`start_loc`、`init_blocks`、`local_blocks` 共同生成 $local\_mask$，对序列头部 / 当前 query 附近若干强制保留的 block 写入高分，保证后续 TopK 一定选中它们。与 Triton raw score kernel 对齐时可将 `init_blocks`、`local_blocks` 置 0，关闭 $local\_mask$。

- **主要计算过程**：

    1. 将每个 query token 的 $Q_{idx}$ 与 KV sparse block 内因果可见的 $K_{idx}$ 做矩阵乘，得到 token 级相关性；int8 场景先按 $scale$ 反量化。
    2. 按 `sparse_mode` 叠加 $atten\_mask$：`sparse_mode=0` 不做因果截断，`sparse_mode=3` 按 rightDownCausal 屏蔽不可见位置。
    3. 在每个长度为 $block\_size$ 的 sparse block 上做 Maxpool，得到逐 block 的重要性分数。
    4. 用 `start_loc`、`init_blocks`、`local_blocks` 生成 $local\_mask$，对强制保留的 block 写入高分，得到输出 `score`。本算子不包含 TopK。

## 参数说明

> **说明：**
>
> - B（Batch Size）表示输入样本批量大小
> - S（Sequence Length）表示序列长度，$S1$ 为 query 侧、$S2$ 为 key 侧
> - T 表示所有 Batch 序列长度累加和，$T1$ 为 query 侧、$T2$ 为 key 侧
> - N（Head Num）表示头数，$N1$ 为 query 侧、$N2$ 为 key 侧
> - D（Head Dim）表示单个注意力头维度
> - PageAttention 场景下 $block\_num$ 为物理 block 总数、$block\_size$ 为每个 block 的 token 数，$maxBlockNumPerSeq$ 为每个 batch 最大逻辑 block 数（通常 $\ge\lceil S2/block\_size\rceil$），$M_b=\lceil S2/block\_size\rceil$ 为逻辑 block 总数

<table style="undefined;table-layout: fixed; width: 1280px"><colgroup>
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 560px">
<col style="width: 320px">
<col style="width: 120px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>query</td>
    <td>输入</td>
    <td>公式中的 $Q_{idx}$。当前仅支持 TND，shape 为 $[T1, N1, D]$。</td>
    <td>BFLOAT16、FLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的 $K_{idx}$。支持 TND（$[T2, N2, D]$）、BNBD（$[block\_num, N2, block\_size, D]$）、BBND（$[block\_num, block\_size, N2, D]$）。A2/A3 与 950 上 PA（BBND/BNBD）允许首轴（物理 page）按 stride 非连续存放；size>1 的非首轴须连续，否则 tiling 拒绝。</td>
    <td>BFLOAT16、FLOAT16、INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>block_table</td>
    <td>可选输入</td>
    <td>PageAttention 的逻辑 block → 物理 page 映射表。PA 场景必须传入，二维，第二维长度不能小于 $maxBlockNumPerSeq$；shape 为 $[B, S2/block\_size]$。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>可选输入</td>
    <td>公式中的 $scale$，反量化系数。非量化必须为空；量化场景必选。PA 为 $[block\_num, N2, block\_size]$ 或 $[block\_num, block\_size, N2]$；TND 为 $[T2, N2]$。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>atten_mask</td>
    <td>可选输入</td>
    <td>控制因果可见的 mask。仅在 <code>sparse_mode=3</code> 时使用；取值为 1 表示该位不参与计算，为 0 表示参与计算；shape 为 $[2048, 2048]$。</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actual_seq_qlen</td>
    <td>可选输入</td>
    <td>每个 Batch 中 Query 的有效 token 数。query 为 TND 时必须传入，单调不减（前缀和），shape 为 $[B+1]$。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actual_seq_klen</td>
    <td>可选输入</td>
    <td>每个 Batch 中 Key 的有效 token 数。key 为 TND 时必须传入（前缀和）；PageAttention 场景下为各请求可见 $S2$，shape 为 $[B]$。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>start_loc</td>
    <td>输入</td>
    <td>当前 query 所在逻辑 block 索引（非 token 前缀），用于生成 $local\_mask$；shape 为 $[B]$。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>layout_key</td>
    <td>属性</td>
    <td>key 布局。<code>"TND"</code> / <code>"BBND"</code> / <code>"BNBD"</code>。aclnn 参数名为 <code>layoutKeyOptional</code>，不传时默认 <code>"BBND"</code>。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparse_mode</td>
    <td>属性</td>
    <td>sparse 模式。0：defaultMask（<code>atten_mask</code> 传空）；3：rightDownCausal（须传入 $[2048, 2048]$ 的 <code>atten_mask</code>）。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>init_blocks</td>
    <td>属性</td>
    <td>$local\_mask$ 强制选中的头部 block 数。对逻辑 block $[0, init\_blocks)$ 写入高分 $1\mathrm{e}30$。可选，默认 $0$。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>local_blocks</td>
    <td>属性</td>
    <td>$local\_mask$ 强制选中的局部窗口长度。窗口为 $[max(0, start\_loc+1-local\_blocks), start\_loc]$，写入高分 $1\mathrm{e}29$（覆盖同位置的 <code>init_blocks</code>）。可选，默认 $1$（对齐 MiniMax HF）；与 Triton raw score 对齐时置 $0$。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>score</td>
    <td>输出</td>
    <td>公式中的 $score$，逐 block 重要性分数；shape 为 $[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 当前 $block\_size$ 仅支持 128。
- `layout_key` 必须显式指定：`"BBND"` / `"BNBD"` / `"TND"`，与 `key` 实际 shape 一致。
- PageAttention（`layout_key` 为 `"BBND"` / `"BNBD"`）场景下，`block_table` 必须传入；TND key 场景不得传入 `block_table`，`actual_seq_klen` 为 `[B+1]` 前缀和。
- 非量化场景下，`key` dtype 与 `query` 相同（BFLOAT16 / FLOAT16；Ascend 950 另支持 HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN），`scale` 必须为空。量化场景仅支持 INT8（fp16 query），`scale` 必选：PA 为 $[block\_num, N2, block\_size]$ 或 $[block\_num, block\_size, N2]$，TND 为 $[T2, N2]$，dtype 为 FLOAT。三种 FP8 仅 950，且 query 与 key 必须同型、不得传 `scale`。
- `sparse_mode` 当前仅支持 0、3：
    - 为 0 时，代表 defaultMask 模式，`atten_mask` 传入空；
    - 为 3 时，代表 rightDownCausal 模式，`atten_mask` 必须传入，shape 为 $[2048, 2048]$，取值为 1 代表该位不参与计算，为 0 代表该位参与计算。
- `init_blocks`、`local_blocks` 必须 $\ge 0$ 且不超过逻辑 block 数（PA 为 `block_table` 第二维；TND 为 score 末维对齐宽度）。两者均为 0 时跳过 $local\_mask$。
- PageAttention `block_table` 第二维可以大于实际 KV 逻辑 block 数；score 末维为 $\mathrm{RoundUp}(width, 16)$。Ascend 950 C2UB 对超过 256 列按 256 列滑窗 flush。
- PageAttention `key` 仅允许 dim0 非连续；dim1 至末维（size>1）stride 必须与紧凑布局一致，否则 tiling 报非法输入。TND `key` 须全连续。
- A2/A3 与 Ascend 950：`q_len` / `kv_len` 允许为 0（含整 batch）。对应请求跳过 QK；空 KV 的 score 填 `-inf`；整 batch `T1=0` 时 `SetBlockDim(1)`。
- A2/A3 与 Ascend 950：按估计 M-task 数启动 MIX（`B=1` 时为 $\mathrm{CeilDiv}(T1\cdot N1, 128)$；多 batch 用 packed+$B$ 上界），短 decode 不打满 AIC。950 短 M 长 S 再按可见 KV stile 切 `kvChunks`。
- 本算子输出止于 block score，**不包含** TopK。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| aclnn接口 | [test_aclnn_msa_index_score](./examples/test_aclnn_msa_index_score.cpp) | 通过[aclnnMsaIndexScore](./docs/aclnnMsaIndexScore.md)接口方式调用算子 |
| PyTorch API | - | 通过[msa_index_score](./docs/torchapi_msa_index_score.md)接口调用msa_index_score算子 |
