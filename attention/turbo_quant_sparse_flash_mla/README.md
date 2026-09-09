# TurboQuantSparseFlashMla

## 产品支持情况

| 产品                                                          | 是否支持 |
| :------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                         |     ×    |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>         |     √    |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>         |     √    |
|<term>Atlas 200I/500 A2推理系列产品</term>                      |     ×    |
|<term>Atlas 推理系列产品</term>                                 |     ×    |
|<term>Atlas 训练系列产品</term>                                 |     ×    |

## 功能说明

- 算子功能：

  `TurboQuantSparseFlashMla`面向DeepSeek DSA的shared-KV稀疏注意力推理场景，在同一个kernel内融合原始窗口KV计算、压缩KV稀疏选取、TurboQuant 4bit（TQ4）反量化和FlashAttention计算。压缩KV的反量化结果不写回GM，从而减少KV cache容量和中间结果访存。

  当前算子仅支持SCFA（Sparse Compressed Flash Attention）模板。调用时必须同时传入原始窗口KV `ori_kv`、TQ4压缩KV `cmp_kv`以及由`SparseAttnSharedkvMetadata`生成的任务列表`metadata`。

  典型调用流程如下：

  1. 使用与`q`相同的数据类型准备分页原始窗口KV `ori_kv`。
  2. 使用`TurboQuantCompressLatent(output_mode=1)`生成TQ4 compact格式的分页压缩KV `cmp_kv`。
  3. 准备`cmp_sparse_indices`、两份block table、序列长度和`sinks`，并通过`SparseAttnSharedkvMetadata`生成`metadata`。
  4. 调用`TurboQuantSparseFlashMla`完成原始窗口与压缩稀疏区域的融合Attention计算。

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$由原始窗口KV与`cmp_sparse_indices`选中的TQ4压缩KV共同组成。`sinks`作为每个query head的额外softmax分母项参与在线softmax，但不向输出贡献value。

- TQ4 compact slot格式：

  每个压缩KV token固定占用258字节，布局如下：

  ```text
  [0, 256)   512个4bit code，偶数维位于低nibble，奇数维位于高nibble
  [256, 258) 1个FLOAT16 corrected_scale
  ```

  反量化过程为：

  ```text
  code[d] = unpack(cmp_kv_slot[d / 2])
  kv_hat[d] = centroid[code[d]] * corrected_scale
  corrected_scale = norm(latent) / norm(centroid[code])
  ```

  query、KV和attention output位于同一个signed-Hadamard基中；调用方在Attention前变换query，并在输出后执行逆变换。TQ4 compact payload相对512维FP16/BF16 KV的理论压缩比为`512 * 2 / 258 = 3.97x`。

## 参数说明
>
> **说明：**<br>
> 参数维度含义：B表示Batch Size，Q_T表示所有batch中query的Total Tokens，Q_N和KV_N分别表示query和key/value的Head Num，ORI_BLOCK_NUM和CMP_BLOCK_NUM分别表示原始与压缩KV的物理block总数，ORI_BLOCK_SIZE和CMP_BLOCK_SIZE分别表示对应分页KV的block size，D表示head dim，K表示每个query token选择的压缩KV索引数。

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 220px">
  <col style="width: 150px">
  <col style="width: 700px">
  <col style="width: 180px">
  <col style="width: 100px">
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
      <td>q</td>
      <td>输入</td>
      <td>对应公式中的Q，shape为[Q_T, Q_N, 512]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_kv</td>
      <td>可选输入</td>
      <td>对应公式中K和V的原始窗口部分。OpDef中声明为可选输入，当前SCFA模板实际必传；shape为[ORI_BLOCK_NUM, ORI_BLOCK_SIZE, 1, 512]或[ORI_BLOCK_NUM, 1, ORI_BLOCK_SIZE, 512]，数据类型必须与q一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_kv</td>
      <td>可选输入</td>
      <td>对应公式中K和V的压缩稀疏部分。OpDef中声明为可选输入，当前SCFA模板实际必传；由TurboQuantCompressLatent(output_mode=1)生成，shape为[CMP_BLOCK_NUM, CMP_BLOCK_SIZE, 1, 258]或[CMP_BLOCK_NUM, 1, CMP_BLOCK_SIZE, 258]。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_sparse_indices</td>
      <td>可选输入</td>
      <td>原始KV的稀疏索引预留输入，当前版本不支持传入非空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_sparse_indices</td>
      <td>可选输入</td>
      <td>表示从cmp_kv中离散取数的batch内逻辑索引，-1表示无效或填充slot。当前SCFA模板必传，shape为[Q_T, 1, K]，K支持512或1024。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_block_table</td>
      <td>可选输入</td>
      <td>PageAttention中ori_kv使用的block映射表。当前SCFA模板必传，shape为[B, MAX_ORI_BLOCKS]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_block_table</td>
      <td>可选输入</td>
      <td>PageAttention中cmp_kv使用的block映射表。当前SCFA模板必传，shape为[B, MAX_CMP_BLOCKS]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>TND布局下各batch的query累积序列长度，当前SCFA模板必传。shape为[B+1]，首元素为0，末元素等于Q_T。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>非分页TND KV布局下ori_kv的累积序列长度预留输入；当前仅支持分页KV布局，不使用该输入。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>非分页TND KV布局下cmp_kv的累积序列长度预留输入；当前仅支持分页KV布局，不使用该输入。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>各batch中q实际参与计算的token数预留输入，当前TND布局不支持传入非空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>可选输入</td>
      <td>各batch中ori_kv实际参与计算的token数，当前SCFA模板必传，shape为[B]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>可选输入</td>
      <td>各batch中cmp_kv实际参与计算的token数预留输入，当前版本不支持传入非空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>可选输入</td>
      <td>用于恢复压缩前KV长度的余数预留输入，当前版本不支持传入非空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>可选输入</td>
      <td>原始KV有效稀疏索引数量预留输入，当前版本不支持传入非空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>可选输入</td>
      <td>压缩KV有效稀疏索引数量预留输入，当前版本不支持传入非空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinks</td>
      <td>可选输入</td>
      <td>每个query head的attention sink值。OpDef中声明为可选输入，当前SCFA模板实际必传，shape为[Q_N]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>可选输入</td>
      <td>SparseAttnSharedkvMetadata生成的任务切分结果。OpDef中声明为可选输入，当前SCFA模板实际必传，shape固定为[1024]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_scale</td>
      <td>可选属性</td>
      <td>对应公式中的softmax_scale，默认值为1.0，通常设置为1/sqrt(512)。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>cmp_kv相对于压缩前KV长度的压缩倍率，用于计算压缩侧因果mask边界。支持4或128，默认值为4；TQ4调用使用4。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>q与ori_kv计算的mask模式，当前仅支持4（Band模式），默认值为4。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>q与cmp_kv计算的mask模式，当前仅支持3（RightDownCausal模式），默认值为3。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>q与ori_kv计算中每个query token向左关注的窗口大小，必须为非负数，默认值为127。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>q与ori_kv计算中每个query token向右关注的窗口大小，当前仅支持0，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>输入q的数据排布格式，当前仅支持"TND"，默认值为"TND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>输入ori_kv和cmp_kv的数据排布格式，支持"PA_BSND"和"PA_BNSD"，默认值为"PA_BSND"；"PA_ND"作为"PA_BSND"的兼容别名保留。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk_value_mode</td>
      <td>可选属性</td>
      <td>表示TopK索引取值模式，当前仅支持1，默认值为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>可选属性</td>
      <td>是否返回softmax的log-sum-exp结果，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_kv_stride</td>
      <td>可选属性</td>
      <td>ori_kv物理block首轴stride，单位为元素。PyTorch封装会根据ori_kv.stride(0)自动设置；直接调用aclnn接口时必须传入能够覆盖一个完整物理block的值。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_kv_stride</td>
      <td>可选属性</td>
      <td>cmp_kv物理block首轴stride，单位为字节。PyTorch封装会根据cmp_kv.stride(0)自动设置；直接调用aclnn接口时必须传入能够覆盖一个完整物理block的值。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kv_quant_mode</td>
      <td>可选属性</td>
      <td>压缩KV量化模式，当前仅支持3（TQ4 compact corrected格式），默认值为3。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attn_out</td>
      <td>输出</td>
      <td>对应公式中的输出O，shape和数据类型与q一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_lse</td>
      <td>可选输出</td>
      <td>softmax的log-sum-exp结果。return_softmax_lse为false时shape为[0]；为true时shape为[Q_T, Q_N, 1]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口用于推理场景，支持动态shape和aclgraph模式，不支持动态rank。
- 当前仅支持SCFA模板，必须同时传入`ori_kv`和`cmp_kv`；`ori_kv`参与滑动窗口Attention，`cmp_kv`参与压缩稀疏Attention。
- 当前SCFA路径必须传入`ori_kv`、`cmp_kv`、`cmp_sparse_indices`、`ori_block_table`、`cmp_block_table`、`cu_seqlens_q`、`seqused_ori_kv`、`sinks`和`metadata`。这些输入虽然为兼容通用SparseFlashMla原型而在OpDef中声明为可选输入，但传入空指针会导致参数校验失败。
- `ori_sparse_indices`、`seqused_q`、`seqused_cmp_kv`、`cmp_residual_kv`、`ori_topk_length`和`cmp_topk_length`必须为nullptr或空Tensor；当前版本不支持传入非空Tensor。
- `q`的layout仅支持TND，shape为[Q_T, Q_N, 512]，数据类型支持FLOAT16和BFLOAT16。Q_N必须为4的倍数。
- `cu_seqlens_q`的shape为[B+1]，首元素固定为0，元素单调不减，末元素等于Q_T。每个batch的query长度由相邻元素之差确定。
- `ori_kv`和`cmp_kv`仅支持PageAttention布局：
  - `layout_kv="PA_BSND"`时，shape分别为[ORI_BLOCK_NUM, ORI_BLOCK_SIZE, 1, 512]和[CMP_BLOCK_NUM, CMP_BLOCK_SIZE, 1, 258]。
  - `layout_kv="PA_BNSD"`时，shape分别为[ORI_BLOCK_NUM, 1, ORI_BLOCK_SIZE, 512]和[CMP_BLOCK_NUM, 1, CMP_BLOCK_SIZE, 258]。
  - 兼容别名`layout_kv="PA_ND"`按`PA_BSND`处理。
- KV_N仅支持1。`ori_kv`的D固定为512，数据类型必须与`q`一致；`cmp_kv`的D固定为258，数据类型必须为UINT8。
- `ori_block_table`和`cmp_block_table`均为二维INT32 Tensor，shape第一维必须等于B，第二维应覆盖对应batch所需的最大逻辑block数量。
- ORI_BLOCK_SIZE和CMP_BLOCK_SIZE的取值范围均为[16, 1024]且必须为16的倍数。
- `seqused_ori_kv`为各batch原始KV的实际token数，shape为[B]。压缩KV的逻辑mask边界由原始KV长度、`cmp_ratio`和`cmp_sparse_indices`共同确定。
- `cmp_sparse_indices`的shape为[Q_T, 1, K]，K仅支持512或1024；索引为batch内cmp_kv逻辑token位置，-1表示无效或填充slot。
- `sinks`的数据类型为FLOAT32，shape固定为[Q_N]；`metadata`的数据类型为INT32，shape固定为[1024]。
- `kv_quant_mode`仅支持3。每个`cmp_kv` token包含256字节packed int4 code和2字节FLOAT16 corrected scale，共258字节；必须与`TurboQuantCompressLatent(output_mode=1)`的输出配套使用。
- `cmp_ratio`支持4或128，当前TQ4调用使用4；`topk_value_mode`仅支持1。
- `ori_mask_mode`仅支持4，`cmp_mask_mode`仅支持3，`ori_win_left`必须大于等于0，`ori_win_right`仅支持0。
- MTP场景中，当单请求query长度大于1时，每个query token独立应用causal sliding-window mask。
- `ori_kv_stride`以元素为单位，必须不小于`ORI_BLOCK_SIZE * KV_N * 512`；`cmp_kv_stride`以字节为单位，必须不小于`CMP_BLOCK_SIZE * KV_N * 258`。允许cache首轴非连续，只要stride能够覆盖一个完整物理block。PyTorch封装会从输入Tensor的`stride(0)`自动设置这两个属性。
- 单请求query长度为0时直接跳过；KV有效长度为0时跳过Attention计算，并将对应`attn_out`清零。
- `return_softmax_lse`为false时，`softmax_lse`为shape `[0]`的FLOAT32占位Tensor；为true时，shape为`q.shape[:-1] + [1]`。
- TQ4 compact payload的3.97倍为理论KV数据压缩比，不等同于整个KV池或端到端模型的显存收益；完整显存占用还包括原始窗口KV、索引、block table、workspace和模型权重。

## 调用说明

| 调用方式   | 样例代码 | 说明 |
| ---------- | -------- | ---- |
| aclnn API  | `aclnnTurboQuantSparseFlashMla` | 通过两阶段GetWorkspaceSize和异步执行接口调用`TurboQuantSparseFlashMla`算子 |
| PyTorch API | [turbo_quant_sparse_flash_mla](./torch_extension/turbo_quant_sparse_flash_mla.py) | 通过`cann_ops_transformer.turbo_quant_sparse_flash_mla`调用`TurboQuantSparseFlashMla`算子 |
