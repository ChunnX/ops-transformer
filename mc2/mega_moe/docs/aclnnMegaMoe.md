# aclnnMegaMoe

## 产品支持情况

| 产品 | 是否支持 |
|:----|:--------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：该算子将 `MoE（Mixture of Experts）`以及 `FFN（Feed-Forward Network）`的完整计算流程融合为单个算子，实现 `Dispatch + GroupMatmul1 + SwiGLUQuant + GroupMatmul2 + Combine` 的端到端融合计算。

- 计算公式：

  MegaMoe 是一个多阶段融合的 MoE 算子，在专家并行场景下依次完成 Token 路由分发与量化、分组矩阵乘法 + SwiGLU 激活、二次分组矩阵乘法与跨 Rank 聚合、以及加权求和，完整计算流程可以分解为以下四个阶段。

  第一阶段对输入 Token 按专家分组收集后做 MXFP8 量化，生成各专家的量化输入与缩放因子：

  $$
  \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
  $$

  说明：根据 `topkIds` 将 Token 按专家排序收集，$\mathcal{T}_e$ 为分配到专家 $e$ 的 Token 索引集合，$E_{local}$表示当前专家收到的最大token数，每个专家数值可能不同，$X[\mathcal{T}_e]$ 为对应的子矩阵。$\mathrm{Q}_{\text{MX}}$ 表示 MX 逐组量化（group size = 32），对每组 32 个元素提取共享指数后量化为 FP8 目标类型（FLOAT8_E5M2 或 FLOAT8_E4M3FN），同时输出 FLOAT8_E8M0 缩放因子。量化后的数据将作为 GMM1 的输入。

  第二阶段对每个专家执行 GMM1 矩阵乘法（将 $W_1$ 沿列方向分为两半分别计算）、SwiGLU 激活和 MX 量化：

  $$
  Z_e^{(x)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(x)}, S_{1,e}^{(x)}), \quad Z_e^{(y)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(y)}, S_{1,e}^{(y)})
  $$

  $$
  U_e = Z_e^{(x)} \odot \sigma\!\left(Z_e^{(x)}\right) \odot Z_e^{(y)}
  $$

  $$
  \hat{U}_e,\ S_{U,e} = \mathrm{Q}_{\text{MX}}(U_e)
  $$

  说明：将 $W_1$ 的前 $N/2$ 列 $W_{1,e}^{(x)}$ 和后 $N/2$ 列 $W_{1,e}^{(y)}$ 分别与 MX 反量化后的输入做矩阵乘法，得到 Swish 分支 $Z_e^{(x)}$ 和门控分支 $Z_e^{(y)}$。SwiGLU 激活对两个分支做逐元素乘积 $x \cdot \sigma(x) \cdot y$，其中 $\sigma$ 为 Sigmoid 函数，将中间维度从 $N$ 减半为 $N/2$。随后对 SwiGLU 输出做 MX 量化，得到 GMM2 的量化输入 $\hat{U}_e$。

  第三阶段对每个专家执行 GMM2 矩阵乘法，并将结果按目标 Rank 分发：

  $$
  O_e = \mathrm{DQ}_{\text{MX}}(\hat{U}_e, S_{U,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
  $$

  说明：将量化后的 SwiGLU 输出与第二组权重 $W_2$ 做 MX 反量化后的矩阵乘法，将 $N/2$ 维中间表示映射回 $H$ 维隐藏空间，得到每个专家的输出 $O_e$。计算完成后通过 RDMA peermem 将结果按目标 Rank 的专家偏移地址写入远端，实现跨 Rank 聚合。

  第四阶段对所有 Token 按路由权重加权求和，恢复为与输入相同形状的输出：

  $$
  Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)]
  $$

  说明：对每个 Token $i$，根据排序后的路由索引 $\pi(i,k)$ 从聚合后的专家结果中取出对应行，按 `topkWeights` 中的权重逐元素加权累加，得到最终输出 $Y$。

  其中，$X$ 表示参数 `x`，$W$ 表示参数 `topkWeights`，$W_1$ 表示参数 `weight1`，$W_2$ 表示参数 `weight2`，$Y$ 表示参数 `y`，$E_{\text{local}}$ 表示属性 `moeExpertNum / epWorldSize`（每个 Rank 的专家数），$K$ 表示 `topkIds` 的第二维度（top-K 值，取值 6 或 8）。

  局部变量说明：
  - $\mathcal{T}_e$：被路由到专家 $e$ 的 Token 索引集合，由 `topkIds` 排序后确定。
  - $\hat{X}_e,\ S_{X,e}$：专家 $e$ 的量化输入及其 MX 缩放因子，第一阶段中间结果。
  - $W_{1,e}^{(x)}$、$W_{1,e}^{(y)}$：$W_1$ 对应专家 $e$ 的前 $N/2$ 列和后 $N/2$ 列子矩阵，由 `weight1` 按 SwiGLU 拆分推导。
  - $S_{1,e}^{(x)}$、$S_{1,e}^{(y)}$：$W_{1,e}^{(x)}$ 和 $W_{1,e}^{(y)}$ 对应的 MX 缩放因子，从 `weightScales1` 按维度截取。
  - $S_{2,e}$：$W_{2,e}$ 对应的 MX 缩放因子，来自参数 `weightScales2`。
  - $Z_e^{(x)},\ Z_e^{(y)}$：GMM1 的两路矩阵乘法输出（Swish 分支和门控分支），中间结果。
  - $U_e$：SwiGLU 激活输出，维度 $m_e \times N/2$，中间结果。
  - $\hat{U}_e,\ S_{U,e}$：量化后的 SwiGLU 输出及其 MX 缩放因子，中间结果。
  - $O_e$：GMM2 的专家级输出，维度 $m_e \times H$，中间结果。
  - $\pi(i, k)$：Token $i$ 的第 $k$ 个 top-k 专家在展开排序后的行索引，由路由排序确定。
  - $\mathrm{Q}_{\text{MX}}(\cdot)$：MX 逐组量化操作，block size = 32，输出 FP8 数据和 E8M0 缩放因子。
  - $\mathrm{DQ}_{\text{MX}}(\cdot)$：MX 逐组反量化操作，在 matmul 内部隐式执行。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnMegaMoeGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMegaMoe"接口执行计算。

```cpp
aclnnStatus aclnnMegaMoeGetWorkspaceSize(
  const aclTensor* context,
  const aclTensor* x,
  const aclTensor* topkIds,
  const aclTensor* topkWeights,
  const aclTensor* weight1,
  const aclTensor* weight2,
  const aclTensor* weightScales1,
  const aclTensor* weightScales2,
  const aclTensor* xActiveMask,
  const aclTensor* scales,
  int64_t          moeExpertNum,
  int64_t          epWorldSize,
  int64_t          cclBufferSize,
  int64_t          maxRecvTokenNum,
  int64_t          dispatchQuantMode,
  int64_t          dispatchQuantOutType,
  int64_t          combineQuantMode,
  const char*      commAlg,
  int64_t          globalBs,
  aclTensor*       y,
  aclTensor*       expertTokenNums,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor
)
```

```cpp
aclnnStatus aclnnMegaMoe(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream
)
```

## aclnnMegaMoeGetWorkspaceSize

- 参数说明

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1500px"><colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 350px">
      <col style="width: 250px">
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>context</td>
        <td>输入</td>
        <td>本卡通信域信息数据。</td>
        <td>要求一个1D的Tensor。</span></li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>本卡发送的 token 数据。</td>
        <td><ul><li>BS（dim0）范围 [1, 512]。</li><li>H（dim1）仅支持 4096、5120、7168。</li><li>要求一个2D的Tensor。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1、FLOAT4_E1M2</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topkIds</td>
        <td>输入</td>
        <td>每个 token 的 topK 个专家索引。</td>
        <td><ul><li>K（dim1）仅支持 6 或 8。</li><li>dim0 必须与 x 的 dim0 相等。</li><li>dim1 必须与 topkWeights 的 dim1 相等。</li><li>要求一个2D的Tensor。</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topkWeights</td>
        <td>输入</td>
        <td>每个 token 的 topK 个专家权重。</td>
        <td><ul><li>dim0 必须与 x 的 dim0 相等。</li><li>dim1 必须与 topkIds 的 dim1 相等。</li><li>要求一个2D的Tensor。</li></ul></td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight1</td>
        <td>输入</td>
        <td>GroupMatmul1 计算的右矩阵，用于计算 SwiGLU 激活前的线性变换。</td>
        <td><ul><li>dim0（expertPerRank）范围 [1, 16]。</li><li>dim1（N）仅支持 1024、2048、3072、4096、7168。</li><li>dim2 必须等于 x 的 H 维度。</li><li>dim1 必须等于 weight2 的 dim2 × 2。</li><li>dim0 必须与 weight2 的 dim0 相等。</li><li>数据类型必须与 weight2 相同。</li><li>要求一个3D的Tensor。</li></ul></td>
        <td>BFLOAT16、FLOAT16、INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1、FLOAT4_E1M2</td>
        <td>ND</td>
        <td>3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weight2</td>
        <td>输入</td>
        <td>GroupMatmul2 计算的右矩阵，用于 SwiGLU 激活后的线性变换。</td>
        <td><ul><li>dim1 必须等于 x 的 H 维度。</li><li>dim2 必须等于 weight1 的 dim1 / 2（SwiGLU 关系）。</li><li>dim0 必须与 weight1 的 dim0 相等。</li><li>数据类型必须与 weight1 相同。</li><li>要求一个3D的Tensor。</li></ul></td>
        <td>与 weight1 一致</td>
        <td>ND</td>
        <td>3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightScales1Optional</td>
        <td>输入</td>
        <td>量化场景需要，GroupMatmul1 右矩阵反量化参数。</td>
        <td><ul><li>shape 为 (expertPerRank, N, CeilDiv(H, 64), 2)。</li><li>dim0 必须与 weightScales2 的 dim0 相等。</li><li>dim2 必须等于 CeilDiv(H, 64)。</li><li>dim3 必须等于 2。</li><li>不可传空指针。</li><li>要求一个3D的Tensor。</li></ul></td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightScales2Optional</td>
        <td>输入</td>
        <td>量化场景需要，GroupMatmul2 右矩阵反量化参数。</td>
        <td><ul><li>shape 为 (expertPerRank, H, CeilDiv(N/2, 64), 2)。</li><li>dim0 必须与 weightScales1 的 dim0 相等。</li><li>dim1 必须等于 H。</li><li>dim2 必须等于 CeilDiv(N/2, 64)。</li><li>dim3 必须等于 2。</li><li>不可传空指针。</li><li>要求一个3D的Tensor。</li></ul></td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>xActiveMaskOptional</td>
        <td>输入</td>
        <td>Token 激活掩码，当前版本不支持，必须传入空指针。</td>
        <td><ul><li>必须传入空指针。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>输入</td>
        <td>输入缩放因子，当前版本不支持，必须传入空指针。</td>
        <td><ul><li>必须传入空指针。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>moeExpertNum</td>
        <td>输入</td>
        <td>MoE 模型的总专家数量。</td>
        <td><ul><li>范围 [epWorldSize, 1024]，且 moeExpertNum % epWorldSize == 0。</li><li>必须满足 moeExpertNum == expertPerRank × epWorldSize。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>专家并行通信域大小。</td>
        <td><ul><li>范围 [2, 768]。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cclBufferSize</td>
        <td>输入</td>
        <td>CCL 通信缓冲区大小。</td>
        <td><ul><li>值必须 >= 0。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maxRecvTokenNum</td>
        <td>输入</td>
        <td>每个 Rank 最大可接收 Token 数，默认值为 0 表示自动计算。</td>
        <td><ul><li>范围 [0, BS × epWorldSize × min(topK, expertPerRank)]。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dispatchQuantMode</td>
        <td>输入</td>
        <td>dispatch 通信时量化模式。</td>
        <td><ul><li>当前仅支持 4（MXFP 模式）。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dispatchQuantOutType</td>
        <td>输入</td>
        <td>dispatch量化后输出的数据类型。</td>
        <td><ul><li>仅支持 23（FLOAT8_E5M2）或 24（FLOAT8_E4M3FN）。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>combineQuantMode</td>
        <td>输入</td>
        <td>combine阶段量化模式。</td>
        <td><ul><li>当前版本不支持，必须为 0。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>commAlg</td>
        <td>输入</td>
        <td>通信算法类型。</td>
        <td><ul><li>当前版本不支持，必须为空字符串 ""。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>globalBs</td>
        <td>输入</td>
        <td>全局 batch size，多卡场景下的总 Token 数，默认值为 0 表示使用单卡 BS。</td>
        <td><ul><li>为 0 时使用单卡 BS；非 0 时需满足 BS × epWorldSize <= globalBs 且 globalBs % epWorldSize == 0。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut</td>
        <td>输出</td>
        <td>公式中的 Y，表示加权聚合后的专家计算结果，shape 为 (BS, H)。</td>
        <td><ul><li>数据类型与 x 相同。</li><li>要求一个2D的Tensor。</li></ul></td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>expertTokenNumsOut</td>
        <td>输出</td>
        <td>每个本地专家分配到的 Token 数量，shape 为 (expertPerRank,)。</td>
        <td><ul><li>expertPerRank = moeExpertNum / epWorldSize。</li><li>要求一个1D的Tensor。</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>
  </div>

- 返回值

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述<span style="color:#d97706;"></span></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>内部异常：参数取值或属性组合不满足约束（如 epWorldSize、moeExpertNum、dispatchQuantMode 等属性校验失败）。</td>
    </tr>
  </tbody>
  </table>

## aclnnMegaMoe

- 参数说明

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
    </colgroup>
    <thead>
    <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>输入和输出的shape不在支持的范围内。</td>
    </tr>
    <tr>
        <td>参数的取值不在支持的范围内。</td>
    </tr>
    </tbody>
    </table>

- 返回值

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- **参数一致性约束**：
    - 调用算子过程中使用的 `epWorldSize`、`globalBs`、`HCCL_BUFFSIZE`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域使用约束**：
    - 仅支持 `EP`域，无 `TP` 域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性。
    - 所有卡的 `moe_expert_num`、`ep_world_size`、`ccl_buffer_size`、`max_recv_token_num`、`dispatch_quant_mode`、`dispatch_quant_out_type`、`global_bs` 参数取值需保持一致。

- **参数约束**：
  - BS（x.dim0）范围 [1, 512]。
  - H（x.dim1）仅支持 4096、5120、7168。
  - topK（topkIds.dim1）仅支持 6 或 8。
  - expertPerRank（weight1.dim0）范围 [1, 16]。
  - N（weight1.dim1）仅支持 1024、2048、3072、4096、7168。
  - epWorldSize 范围 [2, 768]。
  - moeExpertNum 范围 [epWorldSize, 1024]，且 moeExpertNum % epWorldSize == 0。
  - maxRecvTokenNum 范围 [0, BS × epWorldSize × min(topK, expertPerRank)]。
  - dispatchQuantOutType 仅支持 23（FLOAT8_E5M2）或 24（FLOAT8_E4M3FN）。
  - globalBs 为 0 或满足 BS × epWorldSize <= globalBs 且 globalBs % epWorldSize == 0。
  - 当前版本仅支持 MXFP 量化模式（dispatchQuantMode = 4），dispatch 阶段使用 MX 逐组量化（group size = 32），量化缩放因子类型为 FLOAT8_E8M0。
  - xActiveMask 和 scales 参数当前版本必须传入空指针，不支持非空输入。
  - combineQuantMode 必须为 0，commAlg 必须为空字符串 ""。
  - y 的数据类型与 x 相同。
  - weight1 的 dim1（N）必须等于 weight2 的 dim2 的二倍，这是因为 SwiGLU 激活需要将中间维度从 N 减半为 N/2。
  - weightScales1 和 weightScales2 不可为空指针。
  - expertPerRank = moeExpertNum / epWorldSize，必须为整数且在 [1, 16] 范围内。
  - weightScales1 和 weightScales2 不可为空指针。

- **MXFP量化场景约束**：
    - weight1 shape 为 (expertPerRank, N, H)，weight2 shape 为 (expertPerRank, H, N/2)。
    - weightScales1 shape 为 (expertPerRank, N, CeilDiv(H, 64), 2)，其中 CeilDiv(H, 64) = (H + 63) / 64。
    - weightScales2 shape 为 (expertPerRank, H, CeilDiv(N/2, 64), 2)，其中 CeilDiv(N/2, 64) = (N/2 + 63) / 64。
    - weightScales1 的 dim3 和 weightScales2 的 dim3 必须等于 2。
    - MXFP 场景下，dispatchQuantOutType=23 时 weight1 和 weight2 必须为 FLOAT8_E5M2，dispatchQuantOutType=24 时必须为 FLOAT8_E4M3FN。
    - xActiveMask 和 scales 必须为空指针。


## 调用示例

- <term>Ascend 950PR/Ascend 950DT</term>：

    本示例支持算子运行在卡数为[2, 8]的单机环境中，用户可以根据需要在示例代码中设置EP_WORLD_SIZE为卡数，并更改moeExpertNum，使得moeExpertNum可以被EP_WORLD_SIZE整除。

    - 编译算子：算子编译命令如下。
        ```bash
        bash build.sh --pkg --soc=ascend950 --ops=mega_moe
        ```

    - 执行算子样例：示例算子执行命令如下，该命令会执行算子[examples](../examples/)目录下所有的示例代码文件。
        ```bash
        bash build.sh --run_example --ops=mega_moe eager cust
        ```

    - 示例代码：
        ```Cpp
        #include <thread>
        #include <iostream>
        #include <string>
        #include <cstring>
        #include <vector>
        #include <memory>
        #include <cstdio>
        #include "acl/acl.h"
        #include "hccl/hccl.h"
        #include "aclnn/opdev/fp16_t.h"
        #include "aclnnop/aclnn_mega_moe.h"

        #define CHECK_RET(cond, return_expr) \
            do {                             \
                if (!(cond)) {               \
                    return_expr;             \
                }                            \
            } while (0)

        #define LOG_PRINT(message, ...)         \
            do {                                \
                printf(message, ##__VA_ARGS__); \
            } while(0)

        struct Args {
            uint32_t rankId;
            uint32_t epRankId;
            HcclComm hcclEpComm;
            aclrtStream megaMoeStream;
            aclrtContext context;
        };

        constexpr uint32_t EP_WORLD_SIZE = 8;
        constexpr uint32_t DEV_NUM = EP_WORLD_SIZE;

        int64_t GetShapeSize(const std::vector<int64_t> &shape)
        {
            int64_t shape_size = 1;
            for (auto i : shape) {
                shape_size *= i;
            }
            return shape_size;
        }

        template<typename T>
        int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
            aclDataType dataType, aclTensor **tensor)
        {
            auto size = GetShapeSize(shape) * sizeof(T);
            auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
            ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
            std::vector<int64_t> strides(shape.size(), 1);
            for (int64_t i = shape.size() - 2; i >= 0; i--) {
                strides[i] = shape[i + 1] * strides[i + 1];
            }
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                shape.data(), shape.size(), *deviceAddr);
            return 0;
        }

        void DestroyTensor(aclTensor *tensor) {
            if (tensor != nullptr) {
                aclDestroyTensor(tensor);
            }
        }

        void FreeDeviceAddr(void *deviceAddr) {
            if (deviceAddr != nullptr) {
                aclrtFree(deviceAddr);
            }
        }

        int LaunchOneThreadMegaMoe(Args &args)
        {
            int ret = aclrtSetCurrentContext(args.context);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);
            
            char hcomEpName[128] = {0};
            ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
            LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, megaMoeStream = %p, context = %p\n", 
                      args.rankId, hcomEpName, args.megaMoeStream, args.context);

            int64_t BS = 64;
            int64_t H = 7168;
            int64_t K = 8;
            int64_t N = 7168;
            int64_t moeExpertNum = 128;
            int64_t epWorldSize = EP_WORLD_SIZE;
            int64_t expertPerRank = moeExpertNum / epWorldSize;
            int64_t cclBufferSize = 0;
            int64_t maxRecvTokenNum = 0;
            int64_t dispatchQuantMode = 4;
            int64_t dispatchQuantOutType = 23;
            int64_t combineQuantMode = 0;
            std::string commAlg = "";
            int64_t globalBs = BS * epWorldSize;

            void *contextDeviceAddr = nullptr;
            void *xDeviceAddr = nullptr;
            void *topkIdsDeviceAddr = nullptr;
            void *topkWeightsDeviceAddr = nullptr;
            void *weight1DeviceAddr = nullptr;
            void *weight2DeviceAddr = nullptr;
            void *weightScales1DeviceAddr = nullptr;
            void *weightScales2DeviceAddr = nullptr;
            void *yDeviceAddr = nullptr;
            void *expertTokenNumsDeviceAddr = nullptr;

            aclTensor *contextTensor = nullptr;
            aclTensor *x = nullptr;
            aclTensor *topkIds = nullptr;
            aclTensor *topkWeights = nullptr;
            aclTensor *weight1 = nullptr;
            aclTensor *weight2 = nullptr;
            aclTensor *weightScales1 = nullptr;
            aclTensor *weightScales2 = nullptr;
            aclTensor *y = nullptr;
            aclTensor *expertTokenNumsOut = nullptr;

            std::vector<int64_t> contextShape{1};
            std::vector<int64_t> xShape{BS, H};
            std::vector<int64_t> topkIdsShape{BS, K};
            std::vector<int64_t> topkWeightsShape{BS, K};
            std::vector<int64_t> weight1Shape{expertPerRank, N, H};
            std::vector<int64_t> weight2Shape{expertPerRank, H, N / 2};
            std::vector<int64_t> weightScales1Shape{expertPerRank, N, (H + 63) / 64, 2};
            std::vector<int64_t> weightScales2Shape{expertPerRank, H, (N / 2 + 63) / 64, 2};
            std::vector<int64_t> yShape{BS, H};
            std::vector<int64_t> expertTokenNumsShape{expertPerRank};

            int64_t contextShapeSize = GetShapeSize(contextShape);
            int64_t xShapeSize = GetShapeSize(xShape);
            int64_t topkIdsShapeSize = GetShapeSize(topkIdsShape);
            int64_t topkWeightsShapeSize = GetShapeSize(topkWeightsShape);
            int64_t weight1ShapeSize = GetShapeSize(weight1Shape);
            int64_t weight2ShapeSize = GetShapeSize(weight2Shape);
            int64_t weightScales1ShapeSize = GetShapeSize(weightScales1Shape);
            int64_t weightScales2ShapeSize = GetShapeSize(weightScales2Shape);
            int64_t yShapeSize = GetShapeSize(yShape);
            int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);

            std::vector<int32_t> contextHostData(contextShapeSize, 1);
            std::vector<int16_t> xHostData(xShapeSize, 1);
            std::vector<int32_t> topkIdsHostData;
            for (int32_t token_id = 0; token_id < topkIdsShape[0]; token_id++) {
                for (int32_t k_id = 0; k_id < topkIdsShape[1]; k_id++) {
                    topkIdsHostData.push_back((token_id * K + k_id) % moeExpertNum);
                }
            }
            std::vector<float> topkWeightsHostData(topkWeightsShapeSize, 0.125);
            std::vector<uint8_t> weight1HostData(weight1ShapeSize, 1);
            std::vector<uint8_t> weight2HostData(weight2ShapeSize, 1);
            std::vector<uint8_t> weightScales1HostData(weightScales1ShapeSize, 1);
            std::vector<uint8_t> weightScales2HostData(weightScales2ShapeSize, 1);
            std::vector<int16_t> yHostData(yShapeSize, 0);
            std::vector<int32_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);

            ret = CreateAclTensor(contextHostData, contextShape, &contextDeviceAddr, aclDataType::ACL_INT32, &contextTensor);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(topkIdsHostData, topkIdsShape, &topkIdsDeviceAddr, aclDataType::ACL_INT32, &topkIds);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(topkWeightsHostData, topkWeightsShape, &topkWeightsDeviceAddr, aclDataType::ACL_FLOAT, &topkWeights);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(weight1HostData, weight1Shape, &weight1DeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &weight1);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(weight2HostData, weight2Shape, &weight2DeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &weight2);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(weightScales1HostData, weightScales1Shape, &weightScales1DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &weightScales1);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(weightScales2HostData, weightScales2Shape, &weightScales2DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &weightScales2);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT32, &expertTokenNumsOut);
            CHECK_RET(ret == ACL_SUCCESS, return ret);

            uint64_t megaMoeWorkspaceSize = 0;
            aclOpExecutor *megaMoeExecutor = nullptr;
            void *megaMoeWorkspaceAddr = nullptr;

            ret = aclnnMegaMoeGetWorkspaceSize(
                contextTensor, x, topkIds, topkWeights,
                weight1, weight2, weightScales1, weightScales2,
                nullptr, nullptr,
                moeExpertNum, epWorldSize, cclBufferSize,
                maxRecvTokenNum, dispatchQuantMode, dispatchQuantOutType,
                combineQuantMode, commAlg.c_str(), globalBs,
                y, expertTokenNumsOut,
                &megaMoeWorkspaceSize, &megaMoeExecutor
            );
            CHECK_RET(ret == ACL_SUCCESS, 
                LOG_PRINT("[ERROR] aclnnMegaMoeGetWorkspaceSize failed. ret = %d \n", ret); return ret);

            if (megaMoeWorkspaceSize > 0) {
                ret = aclrtMalloc(&megaMoeWorkspaceAddr, megaMoeWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
            }

            ret = aclnnMegaMoe(megaMoeWorkspaceAddr, megaMoeWorkspaceSize, megaMoeExecutor, args.megaMoeStream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMegaMoe failed. ret = %d \n", ret); return ret);

            ret = aclrtSynchronizeStreamWithTimeout(args.megaMoeStream, 10000);
            CHECK_RET(ret == ACL_SUCCESS, 
                LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret); return ret);
            LOG_PRINT("[INFO] device_%d aclnnMegaMoe execute successfully.\n", args.rankId);

            if (megaMoeWorkspaceSize > 0) {
                aclrtFree(megaMoeWorkspaceAddr);
            }
            
            DestroyTensor(contextTensor);
            DestroyTensor(x);
            DestroyTensor(topkIds);
            DestroyTensor(topkWeights);
            DestroyTensor(weight1);
            DestroyTensor(weight2);
            DestroyTensor(weightScales1);
            DestroyTensor(weightScales2);
            DestroyTensor(y);
            DestroyTensor(expertTokenNumsOut);

            FreeDeviceAddr(contextDeviceAddr);
            FreeDeviceAddr(xDeviceAddr);
            FreeDeviceAddr(topkIdsDeviceAddr);
            FreeDeviceAddr(topkWeightsDeviceAddr);
            FreeDeviceAddr(weight1DeviceAddr);
            FreeDeviceAddr(weight2DeviceAddr);
            FreeDeviceAddr(weightScales1DeviceAddr);
            FreeDeviceAddr(weightScales2DeviceAddr);
            FreeDeviceAddr(yDeviceAddr);
            FreeDeviceAddr(expertTokenNumsDeviceAddr);

            HcclCommDestroy(args.hcclEpComm);
            aclrtDestroyStream(args.megaMoeStream);
            aclrtDestroyContext(args.context);
            LOG_PRINT("[INFO] device_%d Destroy.\n", args.rankId);
            aclrtResetDevice(args.rankId);
            LOG_PRINT("[INFO] device_%d Reset.\n", args.rankId);
            return 0;
        }

        int main(int argc, char *argv[])
        {
            LOG_PRINT("[INFO] run_example_on_950PR.\n");
            int ret = aclInit(nullptr);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

            aclrtStream megaMoeStream[DEV_NUM];
            aclrtContext context[DEV_NUM];
            for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
                ret = aclrtSetDevice(rankId);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
                ret = aclrtCreateContext(&context[rankId], rankId);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
                ret = aclrtCreateStream(&megaMoeStream[rankId]);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            }

            int32_t devicesEp[EP_WORLD_SIZE];
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[epId] = epId;
            }

            HcclComm commsEp[EP_WORLD_SIZE];
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret %d\n", ret); return ret);

            Args args[DEV_NUM];
            std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
            for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
                uint32_t epRankId = rankId;

                args[rankId].rankId = rankId;
                args[rankId].epRankId = epRankId;
                args[rankId].hcclEpComm = commsEp[epRankId];
                args[rankId].megaMoeStream = megaMoeStream[rankId];
                args[rankId].context = context[rankId];
                threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadMegaMoe, std::ref(args[rankId])));
            }

            for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
                threads[rankId]->join();
            }

            aclFinalize();
            LOG_PRINT("[INFO] aclFinalize success\n");
            return 0;
        }
        ```
