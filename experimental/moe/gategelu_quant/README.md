# 算子名称：GateGeluQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品                                         | 是       |

## 功能说明

- 算子功能：在 Transformer 模型的 FFN 层中，对 Gated GLU 结构的输出进行融合计算与动态 Per-Channel 量化。将输入按列均分为 Gate 和 Value 两部分，对 Gate 部分施加 GELU 激活函数后与 Value 部分逐元素相乘，再经过可选的截断约束，最后乘以缩放因子量化输出为 INT8，即实现 **GeGLU + Per-Channel Quantization** 的融合算子。

- 数学公式：

  $$intermediate = GELU(input[:, :W]) \odot input[:, W:]$$
  $$if \ constrait: \ intermediate = Clamp(intermediate, -clampValue, clampValue)$$
  $$output = Quantize(intermediate \times scale) \quad \in [-128, 127]$$

  其中 $W = gbW / 2$，$\odot$ 表示逐元素乘法，$Quantize$ 表示四舍五入取整并截断到 INT8 范围。

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>in</td>
      <td>输入</td>
      <td>Gate 和 Value 拼接的输入张量，shape 为 (gbH, gbW)，其中 gbW = 2 × hidden_size，前半部分为 Gate，后半部分为 Value</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>Per-Channel 量化缩放因子，shape 为 (gbW / 2, )</td>
      <td>float32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>GeGLU 计算并量化后的 INT8 结果，shape 为 (gbH, gbW / 2)</td>
      <td>int8_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gbH</td>
      <td>输入</td>
      <td>输入张量的行数（token 数量）</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gbW</td>
      <td>输入</td>
      <td>输入张量的列数（Gate + Value 拼接后的隐藏维度，必须为偶数）</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>constrait</td>
      <td>输入</td>
      <td>是否在量化前对 GeGLU 的 FP32 中间结果进行截断约束，默认为 false</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampValue</td>
      <td>输入</td>
      <td>截断约束的阈值，配合 constrait 使用，将值截断在 [-clampValue, clampValue] 内，默认为 128.0</td>
      <td>float32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockDim</td>
      <td>输入</td>
      <td>AI Core 的数量，如 Ascend910B 为 40</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>Device 端的 stream</td>
      <td>AclrtStream</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量的列数 `gbW` 必须为偶数，以便均分为 Gate 和 Value 两部分。
- 数据类型约束：输入为 float16，Scale 为 float32，输出为 int8_t。

## 价值/作用

在 LLaMA、Qwen、DeepSeek 等大语言模型的 W8A8 量化推理部署中，GeGLU 激活及其输出量化是 FFN 层计算的核心步骤。本算子将 GELU 激活、门控乘法、激活值截断及 Per-Channel 动态量化四个操作融合为一个 Kernel，彻底消除了 FP16/FP32 中间结果的 Global Memory 写回与读入，大幅降低显存带宽压力和 Kernel 启动开销，显著提升端到端推理性能。

## 设计方案

### Tiling 策略

- **分核策略**：
  按行（gbH）维度进行分核。每个 Core 处理 `⌈gbH / blockNum⌉` 行数据，采用向上取整方式分配，尾部行通过 `i * blockNum_ + blockIdx_ < gbH` 判断跳过。

- **分块策略**：
  在列方向（bkW = gbW / 2）上按可用 UB 容量进行分块。每个 Tile 的宽度 `tlW_` 根据 UB 最大可用字节数（184KB）和所需 Buffer（2 个 half 输入、1 个 float scale、1 个 int8 输出）动态计算，并向下对齐到 64 元素（满足 128 字节对齐）。尾部 Tile 宽度 `tlTailW_` 为 `bkW % tlW_`，向上对齐到 64 用于计算，实际有效宽度用于搬入搬出。

### Kernel 侧设计

进行 **Init** 和 **Process** 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、数据搬出（CopyOut）三个阶段。采用单缓冲（BUFFER_NUM = 1）机制。

- **初始化（Init）**
  - 计算分核参数：`bkLoop_`（每个 Core 处理的行数）、`blockIdx_`（当前 Core 编号）。
  - 计算分块参数：`bkW_ = gbW / 2`，根据 UB 容量计算 `tlMaxW_`，进而确定 `tlW_`、`tlTailW_`、`tlAlignTailW_`、`tlLoop_`。
  - 建立 GM Tensor 映射（inGm_、scaleGm_、outGm_）。
  - 初始化 VECIN 队列：`inQueIn_`（Gate和Value输入合并）、`inQueScale_`（缩放因子），以及 VECOUT 队列 `outQueOut_`（INT8 输出）。

- **计算流程（Process）**
  - FOR i = 0 TO bkLoop_（行方向循环）：
    - 判断当前行是否在有效范围内
    - **CopyIn**：
      - 从 GM 搬入拼接的 Gate 和 Value 数据到同一块本地内存 `in_local[0]` 和 `in_local[tlW_]`。
      - 从 GM 搬入当前 Tile 对应的 `scale` 数据。
    - **Compute**（高度融合的量化计算流程）：
      1. `Gelu(in_one, in_one)`：对 Gate 部分计算 GELU 激活。
      2. `Mul(in_two, in_one, in_two)`：Gate 和 Value 逐元素相乘，得到 GeGLU 结果。
      3. `Cast(infloat_local, in_two, CAST_NONE)`：将 FP16 结果转为 FP32，以便进行高精度量化计算。
      4. **[可选约束]** 若 `constrait_` 为 true，则使用 `Mins` 和 `Maxs` 将 FP32 结果截断在 `[-clampValue_, clampValue_]` 范围内。
      5. `Mul(infloat_local, infloat_local, scale_local)`：乘以 Per-Channel 量化缩放因子。
      6. `Cast(infloat_local, infloat_local, CAST_RINT)`：FP32 四舍五入到整数（仍以 FP32 格式存储）。
      7. `Mins` 和 `Maxs`：将值截断到 `[-128.0, 127.0]` 的 INT8 表示范围。
      8. `Cast(in_one_local, infloat_local, CAST_NONE)` 和 `Cast(out_local, in_one_local, CAST_RINT)`：将结果从 FP32 经 FP16 转换为最终 INT8 输出。
    - **CopyOut**：将 INT8 计算结果从 UB 搬回 GM，输出偏移量：`offset = (i * blockNum_ + blockIdx_) * bkW_ + j * tlW_`。