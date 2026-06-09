# 算子名称：LayerNormAddAddQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品                                         | 是       |

## 功能说明

- 算子功能：在大语言模型的 Transformer 层中，将残差加法、层归一化和动态量化三个操作融合为一个 Kernel。具体而言，先对两个输入进行逐元素相加（其中 inTwo 为按行广播的偏置），再与第三个输入相加，随后对相加结果执行 LayerNorm（乘 gamma 加 beta），最后对归一化结果乘以缩放因子 scale 并量化截断为 int8 输出。同时，保留相加后的 fp16 中间结果供后续网络使用。

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
      <td>inOne</td>
      <td>输入</td>
      <td>主输入张量，shape 为 (gbH, gbW)</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inTwo</td>
      <td>输入</td>
      <td>偏置向量，shape 为 (gbW,)，在行方向广播与 inOne 相加</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inThr</td>
      <td>输入</td>
      <td>残差输入张量，shape 为 (gbH, gbW)，与 (inOne + inTwo) 相加</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>LayerNorm 的缩放参数，shape 为 (gbW,)</td>
      <td>float16</td>
      <td>ND</td>
    </tr>    
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>LayerNorm 的平移参数，shape 为 (gbW,)</td>
      <td>float16</td>
      <td>ND</td>
    </tr>    
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>量化缩放因子，shape 为 (gbW,)</td>
      <td>float32</td>
      <td>ND</td>
    </tr>     
    <tr>
      <td>outLynQuant</td>
      <td>输出</td>
      <td>量化后的 int8 结果，shape 为 (gbH, gbW)</td>
      <td>int8_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outAdd</td>
      <td>输出</td>
      <td>相加后的 fp16 中间结果 (inOne + inTwo + inThr)，shape 为 (gbH, gbW)</td>
      <td>float16</td>
      <td>ND</td>
    </tr>    
    <tr>
      <td>gbH</td>
      <td>属性</td>
      <td>输入张量的行数（token 数量）</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gbW</td>
      <td>属性</td>
      <td>输入张量的列数（隐藏维度）</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>LayerNorm 防除零极小值</td>
      <td>float32</td>
      <td>-</td>
    </tr>   
    <tr>
      <td>constrait</td>
      <td>属性</td>
      <td>是否在量化前对中间浮点结果进行 [-128, 128] 的截断约束</td>
      <td>bool</td>
      <td>-</td>
    </tr>      
  </tbody></table>

## 约束说明

## 价值/作用

在 LLM 推理中，Attention 输出与残差分支的相加、LayerNorm 以及 KV Cache 的量化写入是极为频繁的操作。如果分步执行，相加结果需要写回 GM 再由 LN 算子读入，LN 结果再写回 GM 供 Quant 算子读入。本算子将这三步合一：

避免了 outAdd 和 LN 中间结果的 GM 写回与重新搬运，极大节省显存带宽。
将 LN 计算提升到 float32 精度保证数值稳定性，但中间变量均在 UB 中流转，无需额外的 GM 显存分配。
一次性输出 fp16 残差结果和 int8 量化结果，完美契合推理引擎中“残差继续前向传播 + KV Cache 量化存储”的双重需求。

## 设计方案

### Tiling 策略

- **分核策略**：
  按行（gbH）维度进行分核。每个 Core 处理 ⌈gbH / blockNum⌉ 行数据。尾部行通过 i * blockNum_ + blockIdx_ < gbH 判断跳过。

- **分块策略**：
  当前列方向不进行分块（整行处理），bkW_ = gbW，按 64 元素向上对齐得到 bkAlignW_。所有输入输出按整行搬入搬出。

### Kernel 侧设计

进行 **Init** 和 **Process** 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、数据搬出（CopyOut）三个阶段。采用单缓冲（BUFFER_NUM = 1）机制，1D 参数（inTwo, gamma, beta, scale）在 Init 阶段搬入 UB 并常驻，所有行计算复用，减少 GM 读取次数。

- **初始化（Init）**
  - 计算分核与对齐参数。
  - 搬入 1D 参数（inTwo, gamma, beta, scale）到 UB 并出队保存引用，供后续行循环复用。
  - 对 inTwo 的尾部对齐补零。

- **计算流程（Process）**
  - FOR i = 0 TO bkLoop_（行方向循环）：
      - CopyIn：从 GM 搬入当前行的 inOne 和 inThr 到 UB，尾部补零。
      - Compute：
      - 计算 LayerNormNormAddAdd
      - 计算量化结果
      - 将 out_add (half) 和 out_lyn_quant (int8_t) 搬回 GM 对应行

- **数据搬入（CopyIn）**
  - 从 GM 搬入当前行的 inOne 和 inThr 到 UB，尾部补零。

- **数据搬出（CopyOut）**
  - 将 out_add (half) 和 out_lyn_quant (int8_t) 搬回 GM 对应行。