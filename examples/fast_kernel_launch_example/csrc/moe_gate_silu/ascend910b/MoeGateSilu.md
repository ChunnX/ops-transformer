# MoeGateSilu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品                                         | 是       |

## 功能说明

- 在 MoE (Mixture of Experts) 架构的 Expert MLP 阶段，实现 SwiGLU (Gated SiLU) 激活函数的深度融合计算，并原生支持动态 Shape (Dynamic Shape) 下的真实行数读取。

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
      <td>x</td>
      <td>输入</td>
      <td>Expert MLP 的 Gate 和 Up 投影融合输入，shape 为 (ROWS, Hidden)</td>
      <td>BFLOAT16 </td>
      <td>ND</td>
    </tr>
	<tr>
      <td>total_rows</td>
      <td>输入</td>
      <td>动态 Shape 场景下的真实有效行数（Token 数），Shape 为 (local_expert_num)。</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>z</td>
      <td>输出</td>
      <td>SwiGLU 激活后的输出，shape 为 (ROWS, Hidden / 2)</td>
      <td>BFLOAT16 </td>
      <td>ND</td>
    </tr>
	<tr>
      <td>NUM_ROWS</td>
      <td>输出</td>
      <td>静态总行数ROWS，若 use_total_rows 为 true 则被覆盖</td>
      <td>int32_t</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>per_count</td>
      <td>输入</td>
      <td>输入张量的隐藏层维度Hidden</td>
      <td>int32_t</td>
      <td>-</td>
    </tr>
	<tr>
      <td>per_total_count</td>
      <td>输入</td>
      <td>每个ep上专家数, local_expert_num</td>
      <td>int32_t</td>
      <td>-</td>
  </tr>
    <tr>
      <td>block_cnt</td>
      <td>输入</td>
      <td>AI CORE的数量，比如：Ascend910B是40。</td>
      <td>int32_t</td>
      <td>-</td>
    </tr>
	<tr>
      <td>stream</td>
      <td>输入</td>
      <td>Device端的stream</td>
      <td>AclrtStream</td>
      <td>-</td>
    </tr>
	<tr>
      <td>use_total_rows</td>
      <td>输入</td>
      <td>是否启用动态行数读取</td>
      <td>bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明
Hidden 必须小于等于 8192
## 价值/作用
在 LLaMA、Qwen、DeepSeek 等主流大语言模型中，Expert MLP 普遍采用 SwiGLU 作为前馈网络（FFN）的激活函数。在模型实现中，通常会将 Gate 投影和 Up 投影合并为一次矩阵乘法，输出维度翻倍。
本算子将张量拆分、SiLU 激活、逐元素乘法三个操作深度融合在一个 AI Core 内部执行，并原生支持动态 Batch Size。这不仅消除了中间结果在 UB 和 GM 之间的频繁读写，降低了访存带宽压力，还完美适配了推理引擎（如 vLLM/MindIE）中因 PagedAttention 或动态 Padding 导致的真实 Token 数动态变化问题，大幅提升了 MoE Expert FFN 阶段的计算吞吐量

## 设计方案

### Tilling 策略

- 分核策略：
按行（ROWS）维度进行分核。每个核心只分配单 rows 数据。
采用尾部处理机制（Tail Processing）：对于无法整除的余数行（tail_count），分配给前 tail_count 个 Core 处理一行。

### Kernel 侧设计

进行 Init 和 Process 两个阶段。采用双缓冲（Double Buffering, BUFFER_NUM = 2） 机制实现数据搬运（MTE）与计算（Vector）的流水线重叠。

- 初始化（Init & InitTOTAL）
    - 动态行数解析：若 use_total_rows 为 true，优先通过 Process_Total 从 GM 读取 total_rows 标量，更新真实的 real_rows。
    - 维度与对齐计算：计算拆分后的半维度 per_dim_ = per_count / 2，并计算满足 32B 对齐的 align_cnt。
    - UB 容量评估：根据 UB 总大小（188KB）和单次计算所需的 Buffer 数量（输入左、输入右、输出、FP32工作区左、FP32工作区右，共 5~6 份），动态计算 rows_per_core_per_loop。
    - 资源分配：建立 GM Tensor 映射，初始化 VECIN（inQueue_X, inQueue_Y）、VECOUT（outQueue_Z）队列及 VECCALC 工作缓冲区（work_buf, work_buf1 用于 FP32 中间计算）。
- 计算流程（Compute）
    - 数据出队与精度转换：从队列取出 x_left 和 x_right，Cast 为 FP32 格式。
    - SwiGLU  核心计算。
    - 精度回退与入队
- 流程调度（Process）
    - [可选] 读取 total_rows 获取真实行数
    - 动态计算 Tilling 参数 (loop_count, tail_count)
    - FOR i = 0 TO loop_count_:
      - CopyOut(i)
      - Compute(i)
      - CopyOut(i)
