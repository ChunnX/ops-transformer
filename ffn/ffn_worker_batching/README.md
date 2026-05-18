# FfnWorkerBatching

## 产品支持情况

| 产品 | 是否支持 |
| :---------------------------- | :-----------: |
|<term>Ascend 950PR/Ascend 950DT</term>| × |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>| √ |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>| √ |
|<term>Atlas 200I/500 A2 推理产品</term>| × |
|<term>Atlas 推理系列产品</term>| × |
|<term>Atlas 训练系列产品</term>| × |

## 功能说明

- 算子功能：`FFNWorkerBatching` 在 Attention 与 FFN 分离部署场景下，完成 FFN worker 上的 token 重排操作。Attention 将 token 按专家路由发送到对应 FFN worker 的预分配数据区，FFNWorkerBatching 从该数据区中扫描调度信息，按专家维度聚合并重排 token，产出各专家对应的连续 token 数据块。

- 计算步骤：

    1. 对 `expert_ids_in` 中所有 token 的专家 ID 进行排序（被 mask 的 token 初始化为大值），生成 gather 索引。
    2. 多核并行按 gather 索引从 `token_data` 中提取 token 的 hidden states 和 dynamic scale，同时查表得到对应的 `session_id`、`micro_batch_id`、`token_id`。
    3. 单核扫描排序后的专家 ID 序列，查找跳变点，生成 `group_list`（每个专家处理的 token 起止偏移）。

    其中 $Y = A \times BS \times (K+1)$，$A$ 为 Attention worker 数量，$BS$ 为 micro batch size，$K+1$ 为 topK 加共享专家数。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
  <col style="width: 100px">
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
      <td>schedule_context</td>
      <td>输入</td>
      <td>调度上下文数据结构，内含 CommonArea、ControlArea、AttentionArea、FfnArea。算子从 FfnArea 中读取 token_info_buf 和 token_data_buf 获取待重排的 token 数据与描述信息，并获取 layer_id、session_id、micro_batch_id、expert_ids 等路由信息。结构体总大小 1024 字节。</td>
      <td>INT8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expert_num</td>
      <td>属性</td>
      <td>本卡专家总数，等于每层本卡专家数 × layer_num。用于推导 group_list 输出大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_out_shape</td>
      <td>属性</td>
      <td>输出 shape 上限，格式为 {A, BS, topK+1, H}。用于推导 y 输出的 shape 上限 Y = A × BS × (topK+1)，以及 H 值。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>token_dtype</td>
      <td>属性</td>
      <td>输入 token 的数据类型。0 表示 FP16；1 表示 BF16；2 表示 INT8 动态量化（INT8 数据与 FP32 dynamic scale 连续排布）。默认值为 0。取值为 2 时需输出 dynamic_scale。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>need_schedule</td>
      <td>属性</td>
      <td>调度模式。0 表示仅做 batching 不扫描数据；1 表示先扫描数据再做 batching。默认值为 0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layer_num</td>
      <td>属性</td>
      <td>层数，每层专家独立索引。默认值为 0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>重排后的 token hidden states，按专家 ID 排序后连续存放。shape 为 [Y, H]。</td>
      <td>FP16、BF16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>输出</td>
      <td>每个专家处理的 token 范围，shape 为 [expert_num, 2]。每行格式为 [expert_id, expert_token_num]，未使用的专家填 [0, 0]。示例：[[1, 20], [10, 40], [22, 15], ...]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>session_ids</td>
      <td>输出</td>
      <td>每个输出 token 对应的 Attention session ID，shape 为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>micro_batch_ids</td>
      <td>输出</td>
      <td>每个输出 token 对应的 micro batch ID，shape 为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>token_ids</td>
      <td>输出</td>
      <td>每个输出 token 在原始输入中的位置索引，shape 为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_offsets</td>
      <td>输出</td>
      <td>每个输出 token 在其所属专家分组内的偏移，shape 为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dynamic_scale</td>
      <td>输出</td>
      <td>动态量化的 scale 值，仅在 token_dtype=2 时有效。shape 为 [Y]。token_dtype 为 0 或 1 时为空 tensor。</td>
      <td>FP32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_token_num</td>
      <td>输出</td>
      <td>所有专家有效 token 数之和，标量输出。shape 为 []。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 该接口支持图模式（GEIR）和单算子模式（aclnn）。
- 参数 A（Attention worker 数量）支持 ≤ 1024。
- 参数 M（micro batch 数量）支持 ≤ 64。
- 参数 K（topK 数）支持 ≤ 64。
- 参数 BS（micro batch size）和 Y 支持泛化，无硬上限（受内存限制）。
- 参数 H（hidden size）支持泛化。
- token_dtype 为 2 时，输入 int8 数据与 fp32 scale 连续排布。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_ffn_worker_batching.cpp](examples/test_geir_ffn_worker_batching.cpp) | 通过[算子IR](op_graph/ffn_worker_batching_proto.h)构图方式调用FfnWorkerBatching算子。 |

## 调度上下文数据结构

```cpp
struct ScheduleContext {
  struct CommonArea {
    uint32_t session_num;           // Attention 节点数
    uint32_t micro_batch_num;       // micro batch 拆分数量
    uint32_t micro_batch_size;      // batch_size / micro_batch_num
    uint32_t selected_expert_num;   // topK 个数 + 1（含共享专家）
    uint32_t expert_num;            // 每层专家个数
    uint32_t attn_to_ffn_token_size;// 每个 token 在 FFN window 数据区占用大小，对齐到 512
    uint32_t ffn_to_attn_token_size;// 每个 token 在 Attention window 数据区占用大小，对齐到 512
    int32_t  schedule_mode;         // 0:只调度FFN, 1:只调度Attention, 2:同时调度
    int8_t   reserve0[96];          // padding to 128 bytes
  };
  struct ControlArea {
    int32_t run_flag;               // 控制循环退出
    int8_t  reserve1[124];
  };
  struct AttentionArea {
    uint64_t token_info_buf;        // [M, DataDesc]，DataDesc 含 flags[batch_size][topK+1]
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;        // [M, BS, K+1, HS]
    uint64_t token_data_buf_size;
    uint32_t micro_batch_id;        // 轮询用，初始值 micro_batch_num-1
    int8_t   reserve5[92];
  };
  struct FfnArea {
    // FFN 输入区
    uint64_t token_info_buf;        // [A, M, F]，DataDesc 含 flag/layer_id/expert_ids
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;        // [A, M, BS, K+1, HS]
    uint64_t token_data_buf_size;
    uint64_t polling_index;
    int8_t   reserve3[88];
    // FFN 输出区
    uint64_t layer_ids_buf;         // [session_num]
    uint64_t layer_ids_buf_size;
    uint64_t session_ids_buf;       // [session_num]
    uint64_t session_ids_buf_size;
    uint64_t micro_batch_ids_buf;   // [session_num]
    uint64_t micro_batch_ids_buf_size;
    uint64_t expert_ids_buf;        // [session_num, BS, K+1]
    uint64_t expert_ids_buf_size;
    uint32_t out_num;               // 实际收齐的 session 个数
    int8_t   reserve4[60];
  };
  CommonArea    common;
  ControlArea   control;
  AttentionArea attention;
  FfnArea       ffn;
  int8_t        reserve6[384];      // padding to 1024 bytes
}; // 总大小 1024 字节
```
