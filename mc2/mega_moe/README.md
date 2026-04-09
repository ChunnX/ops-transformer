# MoeDispatchFFNCombine

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

算子功能：实现Dispatch + GroupMatmul1 + SwiGLUQuant + GroupMatmul2 + Combine的端到端融合计算。

该算子将MoE FFN（Mixture of Experts Feed-Forward Network）的完整计算流程融合为单个算子，包括：

1. **PermuteQuant**：对输入token执行本地重排操作，进行量化预处理，统计各专家的token分配情况，为后续专家并行计算准备数据布局
2. **Dispatch**：通过AllToAllV通信将token及对应scale值分发至目标专家卡，实现专家并行计算的数据分布
3. **GroupMatmul1**：执行分组矩阵乘法计算（Gate/Up投影），采用硬件优化的分组矩阵乘实现
4. **SwiGLUQuant**：对GroupMatmul1的输出执行SwiGLU激活计算，实现激活结果的量化压缩
5. **GroupMatmul2**：对量化后的激活结果执行Down投影，完成专家特征变换的最后阶段
6. **Combine**：对GroupMatmul2输出进行量化处理，通过AllToAllV通信将结果发送回原卡，保持输出数据与输入数据的对应关系
7. **Unpermute**：对本地token执行加权求和，恢复原始数据排列，将最终结果写入输出地址

计算流程：

$$
x_{permute} = PermuteQuant(x) \\
x_{dispatched} = AllToAllV(x_{permute}, expert\_ids) \\
h_1 = GroupMatmul1(x_{dispatched}, weight_1) \\
h_{act} = SwiGLUQuant(h_1) \\
h_2 = GroupMatmul2(h_{act}, weight_2) \\
y_{combined} = AllToAllV(h_2, expert\_ids)^{-1} \\
y = Unpermute(y_{combined})
$$

其中，$weight_1$为Gate/Up投影权重，$weight_2$为Down投影权重。



## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
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
   <td>context</td>
   <td>输入</td>
   <td>本卡通信域信息数据。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>输入</td>
   <td>本卡发送的token数据。</td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_ids</td>
   <td>输入</td>
   <td>每个token的topK个专家索引。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_scales</td>
   <td>输入</td>
   <td>每个token的topK个专家权重。</td>
   <td>FP32、BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight1</td>
   <td>输入</td>
   <td>GroupMatmul1计算的右矩阵，即专家的Gate/Up投影权重。</td>
   <td>FP8_E5M2、FP8_E4M3</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight2</td>
   <td>输入</td>
   <td>GroupMatmul2计算的右矩阵，即专家的Down投影权重。</td>
   <td>FP8_E5M2、FP8_E4M3</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>scales</td>
   <td>可选输入</td>
   <td>预留参数，暂不支持。</td>
   <td>FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_active_mask</td>
   <td>可选输入</td>
   <td>预留参数，暂不支持。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight_scales1</td>
   <td>可选输入</td>
   <td>量化场景需要，GroupMatmul1右矩阵反量化参数。</td>
   <td>FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight_scales2</td>
   <td>可选输入</td>
   <td>量化场景需要，GroupMatmul2右矩阵反量化参数。</td>
   <td>FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y</td>
   <td>输出</td>
   <td>计算输出结果，与输入x shape相同。</td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_world_size</td>
   <td>属性</td>
   <td>EP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_rank_id</td>
   <td>属性</td>
   <td>EP域本卡Id，取值范围[0, ep_world_size)。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moe_expert_num</td>
   <td>属性</td>
   <td>MoE专家数量。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ccl_buffer_size</td>
   <td>属性</td>
   <td>当前通信域Buffer大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>max_recv_token_num</td>
   <td>可选属性</td>
   <td><li>alltoall接收的最多的token数，0表示使用默认值，内部自行按最大值计算。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_num</td>
   <td>可选属性</td>
   <td><li>共享专家数量。</li><li>默认值为1。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>dispatch_quant_mode</td>
   <td>可选属性</td>
   <td><li>Dispatch通信时量化模式，0表示非量化。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>dispatch_quant_out_type</td>
   <td>可选属性</td>
   <td><li>Dispatch量化后输出的数据类型。</li><li>默认值为0（DT_UNDEFINED）。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>combine_quant_mode</td>
   <td>可选属性</td>
   <td><li>预留参数，暂不支持。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>comm_alg</td>
   <td>可选属性</td>
   <td><li>预留参数，暂不支持。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>global_bs</td>
   <td>可选属性</td>
   <td><li>EP域全局的global_bs大小，若为0，则认为global_bs = BS * ep_world_size。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## 约束说明

- 调用算子过程中使用的`groupEp`、`epWorldSize`、`sharedExpertNum`、`globalBs`参数，`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致。

- 通信域使用约束：
    - 仅支持EP域，无TP域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性。
    - 所有卡的`ep_world_size`、`moe_expert_num`、`ccl_buffer_size`、`shared_expert_num`、`global_bs`、`comm_alg`参数取值需保持一致。

- <term>Ascend 950PR/Ascend 950DT</term>：
    - 参数约束：
        - `epWorldSize`：取值范围[2, 768]。
        - `moeExpertNum`：取值范围(0, 1024]。
        - `dispatchquantMode`支持0（非量化）、1（mx动态量化）。
    - 参数说明里shape格式说明：
        - `K`：表示选取topK个专家，取值范围为0 < `K` ≤ 16。
        - `H`：表示hidden size隐藏层大小，取值范围[1024, 8192]。
        - `Bs`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足>= 2 * (`localExpertNum` * `maxBs` * `epWorldSize` * `Align512`(`Align32`(2 * `H`) + 64) + (`K` + `sharedExpertNum`) * `maxBs` * `Align512`(2 * `H`))，`localExpertNum`需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) * 512，Align32(x) = ((x + 32 - 1) / 32) * 32。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_dispatch_ffn_combine.cpp](./examples/test_aclnn_moe_dispatch_ffn_combine.cpp) | 通过[aclnnMoeDispatchFFNCombine](./docs/aclnnMoeDispatchFFNCombine.md)接口方式调用moe_dispatch_ffn_combine算子。 |
