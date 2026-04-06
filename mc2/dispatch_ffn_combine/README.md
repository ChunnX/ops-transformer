# DispatchFFNCombine

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

算子功能：实现Dispatch + GroupMatmul1 + SwiGLUQuant + GroupMatmul2 + Combine的端到端融合计算。

该算子将MoE FFN（Mixture of Experts Feed-Forward Network）的完整计算流程融合为单个算子，包括：

1. **Dispatch**：将token通过AllToAllV通信分发到对应的专家卡
2. **GroupMatmul1**：对分发后的token执行第一层分组矩阵乘（Gate/Up投影）
3. **SwiGLUQuant**：对GroupMatmul1的输出执行SwiGLU激活函数及量化
4. **GroupMatmul2**：对激活后的结果执行第二层分组矩阵乘（Down投影）
5. **Combine**：将各专家的计算结果通过AllToAllV通信聚合回原始卡

计算流程：

$$
x_{dispatched} = AllToAllV(x, expert\_ids) \\
h_1 = GroupMatmul1(x_{dispatched}, weight_1) \\
h_{act} = SwiGLUQuant(h_1, scales) \\
h_2 = GroupMatmul2(h_{act}, weight_2) \\
y = AllToAllV(h_2, expert\_ids)^{-1}
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
   <td>输入的hidden_states，二维tensor。一个die输入bs个tokens，每个token长度为h。</td>
   <td>BF16、FP16、FP8_E5M2、FP8_E4M3、HIF8、FP4_E2M1、FP4_E1M2</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_ids</td>
   <td>输入</td>
   <td>每个token的topK个专家索引，值为Moe的rankId，即[shared_expert_rank_num, world_size]。</td>
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
   <td>动态输入</td>
   <td>GroupMatmul1计算的右矩阵，即moe专家和共享专家的Gate/Up投影权重，共享专家权重在前。</td>
   <td>BF16、FP16、FP8_E5M2、FP8_E4M3</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight2</td>
   <td>动态输入</td>
   <td>GroupMatmul2计算的右矩阵，即moe专家和共享专家的Down投影权重，共享专家权重在前。</td>
   <td>BF16、FP16、FP8_E5M2、FP8_E4M3</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>scales</td>
   <td>可选输入</td>
   <td>可选smooth参数，非量化不传；动态量化可不传入，传入场景的数据排布顺序为共享专家共有的一行scale+每个moe专家对应scale。</td>
   <td>FP32、FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_active_mask</td>
   <td>可选输入</td>
   <td>表示token是否参与通信，false的不需要发送。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight_scales1</td>
   <td>动态输入</td>
   <td>可选输入，量化场景需要，GroupMatmul1右矩阵反量化参数，包含moe专家和共享专家的scale。</td>
   <td>FP32、FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight_scales2</td>
   <td>动态输入</td>
   <td>可选输入，量化场景需要，GroupMatmul2右矩阵反量化参数，包含moe专家和共享专家的scale。</td>
   <td>FP32、FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y</td>
   <td>输出</td>
   <td>计算输出结果，与输入x shape相同。</td>
   <td>BF16、FP16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_world_size</td>
   <td>必选属性</td>
   <td>EP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_rank_id</td>
   <td>必选属性</td>
   <td>EP域本卡Id，取值范围[0, ep_world_size)。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moe_expert_num</td>
   <td>必选属性</td>
   <td>MoE专家数量。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ccl_buffer_size</td>
   <td>必选属性</td>
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
   <td><li>Combine通信时量化模式，0表示非量化。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>comm_alg</td>
   <td>可选属性</td>
   <td><li>通信亲和内存布局算法。</li><li>默认值为""。</li></td>
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

- 该算子为端到端融合算子，将Dispatch、GroupMatmul1、SwiGLUQuant、GroupMatmul2、Combine五个步骤融合为一次调用，减少通信次数和中间数据搬运。

- 输入shape约束：
    - `x`：`(bs, h)`，其中`h`为hidden size。
    - `expert_ids`：`(bs, k)`，其中`k`为topK专家数。
    - `expert_scales`：`(bs, k)`。
    - `weight1`：`(expertPerRank, H, N)`，Gate/Up投影权重。
    - `weight2`：`(expertPerRank, N/2, N)`，Down投影权重。
    - `y`：`(bs, h)`，与输入`x`相同。

- 通信域使用约束：
    - 所有卡的`ep_world_size`、`moe_expert_num`、`ccl_buffer_size`、`shared_expert_num`、`global_bs`、`comm_alg`参数取值需保持一致。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 该场景下单卡包含双DIE（简称为"晶粒"或"裸片"），因此参数说明里的"本卡"均表示单DIE。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | aclnnDispatchFFNCombineGetWorkspaceSize / aclnnDispatchFFNCombine | 通过aclnn接口方式调用DispatchFFNCombine算子。 |
