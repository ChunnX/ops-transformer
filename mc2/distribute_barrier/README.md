# DistributeBarrier

> 注意：
> 本文档仅仅是算子功能的简介，不支持用户直接调用，因为当前不支持kernel直调，等后续支持再完善文档!!!!!!

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明

算子功能：完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。



## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 <col style="width: 800px">
 <col style="width: 200px">
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
   <td>xRef</td>
   <td>输入</td>
   <td>Device侧的aclTensor，无业务语义，仅用于输入Tensor依赖，接口内不做任何操作。</td>
   <td>BFLOAT16, FLOAT16、FLOAT32、BOOL、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group</td>
   <td>输入</td>
   <td>通信域名称，进行所有卡同步的通信域。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>worldSize</td>
   <td>输入</td>
   <td>通信域大小。</td>
   <td>UINT64</td>
   <td>ND</td>
  </tr>
 </tbody></table>




## 约束说明

- 通信域使用约束：
    - 一个模型中的aclnnDistributeBarrier需要使用单独通信域，该通信域中不允许有其他算子。

- 使用场景说明：
    - 在需要进行全卡同步的网络模型中调用该算子，可以屏蔽快慢卡引入的性能波动问题，协助分析性能。
    - 可以连续调用，入图时，需将上个算子的输入、下个算子的输出作为入参传入接口。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_distribute_barrier.cpp](./example/test_distribute_barrier.cpp) | 通过[aclnnDistributeBarrier](aclnnDistributeBarrier.md)接口方式调用distribute_barrier算子。 |