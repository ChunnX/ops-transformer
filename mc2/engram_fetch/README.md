# EngramFetch

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT</term>                                    |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

算子功能：根据`indices`指定的token索引，从elastic buffer中拷贝指定token数据到`fetched`输出。其中本卡数据通过MTE拷贝完成，远端卡数据通过URMA通信异步读取。该算子需与`EngramFetchWait`配套使用，`EngramFetch`负责发起数据读取，`EngramFetchWait`负责等待远端URMA读取完成。

$$
fetched[i] = commBuffer[ownerRank(indices[i])][localEntryIdx(indices[i])]
$$

其中：
- $indices[i]$为第i个token的全局索引，索引空间覆盖所有卡上的token，按统一的连续编号编排。
- $ownerRank = indices[i] / num\_entries\_per\_rank$，表示该token所在的卡号，通过整除确定目标卡。
- $localEntryIdx = indices[i] \mod num\_entries\_per\_rank$，表示该token在目标卡elastic buffer中的卡内索引，通过取余计算。
- $commBuffer$表示各卡的elastic buffer数据缓冲区，按卡号索引。

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
  </tr></thead>
 <tbody>
  <tr>
   <td>commContext</td>
   <td>输入</td>
   <td>通信上下文，包含rankId、rankSize、通信缓冲区地址及HCOMM句柄，由通信域初始化时生成。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>indices</td>
   <td>输入</td>
   <td>需要获取的token的全局索引，1D Tensor，shape为(numTokens,)。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>hidden_size</td>
   <td>属性</td>
   <td>每个token的隐藏层大小，需为128的整数倍。</td>
   <td>INT</td>
   <td>-</td>
  </tr>
  <tr>
   <td>num_entries_per_rank</td>
   <td>属性</td>
   <td>每个elastic buffer上的token条目数，用于计算token所属rank及本地偏移。</td>
   <td>INT</td>
   <td>-</td>
  </tr>
  <tr>
   <td>fetched</td>
   <td>输出</td>
   <td>获取到的token数据，2D Tensor，shape为(numTokens, hidden_size)。</td>
   <td>BFLOAT16、FLOAT16、FLOAT</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## 约束说明

- 该算子必须与`EngramFetchWait`配套使用，`EngramFetch`发起读取后必须调用`EngramFetchWait`等待完成。
- `commContext`必须为elastic buffer创建的通信上下文，数据类型为INT32。
- `indices`必须为1D Tensor，数据类型为INT32，元素值需在[0, num_entries_per_rank * rankSize)范围内。
- `fetched`必须为2D Tensor，shape为(numTokens, hidden_size)，其中dim0需与`indices`的dim0一致，dim1需与`hidden_size`属性一致。
- `hidden_size`需大于0且为128的整数倍。
- `num_entries_per_rank`取值范围为[0, INT32_MAX / rankSize]。
- 通信方式约束：<term>Ascend 950DT</term>：仅支持URMA通信。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_engram_fetch.cpp](./tests/test_aclnn_engram_fetch.cpp) | 通过`aclnnEngramFetch`接口方式调用engram_fetch算子。 |
