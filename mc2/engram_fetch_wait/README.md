# EngramFetchWait

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

算子功能：等待`EngramFetch`通过URMA通信发起的远端数据读取操作完成。`fetched`输入仅用于构建Tensor依赖，接口内不对`fetched`输入做任何操作，输出`fetched`与输入`fetched`的shape和数据类型一致。

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
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>commContext</td>
   <td>输入</td>
   <td>通信上下文，包含rankId、rankSize、通信缓冲区地址及HCOMM句柄，与`EngramFetch`使用相同的commContext。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>fetched</td>
   <td>输入</td>
   <td>无业务语义，仅用于构建Tensor依赖，接口内不做任何操作。</td>
   <td>BFLOAT16、FLOAT16、FLOAT</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>fetched</td>
   <td>输出</td>
   <td>输出Tensor，shape和数据类型与输入`fetched`一致。</td>
   <td>BFLOAT16、FLOAT16、FLOAT</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## 约束说明

- 该算子必须与`EngramFetch`配套使用，且`EngramFetch`和`EngramFetchWait`需使用相同的`commContext`。
- `commContext`必须为elastic buffer创建的通信上下文，数据类型为INT32。
- 输出`fetched`的shape和数据类型需与输入`fetched`一致。
- 通信方式约束：<term>Ascend 950DT</term>：仅支持URMA通信。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | - | 通过`aclnnEngramFetchWait`接口方式调用engram_fetch_wait算子。 |
