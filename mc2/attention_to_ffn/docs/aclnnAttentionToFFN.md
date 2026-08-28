# aclnnAttentionToFFN

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

将Attention节点上token数据发往FFN节点。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAttentionToFFNGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnAttentionToFFN”接口执行计算。

```c++
aclnnStatus aclnnAttentionToFFNGetWorkspaceSize(
    const aclTensor    *x,
    const aclTensor    *sessionId,
    const aclTensor    *microBatchId,
    const aclTensor    *layerId,
    const aclTensor    *expertIds,
    const aclTensor    *expertRankTable,
    const aclTensor    *scalesOptional,
    const aclTensor    *activeMaskOptional,
    const char         *group,
    int64_t             worldSize,
    const aclIntArray  *ffnTokenInfoTableShape,
    const aclIntArray  *ffnTokenDataShape,
    const aclIntArray  *attnTokenInfoTableShape,
    int64_t             moeExpertNum,
    int64_t             quantMode,
    int64_t             syncFlag,
    int64_t             ffnStartRankId,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnAttentionToFFN(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnAttentionToFFNGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 300px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
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
        <td>x</td>
        <td>输入</td>
        <td>本卡发送的token数据。</td>
        <td>shape为(X, Bs, H)。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sessionId</td>
        <td>输入</td>
        <td>表示当前Attention Worker节点的Id。</td>
        <td>shape为(X, )，取值范围为[0, attentionWorkerNum)。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>microBatchId</td>
        <td>输入</td>
        <td>表示microBatch的Id。</td>
        <td>shape为(X, )。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>layerId</td>
        <td>输入</td>
        <td>表示当前模型层数的Id。</td>
        <td>shape为(X, )。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expertIds</td>
        <td>输入</td>
        <td>表示每个micro batch组中每个token的topK个专家索引。</td>
        <td>shape为(X, Bs, K)，取值区间为[0, moeExpertNum)。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expertRankTable</td>
        <td>输入</td>
        <td>每个micro batch组中专家Id到FFN卡专家部署的映射表（外部需保证值正确）。</td>
        <td>shape为(L, moeExpertNum + sharedExpertNum, M)。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>输入</td>
        <td>每个专家的量化平滑参数。</td>
        <td>非量化场景下必须传空指针，动态量化可传有效数据或空指针，shape为(L, moeExpertNum + sharedExpertNum, H)。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>activeMaskOptional</td>
        <td>输入</td>
        <td>表示token是否参与通信，可传有效数据或空指针。</td>
        <td>传空指针时，默认所有token参与通信；传值时，shape为(X, Bs)，true表示该token参与通信，且true需排在false前。</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>group</td>
        <td>输入</td>
        <td>通信域名称（专家并行）。</td>
        <td>字符串长度范围为[1, 128)。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>worldSize</td>
        <td>输入</td>
        <td>表示通信域大小。</td>
        <td>取值区间[2, 768]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnTokenInfoTableShape</td>
        <td>输入</td>
        <td>表示FFN节点上token信息表格shape大小的列表。</td>
        <td>长度为3，包括Attention节点的数量、microBatchSize的大小以及每个token对应的相关发送状态信息shape的大小。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnTokenDataShape</td>
        <td>输入</td>
        <td>表示FFN节点上token数据表格shape大小的列表。</td>
        <td>长度为5，包括Attention节点的数量、microBatchSize的大小、batchSize大小、每个token需发送的专家数量（包括共享专家）、单个token的长度。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attnTokenInfoTableShape</td>
        <td>输入</td>
        <td>表示Attention节点上token信息表格shape大小的列表。</td>
        <td>长度为3，包括microBatchSize的大小、batchSize大小、每个token需发送的专家数量（包括共享专家）。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>moeExpertNum</td>
        <td>输入</td>
        <td>MoE专家数量。</td>
        <td>取值范围(0, 1024]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>输入</td>
        <td>表示量化模式。</td>
        <td>仅支持0（非量化）、2（动态量化）。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>syncFlag</td>
        <td>输入</td>
        <td>表示FFN节点同步模式。</td>
        <td>仅支持0（同步）、1（异步）。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnStartRankId</td>
        <td>输入</td>
        <td>表示FFN节点的起始ID。</td>
        <td>取值范围为[0, worldSize)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回Device侧需申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回包含算子计算流程的op执行器。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 280px">
    <col style="width: 100px">
    <col style="width: 900px">
      </colgroup><thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
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

## aclnnAttentionToFFN

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 900px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnAttentionToFFNGetWorkspaceSize</code>获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
    </tr>
    </tbody></table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- **确定性约束**：
  - `aclnnAttentionToFFN`默认确定性实现。

- **参数一致性约束**：
  - 调用接口过程中使用的`group`、`worldSize`、`moeExpertNum`、`ffnTokenInfoTableShape`、`ffnTokenDataShape`、`ffnStartRankId`参数及`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致，且和分离场景系列算子对应参数也保持一致。

- **产品特定约束**：

  <!-- npu="A3" id7 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明中的“本卡”均表示单DIE。

  <!-- end id7 -->

- **Shape变量约束**：

  | 变量         | 定义与取值范围                                                                 |
  | :----------- | :----------------------------------------------------------------------------- |
  | X            | 表示micro batch sequence size（token组数），当前版本只支持<code>X = 1</code>。 |
  | H（hidden size） | 表示hidden size隐藏层大小，取值范围为<code>[1024, 8192]</code>。 |
  | Bs           | 表示batch sequence size（本卡最终输出的token数量），取值范围为<code>0 < Bs ≤ 512</code>。 |
  | K            | 表示选取topK个专家，取值范围为<code>0 < K ≤ 16</code>，且<code>0 < K ≤ moeExpertNum</code>。 |
  | L            | 表示模型层数，当前版本只支持<code>L = 1</code>。 |
  | M            | 表示expertRankTable最后一维的长度，具体体现为部署在FFN节点上数量最多的专家部署信息列表的长度，取值范围为<code>1 < M ≤ FFNWorkerNum * 2 + 1</code>。 |
  | moeExpertNum | 表示MoE专家数量，取值范围 <code>(0, 1024]</code>。                   |
  | sharedExpertNum | 表示共享专家数量（一个共享专家可以复制部署到多个ffnRank卡上），取值范围 <code>[0, 4]</code>。                   |
  | moeExpertRankNum | 表示部署MoE专家的FFN节点数量，取值范围 <code>0 < moeExpertRankNum < FFNWorkerNum</code>。                   |
  | sharedExpertRankNum | 表示部署共享专家的FFN节点数量，取值范围 <code>0 < sharedExpertRankNum ≤ FFNWorkerNum</code>。                 |
  | FFNWorkerNum | 表示FFN节点的数量，取值范围为 <code>0 < FFNWorkerNum < worldSize</code>，且满足<code>FFNWorkerNum = moeExpertRankNum + sharedExpertRankNum</code>。                   |
  | AttentionWorkerNum | 表示Attention节点的数量，取值范围 <code>0 < AttentionWorkerNum < worldSize</code>。

- **环境变量约束**：
  - **HCCL_BUFFSIZE**：调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。

- **通信域使用约束**：
  - AttentionToFFN算子的通信域中不允许有其他算子。

## 调用示例

文件准备：

1.新建AttentionToFFNDemo目录，按照下方指导在AttentionToFFNDemo下新建aclnnAttentionToFFNDemo.cpp，AttentionToFFN.sh文件并参考如下代码修改。

2.安装cann包，并根据下方指导编译运行AttentionToFFNDemo。

AttentionToFFN.sh编译脚本

```bash
#!/bin/bash
cann_path="/path/to/cann_env" # 更改cann包环境的路径
g++ "aclnnAttentionToFFNDemo.cpp" -o AttentionToFFNDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                    -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi_math -lop_common -lpthread -lhccl
```

编译与运行：

```bash
# source cann环境
source /path/to/cann_env/latest/bin/setenv.bash

# 编译aclnnAttentionToFFNDemo.cpp
bash AttentionToFFN.sh

./AttentionToFFNDemo
```

示例代码如下，仅供参考

```c++

```
