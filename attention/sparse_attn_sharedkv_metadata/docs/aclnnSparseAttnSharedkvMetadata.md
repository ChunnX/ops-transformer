# aclnnSparseAttnSharedkvMetadata

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- 接口功能：`SparseAttnSharedkvMetadata`算子为`SparseAttnSharedkv`算子的前序算子，负责根据输入的序列长度信息和注意力配置参数，生成负载均衡的分核元数据（metadata）。该元数据包含每个AICore上FlashAttention计算任务的Batch、Head、Query分块和KV分块的索引，以及每个VectorCore上FlashDecode归约任务的索引信息。

- 计算公式：

  该算子为AICPU调度算子，不涉及数值计算。核心流程为：解析各Batch的Q/KV序列长度 → 根据mask模式计算每个S1G块的有效S2范围 → 基于开销模型进行负载均衡分核 → 输出分核元数据。

  输出metadata tensor的shape为(1024,)，数据类型为INT32，内部结构如下：

  - FA Metadata区域（AIC_CORE_NUM × 8个INT32），每个AICore的FA阶段任务信息：

    | 索引 | 含义 |
    | :--- | :--- |
    | 0 | core_enable，该核是否启用 |
    | 1 | bn2_start，BN2起始索引 |
    | 2 | m_start，M（S1G）起始索引 |
    | 3 | s2_start，S2起始索引 |
    | 4 | bn2_end，BN2结束索引 |
    | 5 | m_end，M结束索引 |
    | 6 | s2_end，S2结束索引 |
    | 7 | first_fd_data_workspace_idx，第一份FD归约数据的workspace偏移 |

  - FD Metadata区域（AIV_CORE_NUM × 8个INT32），每个AIVCore的FD归约任务信息：

    | 索引 | 含义 |
    | :--- | :--- |
    | 0 | core_enable，该核是否启用 |
    | 1 | bn2_idx，归约任务的BN2索引 |
    | 2 | m_idx，归约任务的M索引 |
    | 3 | workspace_idx，归约数据在workspace中的存放位置 |
    | 4 | workspace_num，S2核间切分份数 |
    | 5 | m_start，M轴起点 |
    | 6 | m_num，M轴行数 |

- 符号说明

  | 符号                | 含义                                                      |
  | ------------------- | --------------------------------------------------------- |
  | $B$                 | Batch Size                                                |
  | $N1$/$N2$           | Query/KV头数                                              |
  | $D$                 | 每个注意力头的维度                                        |
  | $G$                 | GQA分组比，$G = N1/N2$                                    |
  | $S1$/$S2$           | Query/KV序列长度                                          |
  | S1G                 | S1×G方向的分块索引                                        |
  | mBaseSize           | M轴基本块大小，等于$G$                                    |
  | s2BaseSize          | S2轴基本块大小，固定为512                                  |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnSparseAttnSharedkvMetadataGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnSparseAttnSharedkvMetadata`执行实际计算。

```c++
aclnnStatus aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensOriKvOptional,
    const aclTensor *cuSeqLensCmpKvOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKvOptional,
    int64_t          numHeadsQ,
    int64_t          numHeadsKv,
    int64_t          headDim,
    int64_t          batchSizeOptional,
    int64_t          maxSeqlenQOptional,
    int64_t          maxSeqlenKvOptional,
    int64_t          oriTopKOptional,
    int64_t          cmpTopKOptional,
    int64_t          cmpRatioOptional,
    int64_t          oriMaskModeOptional,
    int64_t          cmpMaskModeOptional,
    int64_t          oriWinLeftOptional,
    int64_t          oriWinRightOptional,
    char            *layoutQOptional,
    char            *layoutKvOptional,
    bool             hasOriKvOptional,
    bool             hasCmpKvOptional,
    const aclTensor *metaData,
    uint64_t       **workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnSparseAttnSharedkvMetadata(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnSparseAttnSharedkvMetadataGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width: 220px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 300px">
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 160px">
    <col style="width: 100px">
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
      <td>cuSeqLensQOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中q的有效token数（前缀和形式）。</td>
      <td>layout_q为TND时必须传入。每个元素表示当前batch与之前所有batch的token数总和。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLensOriKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv的有效token数（前缀和形式）。</td>
      <td>layout_kv为TND时必须传入。当前layout_kv仅支持PA_ND，故设置此参数无效。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLensCmpKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中cmp_kv的有效token数（前缀和形式）。</td>
      <td>layout_kv为TND且存在cmp_kv时必须传入。当前layout_kv仅支持PA_ND，故设置此参数无效。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedQOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中q实际参与运算的token数。</td>
      <td>当前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedKvOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv实际参与运算的token数。</td>
      <td>layout_kv为PA_ND时必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>Query的多头数。</td>
      <td>仅支持64。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv（int64_t）</td>
      <td>输入</td>
      <td>KV的多头数。</td>
      <td>仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim（int64_t）</td>
      <td>输入</td>
      <td>每个注意力头的维度。</td>
      <td>仅支持512。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSizeOptional（int64_t）</td>
      <td>输入</td>
      <td>输入样本批量大小。</td>
      <td>默认值为0。layout_q为TND时从cu_seqlens_q推断，无需手动指定。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQOptional（int64_t）</td>
      <td>输入</td>
      <td>所有Batch中q的最大有效token数。</td>
      <td>默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenKvOptional（int64_t）</td>
      <td>输入</td>
      <td>所有Batch中ori_kv的最大有效token数。</td>
      <td>默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopKOptional（int64_t）</td>
      <td>输入</td>
      <td>从ori_kv中筛选的稀疏token个数。</td>
      <td>当前暂不支持，默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopKOptional（int64_t）</td>
      <td>输入</td>
      <td>从cmp_kv中筛选的稀疏token个数。</td>
      <td>SCFA模式下仅支持512或1024，CFA/SWA模式下为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatioOptional（int64_t）</td>
      <td>输入</td>
      <td>对ori_kv的压缩率。</td>
      <td>仅支持4或128。SWA模式下传入-1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskModeOptional（int64_t）</td>
      <td>输入</td>
      <td>q和ori_kv计算的mask模式。</td>
      <td>仅支持4（Band模式）。默认值为4。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskModeOptional（int64_t）</td>
      <td>输入</td>
      <td>q和cmp_kv计算的mask模式。</td>
      <td>仅支持3（RightDownCausal模式）。默认值为3。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeftOptional（int64_t）</td>
      <td>输入</td>
      <td>滑动窗口向左扩展的token数。</td>
      <td>仅支持127。默认值为127。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRightOptional（int64_t）</td>
      <td>输入</td>
      <td>滑动窗口向右扩展的token数。</td>
      <td>仅支持0。默认值为0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>标识输入q的数据排布格式。</td>
      <td>支持"BSND"和"TND"，默认值为"BSND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional（char*）</td>
      <td>输入</td>
      <td>标识输入KV的数据排布格式。</td>
      <td>支持"PA_ND"、"BSND"和"TND"，默认值为"PA_ND"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKvOptional（bool）</td>
      <td>输入</td>
      <td>是否传入ori_kv。</td>
      <td>默认值为true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKvOptional（bool）</td>
      <td>输入</td>
      <td>是否传入cmp_kv。</td>
      <td>SWA模式为false，CFA/SCFA模式为true。默认值为true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metaData（aclTensor*）</td>
      <td>输出</td>
      <td>分核元数据输出，供SparseAttnSharedkv算子使用。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t**）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 817px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入、输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>输入变量的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>num_heads_kv不为1。</td>
    </tr>
    <tr>
      <td>ori_mask_mode不为4，或cmp_mask_mode不为3。</td>
    </tr>
    <tr>
      <td>ori_win_left不为127。</td>
    </tr>
    <tr>
      <td>cmp_ratio不为4或128（当has_cmp_kv为true时）。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>调用NPU Runtime接口申请内存/创建Tensor失败。</td>
    </tr>
  </tbody>
  </table>

## aclnnSparseAttnSharedkvMetadata

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseAttnSharedkvMetadataGetWorkspaceSize获取。</td>
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
  </tbody>
  </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算

  - aclnnSparseAttnSharedkvMetadata默认采用确定性实现，相同输入多次调用结果一致。

- 规格约束

  | 规格项        | 规格           | 规格说明                                    |
  | :------------ | :------------- | :------------------------------------------ |
  | num_heads_q   | 64             | Query头数仅支持64。                          |
  | num_heads_kv  | 1              | KV头数仅支持1。                              |
  | head_dim      | 512            | 每个注意力头的维度仅支持512。                 |
  | cmp_topk      | 0 / 512 / 1024 | SCFA模式下TopK仅支持512或1024，CFA/SWA为0。 |
  | cmp_ratio     | 4 / 128        | 压缩率仅支持4或128。                          |
  | ori_mask_mode | 4              | 原始KV掩码模式仅支持4（Band模式）。            |
  | cmp_mask_mode | 3              | 压缩KV掩码模式仅支持3（RightDownCausal模式）。 |
  | ori_win_left  | 127            | 滑动窗口向左扩展仅支持127。                    |
  | ori_win_right | 0              | 滑动窗口向右扩展仅支持0。                      |

- 使用约束

  - 该接口支持推理场景下使用。
  - 该接口支持aclgraph模式。
  - Tensor输入不能全传nullptr。
  - layout_q为TND时，`cu_seqlens_q`必须传入。
  - layout_kv为PA_ND时，`seqused_kv`必须传入。
  - layout_kv为TND时，`cu_seqlens_ori_kv`必须传入；若has_cmp_kv为true，`cu_seqlens_cmp_kv`也必须传入。
  - 该算子为AICPU算子，在Host侧CPU上执行，不占用NPU计算资源。

- 三种模板模式属性配置

  | 模式 | has_ori_kv | has_cmp_kv | cmp_topk | cmp_ratio |
  | :--- | :--------- | :--------- | :------- | :-------- |
  | SWA  | true       | false      | 0        | -1        |
  | CFA  | true       | true       | 0        | 4或128    |
  | SCFA | true       | true       | 512或1024 | 4或128   |

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_attn_sharedkv_metadata.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

int main() {
  // 1. 初始化AscendCL环境（略，参考aclrt相关接口）

  // 2. 构造参数
  int64_t B = 4, S1 = 128, S2 = 8192;
  int64_t N1 = 64, N2 = 1, D = 512, K = 512;
  int64_t cmp_ratio = 4;
  int64_t ori_mask_mode = 4, cmp_mask_mode = 3;
  int64_t ori_win_left = 127, ori_win_right = 0;

  // 3. 准备输入Tensor
  // cu_seqlens_q: [B+1], INT32, 前缀和形式
  // seqused_kv: [B], INT32
  // metadata输出: [1024], INT32

  // 4. 调用aclnnSparseAttnSharedkvMetadataGetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
      cuSeqLensQ, nullptr, nullptr,
      nullptr, seqUsedKv,
      N1, N2, D, B, S1, S2,
      0, K, cmp_ratio,
      ori_mask_mode, cmp_mask_mode,
      ori_win_left, ori_win_right,
      "TND", "PA_ND",
      true, true,
      metadataTensor,
      &workspaceSize, &executor);

  // 5. 申请workspace并执行
  void *workspace = nullptr;
  aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclnnSparseAttnSharedkvMetadata(workspace, workspaceSize, executor, stream);

  // 6. metadataTensor中的数据即可传给aclnnSparseAttnSharedkv使用

  // 7. 释放资源（略）
  return 0;
}
```
