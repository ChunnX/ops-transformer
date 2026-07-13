# aclnnGroupedMatmulV5

<style>
abbr {
  border-bottom: 1px dotted #666;
  cursor: help;
}
</style>

[查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul)

---

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |

---

## 专业术语

| 术语 | 简要说明 |
|------|----------|
| 非量化 | x和weight均为浮点类型直接计算 |
| 全量化 | x和weight均为低精度计算，通过scale还原 |
| 伪量化 | weight低精度存储，通过antiquantScale/Offset反量化后计算 |
| A8W4 | A8 表示 Activation 采用 8bit 量化，W4 表示 weight 采用 4bit 量化，同理还有 A8W8、A4W4 |
| A16W8 | A16 表示 Activation 采用 16bit 非量化，W8 表示 weight 采用 8bit 量化，同理还有 A16W4 |
| perchannel | 每列独立缩放因子，详见[《量化介绍》](../../../docs/zh/context/量化介绍.md) |
| pertensor | 一个 Tensor统一缩放因子 |
| pertoken | 每行独立缩放因子 |
| pergroup | 在 reduce 轴上对数据分组，每组使用独立的量化参数，group size 为 gs |
| perblock | 在所有轴上（含 M 方向）对数据分块，每块使用独立的量化参数，block size 为 bs |
| T-C | pertensor-perchannel：输入侧统一缩放，权重侧每列独立缩放 |
| T-T | pertensor-pertensor：输入和权重均为统一缩放 |
| K-C | pertoken-perchannel：输入侧每行独立缩放，权重侧每列独立缩放 |
| K-T | pertoken-pertensor：输入侧每行独立缩放，权重侧统一缩放 |
| G-B | pergroup-perblock：输入侧按 K分组缩放，权重侧按 N×K 分块缩放 |
| MX | pergroup-pergroup（G-G）量化模式，量化参数类型为 FLOAT8_E8M0，group size 为 32 的特例 |
| ND | 常规连续排布 |
| NZ | [FRACTAL_NZ](../../../docs/zh/context/数据格式.md)格式（亲核排布） |

---

<a id="3-功能说明"></a>

## 功能说明

### 分组方式

算子功能：实现分组矩阵乘计算，公式为 $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中 $g$ 为分组数。

- **M轴分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=0）**：$k_i$、$n_i$ 各组相同，$m_i$ 可以不相同。
- **K轴分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=2）**：$m_i$、$n_i$ 各组相同，$k_i$ 可以不相同。
- **不分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=-1）**：各组 shape 完全独立。

### 计算公式

<details>
<summary>非量化场景</summary>

$$
y_i=x_i\times weight_i + bias_i
$$

</details>

<details>
<summary>全量化场景</summary>

- **量化场景（静态量化，T-C && T-T量化，无perTokenScaleOptional）：**
  $$
    y_i=(x_i\times weight_i) * scale_i + offset_i
  $$
  - x为INT8，bias为INT32
    $$
      y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
    $$
  - x为INT8，bias为BFLOAT16/FLOAT16/FLOAT32，无offset
    $$
      y_i=(x_i\times weight_i) * scale_i + bias_i
    $$
- **量化场景（动态量化，T-T && T-C && K-T && K-C量化）：**
  $$
   y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i
  $$
  - x为INT8，bias为INT32
    $$
      y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i
    $$
  - x为INT8，bias为BFLOAT16/FLOAT16/FLOAT32
    $$
      y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i
    $$
- **量化场景（动态量化，MX && G-B量化）：**
  $$
  y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (xSlice_i * weightSlice_i)) * (per\_token\_scale_i[m/gsM, j] * scale_i[j, n/gsN])) + bias_i[n]
  $$
  其中，gsM,gsN和gsK分别代表M/N/K轴的量化的block size，$xSlice_i$代表$x_i$第m行长度为gsK的向量，$weightSlice_i$代表$weight_i$第n列长度为gsK的向量，K轴均从j * gsK起始切片，j的取值范围[0, kLoops), kLoops=ceil($K_i$ / gsK)，支持最后的切片长度不足gsK。

</details>

<details>
<summary>伪量化场景</summary>

- x为Float16、BFloat16，weight为INT4、INT8（仅支持x、weight、y均为单tensor的场景）。

  $$
  y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
  $$

- x为INT8，weight为INT4（仅支持x、weight、y均为单tensor的场景）。其中$bias$为必选参数，是离线计算的辅助结果，且 $bias_i=8\times weight_i  * scale_i$ ，并沿k轴规约。

  $$
  y_i=((x_i - 8) \times weight_i * scale_i+bias_i ) * per\_token\_scale_i
  $$

</details>

### 版本演进

#### Ascend 950PR/Ascend 950DT

- V1 → V4：支持分组轴（groupType）；非量化支持 x/weight转置；支持静态量化（T-C/T-T）BFLOAT16/FLOAT16/FLOAT32 输出 + bias；支持动态量化（K-C/K-T/T-T/T-C/MX/G-B）BFLOAT16/FLOAT16/FLOAT32 输出 + bias；支持伪量化 weight为 INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8（INT4 支持 perchannel 和 pergroup，其余仅 perchannel）。
- V4 → V5：无变化。

#### Atlas A3/A2 系列产品

- V1 → V4：支持分组轴（groupType）；非量化支持 x/weight转置；支持 x/weight/y 均为单Tensor 非量化 FLOAT32 输入；支持伪量化 weight=INT4（perchannel/pergroup 模式）。
- V4 → V5：增加可选参数 `tuningConfigOptional`，数组中第一个值表示各专家处理的 token数预期值，tiling时按该值进行最优切分。

**确定性计算**：GroupedMatmul默认提供确定性实现。


## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 接口获取计算所需 workspace大小以及包含了算子计算流程的执行器，再调用 `aclnnGroupedMatmulV5` 接口执行计算。

```c++
aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *weight,
    const aclTensorList *biasOptional,
    const aclTensorList *scaleOptional,
    const aclTensorList *offsetOptional,
    const aclTensorList *antiquantScaleOptional,
    const aclTensorList *antiquantOffsetOptional,
    const aclTensorList *perTokenScaleOptional,
    const aclTensor     *groupListOptional,
    const aclTensorList *activationInputOptional,
    const aclTensorList *activationQuantScaleOptional,
    const aclTensorList *activationQuantOffsetOptional,
    int64_t              splitItem,
    int64_t              groupType,
    int64_t              groupListType,
    int64_t              actType,
    aclIntArray         *tuningConfigOptional,
    aclTensorList       *out,
    aclTensorList       *activationFeatureOutOptional,
    aclTensorList       *dynQuantScaleOutOptional,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulV5(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

---

## aclnnGroupedMatmulV5GetWorkspaceSize

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  |--------|:---:|------|------|----------|:---:|:---|:---:|
  | x | 输入 | 公式中的输入 $x$ | TensorList长度 [1,128] 或 [1,1024] | FLOAT、FLOAT16、INT16<span title="Ascend 950PR/950DT 不支持"><sup>1</sup></span>、INT8、INT4<span title="Ascend 950PR/950DT 不支持"><sup>1</sup></span>、BFLOAT16、FLOAT8_E5M2<span title="Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT8_E4M3FN<span title="Atlas A3/A2 不支持"><sup>2</sup></span>、HIFLOAT8<span title="Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT4_E2M1<span title="Atlas A3/A2 不支持"><sup>2</sup></span> | <abbr title="常规连续排布">ND</abbr> | 2~6 | √ |
  | weight | 输入 | 公式中的 $weight$ | TensorList长度 [1,128] 或 [1,1024] | FLOAT、FLOAT16、INT16<span title="Ascend 950PR/950DT 不支持"><sup>1</sup></span>、INT8、INT4、BFLOAT16、FLOAT8_E5M2<span title="Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT8_E4M3FN<span title="Atlas A3/A2 不支持"><sup>2</sup></span>、HIFLOAT8<span title="Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT4_E2M1<span title="Atlas A3/A2 不支持"><sup>2</sup></span> | ND/<abbr title="FRACTAL_NZ格式（亲核排布）">NZ</abbr> | 2~3 | √ |
  | biasOptional | 可选输入 | 公式中的 $bias$ | 长度与weight相同 | FLOAT、FLOAT16、INT32、BFLOAT16<span title="Atlas A3/A2 不支持"><sup>2</sup></span> | ND | 1~2 | √ |
  | scaleOptional | 可选输入 | 公式中的 $scale$，代表量化参数中的缩放因子 | 一般情况下，长度与weight相同。综合约束请参见 [约束说明](#7-约束说明) | FLOAT、UINT64、BFLOAT16、FLOAT8_E8M0<span title="Atlas A3/A2 不支持"><sup>2</sup></span>、INT64<span title="Atlas A3/A2 不支持"><sup>2</sup></span> | ND | 1~4 | √ |
  | offsetOptional | 可选输入 | 公式中的 $offset$，代表量化参数中的偏移量 | 长度与weight相同 | FLOAT | ND | 3 | √ |
  | antiquantScaleOptional | 可选输入 | 公式中的 $antiquant\_scale$，代表伪量化参数中的缩放因子 | 长度与weight相同。综合约束请参见 [约束说明](#7-约束说明) | FLOAT16、BFLOAT16 | ND | 1~3 | √ |
  | antiquantOffsetOptional | 可选输入 | 公式中的 $antiquant\_offset$，代表伪量化参数中的偏移量 | 长度与weight相同。综合约束请参见 [约束说明](#7-约束说明) | FLOAT16、BFLOAT16 | ND | 1~3 | √ |
  | perTokenScaleOptional | 可选输入 | 公式中的 $per\_token\_scale$，代表量化参数中由x量化引入的缩放因子 | 一般情况下，只支持 1维且长度与 x的M相同。综合约束请参见 [约束说明](#7-约束说明) | FLOAT、FLOAT8_E8M0<span title="Atlas A3/A2 不支持"><sup>2</sup></span> | ND | 1~3 | √ |
  | groupListOptional | 可选输入 | 代表输入和输出分组轴方向的 matmul大小分布 | 根据 groupListType输入不同格式数据 | INT64 | ND | 1~2 | √ |
  | activationInputOptional | 可选输入 | 代表激活函数的反向输入，当前只支持传入 nullptr | - | - | - | - | - |
  | activationQuantScaleOptional | 可选输入 | 激活函数量化缩放因子（预留参数），当前只支持传入 nullptr | - | - | - | - | - |
  | activationQuantOffsetOptional | 可选输入 | 激活函数量化偏移量（预留参数），当前只支持传入 nullptr | - | - | - | - | - |
  | splitItem | 输入 | 代表输出是否要做 Tensor切分 | `0/1` = 多Tensor 输出；`2/3` = 单Tensor 输出。接口不区分 0/1 和 2/3 的内部差异 | - | - | - | - |
  | groupType | 输入 | 代表需要分组的轴 | 如矩阵乘为 $C[m,n]=A[m,k]\times B[k,n]$：`-1` = 不分组；`0` = M轴分组；`2` = K轴分组 | - | - | - | - |
  | groupListType | 输入 | 代表 groupList输入的分组方式 | `0` = 累积和（cumsum）；`1` = 各组大小；`2` = [组索引, 组大小] 对。详见 [约束说明](#7-约束说明) | - | - | - | - |
  | actType | 输入 | 代表激活函数类型 | `0`=GMM_ACT_TYPE_NONE / `1`=GMM_ACT_TYPE_RELU / `2`=GMM_ACT_TYPE_GELU_TANH / `3`=GMM_ACT_TYPE_GELU_ERR_FUNC（不支持） / `4`=GMM_ACT_TYPE_FAST_GELU / `5`=GMM_ACT_TYPE_SILU | - | - | - | - |
  | tuningConfigOptional | 可选输入 | 第一个数代表各个专家处理的 token数的预期值；第二个数代表 <abbr title="A8 表示 Activation 采用 8bit 量化，W4 表示 weight 采用 4bit 量化，同理还有 A8W8、A4W4">A8W4</abbr> 可选开启 weight先转置的 NZ格式；第三个数代表允许额外使用的内存空间。详见 [约束说明](#7-约束说明) | 兼容历史版本，用户如不使用该参数，不传（即为 nullptr）即可 | INT64 | - | 3 | - |
  | out | 输出 | 公式中的输出 $y$ | TensorList长度 [1,128] 或 [1,1024] | FLOAT、FLOAT16、INT32、INT8、BFLOAT16 | ND | 2~3 | - |
  | activationFeatureOutOptional | 输出 | 激活函数的输入数据，当前只支持传入 nullptr | - | - | - | - | - |
  | dynQuantScaleOutOptional | 输出 | 动态量化缩放因子，当前只支持传入 nullptr | - | - | - | - | - |
  | workspaceSize | 输出 | 返回需要在 Device侧申请的 workspace大小（字节） | - | - | - | - | - |
  | executor | 输出 | 返回 op执行器，包含了算子计算流程 | - | - | - | - | - |

  - **<term>Ascend 950PR/Ascend 950DT</term>**：
    - 上表数据类型列中的角标 <span title="Ascend 950PR/950DT 不支持"><sup>1</sup></span> 代表该系列不支持的数据类型
    - 输入参数 x、weight均不支持 INT16 类型，且 x不支持 INT4 类型
  - **<term>Atlas A3/A2 系列产品</term>**：
    - 上表数据类型列中的角标 <span title="Atlas A3/A2 不支持"><sup>2</sup></span> 代表该系列不支持的数据类型
    - 不支持 FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT8_E8M0类型
    - 输入参数 biasOptional不支持 BFLOAT16
    - 输入参数 scaleOptional不支持 INT64 类型

- **返回值：**

  aclnnStatus：返回状态码，具体参见 [aclnn 返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景时报错。

  <!-- 以下表格使用 HTML 格式以支持 rowspan 合并单元格 -->

  <table>
    <thead>
      <tr>
        <th style="width: 250px">返回值</th>
        <th style="width: 130px">错误码</th>
        <th style="width: 850px">描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="4"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="4"> 161001 </td>
        <td>传入参数是必选输入、输出或者必选属性，且是空指针。</td>
      </tr>
      <tr>
        <td>传入参数 weight的元素存在空指针。</td>
      </tr>
      <tr>
        <td>传入参数 x的元素为空指针，且传出参数 out的元素不为空指针。</td>
      </tr>
      <tr>
        <td>传入参数 x的元素不为空指针，且传出参数 out的元素为空指针。</td>
      </tr>
      <tr>
        <td rowspan="7"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="7"> 161002 </td>
        <td>x、weight、biasOptional、scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、groupListOptional、out的数据类型和数据格式不在支持的范围内。</td>
      </tr>
      <tr>
        <td>weight的长度不在支持范围。</td>
      </tr>
      <tr>
        <td>若 bias不为空，bias的长度不等于weight的长度。</td>
      </tr>
      <tr>
        <td>groupListOptional维度不符合要求。</td>
      </tr>
      <tr>
        <td>splitItem 为 2、3 的场景，out长度不等于1。</td>
      </tr>
      <tr>
        <td>splitItem 为 0、1 的场景，out长度不等于weight的长度，groupListOptional长度不等于weight的长度。</td>
      </tr>
      <tr>
        <td>传入参数 tuningConfigOptional 的元素为负数，或者大于x的行数 m。</td>
      </tr>
    </tbody>
  </table>

## aclnnGroupedMatmulV5

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 |
  |--------|:---:|------|
  | workspace | 输入 | 在 Device侧申请的 workspace内存地址 |
  | workspaceSize | 输入 | 在 Device侧申请的 workspace大小，由第一段接口 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 |
  | executor | 输入 | op执行器，包含了算子计算流程 |
  | stream | 输入 | 指定执行任务的 Stream |

- **返回值：**

  aclnnStatus：返回状态码，具体参见 [aclnn 返回码](../../../docs/zh/context/aclnn返回码.md)。

---

<a id="7-约束说明"></a>

## 约束说明

### 确定性计算

aclnnGroupedMatmulV5 默认确定性实现。

### Ascend 950PR/Ascend 950DT

#### 平台约束

- groupType：支持M轴分组（0）和不分组（-1）。非量化/全量化额外支持K轴分组（2）。
- groupListType：支持 0、1、2。
  - groupListType=0：须为非负单调非递减数列（累积和）。以M=256、E=4（各组大小依次为64、0、128、64）为例：`[64, 64, 192, 256]`
  - groupListType=1：须为非负数列（各组大小）。例如：`[64, 0, 128, 64]`
  - groupListType=2：仅全量化且groupType=0场景下支持，须为非负数列，shape为`[E, 2]`，E表示Group大小，数据排布为`[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]`，非零组前置。例如：`[[0, 64], [2, 128], [3, 64], [1, 0]]`
- tuningConfigOptional：不支持。
- actType（0~5）：
  - 非量化/伪量化仅支持 0。
  - 全量化下 x/weight=INT8 且 out=BFLOAT16/FLOAT16、静态 T-C 或动态 K-C、scale=FLOAT32/BFLOAT16 时支持 0/1/2/4/5；其余仅支持 0。
- 输入参数 x、weight，输出参数 out 在非量化场景支持最多 1024个Tensor，在伪量化支持最多 128个Tensor，在全量化场景最多支持 1个Tensor。

#### 场景速查表

| 场景名 | x | weight | out | 约束说明 |
|--------|:---:|:---:|:---|------|
| 非量化 | FLOAT32/FLOAT16/BFLOAT16 | FLOAT32/FLOAT16/BFLOAT16 | FLOAT32/FLOAT16/BFLOAT16 | [非量化场景约束](#ascend950-非量化场景约束) |
| 静态量化 | INT8 | INT8 | BFLOAT16/FLOAT16/INT32/INT8 | [静态量化场景约束](#ascend950-静态量化场景约束) |
| 静态量化 | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | BFLOAT16/FLOAT16/FLOAT32 | [静态量化场景约束](#ascend950-静态量化场景约束) |
| 动态量化（<abbr title="pertensor-pertensor：输入和权重均为统一缩放">T-T</abbr>/T-C/<abbr title="pertoken-pertensor：输入侧每行独立缩放，权重侧统一缩放">K-T</abbr>/K-C） | INT8 | INT8 | BFLOAT16/FLOAT16 | [动态量化（T-T/T-C/K-T/K-C）场景约束](#ascend950-动态量化-ttck) |
| 动态量化（T-T/T-C/K-T/K-C） | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | BFLOAT16/FLOAT16/FLOAT32 | [动态量化（T-T/T-C/K-T/K-C）场景约束](#ascend950-动态量化-ttck) |
| 动态量化（<abbr title="特殊 pergroup量化（block 粒度 M=1,N=1,K=32）">MX</abbr>） | FLOAT8_E5M2/FLOAT8_E4M3FN / FLOAT4_E2M1 | FLOAT8_E5M2/FLOAT8_E4M3FN / FLOAT4_E2M1 | BFLOAT16/FLOAT16/FLOAT32 | [动态量化（MX）场景约束](#ascend950-动态量化-mx) |
| 动态量化（<abbr title="pergroup-perblock：输入侧按 K分组缩放，权重侧按 N×K 分块缩放">G-B</abbr>） | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | BFLOAT16/FLOAT16/FLOAT32 | [动态量化（G-B）场景约束](#ascend950-动态量化-gb) |
| 伪量化 | FLOAT16 / BFLOAT16 | INT8/INT4 | FLOAT16 / BFLOAT16 | [伪量化场景约束](#ascend950-伪量化场景约束) |
| 伪量化 | FLOAT16 / BFLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | FLOAT16 / BFLOAT16 | [伪量化场景约束](#ascend950-伪量化场景约束) |

<a id="ascend950-非量化场景约束"></a>

<details>
<summary>非量化场景约束</summary>

- 以下入参为空：scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional、activationFeatureOutOptional
- 不为空的参数支持的数据类型组合要满足下表：

| groupType | x | weight | biasOptional | out |
|:---:|:---:|:---:|:---|:---|
| -1/0/2 | BFLOAT16 | BFLOAT16 | BFLOAT16/FLOAT32/null | BFLOAT16 |
| -1/0/2 | FLOAT16 | FLOAT16 | FLOAT16/FLOAT32/null | FLOAT16 |
| -1/0/2 | FLOAT32 | FLOAT32 | FLOAT32/null | FLOAT32 |

</details>

<a id="ascend950-静态量化场景约束"></a>

<details>
<summary>静态量化场景约束</summary>

- 以下入参为空：offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、perTokenScaleOptional、activationInputOptional
- 不为空的参数支持的数据类型组合要满足下表：

| groupType | x | weight | biasOptional | scaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|
| 0 | INT8 | INT8 | INT32/null | UINT64/INT64 | BFLOAT16/FLOAT16/INT8 |
| 0 | INT8 | INT8 | INT32/null | null/UINT64/INT64 | INT32 |
| 0 | INT8 | INT8 | INT32/BFLOAT16/FLOAT32/null | BFLOAT16/FLOAT32 | BFLOAT16 |
| 0 | INT8 | INT8 | INT32/FLOAT16/FLOAT32/null | FLOAT32 | FLOAT16 |
| 0 | HIFLOAT8 | HIFLOAT8 | null | UINT64/INT64 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | HIFLOAT8 | HIFLOAT8 | null | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | UINT64/INT64 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**（$g$=分组数）：

| groupType | 子场景 | shape |
|:---:|:---|:---|
| 0/2 | <abbr title="每列独立缩放因子">perchannel</abbr> | `(g, N)` |
| 0/2 | <abbr title="一个 Tensor统一缩放因子">pertensor</abbr> | `(g, 1)` 或 `(g,)`，输出为 INT8 时不支持 Pertensor场景 |

</details>

<a id="ascend950-动态量化-ttck"></a>

<details>
<summary>动态量化（T-T/T-C/K-T/K-C）场景约束</summary>

- 以下入参为空：offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、activationInputOptional
- 不为空的参数支持的数据类型组合要满足下表：

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| 0 | INT8 | INT8 | INT32/BFLOAT16/FLOAT32/null | BFLOAT16/FLOAT32 | FLOAT32 | BFLOAT16 |
| 0 | INT8 | INT8 | INT32/FLOAT16/FLOAT32/null | FLOAT32 | FLOAT32 | FLOAT16 |
| 0/2 | HIFLOAT8 | HIFLOAT8 | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**（$g$=分组数），推荐在 Pertensor场景 shape 使用 `(g,)`，防止与 G-B量化模式混淆：

  | groupType | 子场景 | shape |
  |:---:|:---|:---|
  | 0/2 | <abbr title="每列独立缩放因子">perchannel</abbr> | `(g, N)` |
  | 0/2 | <abbr title="一个 Tensor统一缩放因子">pertensor</abbr> | `(g, 1)` 或 `(g,)` |

- **perTokenScaleOptional shape**：

| groupType | 子场景 | shape |
|:---:|:---|:---|
| 0 | <abbr title="每行独立缩放因子">pertoken</abbr> | `(M,)` |
| 0 | pertensor | `(g, 1)` 或 `(g,)`，输入为 INT8 时不支持 pertensor 场景 |
| 2 | pertoken | `(g, M)` |
| 2 | pertensor | `(g, 1)` 或 `(g,)` |

</details>

<a id="ascend950-动态量化-mx"></a>

<details>
<summary>动态量化（MX）场景约束</summary>

- 以下入参为空：offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、activationInputOptional
- 计算公式中量化blocksize为：gsM=gsN=1，gsK=32。MX量化是特殊的pergroup量化。
- 不为空的参数支持的数据类型组合要满足下表：

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |
| 0 | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32/null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**：

| groupType | shape |
|:---:|:---|
| 0 | `(g, N, ceil(K/64), 2)`（weight 转置）/ `(g, ceil(K/64), N, 2)`（weight不转置） |
| 2 | `(K/64 + g, N, 2)`，scale_i起始地址偏移为 `((K_0 + K_1 + ... + K_{i-1}) / 64 + g_i) * N * 2`（$g_i$ 为第 i 个分组，下标从 0 开始），即scale_0偏移为0，scale_1偏移为 `(K_0 / 64 + 1) * N * 2`，scale_2偏移为 `((K_0 + K_1) / 64 + 2) * N * 2`，依此类推 |

- **perTokenScaleOptional shape**：

| groupType | shape |
|:---:|:---|
| 0 | `(M, ceil(K/64), 2)` |
| 2 | `(K/64 + g, M, 2)`，起始地址偏移与scale同理 |

- x为 FLOAT4_E2M1 时，K须为偶数且 K≠2；weight 非转置时 N须为偶数。

</details>

<a id="ascend950-动态量化-gb"></a>

<details>
<summary>动态量化（G-B）场景约束</summary>

- 以下入参为空：biasOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、activationInputOptional
- 计算公式量化blocksize为：当前仅支持gsM=1，gsN=gsK=128。
- 不为空的参数支持的数据类型组合要满足下表：

| groupType | x | weight | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|
| 0/2 | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**（$g$=分组数）：

| groupType | shape |
|:---:|:---|
| 0 | `(g, ceil(N/gsN), ceil(K/gsK))`（转置）/ `(g, ceil(K/gsK), ceil(N/gsN))`（非转置），gsN=gsK=128 |
| 2 | `(K/gsK + g, ceil(N/gsN))`，scale_i地址偏移为 `((K_0 + K_1 + ... + K_{i-1}) / gsK + g_i) * ceil(N / gsN)`（$g_i$ 为第 i 个分组，下标从 0 开始），即scale_0偏移为0，scale_1偏移为 `(K_0 / gsK + 1) * ceil(N / gsN)`，scale_2偏移为 `((K_0 + K_1) / gsK + 2) * ceil(N / gsN)`，依此类推 |

- **perTokenScaleOptional shape**：

| groupType | shape |
|:---:|:---|
| 0 | `(M, ceil(K/gsK))` |
| 2 | `(K/gsK + g, M)`，per_token_scale_i地址偏移为 `((K_0 + K_1 + ... + K_{i-1}) / gsK + g_i) * M`（$g_i$ 为第 i 个分组，下标从 0 开始），即per_token_scale_0偏移为0，per_token_scale_1偏移为 `(K_0 / gsK + 1) * M`，per_token_scale_2偏移为 `((K_0 + K_1) / gsK + 2) * M`，依此类推 |

- **动态量化特殊场景处理**：
  - 在动态量化场景 M分组或K分组情况下，当 N等于1 且 scaleOptional 的 shape 为 `(g, 1)` 时，weight既可以pertensor量化也可以perchannel量化时，优先选择pertensor量化模式。
  - 在动态量化场景 M分组情况下，当 g = M 且 perTokenScaleOptional 的 shape 为 `(g,)` 时，x选择pertoken量化模式；当 g = M，K ≤ 128 且 perTokenScaleOptional 的 shape 为 `(g, 1)` 时，根据weight的量化模式选择x的量化模式（weight 如果是 perchannel 或者pertensor量化，x选择pertensor量化；weight 如果是 <abbr title="按 K/N 方向固定大小分块">perblock</abbr> 量化，x选择 <abbr title="按 K 方向等分多组">pergroup</abbr> 量化）。
  - 在动态量化场景 K分组情况下，K小于 128，N小于等于128 且 scaleOptional 的 shape 为 `(g, 1)` 时，按照现有量化模式区分规则，既可以为非pergroup量化，又可以为G-B量化，此种场景现一律按照G-B量化处理。
  - 在动态量化场景 K分组情况下，当 M等于 1 且 perTokenScaleOptional 的 shape 为 `(g, 1)` 时，x既可以pertoken量化也可以pertensor量化时，优先选择pertensor量化模式。
  - 在动态量化场景 K分组情况下，K小于 128，M等于 1 且 perTokenScaleOptional 的 shape 为 `(g, 1)` 时，如果 N小于等于128，x则选择pergroup量化；如果 N大于128，根据weight的量化模式选择x的量化模式（weight 如果是 perchannel 或者pertensor量化，x选择pertensor量化；weight 如果是 perblock量化，x选择pergroup量化）。
  - 在动态量化场景 K分组情况下，K小于 128，M不等于 1 时，如果 N小于等于128，x则选择pergroup量化；如果 N大于128，根据weight的量化模式选择x的量化模式（weight 如果是 perchannel 或者pertensor量化，x选择pertoken量化；weight 如果是 perblock量化，x选择pergroup量化）。

</details>

<a id="ascend950-伪量化场景约束"></a>

<details>
<summary>伪量化场景约束</summary>

- 以下入参为空：scaleOptional、offsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional
- 不为空的参数支持的数据类型组合要满足下表：

| groupType | x | weight | biasOptional | antiquantScaleOptional | antiquantOffsetOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| -1/0 | BFLOAT16 | INT8/INT4 | BFLOAT16/FLOAT32/null | BFLOAT16 | BFLOAT16/null | BFLOAT16 |
| -1/0 | FLOAT16 | INT8/INT4 | FLOAT16/null | FLOAT16 | FLOAT16/null | FLOAT16 |
| 0 | BFLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | BFLOAT16/FLOAT32/null | BFLOAT16 | null | BFLOAT16 |
| 0 | FLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | FLOAT16/null | FLOAT16 | null | FLOAT16 |

- weight为FLOAT8系列时，antiquantOffsetOptional 仅支持空指针或空TensorList，weight仅支持转置。
- 若 weight的类型为 INT4，则 weight中每一组 tensor的最后一维大小都应是偶数。$weight_i$ 的最后一维指 weight不转置时 $weight_i$ 的 N轴或当 weight转置时 $weight_i$ 的 K轴。
- 当 weight为 INT4 时，支持 perchannel 和 pergroup 两种伪量化模式，通过 antiquantScaleOptional 的维度数自动判定：单单单场景维度为 2 时为 perchannel、维度为 3 时为 pergroup；多多多场景维度为 1 时为 perchannel、维度为 2 时为 pergroup。
- pergroup场景下，pergroup数 $G$ 或 $G_i$ 须能整除对应 $k_i$。多Tensor时 pergroup长度 $s_i = k_i / G_i$ 各组相等。
- pergroup场景下，groupSize 取值仅支持 32、64、128、256。weight转置时 pergroup长度 $s_i$ 须为偶数。
- antiquantScaleOptional 和非空的 biasOptional、antiquantOffsetOptional 要满足下表（其中 $g$ 为 matmul组数即分组数，$G$ 为 pergroup数，$G_i$ 为第 i 个 tensor的pergroup数）：

  | groupType | 使用场景 | shape 限制 |
  |:---:|:---|:---|
  | -1 | weight 多 tensor（perchannel） | 每个 tensor1维，shape 为（$n_i$），不允许存在一个 tensorList 中部分 tensor的shape 为（$n_i$）部分 tensor 为空的情况 |
  | -1 | weight 多 tensor（pergroup） | 每个 tensor 2维，shape 为（$G_i$, $n_i$） |
  | 0 | weight 单 tensor（perchannel） | 每个 tensor 2维，shape 为（$g$, N）|
  | 0 | weight 单 tensor（pergroup） | 每个 tensor 3维，shape 为（$g$, $G$, N）|

</details>


#### groupType 支持场景

> x、weight、y 输入为 aclTensorList。"单" = 单元素TensorList，"多" = 多元素TensorList。

| groupType | x | weight | y | splitItem | groupListOptional | 转置 | 其余场景限制 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x不转置；weight可转置（统一） | 非量化 x/out 2维，伪量化 x 2~6维；weight 2维；仅非量化/伪量化；仅 ND进ND出 |
| 0 | 单 | 单 | 单 | 2/3 | 必须传 | x不转置；weight可转置 | weight 3维 `(g,N,K)` 或 `(g,K,N)`；x/out 2维；bias 2维 `(g,N)`；最多 1024 组；仅 ND进ND出 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传（长度 ≤1024） | x不转置；weight可转置（统一） | x/weight/y 2维；weight N轴相等；仅 ND进ND出；仅非量化 |
| 0 | 多 | 多 | 单 | 2 | 可选 | x不转置；weight可转置（统一） | x/weight/y 2维；weight N轴相等；仅 ND进ND出；仅非量化 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传 | x必须转置；weight不转置 | x/weight 2维 `(K,M)`/`(K,N)`；y 3维 `(g,M,N)`；bias传空；仅 ND进ND出；仅非量化/量化 |
| 2 | 单 | 多 | 多 | 0/1 | 可选 | x必须转置；weight不转置 | x/weight/y 2维 `(M,N)`；weight最大 128 组；不支持 bias；仅 ND进ND出；仅非量化 |

### Atlas A3/A2 系列产品

#### 平台约束

- x/weight中每组 Tensor的最后一维 < 65536。
  - $x_i$ 最后一维：x不转置时为K轴，x转置时为M轴。
  - $weight_i$ 最后一维：weight不转置时为N轴，weight转置时为K轴。
- x/weight转置时对应 Tensor必须 [非连续](../../../docs/zh/context/非连续的Tensor.md)。
- weight为 [FRACTAL_NZ](../../../docs/zh/context/数据格式.md) 格式时，shape 须满足 NZ格式要求。
- perTokenScaleOptional：通常仅1维，长度与 x的M一致。仅支持 x/weight/out均为单Tensor 场景。
- groupListOptional：out中 TensorList长度为1时，groupList约束输出数据有效范围，未指定部分不参与更新。
- groupListType：
  - groupListType=0：须为非负单调非递减数列（累积和）。以M=256、E=4（各组大小依次为64、0、128、64）为例：`[64, 64, 192, 256]`
  - groupListType=1：须为非负数列（各组大小）。例如：`[64, 0, 128, 64]`
  - groupListType=2：须为非负数列，shape `[E, 2]`（[组索引, 组大小]），非零组前置。例如：`[[0, 64], [2, 128], [3, 64], [1, 0]]`
- groupType：支持 M轴分组（0）和不分组（-1），非量化额外支持 K轴分组（2）。全量化下 groupType 仅支持 0（M轴分组）。
- tuningConfigOptional：支持，为 Host侧 INT64 aclIntArray。
  - `[0]`：预期各专家处理的 token数，tiling 按此优化，适用场景：A8W4/A8W8/A4W4，x/weight/out 单Tensor。
  - `[1]`：是否开启 weight 亲核格式（先转置再 NZ），适用场景：A8W4。
  - `[2]`：允许额外 workspace 上限（-1=不限），适用场景：A8W8 定轴算法优化。
- actType：仅 A8W8（x、weight 为 INT8）且输出为 FLOAT16/BFLOAT16 时支持激活函数类型 0/1/2/4/5（3 不支持）；其余场景仅支持 0。
- 输入参数 x、weight，输出参数 out支持最多 128个Tensor。

#### 场景速查表

| 场景名 | x | weight | out | 约束说明 |
|--------|:---:|:---:|:---:|------|
| 非量化 | FLOAT32 | FLOAT32 | FLOAT32 | [非量化场景约束](#非量化场景约束) |
| 非量化 | BFLOAT16 | BFLOAT16 | BFLOAT16 | [非量化场景约束](#非量化场景约束) |
| 非量化 | FLOAT16 | FLOAT16 | FLOAT16 | [非量化场景约束](#非量化场景约束) |
| 全量化-<abbr title="A8 表示 Activation 采用 8bit 量化，W4 表示 weight 采用 4bit 量化，同理还有 A8W8、A4W4">A8W8</abbr> | INT8 | INT8 | BFLOAT16/FLOAT16/INT32/INT8 | [A8W8场景约束](#a8w8场景约束) |
| 全量化-<abbr title="A8 表示 Activation 采用 8bit 量化，W4 表示 weight 采用 4bit 量化，同理还有 A8W8、A4W4">A4W4</abbr> | INT4 | INT4 | BFLOAT16/FLOAT16 | [A4W4场景约束](#a4w4场景约束) |
| 伪量化-A8W4 | INT8 | INT4 | BFLOAT16/FLOAT16 | [A8W4场景约束](#a8w4场景约束) |
| 伪量化-<abbr title="A16 表示 Activation 采用 16bit 非量化，W8 表示 weight 采用 8bit 量化，同理还有 A16W4">A16W8</abbr> | BFLOAT16/FLOAT16 | INT8 | BFLOAT16/FLOAT16 | [A16W8场景约束](#a16w8场景约束) |
| 伪量化-<abbr title="A16 表示 Activation 采用 16bit 非量化，W8 表示 weight 采用 8bit 量化，同理还有 A16W4">A16W4</abbr> | BFLOAT16/FLOAT16 | INT4 | BFLOAT16/FLOAT16 | [A16W4场景约束](#a16w4场景约束) |

<a id="非量化场景约束"></a>

<details>
<summary>非量化场景约束</summary>

**数据类型要求：**

| x | weight | bias | groupList | out |
|:---|:---|:---|:---|:---|
| FLOAT | FLOAT (ND) | FLOAT/null | INT64 | FLOAT |
| FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | INT64 | FLOAT16 |
| BFLOAT16 | BFLOAT16(ND/NZ) | FLOAT/null | INT64 | BFLOAT16 |

> 以下参数须传空：scale、offset、antiquantScale、antiquantOffset、perTokenScale、activationInput、activationQuantScale、activationQuantOffset、activationFeatureOut。

**约束**：
- 支持 groupType=-1、0、2，actType=0，groupListType=0/1/2。

</details>

<a id="a8w8场景约束"></a>

<details>
<summary>A8W8 场景约束</summary>

**数据类型要求：**

| x | weight | bias | scale | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|
| INT8 | INT8 (ND) | INT32/null | UINT64 | null | INT8 |
| INT8 | INT8 (ND/NZ) | INT32/null | BFLOAT16 | FLOAT/null | BFLOAT16 |
| INT8 | INT8 (ND/NZ) | INT32/null | FLOAT | FLOAT/null | FLOAT16 |
| INT8 | INT8 (ND/NZ) | INT32/null | null | null | INT32 |

> 以下参数须传空：offset、antiquantScale、antiquantOffset、activationInput、activationQuantScale、activationQuantOffset。

**约束**：
- 仅 groupType=0（M轴分组）。
- x、weight、out 仅支持长度为 1 的 TensorList。
- x不支持转置，2维 `(M, K)`；weight 3维 `(E, K, N)` 或 `(E, N, K)`。
- **定轴算法**（优化性能）需同时满足：
  - shape（满足其一）：`(M, 7168) × (7168, 4096)` 或 `(M, 2048) × (2048, 7168)`
  - tuningConfigOptional：[0] ∈ (128, 512)，[1]=0，[2] = -1 或 ≥ M×N×4

</details>

<a id="a4w4场景约束"></a>

<details>
<summary>A4W4 场景约束</summary>

**数据类型要求：**

| x | weight | bias | scale | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|
| INT4 | INT4 (ND/NZ) | null | UINT64 | FLOAT/null | FLOAT16/BFLOAT16 |

> 以下参数须传空：offset、antiquantScale、antiquantOffset、activationInput、activationQuantScale、activationQuantOffset。

**约束**：
- groupType=0，actType=0，groupListType=0/1/2。
- x/weight/out 仅支持长度为 1 的 TensorList。
- x不转置，2维 `(M, K)`。
- weight 3维 `(E, K, N)`。ND格式时 N为 8 的倍数。
- weight NZ格式支持转置，ND 仅支持非转置。
- perchannel：scale shape `(E, N)`；pergroup：scale shape `(E, G, N)`，G能整除 K 且 K/G 为偶数。
- 开启右矩阵 NZ 转置后：K/G 对齐 64，K 对齐 64，N 对齐 16。

</details>

<a id="a8w4场景约束"></a>

<details>
<summary>A8W4 场景约束</summary>

**数据类型要求：**

| x | weight | bias | scale | offset | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|:---|
| INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | null | FLOAT | BFLOAT16 |
| INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | FLOAT | FLOAT16 |

> 以下参数须传空：antiquantScale、antiquantOffset、activationInput、activationQuantScale、activationQuantOffset。

**约束**：
- groupType=0，actType=0。
- 当前支持x、out均为长度为1的TensorList。
- weight、scaleOptional、biasOptional和offsetOptional支持单Tensor场景（tensorlist长度为1）和多Tensor场景（tensorlist长度大于1）
- x/weight均不转置。x 2维 `(M, K)`；weight 3维 `(E, K, N)`。
- bias 为离线辅助结果：$bias_i = 8 \times weight_i \times scale_i$ 沿 K轴规约，shape `(E, N)`。
- weight=INT32 时，每个 INT32 视为 8 个 INT4。
- tuningConfigOptional[1] 可置 1 开启 weight 亲核格式（仅 perchannel 生效，需 weight shape `(E, N, K)` 经 ND2NZ 转换；K ≥ 2048 && N ≥ 2048 时有性能优势）。

**offset为空（对称量化）：**
- 按 count模式（不检查 groupListType），K为 quantGroupSize=256 的整数倍，K ≤ 18432。
- N为 8 的整数倍。
- scale shape `(E, K/256, N)`。
- tuningConfigOptional[0] > N/4 时性能更优（显存额外增加 $g \times K \times N$ 字节）。

**offset 不为空（非对称量化）：**
- scale shape `(E, 1, N)`。
- offset为离线辅助结果（$antiquantOffset \times scale$），shape `(E, 1, N)`，dtype=FLOAT32。

</details>

<a id="a16w8场景约束"></a>

<details>
<summary>A16W8 场景约束</summary>

**数据类型要求：**

| x | weight | bias | antiquantScale | antiquantOffset | out |
|:---|:---|:---|:---|:---|:---|
| FLOAT16 | INT8 (ND) | FLOAT16/null | FLOAT16 | FLOAT16 | FLOAT16 |
| BFLOAT16 | INT8 (ND) | FLOAT/null | BFLOAT16 | BFLOAT16 | BFLOAT16 |

> 以下参数须传空：scale、offset、perTokenScale、activationInput、activationQuantScale、activationQuantOffset。

**约束**：
- x不转置；groupType=-1/0；actType=0；groupListType=0/1/2。
- 仅 perchannel 模式。
- weight 多Tensor时，pergroup长度 $s_i = K_i / G_i$ 须各组相等。
- antiquantScale/Offset shape：

| 子场景 | shape |
|:---|:---|
| weight 单，perchannel | `(E, N)` |
| weight 多，perchannel | `(N_i)` |

</details>

<a id="a16w4场景约束"></a>

<details>
<summary>A16W4 场景约束</summary>

**数据类型要求：**

| x | weight | bias | antiquantScale | antiquantOffset | out |
|:---|:---|:---|:---|:---|:---|
| FLOAT16 | INT4 (ND) | FLOAT16/null | FLOAT16 | FLOAT16 | FLOAT16 |
| BFLOAT16 | INT4 (ND) | FLOAT/null | BFLOAT16 | BFLOAT16 | BFLOAT16 |

> 以下参数须传空：scale、offset、perTokenScale、activationInput、activationQuantScale、activationQuantOffset。

**约束**：
- x不转置；groupType=-1/0；actType=0；groupListType=0/1/2。
- weight 最后一维为偶数（最后一维指 weight不转置时 $weight_i$ 的 N轴、转置时 $weight_i$ 的 K轴）；pergroup场景下 weight转置时 pergroup长度为偶数。
- 对称量化支持 perchannel 和 pergroup 两种模式；pergroup场景：G须能整除对应 $K_i$。多Tensor时所有 $s_i$ 相等。
- 非对称量化仅 perchannel。
- antiquantScale/Offset shape：

| 子场景 | shape |
|:---|:---|
| weight 单，perchannel | `(E, N)` |
| weight 单，pergroup | `(E, G, N)` |
| weight 多，perchannel | `(N_i)` |
| weight 多，pergroup | `(G_i, N_i)` |

</details>

#### groupType 支持场景

> x、weight、y 输入为 aclTensorList。"单" = 单元素TensorList，"多" = 多元素TensorList。

各量化类型支持的 groupType 速览：
- A16W8、A16W4：仅支持 groupType=-1、0。
- A8W8、A8W4、A4W4：仅支持 groupType=0，且 x 为单 tensor。

| groupType | x | weight | y | splitItem | groupListOptional | 转置 | 其余场景限制 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x不转置；weight可转置（统一） | x 2-6维（维度一致），weight 2维，y 同 x |
| 0 | 单 | 单 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第一维；groupListType=1 总和 ≤ x 第一维；groupListType=2 第二列总和 ≤ x 第一维；最大 1024 组 | x不转置；weight可转置（A8W4/A4W4 不可） | weight 3维，x/y 2维 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第一维；groupListType=1 总和 ≤ x 第一维；groupListType=2 第二列总和 ≤ x 第一维；最大 128 组 | x不转置；weight可转置（统一） | x/weight/y 2维；weight N 轴相等 |
| 0 | 多 | 多 | 单 | 2/3 | 可选；若传则 groupListType=0 差值 = x 第一维、groupListType=1 数值 = x 第一维、groupListType=2 第二列 = x 第一维；最大 128 组 | x不转置；weight可转置（统一） | x/weight/y 2维；weight N 轴相等 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第二维；groupListType=1 总和 ≤ x 第二维；groupListType=2 第二列总和 ≤ x 第二维；最大 1024 组 | x必须转置；weight不转置 | x/weight 2维，y 3维；bias传空 |
| 2 | 单 | 多 | 多 | 0/1 | 必须传空 | x必须转置；weight不转置 | x/weight/y 2维；weight最大 128 组；原始 shape 各 weight 第一维之和 ≤ x 第一维；bias传空 |

---

## 调用示例
<a id="83-示例代码索引"></a>

### 示例代码索引

以下表格提供各场景下的示例代码文件链接，请参考文件中的注释了解对应场景说明。编译与运行请参考 [编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

| 场景 | 产品型号 | 示例文件 | 说明 |
|------|:---:|------|------|
| 非量化 | ALL | [test_aclnn_grouped_matmul.cpp](../examples/test_aclnn_grouped_matmul.cpp) | 非量化基础示例（涵盖通用调用流程） |
| 伪量化-A16W8 | ALL | [test_aclnn_grouped_matmul_a16w8.cpp](../examples/test_aclnn_grouped_matmul_a16w8.cpp) | x=BF16, weight=INT8, antiquantScale=BF16, antiquantOffset=BF16 |
| 伪量化-A16W4 | ALL | [test_aclnn_grouped_matmul_a16w4.cpp](../examples/test_aclnn_grouped_matmul_a16w4.cpp) | x=BF16, weight=INT4, antiquantScale=BF16, antiquantOffset=BF16 |
| MX量化 | Ascend 950 | [arch35/test_aclnn_grouped_matmul_mx_quant.cpp](../examples/arch35/test_aclnn_grouped_matmul_mx_quant.cpp) | Ascend 950 MX量化示例 |
| 全量化（动态 K-C） | Ascend 950 | [arch35/test_aclnn_grouped_matmul_quant_dynamic.cpp](../examples/arch35/test_aclnn_grouped_matmul_quant_dynamic.cpp) | x=INT8, weight=INT8, scale=FLOAT32, perTokenScale=FLOAT |
| G-B量化 | Ascend 950 | [arch35/test_aclnn_grouped_matmul_quant_gb.cpp](../examples/arch35/test_aclnn_grouped_matmul_quant_gb.cpp) | x/w=FLOAT8_E5M2, scale=FLOAT32 (3维), perTokenScale=FLOAT32 (2维) |
| 全量化A8W8 | Atlas A3/A2  | [arch22/test_aclnn_grouped_matmul_a8w8.cpp](../examples/arch22/test_aclnn_grouped_matmul_a8w8.cpp) | x=INT8, weight=INT8, scale=UINT64, out=BF16 |
