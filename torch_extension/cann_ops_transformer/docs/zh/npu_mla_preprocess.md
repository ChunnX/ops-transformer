# npu_mla_preprocess

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- **接口功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程如下：
    - 首先对输入$x$ RmsNormQuant后乘以$W^{DQKV}$进行下采样后分为通路1和通路2。
    - 通路1做RmsNormQuant后乘以$W^{UQ}$后再分为通路3和通路4。
    - 通路3后乘以$W^{uk}$后输出$q^N$。
    - 通路4后经过旋转位置编码后输出$q^R$。
    - 通路2拆分为通路5和通路6。
    - 通路5经过RmsNorm后传入Cache中得到$k^N$。
    - 通路6经过旋转位置编码后传入另一个Cache中得到$k^R$。

**计算公式**：

    RmsNormQuant公式

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    RmsNormQuant(x) = ({RmsNorm}(x) + bias) * deqScale
    $$
  
    Query计算公式，包括W^{DQKV}矩阵乘、W^{UK}矩阵乘、RmsNormQuant和ROPE旋转位置编码处理

    $$
    q^N =  RmsNormQuant(x) \cdot W^{DQKV} \cdot W^{UK}
    $$

    $$
    q^R = ROPE(x^Q)
    $$

    Key计算公式，包括RmsNorm和rope，将计算结果存入cache

    $$
    k^N = Cache({RmsNorm}(RmsNormQuant(x)))
    $$

    $$
    k^R = Cache(ROPE(RmsNormQuant(x)))
    $$

## 函数原型

```python
from cann_ops_transformer.ops import npu_mla_preprocess

npu_mla_preprocess(
    input, gamma0, beta0,
    quantScale0, quantOffset0,
    wdqkv, deScale0, bias0,
    gamma1, beta1,
    quantScale1, quantOffset1,
    wuq, deScale1, bias1,
    gamma2, cos, sin, wuk,
    kvCache, slotMapping,
    kvCacheRope=None,
    ctkvScale=None, qNopeScale=None,
    wdqDim=1536, qRopeDim=64, kRopeDim=64,
    epsilon=1e-5,
    qRotaryCoeff=2, kRotaryCoeff=2,
    transposeWdq=True, transposeWuq=True, transposeWuk=True,
    cacheMode=1, quantMode=0,
    doRmsNorm=True, wdkvSplitCount=1
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

### 输入参数

  <table style="undefined;table-layout: fixed; width: 1700px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 301px">
  <col style="width: 331px">
  <col style="width: 237px">
  <col style="width: 111px">
  <col style="width: 170px">
  <col style="width: 150px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>用于计算Query和Key的x。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[tokenNum,hiddenSize]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma0</td>
      <td>输入</td>
      <td>首次RmsNorm计算中的γ参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[hiddenSize]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta0</td>
      <td>输入</td>
      <td>首次RmsNorm计算中的β参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[hiddenSize]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantScale0</td>
      <td>输入</td>
      <td>首次RmsNorm公式中量化缩放的参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantOffset0</td>
      <td>输入</td>
      <td>首次RmsNorm公式中的量化偏移参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>wdqkv</td>
      <td>输入</td>
      <td>与输入首次做矩阵乘的降维矩阵。</td>
      <td>-</td>
      <td>INT8、FLOAT16、BFLOAT16</td>
      <td>NZ</td>
      <td>[qLoraDim + keyTotalDim,hiddenSize]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deScale0</td>
      <td>输入</td>
      <td>输入首次做矩阵乘的降维矩阵中的系数。</td>
      <td>input输入dtype为FLOAT16支持INT64，输入BFLOAT16时支持FLOAT。</td>
      <td>INT32、FLOAT</td>
      <td>ND</td>
      <td>[qLoraDim + keyTotalDim]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias0</td>
      <td>输入</td>
      <td>输入首次做矩阵乘的降维矩阵中的系数。</td>
      <td>支持传入空tensor，quantMode为1、3时不传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[qLoraDim + keyTotalDim]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma1</td>
      <td>输入</td>
      <td>第二次RmsNorm计算中的γ参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[qLoraDim]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>输入</td>
      <td>第二次RmsNorm计算中的β参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[qLoraDim]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantScale1</td>
      <td>输入</td>
      <td>第二次RmsNorm公式中量化缩放的参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantOffset1</td>
      <td>输入</td>
      <td>第二次RmsNorm公式中的量化偏移参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>wuq</td>
      <td>输入</td>
      <td>权重矩阵。</td>
      <td>-</td>
      <td>INT8、FLOAT16、BFLOAT16</td>
      <td>NZ</td>
      <td>[headNum * (qNoRopeDim + qRopeDim),qLoraDim]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deScale1</td>
      <td>输入</td>
      <td>参与wuq矩阵乘的系数。</td>
      <td>input输入dtype为FLOAT16支持INT64，输入BFLOAT16时支持FLOAT。</td>
      <td>INT64、FLOAT</td>
      <td>ND</td>
      <td>[headNum * (qNoRopeDim + qRopeDim)]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias1</td>
      <td>输入</td>
      <td>参与wuq矩阵乘的系数。</td>
      <td>quantMode为1、3时不传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[headNum * (qNoRopeDim + qRopeDim)]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma2</td>
      <td>输入</td>
      <td>参与RmsNormAndreshapeAndCache计算的γ参数。</td>
      <td>数据类型需要与input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[512]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>用于计算旋转位置编码的正弦参数矩阵。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[tokenNum,64]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>用于计算旋转位置编码的余弦参数矩阵。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[tokenNum,64]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>wuk</td>
      <td>输入</td>
      <td>表示计算Key的上采样权重。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[headNum,qNoRopeDim,512]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvCache</td>
      <td>输入</td>
      <td>与输出的kvCacheOut为同一tensor。</td>
      <td>输入格式随cacheMode变化：<ul>
        <li>cacheMode为0：shape为[blockNum,blockSize,1,576]，dtype与input保持一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为1：shape为[blockNum,blockSize,1,512]，tensor的shape为拆分情况，dtype与input保持一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为2：shape为[blockNum,16,blockSize,32]，dtype为int8，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为NZ。</li>
        <li>cacheMode为3：shape为[blockNum,32,blockSize,16]，dtype与input保持一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为NZ。</li>
        <li>仅支持首轴非连续</li></ul>
      </td>
      <td>INT8、FLOAT16、BFLOAT16</td>
      <td>ND、NZ</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kvCacheRope</td>
      <td>输入</td>
      <td>与输出的krCacheOut为同一tensor。</td>
      <td>可选参数，支出传入空指针，输入格式随cacheMode变化：<ul>
        <li>cacheMode为0：不传入。</li>
        <li>cacheMode为1：shape为[blockNum,blockSize,1,64]，dtype与input保持一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为2或3：shape为[blockNum,4,blockSize,16]，dtype与input保持一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为NZ。</li>
        <li>仅支持首轴非连续</li></ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND、NZ</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>slotMapping</td>
      <td>输入</td>
      <td>表示用于存储kv_cache和kr_cache的索引。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[tokenNum]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ctkvScale</td>
      <td>输入</td>
      <td>输出量化处理中参与计算的系数</td>
      <td>仅在cacheMode为2时传入。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qNopeScale</td>
      <td>输入</td>
      <td>输出量化处理中参与计算的系数。</td>
      <td>仅在cacheMode为2时传入</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>wdqDim</td>
      <td>输入</td>
      <td>表示经过matmul后拆分的dim大小。</td>
      <td>预留参数，目前只支持1536。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qRopeDim</td>
      <td>输入</td>
      <td>表示q传入rope的dim大小。</td>
      <td>预留参数，目前只支持64。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kRopeDim</td>
      <td>输入</td>
      <td>表示k传入rope的dim大小。</td>
      <td>预留参数，目前只支持64。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>表示加在分母上防止除0。</td>
      <td>-</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qRotaryCoeff</td>
      <td>输入</td>
      <td>表示q旋转系数。</td>
      <td>预留参数，目前只支持2。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kRotaryCoeff</td>
      <td>输入</td>
      <td>表示k旋转系数。</td>
      <td>预留参数，目前只支持2。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeWdq</td>
      <td>输入</td>
      <td>表示wdq是否转置。</td>
      <td>预留参数，目前只支持true。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeWuq</td>
      <td>输入</td>
      <td>表示wuq是否转置。</td>
      <td>预留参数，目前只支持true。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeWuk</td>
      <td>输入</td>
      <td>表示wuk是否转置。</td>
      <td>预留参数，目前只支持true。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheMode</td>
      <td>输入</td>
      <td>表示指定cache的类型。</td>
      <td><ul>
        <li>0：kcache和q均经过拼接后输出。</li>
        <li>1：输出的kvCacheOut拆分为kvCacheOut和krCacheOut，qOut拆分为qOut和qRopeOut。</li>
        <li>2：krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型。</li>
        <li>3：krope和ctkv转为NZ格式输出。</li></ul>
      </td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>输入</td>
      <td>表示指定RmsNorm量化的类型。</td>
      <td><ul>
        <li>0：per_tensor静态非对称量化，默认量化类型。</li>
        <li>1：per_token动态对称量化，未实现。</li>
        <li>2：per_token动态非对称量化，未实现。</li>
        <li>3：不量化，浮点输出，未实现。</li></ul>
      </td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>doRmsNorm</td>
      <td>输入</td>
      <td>控制对输入tensor做RmsNormQuant或者做Quant。</td>
      <td><ul>
        <li>false：输入tensor只做Quant不做RmsNorm</li>
        <li>true：输入tensor做RmsNormQuant操作。</li></ul>
      </td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>wdkvSplitCount</td>
      <td>输入</td>
      <td>表示指定wdkv拆分的个数。</td>
      <td>支持[1-3]，分别表示不拆分、拆分为2个、拆分为3个降维矩阵。预留参数，目前只支持1。</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 返回值说明

  <table style="undefined;table-layout: fixed; width: 1700px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 301px">
  <col style="width: 331px">
  <col style="width: 237px">
  <col style="width: 111px">
  <col style="width: 170px">
  <col style="width: 150px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>qOut</td>
      <td>输出</td>
      <td>表示Query的输出tensor，对应计算流图中右侧经过NOPE和矩阵乘后的输出。</td>
      <td>shape和dtype随cacheMode变化：<ul>
        <li>cacheMode为0：shape为[tokenNum, headNum, 576]，dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为1或3：shape为[tokenNum, headNum, 512]，dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为2：shape为[tokenNum, headNum, 512]，dtype为INT8，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND格式。</li></ul>
      </td>
      <td>INT8、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvCacheOut</td>
      <td>输出</td>
      <td>表示Key经过ReshapeAndCache后的输出。</td>
      <td>shape和dtype随cacheMode变化：<ul>
        <li>cacheMode为0：shape为[blockNum, blockSize, 1, 576]， dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为1：shape为[blockNum, blockSize, 1, 512]， dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为2：shape为[blockNum, 16, blockSize, 32]，dtype为INT8，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为NZ。</li>
        <li>cacheMode为3：shape为[blockNum, 32, blockSize, 16]，dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为NZ。</li></ul>
      </td>
      <td>INT8、FLOAT16、BFLOAT16</td>
      <td>ND、NZ</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qRopeOut</td>
      <td>输出</td>
      <td>表示Query经过旋转编码后的输出。</td>
      <td>shape和dtype随cacheMode变化：<ul>
        <li>cacheMode为0：不输出。</li>
        <li>cacheMode为1或3：shape为[tokenNum, headNum, 64]，dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为2：shape为[tokenNum, headNum, 64]，dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li></ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>krCacheOut</td>
      <td>输出</td>
      <td>表示Key经过ROPE和ReshapeAndCache后的输出。</td>
      <td>shape和dtype随cacheMode变化：<ul>
        <li>cacheMode为0：不输出。</li>
        <li>cacheMode为1：shape为[blockNum, blockSize, 1, 64]，dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为ND。</li>
        <li>cacheMode为2或3：shape为[blockNum, 4, blockSize, 16]，dtype与input一致，<a href="../../../docs/zh/context/数据格式.md">数据格式</a>为NZ。</li></ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND、NZ</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- shape格式字段含义及约束
    - tokenNum：tokenNum表示输入样本批量大小，取值范围：0~256
    - hiddenSize：hiddenSize表示隐藏层的大小，取值固定为：2048~10240，为256的倍数
    - headNum：表示多头数，取值范围：1~128
    - blockNum：PagedAttention场景下的块数，取值范围：192
    - blockSize：PagedAttention场景下的块大小，取值范围：128
    - qloraDim：表示Q矩阵的LoRA输入维度，取值范围：32~4096，为32的倍数
    - keyTotalDim：表示Key部分的总维度，取值固定为：576（512主维度+64 rope维度）
    - qRopeDim：表示Q矩阵中旋转编码部分的维度，取值固定为：64
    - qNoRopeDim：表示Q矩阵中无旋转编码部分的维度，取值范围：16~256，为16的倍数
- rope模式约束
    - mla_preprocess算子中的Rotary Embedding（RoPE）操作采用half模式，暂不支持interleave模式

## 调用示例

### cacheMode=0（Key+Rope 合并存储）

```python
import torch
import torch_npu
from cann_ops_transformer.ops.mla_preprocess import npu_mla_preprocess

torch_npu.npu.set_device(0)
dtype = torch.bfloat16

tokenNum, hiddenSize = 8, 7168
headNum, nopeDim = 32, 128
keyTotalDim, kRopeDim = 576, 64
blockNum, blockSize = 192, 128

# 准备输入
inp = torch.randn(tokenNum, hiddenSize, dtype=dtype, device="npu")
g0 = torch.randn(hiddenSize, dtype=dtype, device="npu") + 1.0
b0 = torch.randn(hiddenSize, dtype=dtype, device="npu") * 0.1
qs0 = torch.randn(1, dtype=dtype, device="npu").abs() + 0.005
qo0 = torch.zeros(1, dtype=torch.int8, device="npu")

# 权重必须为 FRACTAL_NZ 格式
wdq = torch.randint(-3, 3, (1536 + keyTotalDim, hiddenSize), dtype=torch.int8)
wdq_nz = torch_npu.empty_with_format(wdq.shape, dtype=torch.int8, device="npu",
                                      acl_format=torch_npu.Format.FRACTAL_NZ)
wdq_nz.copy_(wdq)
ds0 = torch.randn(1536 + keyTotalDim, dtype=torch.float32, device="npu").abs() + 0.001
bi0 = torch.randint(-2, 2, (1536 + keyTotalDim,), dtype=torch.int32, device="npu")

g1 = torch.randn(1536, dtype=dtype, device="npu") + 1.0
b1 = torch.randn(1536, dtype=dtype, device="npu") * 0.1
qs1 = torch.randn(1, dtype=dtype, device="npu").abs() + 0.005
qo1 = torch.zeros(1, dtype=torch.int8, device="npu")

wu = torch.randint(-3, 3, (headNum * (nopeDim + kRopeDim), 1536), dtype=torch.int8)
wu_nz = torch_npu.empty_with_format(wu.shape, dtype=torch.int8, device="npu",
                                     acl_format=torch_npu.Format.FRACTAL_NZ)
wu_nz.copy_(wu)
ds1 = torch.randn(headNum * (nopeDim + kRopeDim), dtype=torch.float32, device="npu").abs() + 0.001
bi1 = torch.randint(-2, 2, (headNum * (nopeDim + kRopeDim),), dtype=torch.int32, device="npu")

g2 = torch.randn(512, dtype=dtype, device="npu") + 1.0
cos = torch.randn(tokenNum, kRopeDim, dtype=dtype, device="npu")
sin = torch.randn(tokenNum, kRopeDim, dtype=dtype, device="npu")
wk = torch.randn(headNum, nopeDim, 512, dtype=dtype, device="npu")

# kvCache: [blockNum, blockSize, 1, keyTotalDim]
kv_cache = torch.zeros(blockNum, blockSize, 1, keyTotalDim, dtype=dtype, device="npu")
slot_mapping = torch.tensor([2350, 23251, 11000, 13627, 21896, 6332, 2534, 11263],
                            dtype=torch.int32, device="npu")

q_out, kv_out, qr_out, kr_out = npu_mla_preprocess(
    inp, g0, b0, qs0, qo0,
    wdq_nz, ds0, bi0,
    g1, b1, qs1, qo1,
    wu_nz, ds1, bi1,
    g2, cos, sin, wk,
    kv_cache, slot_mapping,
    cacheMode=0,
)
torch.npu.synchronize()

print(f"qOut:     {q_out.shape}")       # [8, 32, 576]
print(f"kvOut:    {kv_out.shape}")      # [192, 128, 1, 576]
print(f"qRopeOut: {qr_out.shape}")      # [0]
print(f"krOut:    {kr_out.shape}")      # [0]
print(f"in-place: {kv_out.data_ptr() == kv_cache.data_ptr()}")  # True
```

### cacheMode=1（Key/Rope 分离存储，ND 格式）

```python
# 复用上例中的 inp/g0~g2/wdq_nz/wu_nz/wk/cos/sin/slot_mapping 等公共输入
kv_dim = keyTotalDim - kRopeDim  # = 512

kv_cache = torch.zeros(blockNum, blockSize, 1, kv_dim, dtype=dtype, device="npu")
kp_cache = torch.zeros(blockNum, blockSize, 1, kRopeDim, dtype=dtype, device="npu")

q_out, kv_out, qr_out, kr_out = npu_mla_preprocess(
    inp, g0, b0, qs0, qo0,
    wdq_nz, ds0, bi0,
    g1, b1, qs1, qo1,
    wu_nz, ds1, bi1,
    g2, cos, sin, wk,
    kv_cache, slot_mapping,
    kvCacheRope=kp_cache,
    cacheMode=1,
)
torch.npu.synchronize()

print(f"qOut:     {q_out.shape}")       # [8, 32, 512]
print(f"kvOut:    {kv_out.shape}")      # [192, 128, 1, 512]
print(f"qRopeOut: {qr_out.shape}")      # [8, 32, 64]
print(f"krOut:    {kr_out.shape}")      # [192, 128, 1, 64]
```

### cacheMode=1 配合非连续 kvCache（分页注意力场景）

```python
import math

# 构造交替存储的 key/value cache，模拟分页注意力中的内存布局
hidden = blockSize * 1 * kv_dim        # = 65536
hidden_rope = blockSize * 1 * kRopeDim  # = 8192

# kvCache：分配双倍空间，用 as_strided 在相邻 block 间制造间隙
buf = torch.zeros(2 * blockNum * hidden, dtype=dtype, device="npu")
kv = buf.view(2, blockNum, blockSize, 1, kv_dim)
kv.as_strided_(size=kv.shape, stride=(hidden, 2 * hidden, *kv.stride()[2:]))
kv_cache_nc, _ = kv.unbind(0)

# kvCacheRope：同理
buf_rope = torch.zeros(2 * blockNum * hidden_rope, dtype=dtype, device="npu")
kvr = buf_rope.view(2, blockNum, blockSize, 1, kRopeDim)
kvr.as_strided_(size=kvr.shape, stride=(hidden_rope, 2 * hidden_rope, *kvr.stride()[2:]))
kp_cache_nc, _ = kvr.unbind(0)

print(f"kvCache is_contiguous: {kv_cache_nc.is_contiguous()}")     # False
print(f"kvCache stride[0]:    {kv_cache_nc.stride(0)}")            # 131072（2 × 65536）

# 直接传入非连续张量，算子自动处理 stride
q_out, kv_out, qr_out, kr_out = npu_mla_preprocess(
    inp, g0, b0, qs0, qo0,
    wdq_nz, ds0, bi0,
    g1, b1, qs1, qo1,
    wu_nz, ds1, bi1,
    g2, cos, sin, wk,
    kv_cache_nc, slot_mapping,
    kvCacheRope=kp_cache_nc,
    cacheMode=1,
)
torch.npu.synchronize()

# 输出保留原始非连续布局
print(f"kvOut is_contiguous: {kv_out.is_contiguous()}")  # False
print(f"krOut is_contiguous: {kr_out.is_contiguous()}")  # False
```

### cacheMode=2（量化 int8，NZ 格式）

```python
# 准备 NZ 格式的 kvCache 和 kvCacheRope
kv_cache_nz = torch_npu.empty_with_format(
    (blockNum, 16, blockSize, 32), dtype=torch.int8, device="npu",
    acl_format=torch_npu.Format.FRACTAL_NZ)
kp_cache_nz = torch_npu.empty_with_format(
    (blockNum, 4, blockSize, 16), dtype=dtype, device="npu",
    acl_format=torch_npu.Format.FRACTAL_NZ)

ctkv_scale = torch.randn(1, dtype=dtype, device="npu").abs() + 0.01
qnope_scale = torch.randn(headNum, dtype=dtype, device="npu").abs() + 0.001

q_out, kv_out, qr_out, kr_out = npu_mla_preprocess(
    inp, g0, b0, qs0, qo0,
    wdq_nz, ds0, bi0,
    g1, b1, qs1, qo1,
    wu_nz, ds1, bi1,
    g2, cos, sin, wk,
    kv_cache_nz, slot_mapping,
    kvCacheRope=kp_cache_nz,
    ctkvScale=ctkv_scale,
    qNopeScale=qnope_scale,
    cacheMode=2,
)
torch.npu.synchronize()

print(f"qOut:  {q_out.shape}")   # [8, 32, 512]，dtype=int8
print(f"kvOut: {kv_out.shape}")  # [192, 16, 128, 32]，dtype=int8
print(f"krOut: {kr_out.shape}")  # [192, 4, 128, 16]，dtype=bfloat16
```
