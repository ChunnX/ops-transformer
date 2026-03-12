# aclnnMhcPreBackward

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |


## 功能说明

-   **接口功能**：MhcPreBackward是MhcPre的反向算子，mhc_pre基于一系列计算得到MHC架构中hidden层的Hres和Hpost投影矩阵以及Atten或MLP层的输入矩阵Hin。

-   **计算公式**：公式如下：


### 输入
- 算子输入

    $x ∈ R[B,S,N,D]$ 

    $phi ∈ R[ND,2N+N*N]$

    $gamma ∈ R[N, D]$

    $alpha ∈ R[3]$

    $hc\_eps$ 

    $H\_in\_grad ∈ R[B,S,D]$ （输出梯度）

    $H\_post\_grad ∈ R[B,S,N]$ （输出梯度）

    $H\_res\_grad ∈ R[B,S,N,N]$ （输出梯度）


- 前向缓存张量

    $H\_mix ∈ R[B,S,2N+N^2]$ 

    $H\_pre∈ R[B,S,N]$

    $H\_post∈ R[B,S,N]$ 

    $inv\_rms ∈ R[B,S,1]$ （或 $R[B,S]$ ，需unsqueeze转 $R[B,S,1]$ ）

- 重计算张量

    $[H\_pre\_1,H\_post\_1,H\_res\_2] =  H\_mix * inv\_rms  ∈ R[B,S,N^2+2N]$ 

- 张量切分

    $[alpha\_{pre}, alpha\_{post},alpha\_{res}] = alpha ∈ R[3]$ 

### 输出

- 梯度输出：

    $x\_{grad} ∈ R[B,S,N,D]$ 

    $phi\_{grad} ∈ R[ND,2N+N^2]$ 

    $alpha\_{scales\_grad} ∈ R[3]$ （ $alpha\_pre\_grad, alpha\_post\_grad, alpha\_res\_grad$ 拼接）

    $bias\_{grad} ∈ R[2N+N^2]$ （ $bias\_pre\_grad, bias\_post\_grad, bias\_res\_grad$ 拼接）
    
    $gamma\_grad ∈ [N, D]$

### 输出组合梯度计算

- 正向公式：
$$H\_in = \sum_{i=1}^{N} x[{B,S,i,:}] · H\_pre_n[B,S,i]$$

- 反向计算：
$$
\begin{aligned}
H\_pre\_grad &= \text{Reduce}\left(H\_in\_grad.\text{unsqueeze}(-2) \odot x, \text{dim}=-1\right) \quad ([B,S,N]) \\
x\_grad\_vec3 &= H\_in\_grad \times H\_pre \quad ([B,S,N,D])
\end{aligned}
$$

### 门控激活层梯度计算

#### Sigmoid门控反向（H_pre）
- 正向公式：
$$H\_pre = \text{Sigmoid}(\alpha\_pre * H\_pre\_1 + bias\_pre) + hc\_eps$$

- 反向计算：
$$
\begin{aligned}
s &= H\_pre - hc\_eps \\
H\_pre\_2\_grad &= H\_pre\_grad \odot s \odot (1 - s) \\
H\_pre\_1\_grad &= H\_pre\_2\_grad \cdot \alpha\_pre \\
\alpha\_pre\_grad &= \sum_{b,s,n}^{B,S,N} \left(H\_pre\_2\_grad \cdot H\_pre\_1\right) \\
bias\_pre\_grad &= \sum_{b,s}^{B,S} H\_pre\_2\_grad \quad ([N])
\end{aligned}
$$

#### Sigmoid门控反向（H_post）
- 正向公式：
$$H\_post = \text{Sigmoid}(\alpha\_post * H\_post\_1 + bias\_post) * 2$$

- 反向计算：
$$
\begin{aligned}
H\_post\_2\_grad &= H\_post\_grad \odot \left(H\_post \cdot \left(1 - \frac{H\_post}{2}\right)\right) \\
H\_post\_1\_grad &= H\_post\_2\_grad \cdot \alpha_{post} \\
\alpha_{post\_grad} &= \sum_{b,s,n}^{B,S,N} \left(H\_post\_2\_grad \cdot H_{post\_1}\right) \\
bias\_post\_grad &= \sum_{b,s}^{B,S} H\_post\_2\_grad \quad ([N])
\end{aligned}
$$

#### 残差连接反向（H_res）
- 正向公式：
$$H\_res = \alpha\_res * H\_res\_1 + bias\_res$$

- 反向计算：
$$
\begin{aligned}
H\_res\_2\_grad &= H\_res\_grad \cdot \alpha_{res} \quad ([B,S,N,N]) \\
\alpha\_res\_grad &= \sum_{b,s,i,j}^{B,S,N,N} \left(H\_res\_grad \cdot H\_res\_2\right) \\
bias\_res\_grad &= \sum_{b,s}^{B,S} H\_res\_grad \quad ([N,N]) \\
H\_res\_1\_grad &= \text{Reshape}(H\_res\_2\_grad) \quad ([B,S,N^2])
\end{aligned}
$$

### RMS归一化融合反向

#### RMSNorm Fusion反向
- 正向公式：
$$H\_mix\_tmp = H\_mix * inv\_rms$$

- 反向计算：
$$
\begin{aligned}
H\_mix\_tmp\_grad &= \text{Concat}(H\_pre\_1\_grad, H\_post\_1\_grad, H\_res\_1\_grad) \quad ([B,S,2N+N^2]) \\
H\_mix\_grad &= H\_mix\_tmp\_grad \cdot inv\_rms \\
inv\_rms_{grad} &= \sum_{\text{last\_dim}} \left(H\_mix\_tmp\_grad \cdot H\_mix\right) \quad ([B,S,1])
\end{aligned}
$$

### 线性投影层梯度计算

#### 矩阵乘法反向
- 正向公式：
$$H\_mix = x\_rs @ phi^T$$
$$x\_rs = x * gamma$$

- 反向计算：
$$
\begin{aligned}
x\_rs\_grad &= H\_mix\_grad @ phi \quad ([B,S,ND]) \\
X &= \text{Reshape}(x\_rs, [B\cdot S, ND]) \\
G &= \text{Reshape}(H\_mix\_grad, [B\cdot S, 2N+N^2]) \\
phi_{grad} &= G^T @ X \quad ([2N+N^2, ND])
\end{aligned}
$$

#### 特征缩放反向
- 正向公式：
$$x\_rs = x * gamma$$

- 反向计算：
$$
\begin{aligned}
x\_grad\_mm &= x\_rs\_grad * gamma \\
gamma\_grad &= \sum_{b=1}^{B}\sum_{s=1}^{S} (x * x\_rs\_grad)\quad ([N,D])
\end{aligned}
$$

### RMS归一化梯度计算

- 正向公式：
$$inv\_rms = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + eps}}, \quad 其中\ n = N * D$$

- 反向计算：
$$
\begin{aligned}
x\_rs\_grad\_inv &= - \left(\frac{inv\_rms\_grad \cdot {inv\_rms}^3}{N*D}\right) \cdot x\_rs \\
x\_rs\_grad &= x\_grad\_mm + x\_rs\_grad\_inv \\
x\_grad\_vec1 &= \text{Reshape}(x\_rs\_grad, [B,S,N,D]) \\
x\_grad &= x\_grad\_vec3 + x\_grad\_vec1
\end{aligned}
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMhcPreBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMhcPreBackward”接口执行计算。
```c++
aclnnStatus aclnnMhcPreBackwardGetWorkspaceSize(
    const aclTensor     *x,
    const aclTensor     *phi,
    const aclTensor     *alpha,
    const aclTensor     *h_in_grad,
    const aclTensor     *h_post_grad,
    const aclTensor     *h_res_grad,
    const aclTensor     *inv_rms,
    const aclTensor     *h_mix,
    const aclTensor     *h_pre,
    const aclTensor     *h_post,
    const aclTensor     *gamma,
    double               hc_eps,
    const aclTensor     *x_grad,
    const aclTensor     *phi_grad,
    const aclTensor     *alpha_grad,
    const aclTensor     *bias_grad,
    const aclTensor     *gamma_grad,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```
```c++
aclnnStatus aclnnMhcPreBackward(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream stream)
```

## aclnnMhcPreBackwardGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">  
            <col style="width: 400px">  
            <col style="width: 212px">  
            <col style="width: 100px">
            <col style="width: 290px">
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
            <td>待计算数据，表示网络中mHC层的输入数据。</td>
            <td><li>不支持空Tensor。</li></td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S,N,D)、(T,N,D)<br>
            B：支持泛化；S：支持泛化；T：B*S；N：4、6、8；D：128元素对齐
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>phi</td>
            <td>输入</td>
            <td>mHC的参数矩阵。</td>
            <td><li>不支持空Tensor。</li></td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(2N+N*N,N*D)<br>
            N:与x的N保持一致；D:与x的D保持一致
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>alpha</td>
            <td>输入</td>
            <td>mHC的缩放参数alpha。</td>
            <td><li>不支持空Tensor。</li></td>
            <td>FLOAT32</td>
            <td>-</td>
            <td>(3)
            <td>-</td>
        </tr>
        <tr>
            <td>h_in_grad</td>
            <td>输入</td>
            <td>h_in作为Atten/MLP层的输入。正向输出h_in对应的梯度。</td>
            <td>
            <li>不支持空Tensor。</li>
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S,D)、(T,D)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S；D:与x的D保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>h_post_grad</td>
            <td>输入</td>
            <td>正向输出h_post对应的梯度。</td>
            <td>
            <li>不支持空Tensor。</li>
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S,N)、(T,N)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S；N:与x的N保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>h_res_grad</td>
            <td>输入</td>
            <td>正向输出h_res对应的梯度。</td>
            <td>
            <li>不支持空Tensor。</li>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S,N,N)、(T,N,N)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S；N:与x的N保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>inv_rms</td>
            <td>输入</td>
            <td>正向RmsNorm计算的inv\_rms。</td>
            <td>
            <li>不支持空Tensor。</li>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S)、(T)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>h_mix</td>
            <td>输入</td>
            <td>正向计算流x@phi的结果</td>
            <td>
            <li>不支持空Tensor。</li>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S,2N+N*N)、(T,2N+N*N)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S；N:与x的N保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>h_pre</td>
            <td>输入</td>
            <td>正向sigmoid计算之后的h_pre矩阵</td>
            <td>
            <li>不支持空Tensor。</li>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S,N)、(T,N)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S；N:与x的N保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>h_post</td>
            <td>输入</td>
            <td>正向的h_post输出</td>
            <td>
            <li>不支持空Tensor。</li>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S,N)、(T,N)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S；N:与x的N保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>gamma</td>
            <td>输入</td>
            <td>RmsNorm的缩放系数gamma</td>
            <td>
            <li>不支持空Tensor。</li><li>如果传入nullptr，则表示全1的tensor。</li>
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N,D)<br>
            N:与x的N保持一致；D:与x的D保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>hc_eps</td>
            <td>输入</td>
            <td>H_pre的sigmoid后的eps参数</td>
            <td>
           <li>默认值为1e-6
            </td>
            <td>FLOAT32</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>x_grad</td>
            <td>输出</td>
            <td>x对应的梯度。</td>
            <td>
            <li>与输入x的维度、数据类型保持一致</td>
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S,N,D)、(T,N,D)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S；N:与x的N保持一致；D：与x的D保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>phi_grad</td>
            <td>输出</td>
            <td>phi对应的梯度。</td>
            <td>
            <li>与输入phi的维度保持一致</td>
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S,2N+N*N)、(T,2N+N*N)<br>
            B：与x的B保持一致；S:与x的S保持一致；T：B*S；N:与x的N保持一致。
            </td>
            <td>√</td>
        </tr>
        <tr>  
            <td>alpha_grad</td>
            <td>输出</td>
            <td>alpha对应的梯度。</td>
            <td>
            <li>与输入alpha维度保持一致。</td>
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(3)
            <td>-</td>
        </tr>
          <tr>
            <td>bias_grad</td>
            <td>输出</td>
            <td>bias对应的梯度。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(2N+N*N)<br>
            N:与x的N保持一致。
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>gamma_grad</td>
            <td>输出</td>
            <td>gamma对应的梯度。</td>
            <td>
                <li>当输入gamma不为nullptr时，此变量才会输出。</li>
                <li>与输入gamma的Shape维度保持一致。</li>
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N,D)<br>
            N:与x的N保持一致；D:与x的D保持一致。
            </td>
            <td>√</td>
        </tr>
        </tbody>
    </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>必选参数或者输出是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入变量，如x、phi、gamma、alpha……的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>


## aclnnMhcPreBackward

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPreBackwardGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream流。</td>
        </tr>
    </tbody>
    </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnMhcPreBackward默认采用确定性实现，相同输入多次调用结果一致。
- 公共约束
    - x等输入参数不能为空Tensor，否则返回失败。
    - 所有输入/输出Tensor的数据格式仅支持`ACL_FORMAT_ND`；

- 规格约束
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>T或 B*S</td>
            <td>512~65536</td>
            <td>B、S的乘积与T支持512~65536范围以内</td>
        </tr>
        <tr>
            <td>n</td>
            <td>4， 6 ，8</td>
            <td>n值目前支持4， 6， 8</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512~16384</td>
            <td>D支持512~65536范围以内</td>
        </tr>
        </tbody>
    </table>
- 典型值
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>典型值</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>T或 B*S</td>
            <td>1024/2048/4096</td>
        </tr>
        <tr>
            <td>n</td>
            <td>n = 4</td>
        </tr>
        <tr>
            <td>D</td>
            <td>2560/5120</td>
        </tr>
        </tbody>
    </table>


## 调用示例

调用示例代码如下（以<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_backward.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<short> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %e\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {1024, 4, 512};         // T, N, D
  std::vector<int64_t> phiShape = {24, 2048};           // 2 * N + N * N, N * D
  std::vector<int64_t> alphaShape = {3};                // 固定大小
  std::vector<int64_t> hInGradShape = {1024, 512};      // T, D
  std::vector<int64_t> hPostGradShape = {1024, 4};      // T, N
  std::vector<int64_t> hResGradShape = {1024, 4, 4};    // T, N, N
  std::vector<int64_t> invRmsShape = {1024};            // T
  std::vector<int64_t> hMixShape = {1024, 24};          // T, 2 * N + N * N
  std::vector<int64_t> hPreShape = {1024, 4};           // T, N
  std::vector<int64_t> hPostShape = {1024, 4};          // T, N
  std::vector<int64_t> gammaShape = {4, 512};           // N, D
  std::vector<int64_t> xGradShape = {1024, 4, 512};     // T, N, D
  std::vector<int64_t> phiGradShape = {24, 2048};       // 2 * N + N * N, N * D
  std::vector<int64_t> alphaGradShape = {3};            // 固定大小
  std::vector<int64_t> biasGradShape = {24};            // 2 * N + N * N
  std::vector<int64_t> gammaGradShape = {4, 512};       // N, D

  void* xDeviceAddr = nullptr;
  void* phiDeviceAddr = nullptr;
  void* alphaDeviceAddr = nullptr;
  void* hInGradDeviceAddr = nullptr;
  void* hPostGradDeviceAddr = nullptr;
  void* hResGradDeviceAddr = nullptr;
  void* invRmsDeviceAddr = nullptr;
  void* hMixDeviceAddr = nullptr;
  void* hPreDeviceAddr = nullptr;
  void* hPostDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* xGradDeviceAddr = nullptr;
  void* phiGradDeviceAddr = nullptr;
  void* alphaGradDeviceAddr = nullptr;
  void* biasGradDeviceAddr = nullptr;
  void* gammaGradDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* phi = nullptr;
  aclTensor* alpha = nullptr;
  aclTensor* hInGrad = nullptr;
  aclTensor* hPostGrad = nullptr;
  aclTensor* hResGrad = nullptr;
  aclTensor* invRms = nullptr;
  aclTensor* hMix = nullptr;
  aclTensor* hPre = nullptr;
  aclTensor* hPost = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* xGrad = nullptr;
  aclTensor* phiGrad = nullptr;
  aclTensor* alphaGrad = nullptr;
  aclTensor* biasGrad = nullptr;
  aclTensor* gammaGrad = nullptr;

  std::vector<short> xHostData(1024 * 4 * 512, 1.0);
  std::vector<float> phiHostData(24 * 2048, 1.0);
  std::vector<float> alphaHostData(3, 1.0);
  std::vector<short> hInGradHostData(1024 * 512, 1.0);
  std::vector<float> hPostGradHostData(1024 * 4, 1.0);
  std::vector<float> hResGradHostData(1024 * 4 * 4, 1.0);
  std::vector<float> invRmsHostData(1024, 1.0);
  std::vector<float> hMixHostData(1024 * 24, 1.0);
  std::vector<float> hPreHostData(1024 * 4, 1.0);
  std::vector<float> hPostHostData(1024 * 4, 1.0);
  std::vector<float> gammaHostData(4 * 512, 1.0);
  std::vector<short> xGradHostData(1024 * 4 * 512, 0);
  std::vector<float> phiGradHostData(24 * 2048, 0);
  std::vector<float> alphaGradHostData(3, 0);
  std::vector<float> biasGradHostData(24, 0);
  std::vector<float> gammaGradHostData(4 * 512, 0);

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(phiHostData, phiShape, &phiDeviceAddr, aclDataType::ACL_FLOAT, &phi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hInGradHostData, h_in_gradShape, &hInGradDeviceAddr, aclDataType::ACL_BF16, &hInGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hPostGradHostData, xGradShape, &hPostGradDeviceAddr, aclDataType::ACL_FLOAT, &hPostGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hResGradHostData, phiGradShape, &hResGradDeviceAddr, aclDataType::ACL_FLOAT, &hResGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(invRmsHostData, hPostGradShape, &invRmsDeviceAddr, aclDataType::ACL_FLOAT, &invRms);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hMixHostData, hResGradShape, &hMixDeviceAddr, aclDataType::ACL_FLOAT, &hMix);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hPreHostData, invRmsShape, &hPreDeviceAddr, aclDataType::ACL_FLOAT, &hPre);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hPostHostData, hMixShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPost);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, hPreShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xGradHostData, hPostShape, &xGradDeviceAddr, aclDataType::ACL_BF16, &xGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(phiGradHostData, xShape, &phiGradDeviceAddr, aclDataType::ACL_FLOAT, &phiGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(alphaGradHostData, phiShape, &alphaGradDeviceAddr, aclDataType::ACL_FLOAT, &alphaGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasGradHostData, alphaShape, &biasGradDeviceAddr, aclDataType::ACL_FLOAT, &biasGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaGradHostData, hPreShape, &gammaGradDeviceAddr, aclDataType::ACL_FLOAT, &gammaGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  float hc_eps = 1e-6;
  
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 80 * 1024 * 1024;
  aclOpExecutor* executor;

  // 调用aclnnMhcPreBackward第一段接口
  ret = aclnnMhcPreBackwardGetWorkspaceSize(x, phi, alpha, hInGrad, hPostGrad, hResGrad, invRms,
        hMix, hPre, hPost, gamma, hc_eps, xGrad, phiGrad, alphaGrad, biasGrad, gammaGrad, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnMhcPreBackward第二段接口
  ret = aclnnMhcPreBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreBackward failed. ERROR: %d\n", ret); return ret);
  
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(xShape, &dxDeviceAddr);
  PrintOutResult(phiShape, &dkDeviceAddr);
  PrintOutResult(alphaShape, &dvDeviceAddr);
  PrintOutResult(biasShape, &dxRopeDeviceAddr);
  PrintOutResult(hPostShape, &dkRopeDeviceAddr);
  
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(phi);
  aclDestroyTensor(alpha);
  aclDestroyTensor(hInGrad);
  aclDestroyTensor(hPostGrad);
  aclDestroyTensor(hResGrad);
  aclDestroyTensor(invRms);
  aclDestroyTensor(hMix);
  aclDestroyTensor(hPre);
  aclDestroyTensor(hPost);
  aclDestroyTensor(gamma);
  aclDestroyTensor(xGrad);
  aclDestroyTensor(phiGrad);
  aclDestroyTensor(alphaGrad);
  aclDestroyTensor(biasGrad);
  aclDestroyTensor(gammaGrad);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(phiDeviceAddr);
  aclrtFree(alphaDeviceAddr);
  aclrtFree(hInGradDeviceAddr);
  aclrtFree(hPostGradDeviceAddr);
  aclrtFree(hResGradDeviceAddr);
  aclrtFree(invRmsDeviceAddr);
  aclrtFree(hMixDeviceAddr);
  aclrtFree(hPreDeviceAddr);
  aclrtFree(hPostDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(xGradDeviceAddr);
  aclrtFree(phiGradDeviceAddr);
  aclrtFree(alphaGradDeviceAddr);
  aclrtFree(biasGradDeviceAddr);
  aclrtFree(gammaGradDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return 0;
}
```