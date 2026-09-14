# aclnnMoeTutelCombineX

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
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

- 接口功能：MoE（Mixture of Experts）Tutel 路由反向 combine 的数据梯度接口。将专家缓冲区输出的梯度 y_grad，按路由元数据 indices/locations 收集回原始 token 位置，并逐 token 副本乘以门控权重 gates，得到输入侧梯度 x_grad；无效路由条目（容量丢弃、溢出容量）对应输出行为零。本接口与前向 MoeTutelDispatch 配套使用，gates/indices/locations 须由前向产出并在反向原样重放。
- 计算公式：

  维度与每专家容量 C 推导（C 不经算子属性传递，由张量维度与路由元数据推导）：

  $$
  EC = y\_grad.\mathrm{shape}[-2],\qquad H = y\_grad.\mathrm{shape}[-1],\qquad S = gates.\mathrm{shape}[-1]
  $$

  $$
  E = \max_{b,t}\big(indices[b,t]\big) + 1,\qquad
  C = \begin{cases}
  \left\lfloor \dfrac{EC}{E} \right\rfloor, & E \ge 1 \\[4pt]
  0, & E = 0\ (\text{indices 全为 } -1)
  \end{cases}
  $$

  逐条目计算（对每个 batch 元素 $b \in [0, B)$ 与 token 副本 $t \in [0, S)$；2D 形态即 $B = 1$，省去 $b$ 下标）：

  $$
  r_{b,t} = indices[b,t] \cdot C + locations[b,t]
  $$

  $$
  valid_{b,t} = \big(indices[b,t] \ge 0\big) \wedge \big(0 \le locations[b,t] < C\big) \wedge \big(0 \le r_{b,t} < EC\big)
  $$

  $$
  x\_grad[b,t,:] = \begin{cases}
  gates[b,t] \cdot y\_grad\big[\,b,\ r_{b,t},\ :\,\big], & valid_{b,t} \\[4pt]
  \mathbf{0}, & \neg\, valid_{b,t}
  \end{cases}
  $$

  其中 $y\_grad$ 为专家缓冲区梯度（$[E, C, H]$ 展平为 $[EC, H]$ 或 $[B, EC, H]$ 连续布局），$gates$ 为门控权重，$indices$ 为命中的专家 ID，$locations$ 为专家缓冲区内槽位，$x\_grad$ 为输出梯度。无效条目对应输出行写精确 0，不产生 NaN 污染；有效行 NaN/±Inf 按 IEEE-754 传播。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnMoeTutelCombineXGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTutelCombineX”接口执行计算。

```cpp
aclnnStatus aclnnMoeTutelCombineXGetWorkspaceSize(
  const aclTensor  *yGrad,
  const aclTensor  *gates,
  const aclTensor  *indices,
  const aclTensor  *locations,
  const aclTensor  *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnMoeTutelCombineX(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnMoeTutelCombineXGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 380px">
  <col style="width: 220px">
  <col style="width: 100px">
  <col style="width: 200px">
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
      <td>yGrad（const aclTensor*）</td>
      <td>输入</td>
      <td>专家缓冲区输出的梯度，对应计算公式中y_grad。</td>
      <td><li>支持空Tensor。</li><li>数据类型须与gates、out一致。</li><li>shape为[EC, H]或[B, EC, H]，rank须满足约束说明中的rank契约。</li></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3（[EC, H]或[B, EC, H]）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gates（const aclTensor*）</td>
      <td>输入</td>
      <td>逐token副本的路由权重（softmax归一化门控值），对应计算公式中gates。</td>
      <td><li>支持空Tensor。</li><li>数据类型与yGrad保持一致。</li><li>shape须与indices、locations一致。</li></td>
      <td>数据类型与yGrad保持一致。</td>
      <td>ND</td>
      <td>1-2（[S]或[B, S]）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices（const aclTensor*）</td>
      <td>输入</td>
      <td>逐token副本命中的专家ID，对应计算公式中indices。</td>
      <td><li>支持空Tensor。</li><li>取值[-1, E-1]，-1表示容量丢弃的路由条目。</li><li>shape须与gates一致。</li></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-2（[S]或[B, S]）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>locations（const aclTensor*）</td>
      <td>输入</td>
      <td>逐token副本在其专家缓冲区内的槽位，对应计算公式中locations。</td>
      <td><li>支持空Tensor。</li><li>取值[0, C-1]，不小于C表示溢出容量的无效条目。</li><li>shape须与gates一致。</li></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-2（[S]或[B, S]）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（const aclTensor*）</td>
      <td>输出</td>
      <td>原始token位置的输入梯度，对应计算公式中x_grad。</td>
      <td><li>支持空Tensor。</li><li>数据类型与yGrad保持一致。</li><li>shape须为gates.shape拼接H，即[S, H]或[B, S, H]。</li></td>
      <td>数据类型与yGrad保持一致。</td>
      <td>ND</td>
      <td>2-3（[S, H]或[B, S, H]）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见aclnn返回码。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
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
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>yGrad、gates、indices、locations、out、workspaceSize、executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="4"> ACLNN_ERR_PARAM_INVALID </td>
      <td rowspan="4"> 161002 </td>
      <td>yGrad的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gates或out的数据类型与yGrad不一致，或indices、locations的数据类型不为INT32。</td>
    </tr>
    <tr>
      <td>yGrad、gates、indices、locations、out的数据格式不为ND。</td>
    </tr>
    <tr>
      <td>shape契约违背：rank(yGrad)不在[2,3]范围内或rank(yGrad)不等于rank(gates)+1；gates、indices、locations三者shape不一致；带batch时gates.dim0与yGrad.dim0不一致；out的shape不等于gates.shape拼接H。</td>
    </tr>
  </tbody></table>

## aclnnMoeTutelCombineX

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
          <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
      </thead>
      <tbody>
          <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
          <tr><td>workspaceSize </td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTutelCombineXGetWorkspaceSize获取。</td></tr>
          <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
          <tr><td>stream </td><td>输入</td><td>指定执行任务的Stream。</td></tr>
      </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束说明

- 确定性说明：aclnnMoeTutelCombineX默认确定性实现。

- 数据类型：仅支持yGrad/gates/out同为FLOAT、FLOAT16或BFLOAT16（三者一致，无dtype提升），indices/locations固定为INT32；跨组合混用（如yGrad为FLOAT而gates为FLOAT16）报错。

- 数据格式：yGrad、gates、indices、locations、out仅支持ND格式。

- shape契约：rank(yGrad)∈{2,3}且rank(yGrad)=rank(gates)+1；gates/indices/locations三者shape全等；带batch（rank(yGrad)=3）时gates.dim0=yGrad.dim0；out的shape=gates.shape拼接yGrad最后一维H，即[S, H]或[B, S, H]。

- 容量C派生：C=floor(EC/E)，其中EC=yGrad.shape[-2]、E=max(indices)+1，不经算子属性传递；E=0（indices全为-1）或E大于EC（此时floor(EC/E)恰为0）时C=0，全部条目无效，输出全零；E=1时C=EC，单专家路由正常取行。

- 无效路由条目（indices=-1容量丢弃、locations为负、locations不小于C溢出容量、源行r越界）：零贡献静默跳过，不报错、不访问越界内存，对应输出行为精确0；y_grad中的NaN/±Inf仅在有效行按IEEE-754传播。

- 空Tensor为合法退化：S=0时输出[0, H]空Tensor；H=0时输出[S, 0]空Tensor；B=0时输出[0, S, H]空Tensor；EC=0且S大于0时C=0，输出[S, H]全零。

- 多副本命中同一token行的求和：本shape契约下每条输出行恰对应一条路由条目（单写覆盖），跨副本求和须由调用方经batch维组织或逐chunk多次调用完成。

- gates/indices/locations须由前向MoeTutelDispatch产出并在反向原样重放，不允许反向独立重算路由。

## 调用示例

示例代码如下，仅供参考（2D无batch形态微型数值例：E=2、C=2、EC=4、H=4、S=2，预期输出x_grad = [[0.5, 1, 1.5, 2], [4, 3, 2, 1]]）：

```Cpp
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnn_moe_tutel_combine_x.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    aclDataType dataType, void** deviceAddr, aclTensor** tensor)
{
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.(固定写法)device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出（2D无batch形态：y_grad [EC, H]，元数据 [S]，out [S, H]）
    std::vector<int64_t> yGradShape = {4, 4};  // y_grad: [EC, H]
    std::vector<int64_t> gatesShape = {2};     // gates/indices/locations: [S]
    std::vector<int64_t> outShape = {2, 4};    // out: [S, H] = gates.shape + (H,)
    void* yGradDeviceAddr = nullptr;
    void* gatesDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* locationsDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* yGrad = nullptr;
    aclTensor* gates = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* locations = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> yGradHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 4, 3, 2, 1};
    std::vector<float> gatesHostData = {0.5f, 1.0f};
    std::vector<int32_t> indicesHostData = {0, 1};     // token副本分别命中专家0槽位0、专家1槽位1
    std::vector<int32_t> locationsHostData = {0, 1};
    std::vector<float> outHostData(8, 0.0f);

    // 创建yGrad aclTensor
    ret = CreateAclTensor(yGradHostData, yGradShape, aclDataType::ACL_FLOAT, &yGradDeviceAddr, &yGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gates aclTensor
    ret = CreateAclTensor(gatesHostData, gatesShape, aclDataType::ACL_FLOAT, &gatesDeviceAddr, &gates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建indices aclTensor
    ret = CreateAclTensor(indicesHostData, gatesShape, aclDataType::ACL_INT32, &indicesDeviceAddr, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建locations aclTensor
    ret = CreateAclTensor(locationsHostData, gatesShape, aclDataType::ACL_INT32, &locationsDeviceAddr, &locations);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, &outDeviceAddr, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库两段式接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeTutelCombineX第一段接口
    ret = aclnnMoeTutelCombineXGetWorkspaceSize(yGrad, gates, indices, locations, out,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACLNN_SUCCESS,
              LOG_PRINT("aclnnMoeTutelCombineXGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnMoeTutelCombineX第二段接口
    ret = aclnnMoeTutelCombineX(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("aclnnMoeTutelCombineX failed. ERROR: %d\n", ret); return ret);

    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧
    // 预期结果：x_grad[0,:] = 0.5 * y_grad[0*2+0, :] = [0.5, 1, 1.5, 2]
    //           x_grad[1,:] = 1.0 * y_grad[1*2+1, :] = [4, 3, 2, 1]
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.释放aclTensor
    aclDestroyTensor(yGrad);
    aclDestroyTensor(gates);
    aclDestroyTensor(indices);
    aclDestroyTensor(locations);
    aclDestroyTensor(out);

    // 7.释放device资源
    aclrtFree(yGradDeviceAddr);
    aclrtFree(gatesDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(locationsDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

编译与运行（x86_64服务器，需已安装CANN开发套件并完成本自定义算子包部署，`<op_package_path>`为算子包安装路径）：

```bash
# 编译示例
g++ -std=c++17 demo.cpp -o demo \
    -I${ASCEND_HOME_PATH}/include \
    -I${ASCEND_HOME_PATH}/x86_64-linux/include \
    -I<op_package_path>/vendors/MoeTutelCombineX/op_api/include \
    -L${ASCEND_HOME_PATH}/lib64 \
    -L<op_package_path>/vendors/MoeTutelCombineX/op_api/lib \
    -lascendcl -lnnopbase -lcust_opapi \
    -Wl,-rpath,<op_package_path>/vendors/MoeTutelCombineX/op_api/lib

# 运行前设置自定义算子加载路径
export ASCEND_CUSTOM_OPP_PATH=<op_package_path>/vendors/MoeTutelCombineX
export LD_LIBRARY_PATH=<op_package_path>/vendors/MoeTutelCombineX/op_api/lib:${LD_LIBRARY_PATH}

# 执行
./demo
```
