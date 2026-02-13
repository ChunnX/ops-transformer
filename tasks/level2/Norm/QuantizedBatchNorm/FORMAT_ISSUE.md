# QuantizedBatchNorm 格式问题总结

## 问题描述

在迁移 `quantized_batch_norm` 算子到 NPUKernelBench 框架时，遇到了格式转换问题：虽然我们在 `ConvertType` 函数中设置了 `ACL_FORMAT_NCHW`，但实际创建的 `aclTensor` 格式仍然是 `ACL_FORMAT_ND`，导致精度测试失败。

## 错误信息

```
RuntimeError: call aclnnQuantizedBatchNorm failed, detail:[PID: 1919040] 2026-02-03-01:36:42.916.477 
AclNN_Parameter_Error(EZ1001): Io input dtype or format is not supported, 
get io input info is x(DT_INT8, ND) mean(DT_FLOAT, ND) var(DT_FLOAT, ND) ... out(DT_INT8, ND) 
but supported list is:
    SupportInfo[0] x(DT_INT8, NCHW) mean(DT_FLOAT, ND) ... out(DT_INT8, NCHW)
    SupportInfo[1] x(DT_UINT8, NCHW) mean(DT_FLOAT, ND) ... out(DT_UINT8, NCHW)
    SupportInfo[2] x(DT_INT32, NCHW) mean(DT_FLOAT, ND) ... out(DT_INT32, NCHW)
```

**关键信息：**
- **实际传入格式**：`x(DT_INT8, ND)` - 格式是 `ND`
- **算子期望格式**：`x(DT_INT8, NCHW)` - 格式应该是 `NCHW`

## 算子格式要求

根据 `quantized_batch_norm_def.cpp` 和算子实现：

- **输入 `x`**：要求 `FORMAT_NCHW`（4维张量）
- **输出 `y`**：要求 `FORMAT_NCHW`（4维张量）
- **其他输入**（mean, var, scales等）：`FORMAT_ND`（1维张量）

## 已尝试的解决方案

### 1. 修复 `storage_dims` 参数使用

**问题**：原始 `ConvertType` 使用 storage 大小（单个元素）作为 `storage_dims`，而示例代码使用 shape 数组。

**解决方案**：
- 创建了 `pytorch_npu_helper_fixed.hpp`，修复 `ConvertType` 实现
- 使用 tensor shape 作为 `storage_dims`（而不是 storage size）
- 使用相同的指针给 `view_dims` 和 `storage_dims`（参考 `aclnnNpuFormatCast.md`）

**代码位置**：`/home/xutianhao/CANN/NPUKernelBench/tasks/level2/Norm/QuantizedBatchNorm/validation/pytorch_npu_helper_fixed.hpp`

**关键修改**：
```cpp
// 修复前（原始框架代码）
storageDims.push_back(at_tensor.storage().nbytes() / itemsize);  // 单个元素

// 修复后
std::vector<int64_t> shape_vec(sizes.begin(), sizes.end());
// 使用 shape_vec.data() 作为 storage_dims
```

### 2. 使用计算的连续 strides

**问题**：使用 PyTorch tensor 的原始 strides 可能影响格式识别。

**解决方案**：
- 计算连续的 strides（参考示例代码）
- 使用 offset 0（而不是 tensor 的 storage_offset）

**代码修改**：
```cpp
// 计算连续 strides
std::vector<int64_t> strides_vec(shape_vec.size(), 1);
for (int64_t i = shape_vec.size() - 2; i >= 0; i--) {
    strides_vec[i] = shape_vec[i + 1] * strides_vec[i + 1];
}

// 使用计算的 strides 和 offset 0
aclCreateTensor(shape_vec.data(), shape_vec.size(), acl_data_type, 
               strides_vec.data(), 0, format,  // offset = 0
               shape_vec.data(), shape_vec.size(), ...);
```

### 3. 根据维度设置格式

**实现**：
```cpp
const auto dimNum = sizes.size();
aclFormat format = ACL_FORMAT_ND;
switch (dimNum) {
    case 3: format = ACL_FORMAT_NCL; break;
    case 4: format = ACL_FORMAT_NCHW; break;  // 4维设置为 NCHW
    case 5: format = ACL_FORMAT_NCDHW; break;
    default: format = ACL_FORMAT_ND; break;
}
```

### 4. 确保 tensor 连续性

在 `pybind.cpp` 中添加：
```cpp
x = x.contiguous();  // 确保内存连续
```

## 当前状态

### ✅ 已完成
1. ✅ 创建了修复后的 `ConvertType` 实现（`pytorch_npu_helper_fixed.hpp`）
2. ✅ 修复了 `storage_dims` 参数使用问题
3. ✅ 使用相同指针给 `view_dims` 和 `storage_dims`
4. ✅ 使用计算的连续 strides
5. ✅ 根据维度正确设置格式
6. ✅ 编译通过
7. ✅ 更新了 `pybind.cpp` 使用新的头文件

### ❌ 未解决
1. ❌ **格式仍然是 `ND` 而不是 `NCHW`**
   - 虽然设置了 `ACL_FORMAT_NCHW`，但实际创建的 tensor 格式仍然是 `ND`
   - 精度测试失败

## 问题分析

### 可能的原因

1. **`aclCreateTensor` 实现可能自动推断格式**
   - `aclCreateTensor` 的实现可能根据某些条件（如 strides、storage_dims 等）自动推断格式
   - 格式参数可能在某些情况下被忽略

2. **框架层面的格式处理**
   - 可能在框架层面有格式转换或验证逻辑
   - 格式可能在后续处理中被覆盖

3. **PyTorch tensor 到 ACL tensor 的转换**
   - PyTorch tensor 的内存布局可能影响格式识别
   - 可能需要额外的格式转换步骤

### 参考示例代码

在 `test_aclnn_quantized_batch_norm.cpp` 中：
```cpp
*tensor = aclCreateTensor(
    shape.data(), shape.size(), dataType, strides.data(), 0, 
    aclFormat::ACL_FORMAT_NCHW, 
    shape.data(), shape.size(),  // 相同的指针
    *deviceAddr);
```

我们的实现已经遵循了这个模式，但格式仍然是 `ND`。

## 相关文件

1. **修复后的头文件**：
   - `/home/xutianhao/CANN/NPUKernelBench/tasks/level2/Norm/QuantizedBatchNorm/validation/pytorch_npu_helper_fixed.hpp`

2. **PyBind11 绑定**：
   - `/home/xutianhao/CANN/NPUKernelBench/tasks/level2/Norm/QuantizedBatchNorm/validation/pybind.cpp`

3. **算子定义**：
   - `/home/xutianhao/CANN/NPUKernelBench/tasks/level2/Norm/QuantizedBatchNorm/answer/0/op_host/quantized_batch_norm.cpp`

4. **原始框架代码**（对比参考）：
   - `/home/xutianhao/CANN/NPUKernelBench/libs/common_include/msopgen/validation/pytorch_npu_helper.hpp`

5. **示例代码**（参考）：
   - `/home/xutianhao/CANN/ops-nn/norm/quantized_batch_norm/examples/test_aclnn_quantized_batch_norm.cpp`
   - `/home/xutianhao/CANN/canndev/ops/built-in/op_api/doc/zh/context/aclnnNpuFormatCast.md`

## 调试结果

### 调试输出（已添加）

通过添加调试打印，确认传递给 `aclCreateTensor` 的参数：

```
[ConvertType DEBUG] Calling aclCreateTensor with:
  view_dims: [1, 2, 4, 4], size=4
  data_type=2
  strides: [32, 16, 4, 1]
  offset=0
  format=0 (NCHW)  ✅ 格式设置正确！
  storage_dims: [1, 2, 4, 4], size=4  ✅
  storage_dims pointer == view_dims pointer: true  ✅
[ConvertType DEBUG] aclCreateTensor returned tensor pointer: 0xfffdc7ecc050
```

**关键发现**：
- ✅ format=0 (NCHW) 已正确传递
- ✅ view_dims 和 storage_dims 使用相同的指针和数据
- ✅ strides 计算正确
- ✅ offset=0
- ✅ 所有参数都符合示例代码的模式

**但错误信息仍然显示格式为 ND**：
```
get io input info is x(DT_INT8, ND) ... out(DT_INT8, ND)
```

这说明虽然我们传递了正确的 format 参数，但 `aclCreateTensor` 的实现可能：
1. 根据其他参数（如 storage_dims 的内容）自动推断格式
2. 忽略了 format 参数
3. 在后续处理中被覆盖

## 下一步建议

1. **深入调查 `aclCreateTensor` 实现**
   - ✅ 已确认参数传递正确（通过调试输出）
   - 检查 `aclCreateTensor` 的实现逻辑，看是否有格式推断逻辑
   - 确认格式参数如何被使用，是否会被其他参数覆盖

2. **检查框架层面的格式处理**
   - 查看 `EXEC_NPU_CMD` 宏的实现
   - 检查是否有格式转换或验证步骤
   - 查看是否有环境变量或配置影响格式

3. **尝试其他方法**
   - 检查是否有其他 API 可以设置或验证格式
   - 查看是否有格式转换 API（如 `aclnnNpuFormatCast`）
   - 检查是否需要显式调用格式转换

4. **对比示例代码**
   - ✅ 已对比：参数传递方式与示例代码一致
   - 检查示例代码中是否有其他特殊处理
   - 查看是否有初始化或配置步骤影响格式

## 技术细节

### `aclCreateTensor` 函数签名

```cpp
typedef aclTensor *(*_aclCreateTensor)(
    const int64_t *view_dims,        // 视图维度
    uint64_t view_dims_num,          // 视图维度数量
    aclDataType data_type,           // 数据类型
    const int64_t *stride,           // 步长
    int64_t offset,                  // 偏移量
    aclFormat format,                // 格式（我们设置的是 ACL_FORMAT_NCHW）
    const int64_t *storage_dims,     // 存储维度
    uint64_t storage_dims_num,       // 存储维度数量
    void *tensor_data                // 数据指针
);
```

### 当前实现的关键代码

```cpp
// 确定格式
const auto dimNum = sizes.size();
aclFormat format = ACL_FORMAT_ND;
switch (dimNum) {
    case 4: format = ACL_FORMAT_NCHW; break;
    // ...
}

// 存储 shape
std::vector<int64_t> shape_vec(sizes.begin(), sizes.end());

// 计算连续 strides
std::vector<int64_t> strides_vec(shape_vec.size(), 1);
for (int64_t i = shape_vec.size() - 2; i >= 0; i--) {
    strides_vec[i] = shape_vec[i + 1] * strides_vec[i + 1];
}

// 创建 tensor
auto acl_tensor = aclCreateTensor(
    shape_vec.data(), shape_vec.size(), acl_data_type, 
    strides_vec.data(), 0, format,  // format = ACL_FORMAT_NCHW
    shape_vec.data(), shape_vec.size(),  // 相同的指针
    const_cast<void *>(at_tensor.storage().data())
);
```

## 结论

虽然我们已经修复了 `storage_dims` 的使用问题，并按照示例代码的模式实现了 `ConvertType`，但格式仍然是 `ND` 而不是 `NCHW`。这个问题可能需要进一步调查 `aclCreateTensor` 的实现，或者需要框架层面的支持来解决。
