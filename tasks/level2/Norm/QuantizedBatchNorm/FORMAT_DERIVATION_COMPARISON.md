# Format 推导逻辑对比

## 总结

**Format 推导逻辑本身完全相同**，都是根据 tensor 的维度数量（dimNum）来设置格式：
- 3维 → `ACL_FORMAT_NCL`
- 4维 → `ACL_FORMAT_NCHW`
- 5维 → `ACL_FORMAT_NCDHW`
- 其他 → `ACL_FORMAT_ND`

**真正的区别在于传递给 `aclCreateTensor` 的其他参数**，这些参数可能影响格式的最终识别。

## 详细对比

### 1. Format 推导逻辑

#### 原先（原始框架）
```cpp
const auto dimNum = at_tensor.sizes().size();
aclFormat format = ACL_FORMAT_ND;
switch (dimNum) {
    case 3:     // 3表示NCL
        format = ACL_FORMAT_NCL;
        break;
    case 4:     // 4表示NCHW
        format = ACL_FORMAT_NCHW;
        break;
    case 5:     // 5表示NCDHW
        format = ACL_FORMAT_NCDHW;
        break;
    default:
        format = ACL_FORMAT_ND;
}
```

#### 现在（修复后）
```cpp
const auto dimNum = sizes.size();
aclFormat format = ACL_FORMAT_ND;
switch (dimNum) {
    case 3: format = ACL_FORMAT_NCL; break;
    case 4: format = ACL_FORMAT_NCHW; break;
    case 5: format = ACL_FORMAT_NCDHW; break;
    default: format = ACL_FORMAT_ND; break;
}
```

**结论**：Format 推导逻辑**完全相同**，都是根据维度数量设置格式。

---

### 2. storage_dims 参数

#### 原先（原始框架）
```cpp
c10::SmallVector<int64_t, 5> storageDims;
if (acl_data_type != ACL_STRING) {
    storageDims.push_back(at_tensor.storage().nbytes() / itemsize);
    // storageDims 只包含一个元素：storage 的总字节数除以元素大小
    // 例如：[32] 表示 storage 可以存储 32 个元素
}

// 调用 aclCreateTensor
aclCreateTensor(..., format, storageDims.data(), storageDims.size(), ...);
// storageDims.size() = 1
// storageDims.data() 指向单个元素数组
```

#### 现在（修复后）
```cpp
std::vector<int64_t> shape_vec(sizes.begin(), sizes.end());
// shape_vec 包含完整的 shape 信息
// 例如：[1, 2, 4, 4] 表示 4 维 tensor 的 shape

// 调用 aclCreateTensor
aclCreateTensor(..., format, shape_vec.data(), shape_vec.size(), ...);
// shape_vec.size() = 4 (对于 4 维 tensor)
// shape_vec.data() 指向 shape 数组
```

**区别**：
- **原先**：`storage_dims` 是单个元素（storage 大小）
- **现在**：`storage_dims` 是完整的 shape 数组

**参考示例代码**（`test_aclnn_quantized_batch_norm.cpp`）：
```cpp
*tensor = aclCreateTensor(
    shape.data(), shape.size(), dataType, strides.data(), 0, 
    aclFormat::ACL_FORMAT_NCHW, 
    shape.data(), shape.size(),  // storage_dims 也是 shape！
    *deviceAddr);
```

---

### 3. strides 参数

#### 原先（原始框架）
```cpp
// 直接使用 PyTorch tensor 的原始 strides
aclCreateTensor(..., 
               at_tensor.strides().data(),  // 使用 tensor 的原始 strides
               at_tensor.storage_offset(),   // 使用 tensor 的 storage_offset
               format, ...);
```

#### 现在（修复后）
```cpp
// 计算连续的 strides（参考示例代码）
std::vector<int64_t> strides_vec(shape_vec.size(), 1);
for (int64_t i = shape_vec.size() - 2; i >= 0; i--) {
    strides_vec[i] = shape_vec[i + 1] * strides_vec[i + 1];
}

aclCreateTensor(..., 
               strides_vec.data(),  // 使用计算的连续 strides
               0,                    // offset = 0
               format, ...);
```

**区别**：
- **原先**：使用 PyTorch tensor 的原始 strides（可能不连续）
- **现在**：使用计算的连续 strides（按照 shape 计算）

**参考示例代码**（`test_aclnn_quantized_batch_norm.cpp`）：
```cpp
// 计算连续tensor的strides
std::vector<int64_t> strides(shape.size(), 1);
for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
}

*tensor = aclCreateTensor(..., strides.data(), 0, ...);
```

---

### 4. offset 参数

#### 原先（原始框架）
```cpp
aclCreateTensor(..., 
               at_tensor.storage_offset(),  // 使用 tensor 的 storage_offset
               format, ...);
```

#### 现在（修复后）
```cpp
aclCreateTensor(..., 
               0,  // offset = 0
               format, ...);
```

**区别**：
- **原先**：使用 PyTorch tensor 的 `storage_offset()`（可能不为 0）
- **现在**：使用 `0`（参考示例代码）

---

### 5. view_dims 和 storage_dims 指针

#### 原先（原始框架）
```cpp
// view_dims 和 storage_dims 使用不同的指针
aclCreateTensor(
    at_tensor.sizes().data(),        // view_dims: 指向 sizes
    at_tensor.sizes().size(), 
    ..., 
    format, 
    storageDims.data(),              // storage_dims: 指向 storageDims（不同指针！）
    storageDims.size(), 
    ...);
```

#### 现在（修复后）
```cpp
// view_dims 和 storage_dims 使用相同的指针
aclCreateTensor(
    shape_vec.data(),                 // view_dims: 指向 shape_vec
    shape_vec.size(), 
    ..., 
    format, 
    shape_vec.data(),                 // storage_dims: 也指向 shape_vec（相同指针！）
    shape_vec.size(), 
    ...);
```

**区别**：
- **原先**：`view_dims` 和 `storage_dims` 使用不同的指针和数据
- **现在**：`view_dims` 和 `storage_dims` 使用相同的指针和数据

**参考示例代码**（`aclnnNpuFormatCast.md`）：
```cpp
*tensor = aclCreateTensor(
    shape.data(), shape.size(), dataType, strides.data(), 0,
    format, 
    *storageShape, *storageShapeSize,  // 如果 storageShape 指向 shape，则相同
    *deviceAddr);
```

---

## 完整的 aclCreateTensor 调用对比

### 原先（原始框架）
```cpp
auto acl_tensor = aclCreateTensor(
    at_tensor.sizes().data(),              // view_dims: [1, 2, 4, 4]
    at_tensor.sizes().size(),              // view_dims_num: 4
    acl_data_type,                         // DT_INT8
    at_tensor.strides().data(),            // strides: [32, 16, 4, 1] (示例)
    at_tensor.storage_offset(),            // offset: 可能不为 0
    format,                                // ACL_FORMAT_NCHW
    storageDims.data(),                    // storage_dims: [32] (单个元素！)
    storageDims.size(),                   // storage_dims_num: 1
    const_cast<void *>(at_tensor.storage().data())
);
```

### 现在（修复后）
```cpp
auto acl_tensor = aclCreateTensor(
    shape_vec.data(),                      // view_dims: [1, 2, 4, 4]
    shape_vec.size(),                      // view_dims_num: 4
    acl_data_type,                         // DT_INT8
    strides_vec.data(),                    // strides: [32, 16, 4, 1] (计算的连续 strides)
    0,                                     // offset: 0
    format,                                // ACL_FORMAT_NCHW
    shape_vec.data(),                      // storage_dims: [1, 2, 4, 4] (完整的 shape！)
    shape_vec.size(),                      // storage_dims_num: 4
    const_cast<void *>(at_tensor.storage().data())
);
```

---

## 关键差异总结

| 参数 | 原先（原始框架） | 现在（修复后） | 影响 |
|------|-----------------|---------------|------|
| **format 推导** | 根据 dimNum 设置 | 根据 dimNum 设置 | ✅ **相同** |
| **storage_dims** | `[storage_size]` (单个元素) | `[shape]` (完整 shape) | ⚠️ **不同** |
| **strides** | tensor 原始 strides | 计算的连续 strides | ⚠️ **不同** |
| **offset** | `storage_offset()` | `0` | ⚠️ **不同** |
| **指针一致性** | view_dims ≠ storage_dims | view_dims == storage_dims | ⚠️ **不同** |

---

## 问题分析

虽然 format 推导逻辑相同，但 `aclCreateTensor` 的实现可能根据以下因素来**最终确定格式**：

1. **storage_dims 和 view_dims 的关系**
   - 如果 `storage_dims` 和 `view_dims` 不同，可能被推断为 `ND` 格式
   - 如果 `storage_dims` 和 `view_dims` 相同，格式参数可能被正确使用

2. **strides 的连续性**
   - 非连续的 strides 可能触发格式推断逻辑
   - 连续的 strides 可能更容易识别格式

3. **offset 的影响**
   - 非零 offset 可能影响格式识别

4. **format 参数的优先级**
   - 如果 `aclCreateTensor` 的实现优先根据其他参数推断格式，format 参数可能被忽略

---

## 结论

**Format 推导逻辑本身没有区别**，都是根据维度数量设置格式。但传递给 `aclCreateTensor` 的其他参数（storage_dims、strides、offset、指针一致性）可能影响格式的最终识别。

当前的问题是：虽然我们设置了 `ACL_FORMAT_NCHW`，但实际创建的 tensor 格式仍然是 `ND`。这可能是因为 `aclCreateTensor` 的实现根据其他参数（特别是 `storage_dims` 的内容）自动推断格式，而忽略了 format 参数。
