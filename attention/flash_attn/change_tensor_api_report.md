# C_Api → Tensor_Api 修改报告

## 1. 背景

将 Ascend C C-API（`LocalTensor`、`GlobalTensor`、`SetGlobalBuffer`）替换为模板参数，使代码兼容 Tensor API。

### Tensor API 关键差异

| 特性 | C-API | Tensor API |
|------|-------|-----------|
| Local Memory 张量 | `LocalTensor<T>` | `LocalTensor<TensorAttribute<Engine, Layout>>` |
| Global Memory 张量 | `GlobalTensor<T>` | `MakeTensor(MakeMemPtr<GM>(addr), layout)` |
| 设置 GM 地址 | `SetGlobalBuffer(ptr, size)` | `MakeTensor(MakeMemPtr<GM>(ptr), layout)` |
| 获取物理地址 | `GetPhyAddr()` | `Data().GetPhyAddr()` |

## 2. 涉及文件

| 文件 | 路径 | 角色 |
|------|------|------|
| kernel_cube | `flash_attn/op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h` | 主 kernel 文件 |
| fa_gm_tensor | `common/op_kernel/memcopy/fa_gm_tensor.h` | GM 张量封装 |
| fa_l1_tensor | `common/op_kernel/memcopy/fa_l1_tensor.h` | L1 张量封装 |
| fa_ub_tensor | `common/op_kernel/memcopy/fa_ub_tensor.h` | UB 张量封装 |
| parser | `common/op_kernel/memcopy/parser.h` | 序列长度解析器 |
| offset_calculator_v2 | `common/op_kernel/memcopy/offset_calculator_v2.h` | 偏移计算器 |
| copy_gm_to_l1 | `common/op_kernel/memcopy/copy_gm_to_l1.h` | GM→L1 数据搬运 |

## 3. 修改明细

### 3.1 fa_gm_tensor.h

| 行 | 修改点 | 修改前 | 修改后 |
|----|--------|--------|--------|
| 29-33 | 结构体模板参数 | `template <typename Q_T, GmFormat FORMAT, typename ACTLEN_T, bool WITH_ZERO_HEAD>` | 新增 `typename GmTensorType = GlobalTensor<Q_T>` |
| 31 | 成员变量类型 | `GlobalTensor<Q_T> gmTensor` | `GmTensorType gmTensor` |

### 3.2 fa_l1_tensor.h

| 行 | 修改点 | 修改前 | 修改后 |
|----|--------|--------|--------|
| 37-41 | 结构体模板参数 | `template <typename Q_T, L1Format FORMAT>` | 新增 `typename L1TensorType = LocalTensor<Q_T>` |
| 39 | 成员变量类型 | `LocalTensor<Q_T> tensor` | `L1TensorType tensor` |

### 3.3 fa_ub_tensor.h

| 行 | 修改点 | 修改前 | 修改后 |
|----|--------|--------|--------|
| 33-38 | 结构体模板参数 | `template <typename OUT_T>` | 新增 `typename UbTensorType = LocalTensor<OUT_T>` |
| 35 | 成员变量类型 | `LocalTensor<OUT_T> tensor` | `UbTensorType tensor` |

### 3.4 parser.h

| 行 | 修改点 | 修改前 | 修改后 |
|----|--------|--------|--------|
| 34-36 | 主模板 | `template <MODE, ACTLEN_T, WITH_ZERO_HEAD>` | 新增第4参数 `typename GmTensorType = GlobalTensor<ACTLEN_T>` |
| 38-90 | ACCUM 特化 | 固定 `GlobalTensor<ACTLEN_T>` 成员和参数 | 模板参数 `GmTensorType`，成员和 Init 参数均替换 |
| 92-123 | BY_BATCH 特化 | 同上 | 同上 |
| 126-143 | BlockTableParser | 非模板类，`GlobalTensor<int32_t>` 成员 | `template <typename GmTensorType = GlobalTensor<int32_t>>`，成员和 Init 参数替换 |
| 146-187 | WITH_ZERO_HEAD=true 特化 | 固定 `GlobalTensor<ACTLEN_T>` 成员和参数 | 模板参数 `GmTensorType`，成员和 Init 参数均替换 |

### 3.5 offset_calculator_v2.h

| 行 | 修改点 | 修改前 | 修改后 |
|----|--------|--------|--------|
| 183-193 | BNGSD Init | `Init(..., GlobalTensor<ACTLEN_T> actualSeqLengthsGmQ, ...)` | `template <typename GmTensorType> Init(..., GmTensorType, ...)` |
| 278-283 | TND Q Init | `Init(..., GlobalTensor<ACTLEN_T> actualSeqLengthsGmQ, ...)` | 同上 |
| 360-370 | BNSD KV Init | `Init(..., GlobalTensor<ACTLEN_T> actualSeqLengthsGm, ...)` | 同上 |
| 438-443 | TND KV Init | `Init(..., GlobalTensor<ACTLEN_T> actualSeqLengthsGmKV, ...)` | 同上 |
| 503-508 | V_SCALE Init | `Init(..., GlobalTensor<ACTLEN_T> actualSeqLengthsGmKV, ...)` | 同上 |
| 568-573 | PA_BNBD Init | `Init(..., GlobalTensor<int32_t> blockTableGm, ...)` | 同上 |
| 637-642 | PA_NZ Init | `Init(..., GlobalTensor<int32_t> blockTableGm, ...)` | 同上 |
| 712-717 | K_SCALE_PA_NZ Init | `Init(..., GlobalTensor<int32_t> blockTableGm, ...)` | 同上 |
| 978-982 | ANTIQ_BnBs Init | `Init(..., GlobalTensor<int32_t> blockTableGm, ...)` | 同上 |
| 1020-1025 | ANTIQ_BnNBs Init | `Init(..., GlobalTensor<int32_t> blockTableGm, ...)` | 同上 |
| 1076-1079 | PSE Init | `Init(..., GlobalTensor<ACTLEN_T> actualSeqLengthsGmQ, ...)` | 同上 |

### 3.6 copy_gm_to_l1.h

| 行 | 修改点 | 修改前 | 修改后 |
|----|--------|--------|--------|
| 25-27 | CopySingleMatrixNDToNZ | `template <typename T>` + `LocalTensor<T>` / `GlobalTensor<T>` 参数 | `template <typename T, typename L1TensorType, typename GmTensorType>` + 模板类型参数 |
| 74-78 | CopyMultiMatrixNDToNZ | 同上 | 同上 |
| 447-449 | CopySingleMXScaleDNToNZ | 同上 | 同上 |
| 467-471 | CopyMultiMXScaleDNToNZ | 同上 | 同上 |
| 687-689 | CopySingleMXScaleNDToNZ | 同上 | 同上 |

### 3.7 flash_attn_block_cube_noquant_gqa.h（kernel 文件）

| 行 | 修改点 | 修改前 | 修改后 |
|----|--------|--------|--------|
| 464 | CopyQuerySlice 参数 | `const LocalTensor<Q_T> &dstTensor` | `template <typename DstTensorType>` + `const DstTensorType &dstTensor` |
| 483 | CopyQueryTile 参数 | `const LocalTensor<Q_T> &dstTensor` | 同上 |
| 490 | CopyKeySlice 参数 | `const LocalTensor<KV_T> &dstTensor` | 同上 |
| 512 | CopyValueSlice 参数 | `const LocalTensor<KV_T> &dstTensor` | 同上 |
| 528 | CopyKeyTile 参数 | `const LocalTensor<KV_T> &dstTensor` | 同上 |
| 534 | CopyValueTile 参数 | `const LocalTensor<KV_T> &dstTensor` | 同上 |
| 588 | FixpipeMm1 参数 | `const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C` | `template <typename DstTensorType, typename L0CTensorType>` |
| 900 | FixpipeMm1Dn 参数 | 同上 | 同上 |
| 1031 | FixpipeMm2PartialN 参数 | `const LocalTensor<T> &l0C` | `template <typename L0CTensorType>` |
| 651 | IterateBmm1NdL0Split 局部变量 | `LocalTensor<Q_T> mm1ATensor = ...` | `auto mm1ATensor = ...` |
| 655 | 同上 | `LocalTensor<KV_T> mm1BTensor = ...` | `auto mm1BTensor = ...` |
| 698 | 同上 | `LocalTensor<Q_T> mm1ResL0CTensor = ...` | `auto mm1ResL0CTensor = ...` |
| 718 | 同上 | `LocalTensor<T> mmRes = ...` | `auto mmRes = ...` |
| 818 | IterateBmm1NdL1SplitK 局部变量 | `LocalTensor<Q_T> mm1ATensor = ...` | `auto mm1ATensor = ...` |
| 829 | 同上 | `LocalTensor<KV_T> mm1BTensor = ...` | `auto mm1BTensor = ...` |
| 864 | IterateBmm1DnSplitK 局部变量 | `LocalTensor<Q_T> mm1BTensor = ...` | `auto mm1BTensor = ...` |
| 874 | 同上 | `LocalTensor<KV_T> mm1ATensor = ...` | `auto mm1ATensor = ...` |
| 928 | IterateBmm1Dn 局部变量 | `LocalTensor<Q_T> mm1BTensor = ...` | `auto mm1BTensor = ...` |
| 938 | 同上 | `LocalTensor<KV_T> mm1ATensor = ...` | `auto mm1ATensor = ...` |
| 1012 | IterateBmm2L1SplitN 局部变量 | `LocalTensor<KV_T> mm2BTensor = ...` | `auto mm2BTensor = ...` |
| 1075 | IterateBmm2l0Split 局部变量 | `LocalTensor<KV_T> mm2BTensor = ...` | `auto mm2BTensor = ...` |

## 4. 未修改项（需后续处理）

### 4.1 成员变量（保留 GlobalTensor）

| 文件 | 行 | 内容 | 原因 |
|------|-----|------|------|
| kernel_cube | 308 | `GlobalTensor<int32_t> blockTableGm` | 类成员，需配合 SetGlobalBuffer 使用 |
| kernel_cube | 399 | `blockTableGm.SetGlobalBuffer(...)` | 初始化 GM 地址，Tensor API 需用 MakeTensor 替代 |
| kernel_cube | 418 | `qGmTensor.gmTensor.SetGlobalBuffer(...)` | 同上 |
| kernel_cube | 430 | `kvGmTensor.gmTensor.SetGlobalBuffer(...)` | 同上 |
| kernel_cube | 551 | `keyGm.gmTensor.SetGlobalBuffer(...)` | 同上 |
| kernel_cube | 560 | `valueGm.gmTensor.SetGlobalBuffer(...)` | 同上 |

### 4.2 Scale 函数内部 SetGlobalBuffer（copy_gm_to_l1.h）

| 行 | 内容 | 原因 |
|-----|------|------|
| 462-463 | `GlobalTensor<bfloat16_t> gmTensorCast; gmTensorCast.SetGlobalBuffer(...)` | 内部类型转换创建新 GlobalTensor，需 Tensor API 的 MakeMemPtr + MakeTensor 替代 |
| 484-485 | 同上 | 同上 |
| 702-703 | 同上 | 同上 |

### 4.3 迁移建议

对于 SetGlobalBuffer 的 Tensor API 替代方案：

```cpp
// C-API:
GlobalTensor<T> tensor;
tensor.SetGlobalBuffer((__gm__ T*)ptr, size);

// Tensor API:
auto layout = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<T>>(rows, cols);
auto tensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(ptr), layout);
```

需要在调用点提供完整的 Layout 信息（行列数等），因此建议在实际迁移到 Tensor API 时，将 SetGlobalBuffer 调用重构为带 Layout 的 MakeTensor 调用。

## 5. 修改统计

| 文件 | 修改点数 |
|------|---------|
| fa_gm_tensor.h | 2 |
| fa_l1_tensor.h | 2 |
| fa_ub_tensor.h | 2 |
| parser.h | 10 |
| offset_calculator_v2.h | 11 |
| copy_gm_to_l1.h | 5 |
| flash_attn_block_cube_noquant_gqa.h | 21 |
| **合计** | **53** |
