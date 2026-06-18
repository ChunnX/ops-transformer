# MXFP8 场景 Weights 输入改为 float32 修改方案

---

## 一、背景

当前 MXFP8 场景下，weights 的类型为 `bfloat16_t`。需求是将 weights 改为 `float32` 类型输入。

### 当前 MXFP8 类型配置

| 参数 | 当前类型 | 目标类型 |
|------|---------|---------|
| Q/K 数据 | `fp8_e4m3fn_t` | 不变 |
| Q/K Scale | `fp8_e8m0_t` | 不变 |
| **Weights** | **`bfloat16_t`** | **`float32_t`** |
| QK 输出 | `float32_t` | 不变 |

---

## 二、影响范围分析

### 2.1 不需要修改的部分（已兼容）

| 模块 | 文件 | 原因 |
|------|------|------|
| **VF 计算函数** | `vf/quant_lightning_indexer_vector1.h` | `MulWeightAndReduceSum_MxFP8` 和 `MulWeightAndReduceSum2_MxFP8` 已有 `float*` 重载（lines 879-949, 1028-1110） |
| **weightBuf_ 大小** | `service_vector.h:145` | `2 × CeilDiv(s1Base/CVRATIO) × UB_BANK_DEPTH_STRIDE(512)` = 每行 512B，float32 只需 `gSize×4=256B`，空间足够 |
| **DataCopyPad dstStride** | `service_vector.h:370` | `(512 - gSize×4) / 32 = 8`，合法 |
| **weightsCoreOffset** | `kernel.h:564` | 按元素计算，`GlobalTensor<float>` 自动处理字节偏移 |
| **weightFloatBuf_** | `service_vector.h:147` | MXFP8 MxFP8 路径不使用此 buffer |
| **service_cube.h** | `service_cube.h` | Cube 核不处理 weights，完全不受影响 |

### 2.2 需要修改的部分（共 3 处）

| # | 文件 | 行号 | 修改内容 |
|---|------|------|---------|
| 1 | `op_kernel/quant_lightning_indexer.cpp` | 57 | Kernel Entry 模板实例化 |
| 2 | `op_host/quant_lightning_indexer_def.cpp` | 102 | Op Def 数据类型声明 |
| 3 | `op_host/quant_lightning_indexer_tiling.cpp` | 287-294 | Tiling 类型校验 |

---

## 三、详细修改方案

### 修改 1：Kernel Entry 模板实例化

**文件**：`op_kernel/quant_lightning_indexer.cpp`  
**行号**：57

```cpp
// ========== 修改前 ==========
if (ORIG_DTYPE_QUERY_DEQUANT_SCALE == DT_FLOAT8_E8M0) {
    // MXFP8: q/k=fp8_e4m3, scale=fp8_e8m0, mmad_mx 输出 float
    INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
        PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
        bfloat16_t, fp8_e8m0_t, float32_t);
        // ^W_T=bfloat16_t  ← 需要修改
}

// ========== 修改后 ==========
if (ORIG_DTYPE_QUERY_DEQUANT_SCALE == DT_FLOAT8_E8M0) {
    // MXFP8: q/k=fp8_e4m3, scale=fp8_e8m0, mmad_mx 输出 float
    INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
        PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
        float32_t, fp8_e8m0_t, float32_t);
        // ^W_T=float32_t  ← 已修改
}
```

**说明**：`QLIType` 模板参数顺序为 `<Q_T, K_T, OUT_T, PAGE_ATTENTION, Q_LAYOUT, K_LAYOUT, W_T, SCALE_T, QK_T>`，第 7 个参数是 `W_T`。

---

### 修改 2：Op Def 数据类型声明

**文件**：`op_host/quant_lightning_indexer_def.cpp`  
**行号**：100-104

```cpp
// ========== 修改前 ==========
// ascend950 类型位置说明:
// 位置0: fp8_e4m3 + bf16 + float32 scale
// 位置1: hifloat8 + bf16 + float32 scale
// 位置2: fp8_e4m3 + float16 + float16 scale
// 位置3: int8 + float16 + float16 scale
// 位置4: fp8_e4m3 + bf16 + fp8_e8m0 scale (MXFP8)
aicore_config_95.Input("weights")
    .ParamType(REQUIRED)
    .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16})
    //                                                                       ^^^^^^^^^
    //                                                                       位置4: MXFP8
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();

// ========== 修改后 ==========
// ascend950 类型位置说明:
// 位置0: fp8_e4m3 + bf16 + float32 scale
// 位置1: hifloat8 + bf16 + float32 scale
// 位置2: fp8_e4m3 + float16 + float16 scale
// 位置3: int8 + float16 + float16 scale
// 位置4: fp8_e4m3 + float32 + fp8_e8m0 scale (MXFP8)  ← 已修改
aicore_config_95.Input("weights")
    .ParamType(REQUIRED)
    .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
    //                                                                       ^^^^^^^^^^
    //                                                                       位置4: MXFP8
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

**说明**：位置 4 对应 MXFP8 配置，从 `DT_BF16` 改为 `DT_FLOAT`。

---

### 修改 3：Tiling 类型校验

**文件**：`op_host/quant_lightning_indexer_tiling.cpp`  
**行号**：286-294

```cpp
// ========== 修改前 ==========
if (inputQType_ == ge::DT_FLOAT8_E4M3FN) {
    OP_CHECK_IF((weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT) && 
                (weightsType_ != ge::DT_FLOAT16|| inputQueryScaleType_ != ge::DT_FLOAT16) &&
                (weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT8_E8M0),
                //              ^^^^^^^^^^^^^
                //              MXFP8: bf16 + fp8_e8m0
        OP_LOGE(opName_, "When input query and key are float8_e4m3, ",
                "the data type of the input weights, query_dequant_scale and key_dequant_scale ",
                "must be (bfloat16, float32, float32), (float16, float16, float16) or ",
                "(bfloat16, float8_e8m0, float8_e8m0)."),
        return ge::GRAPH_FAILED);
}

// ========== 修改后 ==========
if (inputQType_ == ge::DT_FLOAT8_E4M3FN) {
    OP_CHECK_IF((weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT) && 
                (weightsType_ != ge::DT_FLOAT16|| inputQueryScaleType_ != ge::DT_FLOAT16) &&
                (weightsType_ != ge::DT_FLOAT || inputQueryScaleType_ != ge::DT_FLOAT8_E8M0),
                //              ^^^^^^^^^^^^^^
                //              MXFP8: float32 + fp8_e8m0
        OP_LOGE(opName_, "When input query and key are float8_e4m3, ",
                "the data type of the input weights, query_dequant_scale and key_dequant_scale ",
                "must be (bfloat16, float32, float32), (float16, float16, float16) or ",
                "(float32, float8_e8m0, float8_e8m0)."),
                // ^^^^^^^^
        return ge::GRAPH_FAILED);
}
```

**说明**：MXFP8 分支的 weights 类型校验从 `DT_BF16` 改为 `DT_FLOAT`，错误日志同步更新。

---

## 四、验证清单

### 4.1 编译验证

```bash
cd /home/hpc/code/ops-transformer_mxfp8/attention/quant_lightning_indexer
make clean && make
```

### 4.2 功能验证

- [ ] 单元测试：MXFP8 + float32 weights 场景
- [ ] 集成测试：完整 QLI 算子调用流程
- [ ] 精度验证：对比 bf16 weights 和 float32 weights 的输出一致性

### 4.3 性能验证

- [ ] 对比修改前后的性能（预期无明显变化，weights 仅在 Vector 核使用）
- [ ] 验证 UB 空间使用未超限

---

## 五、技术细节说明

### 5.1 为什么 VF 计算函数不需要修改？

`MulWeightAndReduceSum_MxFP8` 和 `MulWeightAndReduceSum2_MxFP8` 已有 `float*` 重载：

```cpp
// vector1.h:879-949
template <typename QK_T>
__aicore__ inline void MulWeightAndReduceSum_MxFP8(
    __ubuf__ uint16_t* out, __ubuf__ float* qk, uint32_t qkVLStride,
    __ubuf__ float* weight,  // ← float* 重载已存在
    int gSize)
{
    // ... 实现
}
```

模板参数 `W_T` 会自动推导到正确的重载版本。

### 5.2 为什么 weightBuf_ 大小足够？

```cpp
// service_vector.h:145
pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE);
```

- `UB_BANK_DEPTH_STRIDE = 512` bytes
- 每行分配 512 bytes
- float32 weights 每行需要：`gSize × sizeof(float) = 64 × 4 = 256` bytes
- **剩余 256 bytes**，空间充足

### 5.3 为什么 DataCopyPad 参数不需要修改？

```cpp
// service_vector.h:365-374
wDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);  // 自动适配：64 × 4 = 256 bytes
wDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
//                             = (512 - 256) / 32 = 8，合法
```

`sizeof(W_T)` 会自动从 2（bf16）变为 4（float32），所有计算自动适配。

---

## 六、风险评估

| 风险项 | 风险等级 | 说明 |
|--------|---------|------|
| UB 空间溢出 | 🟢 低 | weightBuf_ 每行 512B，float32 仅需 256B，充足 |
| 精度变化 | 🟡 中 | float32 精度高于 bf16，预期输出更精确，需验证 |
| 性能影响 | 🟢 低 | weights 仅在 Vector 核使用，不影响 Cube 核和整体流水 |
| 兼容性 | 🟢 低 | 仅修改 MXFP8 分支，不影响其他量化模式 |

---

## 七、修改统计

| 类别 | 数量 |
|------|------|
| 需修改文件 | 3 个 |
| 需修改代码行 | 3 处 |
| 新增代码行 | 0 |
| 删除代码行 | 0 |
| 预计修改时间 | 5 分钟 |
| 预计测试时间 | 30 分钟 |

---

## 八、相关文件索引

| 文件路径 | 作用 |
|---------|------|
| `op_kernel/quant_lightning_indexer.cpp` | Kernel Entry，模板实例化 |
| `op_host/quant_lightning_indexer_def.cpp` | Op Def，数据类型声明 |
| `op_host/quant_lightning_indexer_tiling.cpp` | Tiling，类型校验 |
| `op_kernel/quant_lightning_indexer_common.h` | `QLIType` 模板定义 |
| `op_kernel/arch35/quant_lightning_indexer_service_vector.h` | Vector 核 weights 处理 |
| `op_kernel/arch35/vf/quant_lightning_indexer_vector1.h` | VF 计算函数（已有 float* 重载） |

---

**文档版本**：1.0  
**创建日期**：2026-06-17  
**维护者**：OpenCode AI Assistant
