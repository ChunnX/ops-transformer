# FIA Antiquant 算子 PA 场景 0/1 轴非连续支持方案

## 1. 目标

在伪量化(antiquant) FA 算子上，支持 PA 场景下 `k`/`v`/`k_descale`/`v_descale` 的 0轴(Bn)和1轴(N2)非连续。

**不支持** `k_rope` 的非连续（kRopeStrides 字段保留对齐但 kernel 不做非连续处理）。

支持 **3 种 PA 布局**（术语与代码 `KvCacheLayout` 枚举对齐，见 `processor.h:32-36`）：

| 布局(代码术语) | shape | paKvShapeType 枚举值 | 支持的非连续轴 |
|------|-------|---------------------|---------------|
| BNBD (KV_CACHE_BNSD) | `[Bn, N2, Bs, D]` | 1 | 0轴 + 1轴 |
| BBH (KV_CACHE_BSH) | `[Bn, Bs, N2, D]` | 0 | **仅0轴** |
| NZ (KV_CACHE_NZ) | `[Bn, N2, D1, Bs, D0]` | 2 | 0轴 + 1轴 |

> **BBH 仅支持 0 轴非连续的原因**：`[Bn, Bs, N2, D]` 中 N2 在 Bs 之后，N2 非连续会使 `bsStride = N2 × n2Stride` 变化，而 Bs 是 block 内部维度（2+轴），破坏"block 内必须连续"的约束。BNBD/NZ 中 N2 在 Bs 之前，不影响 block 内连续性。

约束：非 PA 不支持非连续；BBH 仅 0 轴可非连续；BNBD/NZ 仅 0/1 轴可非连续；block 内(2+轴)必须连续。

---

## 2. 三种布局的 stride 语义

| | BNBD `[Bn,N2,Bs,D]` | BBH `[Bn,Bs,N2,D]` | NZ `[Bn,N2,D1,Bs,D0]` |
|---|---|---|---|
| n2Stride 连续值 | Bs×D | D | D1×Bs×D0 = D×Bs |
| n2Stride 是否可非连续 | **是** | **否(仅0轴)** | **是** |
| bnStride 连续值 | N2×Bs×D | Bs×N2×D | N2×D×Bs |
| bnStride 约束 | ≥ N2×n2Stride | ≥ Bs×N2×D (固定) | ≥ N2×n2Stride |
| bsStride | D (固定) | N2×D (固定,因n2Stride不可非连续) | D0 (固定) |

> **BBH 无 bsStride 依赖问题**：因为 BBH 的 n2Stride 必须连续（仅支持 0 轴），bsStride = N2×D 固定不变。

> **stride 单位体系**：
> - `constInfo`/`inputParamsRegbase` 中的 stride 字段（如 `keyBnStride`/`valueN2Stride`）：**元素数**
> - `DataCopyExtParams.srcStride`：**字节**，需 `× sizeof(KV_T)` 转换
> - `CopyKVPageAttention` 的 `curOffset`：**元素数**（直接用于 `srcGm[offset]` 索引）

---

## 2.1 stride 数据流概览

stride 从用户输入到 kernel 搬运的完整传播链：

```
用户 tensor (view, 含 stride)
  │  context_->GetDynamicInputStride / GetInputStride
  ▼
fiaInfo.keyStrides / kScaleStrides / ...  (gert::Stride*, parser 层)
  │  SetFATilingData: fiaInfo.kScaleStrides->GetStride(0) → inputParams.set_bnStride()
  ▼
inputParamsRegbase.keyStrides / kScaleStrides  (StridesParams, tiling 序列化层, 文件1)
  │  ComputeConstexpr: inputParamsRegbase.keyStrides.get_bnStride() → constInfo.keyBnStride
  ▼
constInfo.keyBnStride / valueN2Stride / kScaleBnStride / ...  (uint64_t, kernel 运行时常量, 文件2/5)
  │  CalcValueCoreOffset* / ComputeOffset* / CopyKVPageAttention / CopyAntiqParam
  ▼
runParam.valueCoreOffset / runInfo.keyOffset / taskParam.kvGmOffset / curOffset  (最终 GM 地址)
```

**两层结构**：
- **Tiling 序列化层**（文件1 `StridesParams` + 文件3 `SetFATilingData`）：stride 从 `gert::Stride*` 提取为 `uint64_t`，序列化到 tiling data
- **Kernel 运行时层**（文件2 `COMMON_CONST_INFO` + 文件5 `ComputeConstexpr`）：从 tiling data 反序列化到 `constInfo`，供 offset 计算和数据搬运使用

**与 PR 6430 (mxfp8 fullquant) 的关键差异**：PR 6430 在 fullquant 路径有 `OffsetCalculator` + `GmLayout` 统一抽象（`offset_calculator_v2.h:597-606`），stride 注入 `GmLayout.MakeLayout` 后 `GetOffset` 自动算出最终地址。antiquant 路径**无此抽象**，需在 offset 计算函数（文件6）和数据搬运函数（文件8）中分别处理 stride——这是本设计散布式修改的根本原因。

---

## 3. Phase 1: Tiling 结构层（文件1 + 文件2）

本阶段在两个结构中新增 `kScaleStrides`/`vScaleStrides` 字段，构建 stride 从 host 到 kernel 的序列化通道：

- **文件1 `flash_attention_score_tiling_regbase.h`**：定义 `InputParamsRegbase`（tiling 序列化结构），新增 `StridesParams kScaleStrides/vScaleStrides` 字段 + getter/setter。`StridesParams`（line 61-70）已有 `bnStride`/`n2Stride` 两个 `uint64_t` 字段，本阶段复用该类型。
- **文件2 `util_regbase.h`**：定义 `COMMON_CONST_INFO` 宏（kernel 侧 `ConstInfo` 的字段声明），新增 `kScaleBnStride`/`kScaleN2Stride`/`vScaleBnStride`/`vScaleN2Stride` 四个 `uint64_t` 字段。同时同步更新 `CVSharedParams<true, true>`。

### 文件1: `flash_attention_score_tiling_regbase.h`

**路径**: `attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h`

### 修改点1: InputParamsRegbase 新增字段 (line 153-156)

**现有代码**:
```cpp
    // 增加strides参数
    StridesParams keyStrides;
    StridesParams valueStrides;
    StridesParams kRopeStrides;
```

**改为**:
```cpp
    // 增加strides参数
    StridesParams keyStrides;
    StridesParams valueStrides;
    StridesParams kRopeStrides;
    StridesParams kScaleStrides;
    StridesParams vScaleStrides;
```

### 修改点2: 新增 getter/setter (line 314-324, 在 `}` 闭合 `InputParamsRegbase` 类之前)

**现有代码** (line 314-324):
```cpp
    void set_keyStrides(StridesParams keyStrides) {this->keyStrides = keyStrides;}
    StridesParams get_keyStrides() const {return keyStrides;}

    void set_valueStrides(StridesParams valueStrides) {this->valueStrides = valueStrides;}

    StridesParams get_valueStrides() const {return valueStrides;}

    void set_kRopeStrides(StridesParams kRopeStrides) {this->kRopeStrides = kRopeStrides;}

    StridesParams get_kRopeStrides() const {return kRopeStrides;}
};
```

**改为**:
```cpp
    void set_kRopeStrides(StridesParams kRopeStrides)
    {
        this->kRopeStrides = kRopeStrides;
    }

    StridesParams get_kRopeStrides() const
    {
        return kRopeStrides;
    }

    void set_kScaleStrides(StridesParams kScaleStrides)
    {
        this->kScaleStrides = kScaleStrides;
    }

    StridesParams get_kScaleStrides() const
    {
        return kScaleStrides;
    }

    void set_vScaleStrides(StridesParams vScaleStrides)
    {
        this->vScaleStrides = vScaleStrides;
    }

    StridesParams get_vScaleStrides() const
    {
        return vScaleStrides;
    }
};
```

---

### 文件2: `util_regbase.h`

**路径**: `attention/common/op_kernel/arch35/util_regbase.h`

### 修改点1: COMMON_CONST_INFO 宏 (line 303-309)

**现有代码**:
```cpp
    /* strides */                                                                                                      \
    uint64_t keyBnStride;                                                                                              \
    uint64_t keyN2Stride;                                                                                              \
    uint64_t valueBnStride;                                                                                            \
    uint64_t valueN2Stride;                                                                                              \
    uint64_t keyRopeBnStride;                                                                                          \
    uint64_t keyRopeN2Stride
```

**改为**:
```cpp
    /* strides */                                                                                                      \
    uint64_t keyBnStride;                                                                                              \
    uint64_t keyN2Stride;                                                                                              \
    uint64_t valueBnStride;                                                                                            \
    uint64_t valueN2Stride;                                                                                              \
    uint64_t keyRopeBnStride;                                                                                          \
    uint64_t keyRopeN2Stride;                                                                                          \
    uint64_t kScaleBnStride;                                                                                           \
    uint64_t kScaleN2Stride;                                                                                           \
    uint64_t vScaleBnStride;                                                                                           \
    uint64_t vScaleN2Stride
```

> 注意：`COMMON_CONST_INFO` 宏被所有 4 个 `ConstInfo` 特化（`<true,true>`/`<true,false>`/`<false,true>`/`<false,false>`）通过 `COMMON_CONST_INFO;` 展开（`util_regbase.h:420-448`），在宏中新增字段会自动注入所有 ConstInfo 特化。

### 修改点2: CVSharedParams<true, true> (line 531-536)

**现有代码**:
```cpp
    uint64_t keyBnStride;
    uint64_t keyN2Stride;
    uint64_t valueBnStride;
    uint64_t valueN2Stride;
    uint64_t keyRopeBnStride;
    uint64_t keyRopeN2Stride;
```

**改为**:
```cpp
    uint64_t keyBnStride;
    uint64_t keyN2Stride;
    uint64_t valueBnStride;
    uint64_t valueN2Stride;
    uint64_t keyRopeBnStride;
    uint64_t keyRopeN2Stride;
    uint64_t kScaleBnStride;
    uint64_t kScaleN2Stride;
    uint64_t vScaleBnStride;
    uint64_t vScaleN2Stride;
```

---

## 4. Phase 2: Host 层（文件3 + 文件4）

本阶段将 stride 从 `fiaInfo`（parser 已填充）写入 `inputParamsRegbase`（tiling 序列化），并新增 checker 校验非连续轴约束。

### 文件3: `fused_infer_attention_score_tiling_impl.cpp`

**路径**: `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp`

### 修改点: SetFATilingData (line 1931-1940)

> stride 通过 `fiaInfo.isTensorV1` 守卫：`isTensorV1 == true`（老版本 opbase，PR 6430 引入于 `fia_tiling_info.h:241`）时无 stride 信息，直接跳过；`isTensorV1 == false` 时 stride 已由 parser 填充。现有代码已用 `if (!fiaInfo.isTensorV1)` 保护 key/value/kRope stride 赋值，本修改点在同一守卫内补全 kScale/vScale stride。

**现有代码** (line 1931-1940):
```cpp
    if (!fiaInfo.isTensorV1) {
        inputParams.keyStrides.set_bnStride(fiaInfo.keyStrides->GetStride(0));
        inputParams.keyStrides.set_n2Stride(fiaInfo.keyStrides->GetStride(1));
        inputParams.valueStrides.set_bnStride(fiaInfo.valueStrides->GetStride(0));
        inputParams.valueStrides.set_n2Stride(fiaInfo.valueStrides->GetStride(1));
        if (fiaInfo.kRopeStrides != nullptr) {
            inputParams.kRopeStrides.set_bnStride(fiaInfo.kRopeStrides->GetStride(0));
            inputParams.kRopeStrides.set_n2Stride(fiaInfo.kRopeStrides->GetStride(1));
        }
    }
```

**改为**:
```cpp
    if (!fiaInfo.isTensorV1) {
        inputParams.keyStrides.set_bnStride(fiaInfo.keyStrides->GetStride(0));
        inputParams.keyStrides.set_n2Stride(fiaInfo.keyStrides->GetStride(1));
        inputParams.valueStrides.set_bnStride(fiaInfo.valueStrides->GetStride(0));
        inputParams.valueStrides.set_n2Stride(fiaInfo.valueStrides->GetStride(1));
        if (fiaInfo.kRopeStrides != nullptr) {
            inputParams.kRopeStrides.set_bnStride(fiaInfo.kRopeStrides->GetStride(0));
            inputParams.kRopeStrides.set_n2Stride(fiaInfo.kRopeStrides->GetStride(1));
        }
        if (fiaInfo.kScaleStrides != nullptr && fiaInfo.kScaleStrides->GetDimNum() > 0) {
            inputParams.kScaleStrides.set_bnStride(fiaInfo.kScaleStrides->GetStride(0));
            inputParams.kScaleStrides.set_n2Stride(
                fiaInfo.kScaleStrides->GetDimNum() > 1 ? fiaInfo.kScaleStrides->GetStride(1) : 0);
        }
        if (fiaInfo.vScaleStrides != nullptr && fiaInfo.vScaleStrides->GetDimNum() > 0) {
            inputParams.vScaleStrides.set_bnStride(fiaInfo.vScaleStrides->GetStride(0));
            inputParams.vScaleStrides.set_n2Stride(
                fiaInfo.vScaleStrides->GetDimNum() > 1 ? fiaInfo.vScaleStrides->GetStride(1) : 0);
        }
    }
```

> `fiaInfo.kScaleStrides`/`vScaleStrides` 字段已由 PR 6430 在 `fia_tiling_info.h:239-240` 声明，并由 parser 的 `GetStrides()`（`tiling_info_parser.cpp:235-236`）填充，拷贝到 fiaInfo（`tiling_info_parser.cpp:1500-1501`）。但 `SetFATilingData` 当前完全不引用这两个字段——本修改点补全赋值链路。

---

### 文件4: `dequant_checker.cpp` / `dequant_checker.h`

**路径**: `attention/fused_infer_attention_score/op_host/checkers/`

### 修改点1: `dequant_checker.h` 新增声明 (line 137 后)

**现有代码** (line 132-137):
```cpp
    // Feature
    ge::graphStatus CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureQuerySForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureRopeForAntiquant(const FiaTilingInfo &fiaInfo) const;
```

**改为** (新增一行):
```cpp
    // Feature
    ge::graphStatus CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureQuerySForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureRopeForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckStrideForAntiquant(const FiaTilingInfo &fiaInfo);
```

### 修改点2: `dequant_checker.cpp` CheckFeatureForAntiquant (line 2498-2508)

**现有代码**:
```cpp
ge::graphStatus DequantChecker::CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureLayoutForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureQuerySForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeaturePAForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureRopeForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
```

**改为**:
```cpp
ge::graphStatus DequantChecker::CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureLayoutForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureQuerySForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeaturePAForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureRopeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckStrideForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
```

### 修改点3: 新增 `CheckStrideForAntiquant` 函数 (在 line 2508 后插入)

> `CheckTensorContiguous`（`base_checker.h:46`，PR 6430 引入）已声明为 `BaseChecker` 的 protected 方法，`DequantChecker` 可直接调用。antiquant 路径的 PA checker（`CheckFeaturePAForAntiquant`等）当前不区分 BBH/BNBD/NZ，本函数通过 `fiaInfo.kvLayout` 获取布局信息。

```cpp
ge::graphStatus DequantChecker::CheckStrideForAntiquant(const FiaTilingInfo &fiaInfo)
{
    // isTensorV1==true 表示老版本opbase，无stride信息，直接跳过
    if (fiaInfo.isTensorV1) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    const uint32_t keyDimNum = keyShape.GetDimNum();
    const gert::Shape valueShape = fiaInfo.opParamInfo.value.shape->GetStorageShape();
    const uint32_t valueDimNum = valueShape.GetDimNum();
    int32_t dimIndex = 0;

    // 非 PA 场景：不支持任何非连续
    if (!fiaInfo.pageAttentionFlag) {
        OP_CHECK_IF((CheckTensorContiguous(keyDimNum, keyShape, fiaInfo.keyStrides, dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous key is not supported."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((CheckTensorContiguous(valueDimNum, valueShape, fiaInfo.valueStrides, dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous value is not supported."),
            return ge::GRAPH_FAILED);
        if (fiaInfo.opParamInfo.keyAntiquantScale.tensor != nullptr) {
            const auto &scaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
            OP_CHECK_IF((CheckTensorContiguous(scaleShape.GetDimNum(), scaleShape, fiaInfo.kScaleStrides, dimIndex) != ge::GRAPH_SUCCESS),
                OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous k_descale is not supported."),
                return ge::GRAPH_FAILED);
        }
        if (fiaInfo.opParamInfo.valueAntiquantScale.tensor != nullptr) {
            const auto &scaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();
            OP_CHECK_IF((CheckTensorContiguous(scaleShape.GetDimNum(), scaleShape, fiaInfo.vScaleStrides, dimIndex) != ge::GRAPH_SUCCESS),
                OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous v_descale is not supported."),
                return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    // PA 场景：按布局区分允许的非连续轴
    // BBH(KV_CACHE_BSH=0)：仅允许 dimIndex==0
    // BNBD(KV_CACHE_BNSD=1)/NZ(KV_CACHE_NZ=2)：允许 dimIndex==0||1
    const bool isBbhLayout = (fiaInfo.kvLayout == FiaLayout::BnBsH);
    auto checkStrideAllowed = [&](const char* tensorName, uint32_t dimNum, const gert::Shape& shape,
                                  const gert::Stride* strides) -> ge::graphStatus {
        if (strides == nullptr || strides->GetDimNum() == 0) {
            return ge::GRAPH_SUCCESS;
        }
        if (CheckTensorContiguous(dimNum, shape, strides, dimIndex) != ge::GRAPH_SUCCESS) {
            if (isBbhLayout) {
                OP_LOGE(fiaInfo.opName,
                    "In PA BBH antiquant scenarios, %s only supports non-contiguous in dimension 0, but got dimension %d.",
                    tensorName, dimIndex);
                return ge::GRAPH_FAILED;
            } else if (dimIndex != 0 && dimIndex != 1) {
                OP_LOGE(fiaInfo.opName,
                    "In PA BNBD/NZ antiquant scenarios, %s only supports non-contiguous in dimensions 0/1, but got dimension %d.",
                    tensorName, dimIndex);
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    };

    OP_CHECK_IF(checkStrideAllowed("key", keyDimNum, keyShape, fiaInfo.keyStrides) != ge::GRAPH_SUCCESS,
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkStrideAllowed("value", valueDimNum, valueShape, fiaInfo.valueStrides) != ge::GRAPH_SUCCESS,
        return ge::GRAPH_FAILED);
    if (fiaInfo.opParamInfo.keyAntiquantScale.tensor != nullptr) {
        const auto &scaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
        OP_CHECK_IF(checkStrideAllowed("k_descale", scaleShape.GetDimNum(), scaleShape, fiaInfo.kScaleStrides) != ge::GRAPH_SUCCESS,
            return ge::GRAPH_FAILED);
    }
    if (fiaInfo.opParamInfo.valueAntiquantScale.tensor != nullptr) {
        const auto &scaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();
        OP_CHECK_IF(checkStrideAllowed("v_descale", scaleShape.GetDimNum(), scaleShape, fiaInfo.vScaleStrides) != ge::GRAPH_SUCCESS,
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
```

---

## 5. Phase 3: Kernel 接线层（文件5）

本阶段在 `ComputeConstexpr` 中从 `inputParamsRegbase` 读取 stride 并赋值到 `constInfo`。这是 stride 从 tiling data 进入 kernel 运行时的唯一入口。

> **关键修正**：`COMMON_CONST_INFO` 宏（`util_regbase.h:303-309`）已声明 `keyBnStride`/`keyN2Stride`/`valueBnStride`/`valueN2Stride`/`keyRopeBnStride`/`keyRopeN2Stride` 共 6 个 stride 字段，但 antiquant 的 `ComputeConstexpr`（`antiquant_kernel.h:173-385`）**从未给这些字段赋值**——当前是未初始化垃圾值。对比 noquant_mla 路径（`flash_attention_kernel_noquant_mla.h:359-362`）已有赋值。本修改点补全 antiquant 路径的 stride 赋值。
>
> 注意：不应加 `if (get_bnStride() != 0)` 条件包裹——stride 字段默认为 0（`StridesParams` 构造函数 `bnStride = 0`），0 表示连续，后续 kernel 代码用 `!= 0` 判断是否非连续。无条件赋值即可，0 值天然表示连续。

### 文件5: `flash_attention_score_antiquant_kernel.h`

### 修改点: ComputeConstexpr (line 373-385 之间)
```cpp
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        this->constInfo.blockSize = inputParamsRegbase.blockSize;
        this->constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
        this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }

    this->constInfo.transposeLayout = inputParamsRegbase.transposeLayout;
```

**改为** (在 `}` 和空行之间插入):
```cpp
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        this->constInfo.blockSize = inputParamsRegbase.blockSize;
        this->constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
        this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }
    // 从 tiling data 读取 stride（0=连续，非0=非连续步长，单位：元素数）
    constInfo.keyBnStride = inputParamsRegbase.keyStrides.get_bnStride();
    constInfo.keyN2Stride = inputParamsRegbase.keyStrides.get_n2Stride();
    constInfo.valueBnStride = inputParamsRegbase.valueStrides.get_bnStride();
    constInfo.valueN2Stride = inputParamsRegbase.valueStrides.get_n2Stride();
    constInfo.kScaleBnStride = inputParamsRegbase.kScaleStrides.get_bnStride();
    constInfo.kScaleN2Stride = inputParamsRegbase.kScaleStrides.get_n2Stride();
    constInfo.vScaleBnStride = inputParamsRegbase.vScaleStrides.get_bnStride();
    constInfo.vScaleN2Stride = inputParamsRegbase.vScaleStrides.get_n2Stride();

    this->constInfo.transposeLayout = inputParamsRegbase.transposeLayout;
```

---

## 6. Phase 4: Offset 计算层（文件6 + 文件7）

本阶段在 offset 计算函数中引入 stride，使 batch/head 级 core offset 能反映非连续步长。分为两个子层：

- **文件6 `infer_flash_attention_kvcache.h`**：计算 `valueCoreOffset`/`keyCoreOffset`（batch/head 级）和 `valueOffset`/`keyOffset`（循环步进级）。这些 offset 是 stride 的**主要作用点**——bnStride 影响 batch 级偏移，n2Stride 影响 head 级偏移。
- **文件7 `flash_attention_score_antiquant_block_vec.h`**：`SetAntiqParamCommon` 将 constInfo 的 stride 传递到 `taskParam`，供文件8搬运层使用。

### 文件6: `infer_flash_attention_kvcache.h`

**路径**: `attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h`

> **架构说明**：本文件的 offset 函数分两个层级：
> - **Core offset 层**（修改点1-4b）：`CalcValueCoreOffset*` / `GetKeyCoreOffsetParam` / `SyncKeyCoreOffset`——计算 batch/head 级基础偏移，stride 的 bnStride/n2Stride 在此生效
> - **Loop offset 层**（修改点5-7）：`ComputeOffsetForAntiquant` / `ComputeOffsetPrefixAIC` / `ComputeOffsetNonPrefixAIC`——在 core offset 基础上叠加循环步进，需拆分 key/value 的步进（原设计 `keyOffset = valueOffset` 复用假设 dSize==dSizeV）
>
> **PA 布局区分层级**：BBH/BNBD/NZ 三种 PA 布局的区分**只在搬运层** `CopyKVPageAttention`（`processor.h:608-625`）按 `paKvShapeType` 动态查表。本文件的函数只按编译期 `layout`（BSH/TND/BNSD）和 `isPa` 分支——stride 的 bnStride/n2Stride 影响这一层，bsStride 影响搬运层。
>
> **`*Dv` 字段语义**：antiquant 的 `n2Dv`/`s2Dv`/`n2S2Dv`/`s2BaseN2Dv` 实际用 `dSize` 而非 `dSizeV`（`antiquant_kernel.h:207-214`），靠 `dSize==dSizeV` 假设兜底。BBH 的 n2Stride 始终为 0（仅支持 0 轴），BSH 分支的 n2Stride 替换不会进入，但保留防御性。

### 修改点1: CalcValueCoreOffsetBSH (line 305-313)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBSH(RunParamStr<isInfer> &runParam,
                                              const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx)
{
    uint64_t valueInnerOffsetSize = 0;
    if (constInfo.isKvContinuous == 1) {
        valueInnerOffsetSize = bIdx * constInfo.n2S2Dv + runParam.kvLeftPaddingSize * constInfo.n2Dv;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBSH(RunParamStr<isInfer> &runParam,
                                              const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx)
{
    uint64_t valueInnerOffsetSize = 0;
    uint64_t headStride = (constInfo.valueN2Stride != 0) ? constInfo.valueN2Stride : constInfo.dSizeV;
    if (constInfo.isKvContinuous == 1) {
        uint64_t batchStride = (constInfo.valueBnStride != 0) ? constInfo.valueBnStride : constInfo.n2S2Dv;
        valueInnerOffsetSize = bIdx * batchStride + runParam.kvLeftPaddingSize * constInfo.n2Dv;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStride;
}
```

### 修改点2: CalcValueCoreOffsetTND (line 316-326)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetTND(RunParamStr<isInfer> &runParam,
                                              const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx,
                                              __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t valueInnerOffsetSize = 0;
    if constexpr (!isPa) {
        valueInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2Dv;
    } else {
        valueInnerOffsetSize = bIdx * constInfo.n2S2Dv;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetTND(RunParamStr<isInfer> &runParam,
                                              const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx,
                                              __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t valueInnerOffsetSize = 0;
    uint64_t headStride = (constInfo.valueN2Stride != 0) ? constInfo.valueN2Stride : constInfo.dSizeV;
    if constexpr (!isPa) {
        valueInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2Dv;
    } else {
        uint64_t batchStride = (constInfo.valueBnStride != 0) ? constInfo.valueBnStride : constInfo.n2S2Dv;
        valueInnerOffsetSize = bIdx * batchStride;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStride;
}
```

### 修改点3: CalcValueCoreOffsetBNSD (line 346-360)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBNSD(RunParamStr<isInfer> &runParam,
                                               const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx)
{
    uint64_t headStrideV = 0;
    uint64_t valueInnerOffsetSize = 0;
    if (constInfo.isKvContinuous == 1) {
        headStrideV = constInfo.s2Dv;
        valueInnerOffsetSize = bIdx * constInfo.n2Size * headStrideV + runParam.kvLeftPaddingSize * constInfo.dSizeV;
    } else {
        headStrideV = constInfo.dSizeV * runParam.s2InCurrentBatch;
        valueInnerOffsetSize = 0;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStrideV;
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBNSD(RunParamStr<isInfer> &runParam,
                                               const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx)
{
    uint64_t headStrideV = 0;
    uint64_t valueInnerOffsetSize = 0;
    if (constInfo.isKvContinuous == 1) {
        headStrideV = (constInfo.valueN2Stride != 0) ? constInfo.valueN2Stride : constInfo.s2Dv;
        uint64_t batchStride = (constInfo.valueBnStride != 0) ? constInfo.valueBnStride : constInfo.n2Size * headStrideV;
        valueInnerOffsetSize = bIdx * batchStride + runParam.kvLeftPaddingSize * constInfo.dSizeV;
    } else {
        headStrideV = constInfo.dSizeV * runParam.s2InCurrentBatch;
        valueInnerOffsetSize = 0;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStrideV;
}
```

### 修改点4: SyncKeyCoreOffset (line 363-371)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void SyncKeyCoreOffset(RunParamStr<isInfer> &runParam, const ConstInfo<isInfer, hasRope> &constInfo,
                                         int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        GetKeyCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void SyncKeyCoreOffset(RunParamStr<isInfer> &runParam, const ConstInfo<isInfer, hasRope> &constInfo,
                                         int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    if (unlikely(constInfo.dSize != constInfo.dSizeV) ||
        constInfo.keyBnStride != 0 || constInfo.keyN2Stride != 0 ||
        constInfo.valueBnStride != 0 || constInfo.valueN2Stride != 0) {
        GetKeyCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}
```

> `GetKeyCoreOffsetParam` 也需同步修改，与 `CalcValueCoreOffsetBSH/BNSD/TND` 对称。详见修改点4b。

### 修改点4b: GetKeyCoreOffsetParam (line 269-302)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void GetKeyCoreOffsetParam(RunParamStr<isInfer> &runParam,
                                             const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx,
                                             __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t keyInnerOffsetSize = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        if (constInfo.isKvContinuous == 1) {
            keyInnerOffsetSize = bIdx * constInfo.n2S2D + runParam.kvLeftPaddingSize * constInfo.n2D;
        } else {
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * constInfo.dSize;
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!isPa) {
            keyInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2D;
        } else {
            keyInnerOffsetSize = bIdx * constInfo.n2S2D;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * constInfo.dSize;
    } else {
        uint64_t headStrideK = 0;
        if (constInfo.isKvContinuous == 1) {
            headStrideK = constInfo.s2D;
            keyInnerOffsetSize = bIdx * constInfo.n2Size * headStrideK + runParam.kvLeftPaddingSize * constInfo.dSize;
        } else {
            headStrideK = constInfo.dSize * runParam.s2InCurrentBatch;
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * headStrideK;
    }
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void GetKeyCoreOffsetParam(RunParamStr<isInfer> &runParam,
                                             const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx,
                                             __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t keyInnerOffsetSize = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        uint64_t headStride = (constInfo.keyN2Stride != 0) ? constInfo.keyN2Stride : constInfo.dSize;
        if (constInfo.isKvContinuous == 1) {
            uint64_t batchStride = (constInfo.keyBnStride != 0) ? constInfo.keyBnStride : constInfo.n2S2D;
            keyInnerOffsetSize = bIdx * batchStride + runParam.kvLeftPaddingSize * constInfo.n2D;
        } else {
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * headStride;
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        uint64_t headStride = (constInfo.keyN2Stride != 0) ? constInfo.keyN2Stride : constInfo.dSize;
        if constexpr (!isPa) {
            keyInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2D;
        } else {
            uint64_t batchStride = (constInfo.keyBnStride != 0) ? constInfo.keyBnStride : constInfo.n2S2D;
            keyInnerOffsetSize = bIdx * batchStride;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * headStride;
    } else {
        uint64_t headStrideK = 0;
        if (constInfo.isKvContinuous == 1) {
            headStrideK = (constInfo.keyN2Stride != 0) ? constInfo.keyN2Stride : constInfo.s2D;
            uint64_t batchStride = (constInfo.keyBnStride != 0) ? constInfo.keyBnStride : constInfo.n2Size * headStrideK;
            keyInnerOffsetSize = bIdx * batchStride + runParam.kvLeftPaddingSize * constInfo.dSize;
        } else {
            headStrideK = constInfo.dSize * runParam.s2InCurrentBatch;
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * headStrideK;
    }
}
```

### 修改点5: ComputeOffsetForAntiquant (line 925-972)

> 此函数有两个大分支：`enableKVPrefix`（line 928-950）和 `!enableKVPrefix`（line 951-971）。
> prefix 分支中的 **非 prefix loop 子分支**（退出 prefix 后的迭代）也用 `s2BaseN2Dv`/`s2BaseDv`。`keyOffset = valueOffset` 需拆分（共 4 处：938、948、957、966）。
>
> **BBH 简化**：BBH 的 n2Stride 始终为 0（仅支持 0 轴），BSH 分支的 `s2BaseStrideV` n2Stride 替换不会进入。但 TND 在 PA 下可能映射到 BBH/BNBD，保留防御性。

**现有代码** (完整函数, line 925-972):
```cpp
TEMPLATE_INTF
__aicore__ inline void ComputeOffsetForAntiquant(const RunParamStr<isInfer> &runParam,
                                                 const ConstInfo<isInfer, hasRope> &constInfo, uint32_t sInnerLoopIdx,
                                                 RunInfo<isInfer> &runInfo)
{
    if constexpr (enableKVPrefix) {
        const bool inPrefixLoop = sInnerLoopIdx < constInfo.prefixLoopCount;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            if (inPrefixLoop) {
                runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
                runInfo.valueOffset = 0;
                runInfo.keyOffset = 0;
            } else {
                runInfo.prefixOffset = 0;
                runInfo.valueOffset =
                    runParam.valueCoreOffset + (sInnerLoopIdx - constInfo.prefixLoopCount) * constInfo.s2BaseN2Dv;
                runInfo.keyOffset = runInfo.valueOffset;
            }
        } else {
            if (inPrefixLoop) {
                runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
                runInfo.valueOffset = 0;
                runInfo.keyOffset = 0;
            } else {
                runInfo.prefixOffset = 0;
                runInfo.valueOffset =
                    runParam.valueCoreOffset + (sInnerLoopIdx - constInfo.prefixLoopCount) * constInfo.s2BaseDv;
                runInfo.keyOffset = runInfo.valueOffset;
            }
        }
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            if constexpr (isFd) {
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.n2D;
            }
            runInfo.keyOffset = runInfo.valueOffset;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseN2DR;
            }
        } else {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            if constexpr (isFd) {
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSize;
            }
            runInfo.keyOffset = runInfo.valueOffset;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
            }
        }
    }
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void ComputeOffsetForAntiquant(const RunParamStr<isInfer> &runParam,
                                                 const ConstInfo<isInfer, hasRope> &constInfo, uint32_t sInnerLoopIdx,
                                                 RunInfo<isInfer> &runInfo)
{
    if constexpr (enableKVPrefix) {
        const bool inPrefixLoop = sInnerLoopIdx < constInfo.prefixLoopCount;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            if (inPrefixLoop) {
                runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
                runInfo.valueOffset = 0;
                runInfo.keyOffset = 0;
            } else {
                runInfo.prefixOffset = 0;
                uint64_t s2BaseStrideV = constInfo.s2BaseN2Dv;
                if (constInfo.valueN2Stride != 0) {
                    s2BaseStrideV = s2BaseSize * constInfo.n2Size * constInfo.valueN2Stride;
                }
                runInfo.valueOffset =
                    runParam.valueCoreOffset + (sInnerLoopIdx - constInfo.prefixLoopCount) * s2BaseStrideV;
                uint64_t s2BaseStrideK = constInfo.s2BaseN2D;
                if (constInfo.keyN2Stride != 0) {
                    s2BaseStrideK = s2BaseSize * constInfo.n2Size * constInfo.keyN2Stride;
                }
                runInfo.keyOffset =
                    runParam.keyCoreOffset + (sInnerLoopIdx - constInfo.prefixLoopCount) * s2BaseStrideK;
            }
        } else {
            if (inPrefixLoop) {
                runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
                runInfo.valueOffset = 0;
                runInfo.keyOffset = 0;
            } else {
                runInfo.prefixOffset = 0;
                runInfo.valueOffset =
                    runParam.valueCoreOffset + (sInnerLoopIdx - constInfo.prefixLoopCount) * constInfo.s2BaseDv;
                runInfo.keyOffset =
                    runParam.keyCoreOffset + (sInnerLoopIdx - constInfo.prefixLoopCount) * constInfo.s2BaseD;
            }
        }
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
            uint64_t s2BaseStrideV = constInfo.s2BaseN2Dv;
            if (constInfo.valueN2Stride != 0) {
                s2BaseStrideV = s2BaseSize * constInfo.n2Size * constInfo.valueN2Stride;
            }
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * s2BaseStrideV;
            if constexpr (isFd) {
                uint64_t fdStride = (constInfo.valueN2Stride != 0)
                    ? constInfo.sInnerLoopSize * constInfo.n2Size * constInfo.valueN2Stride
                    : constInfo.sInnerLoopSize * constInfo.n2D;
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * fdStride;
            }
            uint64_t s2BaseStrideK = constInfo.s2BaseN2D;
            if (constInfo.keyN2Stride != 0) {
                s2BaseStrideK = s2BaseSize * constInfo.n2Size * constInfo.keyN2Stride;
            }
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * s2BaseStrideK;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseN2DR;
            }
        } else {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            if constexpr (isFd) {
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSize;
            }
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseD;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
            }
        }
    }
}
```

> BNSD 的 s2BaseDv/s2BaseD 不受 n2Stride 影响（bsStride=D 固定），保持原值。keyOffset 改为用 `keyCoreOffset` 独立计算。prefix loop 分支（`inPrefixLoop=true`）不修改——prefix GM 是独立连续缓冲区，用户 stride 不影响。

### 修改点6: ComputeOffsetPrefixAIC (line 835-861) — **AIC 侧补全**

> 原设计遗漏。此函数有 2 处 `keyOffset = valueOffset`（line 848、858），与 `ComputeOffsetForAntiquant` 的 BSH 分支同模式，需同样拆分。

**现有代码** (line 835-861):
```cpp
TEMPLATE_INTF
__aicore__ inline void ComputeOffsetPrefixAIC(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer>& runInfo)
{
    const bool inPrefixLoop =
        (sInnerLoopIdx * static_cast<int32_t>(s2TemplateType) < constInfo.actualKVPrefixSize);
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        if (inPrefixLoop) {
            runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            runInfo.valueOffset = 0;
            runInfo.keyOffset = 0;
        } else {
            runInfo.prefixOffset = 0;
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            runInfo.keyOffset = runInfo.valueOffset;
        }
    } else {
        if (inPrefixLoop) {
            runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            runInfo.valueOffset = 0;
            runInfo.keyOffset = 0;
        } else {
            runInfo.prefixOffset = 0;
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            runInfo.keyOffset = runInfo.valueOffset;
        }
    }
}
```

**改为**（拆分 keyOffset，与修改点5对称）:
```cpp
TEMPLATE_INTF
__aicore__ inline void ComputeOffsetPrefixAIC(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer>& runInfo)
{
    const bool inPrefixLoop =
        (sInnerLoopIdx * static_cast<int32_t>(s2TemplateType) < constInfo.actualKVPrefixSize);
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        if (inPrefixLoop) {
            runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            runInfo.valueOffset = 0;
            runInfo.keyOffset = 0;
        } else {
            runInfo.prefixOffset = 0;
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2D;
        }
    } else {
        if (inPrefixLoop) {
            runInfo.prefixOffset = runParam.prefixCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            runInfo.valueOffset = 0;
            runInfo.keyOffset = 0;
        } else {
            runInfo.prefixOffset = 0;
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseD;
        }
    }
}
```

> BSH 分支用 `s2BaseN2D`（dSize-based）算 keyOffset，与 `s2BaseN2Dv` 对称。当 `dSize==dSizeV` 且 stride 连续时，两者等价。prefix loop 分支不改（prefix GM 独立连续）。

### 修改点7: ComputeOffsetNonPrefixAIC (line 864-907) — **AIC 侧补全**

> 原设计遗漏。此函数有 3 处 `keyOffset = valueOffset`（line 875、888、901），但已有 `if (unlikely(constInfo.dSize != constInfo.dSizeV))` 判断部分分离 key/value。需在条件中追加 stride 非连续检查。

**现有代码** (line 864-907):
```cpp
TEMPLATE_INTF
__aicore__ inline void ComputeOffsetNonPrefixAIC(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer>& runInfo)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
        if constexpr (isFd) {
            runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.n2D;
        }
        if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2D;
        } else {
            runInfo.keyOffset = runInfo.valueOffset;
        }
        if constexpr (hasRope) {
            runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseN2DR;
        }
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
        // ... (类似结构,用 s2BaseDv/s2BaseD)
    } else {
        // ... (BNSD,用 s2BaseDv/s2BaseD)
    }
}
```

**改为**（3 处 `if (unlikely(constInfo.dSize != constInfo.dSizeV))` 追加 stride 条件）:
```cpp
TEMPLATE_INTF
__aicore__ inline void ComputeOffsetNonPrefixAIC(const RunParamStr<isInfer>& runParam,
    const ConstInfo<isInfer, hasRope>& constInfo, uint32_t sInnerLoopIdx, RunInfo<isInfer>& runInfo)
{
    // stride 非连续时必须独立计算 key/value offset
    const bool needSplitKV = (constInfo.dSize != constInfo.dSizeV) ||
        constInfo.keyBnStride != 0 || constInfo.keyN2Stride != 0 ||
        constInfo.valueBnStride != 0 || constInfo.valueN2Stride != 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
        if constexpr (isFd) {
            runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.n2D;
        }
        if (unlikely(needSplitKV)) {
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2D;
        } else {
            runInfo.keyOffset = runInfo.valueOffset;
        }
        if constexpr (hasRope) {
            runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseN2DR;
        }
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
        runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
        if constexpr (isFd) {
            runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSizeV;
        }
        if (unlikely(needSplitKV)) {
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
        } else {
            runInfo.keyOffset = runInfo.valueOffset;
        }
        if constexpr (hasRope) {
            runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
        }
    } else {
        runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
        if constexpr (isFd) {
            runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSize;
        }
        if (unlikely(needSplitKV)) {
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseD;
        } else {
            runInfo.keyOffset = runInfo.valueOffset;
        }
        if constexpr (hasRope) {
            runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
        }
    }
}
```

> `needSplitKV` 提取为局部变量，避免 3 处重复条件。当 stride 全为 0（连续）且 `dSize==dSizeV` 时，走原 `keyOffset = valueOffset` 快速路径，不退化。

---

### 文件7: `flash_attention_score_antiquant_block_vec.h`

**路径**: `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h`

> `SetAntiqParamCommon` 是 stride 从 `constInfo` 进入 `taskParam` 的桥梁——后续搬运层（文件8）全部从 `taskParam` 读取 stride。

### 修改点1: AntiquantKey (line 561)

**现有代码**:
```cpp
    SetAntiqParamCommon(runInfo, runInfo.keyOffset, constInfo);
```

**改为**:
```cpp
    if (constInfo.keyN2Stride != 0 && layout != LayOutTypeEnum::LAYOUT_BNSD) {
        taskParam.kvStep = constInfo.n2Size * constInfo.keyN2Stride;
    }
    SetAntiqParamCommon(runInfo, runInfo.keyOffset, constInfo, true);
```

### 修改点2: AntiquantValue (line 907)

**现有代码**:
```cpp
    SetAntiqParamCommon(runInfo, runInfo.valueOffset, constInfo);
```

**改为**:
```cpp
    if (constInfo.valueN2Stride != 0 && layout != LayOutTypeEnum::LAYOUT_BNSD) {
        taskParam.kvStep = constInfo.n2Size * constInfo.valueN2Stride;
    }
    SetAntiqParamCommon(runInfo, runInfo.valueOffset, constInfo, false);
```

### 修改点3: SetAntiqParamCommon 声明 (line 118-119) + 函数体 (line 583-624)

**现有声明** (line 118-119):
```cpp
    __aicore__ inline void SetAntiqParamCommon(const RunInfo<isInfer> &runInfo, int64_t kvOffset,
                                               ConstInfo<isInfer, hasRope> &constInfo);
```

**改为**:
```cpp
    __aicore__ inline void SetAntiqParamCommon(const RunInfo<isInfer> &runInfo, int64_t kvOffset,
                                               ConstInfo<isInfer, hasRope> &constInfo, bool isKey);
```

**现有代码** (line 583-624):
```cpp
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SetAntiqParamCommon(const RunInfo<isInfer> &runInfo, int64_t kvOffset,
                                                                  ConstInfo<isInfer, hasRope> &constInfo)
{
    if (isBeforeHalf) {
        taskParam.copyTotalS = GetRealDealSize(runInfo.s2RealSize); // 2 is Vec num
    } else {
        taskParam.copyTotalS = runInfo.s2RealSize - (GetRealDealSize(runInfo.s2RealSize)); // 2 is Vec num
    }
    uint32_t curSequence = constInfo.s2BaseSize * runInfo.s2LoopCount + runInfo.kvLeftPaddingSize +
                           constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize);
    if constexpr (isInfer) {
        curSequence += runInfo.s2StartIdx;
    }
    taskParam.flashDecodeS2Idx = runInfo.flashDecodeS2Idx;
    if constexpr (isFd) {
        curSequence += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize;
    }
    taskParam.kvGmOffset = kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
    taskParam.s2BatchOffset = curSequence;
    taskParam.kvPaddingBeginOffset = runInfo.kvLeftPaddingSize;
    taskParam.antiqParamOffset = runInfo.n2oIdx * taskParam.headDim;
    taskParam.bIdx = runInfo.boIdx;
    taskParam.n2Idx = runInfo.n2oIdx;
    taskParam.s2Idx = runInfo.s2LoopCount;
    if constexpr (isInfer) {
        taskParam.s2Idx += runInfo.s2StartIdx / constInfo.s2BaseSize;
        if constexpr (enableKVPrefix) {
            taskParam.isPrefixLoop = ((runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize) <
                                      constInfo.prefixLoopCount); // 判断当前s2LoopCount是否处于prefixLoopCount
            if (taskParam.isPrefixLoop) {
                taskParam.kvGmOffset = runInfo.prefixOffset +
                                       constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
            } else {
                taskParam.kvGmOffset =
                    kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
                taskParam.s2Idx -= constInfo.prefixLoopCount;
            }
        }
    }
}
```

**改为** (新增 `bool isKey` 参数 + stride 传递 + antiqParamOffset 按 scaleN2Stride 修改):
```cpp
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SetAntiqParamCommon(const RunInfo<isInfer> &runInfo, int64_t kvOffset,
                                                                  ConstInfo<isInfer, hasRope> &constInfo, bool isKey)
{
    if (isBeforeHalf) {
        taskParam.copyTotalS = GetRealDealSize(runInfo.s2RealSize); // 2 is Vec num
    } else {
        taskParam.copyTotalS = runInfo.s2RealSize - (GetRealDealSize(runInfo.s2RealSize)); // 2 is Vec num
    }
    uint32_t curSequence = constInfo.s2BaseSize * runInfo.s2LoopCount + runInfo.kvLeftPaddingSize +
                           constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize);
    if constexpr (isInfer) {
        curSequence += runInfo.s2StartIdx;
    }
    taskParam.flashDecodeS2Idx = runInfo.flashDecodeS2Idx;
    if constexpr (isFd) {
        curSequence += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize;
    }
    taskParam.kvGmOffset = kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
    taskParam.s2BatchOffset = curSequence;
    taskParam.kvPaddingBeginOffset = runInfo.kvLeftPaddingSize;
    taskParam.bIdx = runInfo.boIdx;
    taskParam.n2Idx = runInfo.n2oIdx;
    taskParam.s2Idx = runInfo.s2LoopCount;
    // 按 isKey 区分 key/value 的 stride
    if (isKey) {
        taskParam.kvBnStride = constInfo.keyBnStride;
        taskParam.kvN2Stride = constInfo.keyN2Stride;
        taskParam.scaleBnStride = constInfo.kScaleBnStride;
        taskParam.scaleN2Stride = constInfo.kScaleN2Stride;
        // antiqParamOffset: scale 的 head 偏移，连续时 = n2oIdx * headDim
    } else {
        taskParam.kvBnStride = constInfo.valueBnStride;
        taskParam.kvN2Stride = constInfo.valueN2Stride;
        taskParam.scaleBnStride = constInfo.vScaleBnStride;
        taskParam.scaleN2Stride = constInfo.vScaleN2Stride;
    }
    // antiqParamOffset 按 scaleN2Stride 修正（连续时 scaleN2Stride=0，走原逻辑 n2oIdx*headDim）
    taskParam.antiqParamOffset = (taskParam.scaleN2Stride != 0)
        ? runInfo.n2oIdx * taskParam.scaleN2Stride
        : runInfo.n2oIdx * taskParam.headDim;
    if constexpr (isInfer) {
        taskParam.s2Idx += runInfo.s2StartIdx / constInfo.s2BaseSize;
        if constexpr (enableKVPrefix) {
            taskParam.isPrefixLoop = ((runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize) <
                                      constInfo.prefixLoopCount); // 判断当前s2LoopCount是否处于prefixLoopCount
            if (taskParam.isPrefixLoop) {
                taskParam.kvGmOffset = runInfo.prefixOffset +
                                       constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
            } else {
                taskParam.kvGmOffset =
                    kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
                taskParam.s2Idx -= constInfo.prefixLoopCount;
            }
        }
    }
}
```

---

## 7. Phase 5: 数据搬运层（文件8）

**路径**: `attention/common/op_kernel/arch35/flash_attention_score_antiquant_processor.h`

本阶段在数据搬运函数中使用 `taskParam` 中的 stride 计算 GM 地址。搬运层是 PA 布局区分（BBH/BNBD/NZ）和 blockTable 查表的**唯一发生地**——offset 计算层（Phase 4）只算 batch/head 级 core offset，本层在此基础上叠加 block 偏移和 stride 步长。

### 修改点1: AntiquantTaskParamBaseAPI 新增字段 (line 83, `bool isPrefixLoop = 0;` 后)

**现有代码** (line 81-84):
```cpp
    uint64_t kvPaddingBeginOffset;

    bool isPrefixLoop = 0;
};
```

**改为**:
```cpp
    uint64_t kvPaddingBeginOffset;

    bool isPrefixLoop = 0;
    uint64_t kvBnStride = 0;
    uint64_t kvN2Stride = 0;
    uint64_t scaleBnStride = 0;
    uint64_t scaleN2Stride = 0;
};
```

### 修改点2: CopyKV 签名变更 + srcStride 实现 (line 138 声明, line 522 实现, line 543-555 srcStride 逻辑)

**现有声明** (line 138):
```cpp
    __aicore__ inline void CopyKV(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T> &srcGm, uint64_t offset,
                                  uint32_t rowCnt, uint32_t headDim, uint32_t kvHeadNum, uint32_t paKvShapeType);
```

**改为**:
```cpp
    __aicore__ inline void CopyKV(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T> &srcGm, uint64_t offset,
                                  uint32_t rowCnt, uint32_t headDim, uint32_t kvHeadNum, uint32_t paKvShapeType,
                                  uint64_t kvN2Stride = 0);
```

**现有实现签名** (line 522):
```cpp
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyKV(
    LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T> &srcGm, uint64_t offset, uint32_t rowCnt, uint32_t headDim,
    uint32_t kvHeadNum, uint32_t paKvShapeType)
```

**改为**:
```cpp
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyKV(
    LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T> &srcGm, uint64_t offset, uint32_t rowCnt, uint32_t headDim,
    uint32_t kvHeadNum, uint32_t paKvShapeType, uint64_t kvN2Stride)
```

> 默认值 `= 0` 仅在声明处给出，实现处不重复默认值。所有调用点需传入 `taskParam.kvN2Stride`。

### 修改点2a: AntiquantBaseAPI 中 CopyKV 调用 (line 465)

**现有代码**:
```cpp
        CopyKV(tmpUb, kvGm, kvOffset, dealRowCount, taskParam.headDim, taskParam.kvHeadNum, taskParam.paKvShapeType);
```

**改为**:
```cpp
        CopyKV(tmpUb, kvGm, kvOffset, dealRowCount, taskParam.headDim, taskParam.kvHeadNum, taskParam.paKvShapeType,
               taskParam.kvN2Stride);
```

> 此调用点在非 PA 分支（`!PAGE_ATTENTION`），非 PA 场景 stride 始终为 0（设计约束），但仍需传参以保持签名一致。

**CopyKV 实现体 srcStride 改为** (line 543-555):
```cpp
    if constexpr (PAGE_ATTENTION) {
        if (paKvShapeType == 0) {  // PA + BBH
            if (kvN2Stride != 0) {
                copyInParams.srcStride = static_cast<int64_t>(kvHeadNum - 1) * kvN2Stride * sizeof(KV_T);
            } else {
                copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
            }
        } else {  // PA + BNBD/NZ
            copyInParams.srcStride = 0;
        }
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            copyInParams.srcStride = 0;
        } else {
            if (kvN2Stride != 0) {
                copyInParams.srcStride = static_cast<int64_t>(kvHeadNum - 1) * kvN2Stride * sizeof(KV_T);
            } else {
                copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
            }
        }
    }
```

> **srcStride 单位**：`copyInParams.blockLen` 已是字节（`headDim * sizeof(KV_T)`），`copyInParams.srcStride` 也是字节。`kvN2Stride` 是元素数，需 `× sizeof(KV_T)` 转换。

> 所有调用 `CopyKV` 的地方需传入 `taskParam.kvN2Stride`（共 2 处）：
> - `AntiquantBaseAPI` line 465: 已在修改点2a中改
> - `CopyKVPageAttention` line 623-624: 见修改点3

### 修改点3: CopyKVPageAttention offset (line 607-625)

> **关键修正（B6）**：原设计把 `kvBnStride` 检查包在 `if (kvN2Stride != 0)` 里。但 BBH 仅支持 bnStride 非连续（n2Stride 始终为 0），`if (kvN2Stride != 0)` 不会进入，**kvBnStride 永远无法生效**！必须把 `kvBnStride` 检查独立出来。NZ 同理——NZ 可能只有 bnStride 非连续而 n2Stride 连续。

**现有代码** (line 607-625):
```cpp
        uint64_t curOffset = 0;
        if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_NZ)) { // NZ
            curOffset = idInBlockTable * taskParam.kvHeadNum * taskParam.headDim * taskParam.kvCacheBlockSize +
                        (uint64_t)(taskParam.n2Idx * taskParam.headDim * taskParam.kvCacheBlockSize) +
                        reaminRowCnt * typeElementSize;
            CopyKVPaNz(dstLocal[copyFinishRowCnt * typeElementSize], srcGm, curOffset, copyRowCnt, dealRowCount,
                       taskParam);
        } else {
            if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_BSH)) { // BBH
                curOffset = (idInBlockTable * taskParam.kvCacheBlockSize + reaminRowCnt) * taskParam.kvHeadNum *
                                taskParam.headDim +
                            (uint64_t)(taskParam.n2Idx * taskParam.headDim);
            } else { // BNBD
                curOffset = idInBlockTable * taskParam.kvHeadNum * taskParam.kvCacheBlockSize * taskParam.headDim +
                            (uint64_t)(taskParam.n2Idx * taskParam.kvCacheBlockSize * taskParam.headDim) +
                            reaminRowCnt * taskParam.headDim;
            }
            CopyKV(dstLocal[copyFinishRowCnt * dBaseSize], srcGm, curOffset, copyRowCnt, taskParam.headDim,
                   taskParam.kvHeadNum, taskParam.paKvShapeType);
        }
```

**改为**（kvBnStride 检查独立，不依赖 kvN2Stride）:
```cpp
        uint64_t curOffset = 0;
        if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_NZ)) { // NZ
            // NZ [Bn, N2, D1, Bs, D0]: n2Stride 连续值 = D1*Bs*D0 = headDim*blockSize
            uint64_t nzHeadStride = taskParam.headDim * taskParam.kvCacheBlockSize;
            uint64_t nzBlockStride = taskParam.kvHeadNum * nzHeadStride;
            // n2Stride 非连续时替换 headStride
            if (taskParam.kvN2Stride != 0) {
                nzHeadStride = taskParam.kvN2Stride;
            }
            // bnStride 非连续时替换 blockStride（独立检查，不依赖 n2Stride）
            if (taskParam.kvBnStride != 0) {
                nzBlockStride = taskParam.kvBnStride;
            } else if (taskParam.kvN2Stride != 0) {
                nzBlockStride = taskParam.kvHeadNum * nzHeadStride;  // bnStride 连续但 n2Stride 非连续需重算
            }
            curOffset = idInBlockTable * nzBlockStride +
                        (uint64_t)(taskParam.n2Idx * nzHeadStride) +
                        reaminRowCnt * typeElementSize;
            CopyKVPaNz(dstLocal[copyFinishRowCnt * typeElementSize], srcGm, curOffset, copyRowCnt, dealRowCount,
                       taskParam);
        } else {
            if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_BSH)) { // BBH
                // BBH [Bn, Bs, N2, D]: bsStride=N2*D(固定), blockStride=Bs*N2*D
                // BBH 仅支持 bnStride 非连续，n2Stride 始终连续
                uint64_t bsStride = taskParam.kvHeadNum * taskParam.headDim;  // = N2*D
                uint64_t bnStride = taskParam.kvCacheBlockSize * bsStride;     // = Bs*N2*D
                if (taskParam.kvBnStride != 0) {
                    bnStride = taskParam.kvBnStride;
                }
                // kvN2Stride 对 BBH 始终为 0（checker 保证），无需处理
                curOffset = idInBlockTable * bnStride +
                            reaminRowCnt * bsStride +
                            (uint64_t)(taskParam.n2Idx * taskParam.headDim);
            } else { // BNBD
                // BNBD [Bn, N2, Bs, D]: headStride=Bs*D, blockStride=N2*Bs*D, bsStride=D(固定)
                uint64_t headStride = taskParam.kvCacheBlockSize * taskParam.headDim;
                uint64_t blockStride = taskParam.kvHeadNum * headStride;
                if (taskParam.kvN2Stride != 0) {
                    headStride = taskParam.kvN2Stride;
                }
                if (taskParam.kvBnStride != 0) {
                    blockStride = taskParam.kvBnStride;
                } else if (taskParam.kvN2Stride != 0) {
                    blockStride = taskParam.kvHeadNum * headStride;
                }
                curOffset = idInBlockTable * blockStride +
                            (uint64_t)(taskParam.n2Idx * headStride) +
                            reaminRowCnt * taskParam.headDim;
            }
            CopyKV(dstLocal[copyFinishRowCnt * dBaseSize], srcGm, curOffset, copyRowCnt, taskParam.headDim,
                   taskParam.kvHeadNum, taskParam.paKvShapeType, taskParam.kvN2Stride);
        }
```

> **与 PR 6430 的对应关系**：PR 6430 在 fullquant 路径用 `OffsetCalculator.GetOffset(bIdx, n2Idx, s2Idx, dIdx)` 统一算 offset（`offset_calculator_v2.h:597-606`），内部用 `blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + bsIdx * GetStrideBlockSize() + dIdx`。antiquant 无 OffsetCalculator，本修改点手动实现等价逻辑：`idInBlockTable * blockStride + n2Idx * headStride + reaminRowCnt * bsStride`。

### 修改点4: CopyAntiqParam (line 394-417)

**现有代码** (完整函数):
```cpp
template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiqParam(
    LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T> &srcGm, uint32_t rowCnt,
    const AntiquantTaskParamBaseAPI &taskParam)
{
    if (taskParam.isPertensor) {
        Duplicate(dstLocal, srcGm.GetValue(0), rowCnt * dBaseSize);
    } else if (taskParam.isPerHead) {
        Duplicate(dstLocal, srcGm.GetValue(taskParam.n2Idx), rowCnt * dBaseSize);
    } else {
        constexpr uint64_t BYTE_BLOCK = 32UL;

        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams;
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = taskParam.headDim * sizeof(ANTIQ_PARAMS_T);
        copyInParams.srcStride = static_cast<int64_t>(copyInParams.blockLen) * (-1);
        copyInParams.dstStride = (dBaseSize - taskParam.headDim) * sizeof(ANTIQ_PARAMS_T) / BYTE_BLOCK;

        copyInPadParams.isPad = false;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = 0;
        copyInPadParams.paddingValue = 0;

        DataCopyPad(dstLocal, srcGm[taskParam.antiqParamOffset], copyInParams, copyInPadParams);
    }
}
```

**改为**:
```cpp
template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiqParam(
    LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T> &srcGm, uint32_t rowCnt,
    const AntiquantTaskParamBaseAPI &taskParam)
{
    if (taskParam.isPertensor) {
        Duplicate(dstLocal, srcGm.GetValue(0), rowCnt * dBaseSize);
    } else if (taskParam.isPerHead) {
        uint64_t perHeadOffset = taskParam.n2Idx;
        if (taskParam.scaleN2Stride != 0) {
            perHeadOffset = taskParam.n2Idx * taskParam.scaleN2Stride;
        }
        Duplicate(dstLocal, srcGm.GetValue(perHeadOffset), rowCnt * dBaseSize);
    } else {
        constexpr uint64_t BYTE_BLOCK = 32UL;

        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams;
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = taskParam.headDim * sizeof(ANTIQ_PARAMS_T);
        copyInParams.srcStride = static_cast<int64_t>(copyInParams.blockLen) * (-1);
        copyInParams.dstStride = (dBaseSize - taskParam.headDim) * sizeof(ANTIQ_PARAMS_T) / BYTE_BLOCK;

        copyInPadParams.isPad = false;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = 0;
        copyInPadParams.paddingValue = 0;

        uint64_t antiqOffset = taskParam.antiqParamOffset;
        if (taskParam.scaleN2Stride != 0) {
            antiqOffset = taskParam.n2Idx * taskParam.scaleN2Stride;
        }
        DataCopyPad(dstLocal, srcGm[antiqOffset], copyInParams, copyInPadParams);
    }
}
```

> per-tensor 不受 stride 影响（标量广播 `GetValue(0)`）。per-head 新增 `scaleN2Stride` 修正 `GetValue` 偏移。per-channel 的负 srcStride（`-blockLen`）控制 head 内行重复读取，与 offset 独立，不需要修改。

### 修改点5: CopyAntiquantParamsPageAttention (line 511-512)

**现有代码**:
```cpp
        uint64_t srcOffset = idInBlockTable * taskParam.kvCacheBlockSize * useKvHeadNum +
                             taskParam.kvCacheBlockSize * useN2Idx + remainElmenCnt;
```

**改为**:
```cpp
        uint64_t blockStride = taskParam.kvCacheBlockSize * useKvHeadNum;
        uint64_t headStride = taskParam.kvCacheBlockSize;
        if (taskParam.scaleBnStride != 0) {
            blockStride = taskParam.scaleBnStride;
        }
        if (taskParam.scaleN2Stride != 0) {
            headStride = taskParam.scaleN2Stride;
        }
        uint64_t srcOffset = idInBlockTable * blockStride +
                             useN2Idx * headStride + remainElmenCnt;
```

### 修改点6: LoadAntiquantParamsPerToken scaleOffset (line 310-392)

**现有代码** (per-head 分支, line 323-335):
```cpp
    if (taskParam.isPerHead) {
        if constexpr (enableKVPrefix) {
            if (taskParam.isPrefixLoop) {
                scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.antiqSeqSize +
                              taskParam.n2Idx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                              taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
            } else {
                scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.antiqSeqSize +
                              taskParam.n2Idx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                              (taskParam.antiqSeqSize - taskParam.seqSize) + taskParam.kvPaddingBeginOffset +
                              subBlockIdx * GetRealDealSize(s2RealSize);
            }
        } else {
            scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize +
                          taskParam.n2Idx * taskParam.seqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                          taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
        }
    } else {
```

**改为**:
```cpp
    if (taskParam.isPerHead) {
        uint64_t batchStride = taskParam.kvHeadNum * taskParam.antiqSeqSize;
        uint64_t headStride = taskParam.antiqSeqSize;
        uint64_t seqStride = taskParam.singleSInnerSize;
        if (taskParam.scaleBnStride != 0) {
            batchStride = taskParam.scaleBnStride;
        }
        if (taskParam.scaleN2Stride != 0) {
            headStride = taskParam.scaleN2Stride;
        }
        if constexpr (enableKVPrefix) {
            if (taskParam.isPrefixLoop) {
                scaleOffset = taskParam.bIdx * batchStride +
                              taskParam.n2Idx * headStride + taskParam.s2Idx * seqStride +
                              taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
            } else {
                scaleOffset = taskParam.bIdx * batchStride +
                              taskParam.n2Idx * headStride + taskParam.s2Idx * seqStride +
                              (taskParam.antiqSeqSize - taskParam.seqSize) + taskParam.kvPaddingBeginOffset +
                              subBlockIdx * GetRealDealSize(s2RealSize);
            }
        } else {
            uint64_t batchStrideNP = taskParam.kvHeadNum * taskParam.seqSize;
            if (taskParam.scaleBnStride != 0) {
                batchStrideNP = taskParam.scaleBnStride;
            }
            uint64_t headStrideNP = taskParam.seqSize;
            if (taskParam.scaleN2Stride != 0) {
                headStrideNP = taskParam.scaleN2Stride;
            }
            scaleOffset = taskParam.bIdx * batchStrideNP +
                          taskParam.n2Idx * headStrideNP + taskParam.s2Idx * seqStride +
                          taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
        }
    } else {
```

### 修改点7: LoadAntiquantParamsPerToken 非 per-head 分支 (line 336-348)

**现有代码** (非 per-head, !enableKVPrefix 分支, line 345-347):
```cpp
        } else {
            scaleOffset = taskParam.bIdx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                          taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
        }
```

**改为**:
```cpp
        } else {
            uint64_t batchStride = taskParam.antiqSeqSize;
            if (taskParam.scaleBnStride != 0) {
                batchStride = taskParam.scaleBnStride;
            }
            scaleOffset = taskParam.bIdx * batchStride + taskParam.s2Idx * taskParam.singleSInnerSize +
                          taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
        }
```

> enableKVPrefix 的 isPrefixLoop / !isPrefixLoop 子分支 (line 337-343) 同理：`bIdx * antiqSeqSize` → `bIdx * (scaleBnStride != 0 ? scaleBnStride : antiqSeqSize)`。非 per-head 模式无 n2Idx 项（所有 head 共享同一 scale），不需 scaleN2Stride。

### 修改点8: LoadAntiquantParamsPerTokenGroup (line 244-308)

**现有代码** (NZ 分支, line 255-258):
```cpp
        scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize * grpNum +
                      taskParam.n2Idx * taskParam.seqSize * grpNum + taskParam.s2Idx * taskParam.singleSInnerSize * 1 +
                      taskParam.kvPaddingBeginOffset * 1 + subBlockIdx * GetRealDealSize(s2RealSize) * 1;
```

**改为**:
```cpp
        uint64_t batchStride = taskParam.kvHeadNum * taskParam.seqSize * grpNum;
        uint64_t headStride = taskParam.seqSize * grpNum;
        if (taskParam.scaleBnStride != 0) {
            batchStride = taskParam.scaleBnStride * grpNum;
        }
        if (taskParam.scaleN2Stride != 0) {
            headStride = taskParam.scaleN2Stride * grpNum;
        }
        scaleOffset = taskParam.bIdx * batchStride +
                      taskParam.n2Idx * headStride + taskParam.s2Idx * taskParam.singleSInnerSize * 1 +
                      taskParam.kvPaddingBeginOffset * 1 + subBlockIdx * GetRealDealSize(s2RealSize) * 1;
```

**现有代码** (非 NZ, !enableKVPrefix 分支, line 260-265):
```cpp
            scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize * grpNum +
                          taskParam.n2Idx * taskParam.seqSize * grpNum +
                          taskParam.s2Idx * taskParam.singleSInnerSize * grpNum +
                          taskParam.kvPaddingBeginOffset * grpNum + subBlockIdx * GetRealDealSize(s2RealSize) * grpNum;
```

**改为**:
```cpp
            uint64_t batchStrideNP = taskParam.kvHeadNum * taskParam.seqSize * grpNum;
            uint64_t headStrideNP = taskParam.seqSize * grpNum;
            if (taskParam.scaleBnStride != 0) {
                batchStrideNP = taskParam.scaleBnStride * grpNum;
            }
            if (taskParam.scaleN2Stride != 0) {
                headStrideNP = taskParam.scaleN2Stride * grpNum;
            }
            scaleOffset = taskParam.bIdx * batchStrideNP +
                          taskParam.n2Idx * headStrideNP +
                          taskParam.s2Idx * taskParam.singleSInnerSize * grpNum +
                          taskParam.kvPaddingBeginOffset * grpNum + subBlockIdx * GetRealDealSize(s2RealSize) * grpNum;
```

> enableKVPrefix 的 isPrefixLoop / !isPrefixLoop 子分支 (line 250-255) 同理：用 `scaleBnStride * grpNum` 替代 `kvHeadNum * antiqSeqSize * grpNum`，用 `scaleN2Stride * grpNum` 替代 `antiqSeqSize * grpNum`。

---

## 8. 实施顺序

```
Phase 1: Tiling 结构 (第3节, 文件1+2) — InputParamsRegbase + COMMON_CONST_INFO + CVSharedParams 新增 kScale/vScale stride 字段
Phase 2: Host (第4节, 文件3+4) — SetFATilingData 写 stride(用 !isTensorV1 替代 hasViewStride) + CheckStrideForAntiquant(按 BBH/BNBD/NZ 区分)
Phase 3: Kernel 接线 (第5节, 文件5) — ComputeConstexpr 无条件赋值 constInfo stride 字段(补全 antiquant 路径缺失的赋值)
Phase 4: Offset 计算 (第6节, 文件6+7) — CalcValueCoreOffset* + GetKeyCoreOffsetParam + SyncKeyCoreOffset + ComputeOffsetForAntiquant + ComputeOffsetPrefixAIC + ComputeOffsetNonPrefixAIC(AIC 侧补全) + SetAntiqParamCommon
Phase 5: 数据搬运 (第7节, 文件8) — AntiquantTaskParamBaseAPI + CopyKV + CopyKVPageAttention(kvBnStride 独立检查) + CopyAntiqParam + CopyAntiquantParamsPageAttention + LoadAntiquantParams*
Phase 6: 测试 — 冒烟(连续不退化) → PA BNBD/NZ 0轴/1轴非连续 → PA BBH 仅0轴 → 非 PA 报错 → BBH 1轴非连续报错
```

---

## 9. 核心风险

1. **连续不退化**：stride=0 走原逻辑，constInfo 字段无条件赋值(0=连续)，`!= 0` 判断天然跳过非连续分支
2. **BBH 仅 0 轴**：BBH 的 n2Stride 始终为 0（checker 保证），bsStride 固定，CopyKVPageAttention 的 BBH 分支 kvBnStride 检查必须独立（不能包在 kvN2Stride!=0 里）
3. **srcStride 单位**：constInfo stride 是元素数，`copyInParams.srcStride` 是字节，需 `× sizeof(KV_T)` 转换
4. **CopyKV 签名变更**：新增 `kvN2Stride` 参数，2 个调用点(AntiquantBaseAPI line 465 + CopyKVPageAttention line 623)同步修改
5. **SetAntiqParamCommon 签名变更**：新增 `bool isKey` 参数区分 key/value stride + antiqParamOffset 按 scaleN2Stride 修正
6. **AIC 侧补全**：ComputeOffsetPrefixAIC(2处) + ComputeOffsetNonPrefixAIC(3处) 的 keyOffset=valueOffset 需拆分，否则 AIC 侧读错 key offset
7. **constInfo stride 初始化**：antiquant 的 ComputeConstexpr 从未给 stride 字段赋值(垃圾值)，必须在 Phase 3 补全无条件赋值
8. **hasViewStride 不存在**：全代码库无此字段，必须用 `!fiaInfo.isTensorV1` 替代

---

## 10. 文件清单

| # | 文件 | 路径 | 修改点数 | Phase/节 |
|---|------|------|---------|-------|
| 1 | `flash_attention_score_tiling_regbase.h` | `attention/common/op_kernel/arch35/` | 2 | P1/§3 |
| 2 | `util_regbase.h` | `attention/common/op_kernel/arch35/` | 2 | P1/§3 |
| 3 | `fused_infer_attention_score_tiling_impl.cpp` | `attention/fused_infer_attention_score/op_host/arch35/` | 1 | P2/§4 |
| 4 | `dequant_checker.cpp` + `.h` | `attention/fused_infer_attention_score/op_host/checkers/` | 3 | P2/§4 |
| 5 | `flash_attention_score_antiquant_kernel.h` | `attention/common/op_kernel/arch35/` | 1 | P3/§5 |
| 6 | `infer_flash_attention_kvcache.h` | `attention/common/op_kernel/arch35/` | 7(+2 AIC) | P4/§6 |
| 7 | `flash_attention_score_antiquant_block_vec.h` | `attention/common/op_kernel/arch35/` | 4 | P4/§6 |
| 8 | `flash_attention_score_antiquant_processor.h` | `attention/common/op_kernel/arch35/` | 10 | P5/§7 |
