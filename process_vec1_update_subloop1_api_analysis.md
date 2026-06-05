# process_vec1_update_subloop1 函数 Ascend API 接口技术文档

## 1. 函数概述

- **函数名**: `process_vec1_update_subloop1`
- **源文件**: `/Users/cc/Downloads/ops-transformer/ops-transformer/attention/mx_fp8.h`
- **代码行号**: L1967-L2328
- **模板参数**: `T = float (fp32)`, `T2 = fp8_e4m3fn_t (fp8)`
- **功能描述**: 对注意力分数进行 scale 缩放 → 应用 attention mask → 计算行最大值 → 计算 exp(max_old - max_new) 更新因子 → 计算 softmax exp 值 → 将 float 转 fp8 并合并 even/odd → 存储到 UB → 更新 softmax 的 sum 和 max 累积值

---

## 2. 高阶 API 识别总表

| 序号 | API 类别 | API 名称 | 调用位置(行号) | 模板参数实例化 |
|------|---------|---------|---------------|--------------|
| 1 | Mask操作 | `CreateMask` | L2050, L2051, L2062, L2283 | `<float, ALL>`, `<uint16_t, ALL>`, `<int8_t, ALLF>`, `<T2, ALL>` |
| 2 | Mask操作 | `UpdateMask` | L2054, L2055, L2056, L2285 | `<T>(pltOriTailN1)`, `<T>(pltTailN2)`, `<T>(pltOriTailN2)`, `<T2>(128)` |
| 3 | 数据加载 | `LoadAlign` | L2086-2089, L2152-2159, L2204, L2206, L2209, L2220, L2222-2224, L2294, L2319, L2321, L2322 | `<T>`, `<T, DIST_BRC_B32>`, `<T, DIST_DINTLV_B32>`, `<uint32_t, POST_MODE_UPDATE, DIST_DS>`, `<uint8_t>` |
| 4 | 数据存储 | `StoreAlign` | L2166-2173, L2211-2212, L2297-2300, L2327 | `<T, DIST_NORM_B32>`, `<T2, DATA_BLOCK_COPY, POST_MODE_UPDATE>` |
| 5 | 数据存储 | `StoreUnAlign` | L2199-2200, L2236-2237 | `<float, POST_MODE_UPDATE>` |
| 6 | 数据存储 | `StoreUnAlignPost` | L2202, L2316 | `<float, POST_MODE_UPDATE>` |
| 7 | 标量广播 | `Duplicate` | L2070, L2073 | `<T, U, S>` |
| 8 | 数学运算 | `Ln` | L2071 | `<T, auto mode, U>` |
| 9 | 数学运算 | `Muls` | L2091-2094 | `<T, U, mode, S>` |
| 10 | 数学运算 | `Select` | L2160-2165 | `<T, U>` |
| 11 | 数学运算 | `Max` | L2174-2176, L2207 | `<T, mode, U>` |
| 12 | 数学运算 | `ExpSub` | L2208, L2210, L2225-2228 | `<T, U, layout, mode, S, V>` |
| 13 | 数学运算 | `Sub` | L2221 | `<T, mode, U>` |
| 14 | 数学运算 | `Add` | L2231-2233, L2324, L2326 | `<T, mode, U>` |
| 15 | 数学运算 | `Mul` | L2323, L2325 | `<T, mode, U>` |
| 16 | 数学运算 | `Or` | L2290-2293 | `<T, mode, U>` |
| 17 | 类型转换 | `Cast` | L2286-2289 | `<T2, T, castTraitRintZero>`, `<T2, T, castTraitRintTwo>` |
| 18 | 规约运算 | `Reduce` | L2177-2178, L2234-2235 | `<ReduceType::MAX, float, float, ZEROING>`, `<ReduceType::SUM, float, float, ZEROING>` |
| 19 | 数据重排 | `Gather` | L2295-2296 | `<T, U, S, V>` |
| 20 | 内存同步 | `LocalMemBar` | L2205, L2217, L2318 | `<MemType::VEC_STORE, MemType::VEC_LOAD>` |

---

## 3. 调用链路追踪与底层实现定位

### 3.1 CreateMask

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `CreateMask<float, MaskPattern::ALL>()` | L2050 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_maskreg_intf_impl.h` | `CreateMask<T, mode, regTrait>()` → 调用 `CreateMaskImpl<T, mode, regTrait>()` | L43-47 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_maskreg_impl.h` | `CreateMaskImpl<T, mode, regTrait>()` | L83-109 |

**底层实现细节**:
- 当 `T = float (sizeof=4)`, `mode = MaskPattern::ALL` 时:
  - 调用 `pset_b32(modeValue)`, 其中 `modeValue` 为 `std::integral_constant<::Pat, static_cast<::Pat>(MaskPattern::ALL)>`
- 当 `T = uint16_t (sizeof=2)` 时:
  - 调用 `pset_b16(modeValue)`
- 当 `T = int8_t (sizeof=1)`, `mode = MaskPattern::ALLF` 时:
  - 调用 `pset_b8(modeValue)`
- 当 `T = fp8_e4m3fn_t (sizeof=1)` 时:
  - 调用 `pset_b8(modeValue)`

**调用关系**: `CreateMask` → `CreateMaskImpl` → `pset_b32/pset_b16/pset_b8`

---

### 3.2 UpdateMask

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `UpdateMask<T>(pltOriTailN1)` | L2054 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_maskreg_intf_impl.h` | `UpdateMask<T, regTrait>(scalarValue)` → 调用 `UpdateMaskImpl<T, regTrait>(scalarValue)` | L37-41 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_maskreg_impl.h` | `UpdateMaskImpl<T, regTrait>(scalarValue)` | L27-62 |

**底层实现细节**:
- 当 `T = float (sizeof=4)` 时:
  - 调用 `plt_b32(scalarValue, POST_UPDATE)`
- 当 `T = fp8_e4m3fn_t (sizeof=1)` 时:
  - 调用 `plt_b8(scalarValue, POST_UPDATE)`

**调用关系**: `UpdateMask` → `UpdateMaskImpl` → `plt_b32/plt_b16/plt_b8`

---

### 3.3 LoadAlign

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `LoadAlign(vreg, addr)` 等 | L2086-2322 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_datacopy_intf_impl.h` | `LoadAlign<T, dist, U>()` → 调用 `DataCopyImpl<T, dist, U>()` | L48-106 |
| **底层实现(向量寄存器)** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_datacopy_load_impl.h` | `DataCopyImpl<U>(dstReg, srcAddr)` 等 | L47-150+ |
| **底层实现(MaskReg)** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_datacopy_impl.h` | `DataCopyImpl<T, dist>(mask, srcAddr)` 等 | L194-224 |

**底层实现细节** (按使用场景):

| 使用场景 | 底层内联调用 | 实现文件行号 |
|---------|------------|------------|
| `LoadAlign<T>(vreg, addr)` (float, 普通加载) | `vlds(dstReg, srcAddr, 0, distValue)` | datacopy_load_impl.h L47-84 |
| `LoadAlign<T, DIST_BRC_B32>(vreg, addr)` (广播加载) | `vlds(dstReg, srcAddr, 0, distValue)` (dist=BRC_B32) | datacopy_load_impl.h L47-84 |
| `LoadAlign<T, DIST_DINTLV_B32>(vreg1, vreg2, addr)` (解交织加载) | `vlds(dstReg0, dstReg1, srcAddr, 0, dintlvDist)` | datacopy_load_impl.h (双输出重载) |
| `LoadAlign<uint32_t, POST_MODE_UPDATE, DIST_DS>(maskReg, addr, stride)` | `plds(mask, srcAddr, offset, distValue, postValue)` | datacopy_impl.h L215-224 |
| `LoadAlign<uint8_t>(vreg, addr)` | `vlds((RegTensor<uint8_t>&)dstReg, srcAddr, 0, distValue)` | datacopy_load_impl.h L55-56 |

**调用关系**: `LoadAlign` → `DataCopyImpl` → `vlds/vld/plds/pld`

---

### 3.4 StoreAlign

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `StoreAlign<T, DIST_NORM_B32>(addr, vreg, mask)` 等 | L2166-2327 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_datacopy_intf_impl.h` | `StoreAlign<T, dist, U>()` → 调用 `DataCopyImpl<T, dist, U>()` | L116-215 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_datacopy_store_impl.h` | `DataCopyImpl<T, dist, U>(dstAddr, srcReg, mask)` 等 | L46-325 |

**底层实现细节**:

| 使用场景 | 底层内联调用 | 实现文件行号 |
|---------|------------|------------|
| `StoreAlign<T, DIST_NORM_B32>(addr, vreg, mask)` (float) | `vsts((RegTensor<int32_t>&)srcReg, dstAddr, 0, distValue, mask)` | datacopy_store_impl.h L46-99 |
| `StoreAlign<T2, DATA_BLOCK_COPY, POST_MODE_UPDATE>(addr, vreg, blockStride, repeatStride, mask)` (fp8) | `vsstb(srcReg, dstAddr, (dataBlockStride<<16)\|(repeatStride&0xFFFF), mask, postValue)` | datacopy_store_impl.h L296-325 |

**调用关系**: `StoreAlign` → `DataCopyImpl` → `vsts/vst/vsstb`

---

### 3.5 StoreUnAlign

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `StoreUnAlign<float, POST_MODE_UPDATE>(addr, vreg, ureg, count)` | L2199-2200, L2236-2237 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_datacopy_intf_impl.h` | `StoreUnAlign<T, postMode, U>()` → 调用 `DataCopyUnAlignImpl<T, postMode, U>()` | L290-350 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_datacopy_store_impl.h` | `DataCopyUnAlignImpl<T, postMode, U>(dstAddr, srcReg, ureg, postUpdateStride)` | L328-373 |

**底层实现细节**:
- 当 `T = float (sizeof=4)` 时:
  - 调用 `vstus(ureg, postUpdateStride, (RegTensor<int32_t>&)srcReg, dstAddr, postValue)`

**调用关系**: `StoreUnAlign` → `DataCopyUnAlignImpl` → `vstus`

---

### 3.6 StoreUnAlignPost

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `StoreUnAlignPost<float, POST_MODE_UPDATE>(addr, ureg, offset)` | L2202, L2316 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_datacopy_intf_impl.h` | `StoreUnAlignPost<T, postMode>()` → 调用 `DataCopyUnAlignPostImpl<T, postMode>()` | L301-361 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_datacopy_store_impl.h` | `DataCopyUnAlignPostImpl<T, postMode>(dstAddr, ureg, postUpdateStride)` | L375-399 |

**底层实现细节**:
- 当 `T = float (sizeof=4)` 时:
  - 调用 `vstas(ureg, dstAddr, postUpdateStride, POST_UPDATE)`

**调用关系**: `StoreUnAlignPost` → `DataCopyUnAlignPostImpl` → `vstas`

---

### 3.7 Duplicate

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Duplicate(vreg, value)` | L2070, L2073 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_duplicate_intf_impl.h` | `Duplicate<T, U, S>(dstReg, scalarValue)` → 调用 `DuplicateImpl<T, U, S>(dstReg, scalarValue)` | L38-40 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_duplicate_impl.h` | `DuplicateImpl<T, U, S>(dstReg, scalarValue)` | L63-101 |

**底层实现细节**:
- 当 `T = float (sizeof=4)`, 无 mask 版本:
  - 调用 `vbr(dstReg, scalarValue)`

**调用关系**: `Duplicate` → `DuplicateImpl` → `vbr` (无mask) / `vdup` (有mask)

---

### 3.8 Ln

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Ln(vreg_ln_p_scale, vreg_p_scale, preg_all)` | L2071 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_unary_intf_impl.h` | `Ln<T, mode, U>(dstReg, srcReg, mask)` → 调用 `LogImpl<T, mode, U>(dstReg, srcReg, mask)` | L68-71 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_unary_impl.h` | `LogImpl<T, mode, U>(dstReg, srcReg, mask)` | L517-547 |

**底层实现细节**:
- 当 `T = float`, `mode = MaskMergeMode::ZEROING` 时:
  - 调用 `vln(dstReg, srcReg, mask, modeValue)`

**调用关系**: `Ln` → `LogImpl` → `vln`

---

### 3.9 Muls

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Muls(vreg, vreg, scale, preg_all)` | L2091-2094 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_binary_scalar_intf_impl.h` | `Muls<T, U, mode, S>(dstReg, srcReg, scalarValue, mask)` → 调用 `MulsImpl<T, U, mode, S>(dstReg, srcReg, scalarValue, mask)` | L43-47 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_scalar_impl.h` | `MulsImpl<T, U, mode, S>(dstReg, srcReg, scalarValue, mask)` | L139-191 |

**底层实现细节**:
- 当 `T = float (sizeof=4)`, 非 complex 时:
  - 调用 `vmuls(dstReg, srcReg, scalarValue, mask, modeValue)`

**调用关系**: `Muls` → `MulsImpl` → `vmuls`

---

### 3.10 Select

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Select(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1)` | L2160-2165 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_cmpsel_intf_impl.h` | `Select<T, U>(dstReg, srcReg0, srcReg1, mask)` → 调用 `SelectImpl<T, U>(dstReg, srcReg0, srcReg1, mask)` | L54-58 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_cmpsel_impl.h` | `SelectImpl<T, U>(dstReg, srcReg0, srcReg1, mask)` | L191-225 |

**底层实现细节**:
- 当 `T = float (sizeof=4)` 时:
  - 调用 `vsel(dstReg, srcReg0, srcReg1, mask)` (RegTensor\<uint32_t\>)

**调用关系**: `Select` → `SelectImpl` → `vsel`

---

### 3.11 Max

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Max(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all)` | L2174-2176, L2207 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_binary_intf_impl.h` | `Max<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` → 调用 `MaxImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L64-68 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h` | `MaxImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L1460-1476 |

**底层实现细节**:
- 当 `T = float (sizeof=4)`, `mode = MaskMergeMode::ZEROING` 时:
  - `MaxImpl` → `MaxOperator` → 调用 `vmax(dstReg, srcReg0, srcReg1, mask, modeValue)`

**调用关系**: `Max` → `MaxImpl` → `MaxOperator` → `vmax`

---

### 3.12 ExpSub

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `ExpSub(vreg_subloop_update, vreg_in_max, vreg_max_new, preg_all)` | L2208, L2210, L2225-2228 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_fused_intf_impl.h` | `ExpSub<T, U, layout, mode, S, V>(dstReg, srcReg0, srcReg1, mask)` → 调用 `FusedExpSubImpl<T, U, layout, mode, S, V>(dstReg, srcReg0, srcReg1, mask)` | L66-70 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_fused_impl.h` | `FusedExpSubImpl<T, U, layout, mode, S, V>(dstReg, srcReg0, srcReg1, mask)` | L78-99 |

**底层实现细节**:
- 当 `T = float`, `U = float`, `layout = RegLayout::ZERO`, `mode = MaskMergeMode::ZEROING` 时:
  - 调用 `vexpdif(dstReg, srcReg0, srcReg1, mask, partModeValue)`

**调用关系**: `ExpSub` → `FusedExpSubImpl` → `vexpdif`

---

### 3.13 Sub

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Sub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all)` | L2221 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_binary_intf_impl.h` | `Sub<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` → 调用 `SubImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L46-50 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h` | `SubImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L226-256 |

**底层实现细节**:
- 当 `T = float (sizeof=4)`, `mode = MaskMergeMode::ZEROING` 时:
  - 调用 `vsub(dstReg, srcReg0, srcReg1, mask, modeValue)`

**调用关系**: `Sub` → `SubImpl` → `vsub`

---

### 3.14 Add

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all)` | L2231-2233, L2324, L2326 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_binary_intf_impl.h` | `Add<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` → 调用 `AddImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L40-44 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h` | `AddImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L144-161 |

**底层实现细节**:
- 当 `T = float (sizeof=4)`, `mode = MaskMergeMode::ZEROING` 时:
  - `AddImpl` → `AddOperator` → 调用 `vadd(dstReg, srcReg0, srcReg1, mask, modeValue)`

**调用关系**: `Add` → `AddImpl` → `AddOperator` → `vadd`

---

### 3.15 Mul

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Mul(vreg_first_sum, vreg_subloop_update, vreg_first_sum, preg_all)` | L2323, L2325 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_binary_intf_impl.h` | `Mul<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` → 调用 `MulImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L52-56 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h` | `MulImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L347-377 |

**底层实现细节**:
- 当 `T = float (sizeof=4)`, `mode = MaskMergeMode::ZEROING` 时:
  - 调用 `vmul(dstReg, srcReg0, srcReg1, mask, modeValue)`

**调用关系**: `Mul` → `MulImpl` → `vmul`

---

### 3.16 Or

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Or((RegTensor<uint8_t>&)dst, (RegTensor<uint8_t>&)src0, (RegTensor<uint8_t>&)src1, preg_all_b8)` | L2290-2293 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_binary_intf_impl.h` | `Or<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` → 调用 `OrImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L94-98 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h` | `OrImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask)` | L1741-1772 |

**底层实现细节**:
- 当 `ActualT = uint8_t (sizeof=1)`, `mode = MaskMergeMode::ZEROING` 时:
  - 调用 `vor(dstReg, srcReg0, srcReg1, mask, modeValue)`

**调用关系**: `Or` → `OrImpl` → `vor`

---

### 3.17 Cast

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Cast<T2, T, castTraitRintZero>(dst, src, mask)` / `Cast<T2, T, castTraitRintTwo>(dst, src, mask)` | L2286-2289 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_vconv_intf_impl.h` | `Cast<T, U, trait, S, V>(dstReg, srcReg, mask)` → 调用 `CastImpl<T, U, trait>(dstReg, srcReg, mask)` | L45-46 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_vconv_impl.h` | `CastImpl<T, U, trait, S, V>(dstReg, srcReg, mask)` (通用分发版本) | L501-572 |

**底层实现细节**:
- 当 `T2 = fp8_e4m3fn_t`, `T = float`, `trait = castTraitRintZero` (向零舍入) 时:
  - 最终调用 `vcvt(dstReg, srcReg, mask, roundMode, satMode, partMode, modeValue)`, roundMode 为 `ROUND_Z` (向零舍入)
- 当 `T2 = fp8_e4m3fn_t`, `T = float`, `trait = castTraitRintTwo` (向偶舍入) 时:
  - 最终调用 `vcvt(dstReg, srcReg, mask, roundMode, satMode, partMode, modeValue)`, roundMode 为 `ROUND_R` (向偶舍入)

**调用关系**: `Cast` → `CastImpl` (通用分发) → `CastImpl` (含RoundMode+SatMode+layoutMode+mode重载) → `vcvt`

---

### 3.18 Reduce

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Reduce<ReduceType::MAX, float, float, ZEROING>(dst, src, mask)` / `Reduce<ReduceType::SUM, float, float, ZEROING>(dst, src, mask)` | L2177-2178, L2234-2235 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_vec_reduce_intf_impl.h` | `Reduce<type, T, U, mode, S, V>(dstReg, srcReg, mask)` → 根据 type 分发到 `ReduceMaxImpl` 或 `ReduceSumImpl` | L57-65 |
| **底层实现(MAX)** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_reduce_impl.h` | `ReduceMaxImpl<T, mode, U>(dstReg, srcReg, mask)` | L99-127 |
| **底层实现(SUM)** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_reduce_impl.h` | `ReduceSumImpl<T, U, mode, S, V>(dstReg, srcReg, mask)` | L33-65 |

**底层实现细节**:
- `Reduce<MAX>`: 当 `T = float (sizeof=4)` 时 → 调用 `vcmax(dstReg, srcReg, mask)`
- `Reduce<SUM>`: 当 `T = float (sizeof=4)` 时 → 调用 `vcadd(dstReg, srcReg, mask)`

**调用关系**:
- `Reduce<MAX>` → `ReduceMaxImpl` → `vcmax`
- `Reduce<SUM>` → `ReduceSumImpl` → `vcadd`

---

### 3.19 Gather

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `Gather(vreg_dst, vreg_src, vreg_indexes)` | L2295-2296 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_gather_mask_intf_impl.h` | `Gather<T, U, S, V>(dstReg, srcReg, indexReg)` → 调用 `GatherImpl<T, U, S, V>(dstReg, srcReg, indexReg)` | L72-74 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_gather_mask_impl.h` | `GatherImpl<T, U, S, V>(dstReg, srcReg, indexReg)` | L69-91 |

**底层实现细节**:
- 当 `ActualT = fp8_e4m3fn_t (sizeof=1)` 时:
  - 调用 `vselr((RegTensor<uint8_t>&)dstReg, (RegTensor<uint8_t>&)srcReg, indexReg)`

**调用关系**: `Gather` → `GatherImpl` → `vselr`

---

### 3.20 LocalMemBar

| 层级 | 文件 | 函数 | 行号 |
|------|------|------|------|
| **高阶API** | `mx_fp8.h` | `LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()` | L2205, L2217, L2318 |
| **接口层** | `asc-devkit/impl/basic_api/reg_compute/kernel_reg_compute_membar_intf_impl.h` | `LocalMemBar<src, dst>()` → 调用 `LocalMemBarImpl<src, dst>()` | L46-48 |
| **底层实现** | `asc-devkit/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_membar_impl.h` | `LocalMemBarImpl<src, dst>()` | L31-71 |

**底层实现细节**:
- 当 `src = VEC_STORE`, `dst = VEC_LOAD` 时:
  - 调用 `mem_bar(VST_VLD)`

**调用关系**: `LocalMemBar` → `LocalMemBarImpl` → `mem_bar`

---

## 4. 模板参数验证

### 4.1 验证总表

| API | 模板参数实例化 | sizeof(T) | 底层分支 | 是否匹配 |
|-----|--------------|-----------|---------|---------|
| `CreateMask<float, ALL>` | T=float | 4 | `pset_b32` | ✅ |
| `CreateMask<uint16_t, ALL>` | T=uint16_t | 2 | `pset_b16` | ✅ |
| `CreateMask<int8_t, ALLF>` | T=int8_t | 1 | `pset_b8` | ✅ |
| `CreateMask<T2, ALL>` | T2=fp8_e4m3fn_t | 1 | `pset_b8` | ✅ |
| `UpdateMask<T>(...)` | T=float | 4 | `plt_b32` | ✅ |
| `UpdateMask<T2>(128)` | T2=fp8_e4m3fn_t | 1 | `plt_b8` | ✅ |
| `LoadAlign<T>(vreg, addr)` | T=float | 4 | `vlds` | ✅ |
| `LoadAlign<T, DIST_BRC_B32>` | T=float | 4 | `vlds` (BRC_B32) | ✅ |
| `LoadAlign<T, DIST_DINTLV_B32>` | T=float | 4 | `vlds` (DINTLV_B32) | ✅ |
| `LoadAlign<uint32_t, POST, DIST_DS>` | T=uint32_t | 4 | `plds` | ✅ |
| `LoadAlign<uint8_t>` | T=uint8_t | 1 | `vlds` | ✅ |
| `StoreAlign<T, DIST_NORM_B32>` | T=float | 4 | `vsts` | ✅ |
| `StoreAlign<T2, DATA_BLOCK_COPY, POST>` | T2=fp8_e4m3fn_t | 1 | `vsstb` | ✅ |
| `StoreUnAlign<float, POST>` | T=float | 4 | `vstus` | ✅ |
| `StoreUnAlignPost<float, POST>` | T=float | 4 | `vstas` | ✅ |
| `Duplicate<T, U, S>` | T=float | 4 | `vbr` | ✅ |
| `Ln<T, mode, U>` | T=float | 4 | `vln` | ✅ |
| `Muls<T, U, mode, S>` | T=float | 4 | `vmuls` | ✅ |
| `Select<T, U>` | T=float | 4 | `vsel` | ✅ |
| `Max<T, mode, U>` | T=float | 4 | `vmax` | ✅ |
| `ExpSub<T, U, layout, mode, S, V>` | T=float, U=float | 4 | `vexpdif` | ✅ |
| `Sub<T, mode, U>` | T=float | 4 | `vsub` | ✅ |
| `Add<T, mode, U>` | T=float | 4 | `vadd` | ✅ |
| `Mul<T, mode, U>` | T=float | 4 | `vmul` | ✅ |
| `Or<T, mode, U>` | ActualT=uint8_t | 1 | `vor` | ✅ |
| `Cast<T2, T, castTraitRintZero>` | T2=fp8_e4m3fn_t, T=float | 1/4 | `vcvt` | ✅ |
| `Cast<T2, T, castTraitRintTwo>` | T2=fp8_e4m3fn_t, T=float | 1/4 | `vcvt` | ✅ |
| `Reduce<MAX, float, float, ZEROING>` | T=float | 4 | `vcmax` | ✅ |
| `Reduce<SUM, float, float, ZEROING>` | T=float | 4 | `vcadd` | ✅ |
| `Gather<T, U, S, V>` | ActualT=fp8_e4m3fn_t | 1 | `vselr` | ✅ |
| `LocalMemBar<VEC_STORE, VEC_LOAD>` | - | - | `mem_bar(VST_VLD)` | ✅ |

### 4.2 验证结论

所有模板参数类型均与 `T=fp32`、`T2=fp8_e4m3fn_t` 的使用场景相匹配：
- `sizeof(float) == 4` → 所有 float 相关操作均走 b32 分支
- `sizeof(fp8_e4m3fn_t) == 1` → 所有 fp8 相关操作均走 b8 分支
- `Cast<fp8_e4m3fn_t, float, ...>` → 从 float (b32) 转换到 fp8_e4m3fn_t (b8)，合法
- `Or` 操作中对 `(RegTensor<uint8_t>&)` 的强制转换 → fp8_e4m3fn_t 底层存储为 uint8_t，逐位或运算用于合并 even/odd 部分，合法

---

## 5. 接口关系总览

### 5.1 调用链路汇总图

```
process_vec1_update_subloop1 (mx_fp8.h:1967-2328)
│
├── CreateMask<T, mode>  (maskreg_intf_impl.h:43-47)
│   └── CreateMaskImpl<T, mode, regTrait>  (dav_3510/maskreg_impl.h:83-109)
│       └── pset_b32 / pset_b16 / pset_b8
│
├── UpdateMask<T>(scalarValue)  (maskreg_intf_impl.h:37-41)
│   └── UpdateMaskImpl<T, regTrait>  (dav_3510/maskreg_impl.h:27-62)
│       └── plt_b32 / plt_b16 / plt_b8
│
├── LoadAlign<T, dist>(vreg, addr)  (datacopy_intf_impl.h:48-106)
│   └── DataCopyImpl<T, dist, U>  (dav_3510/datacopy_load_impl.h:47-150+)
│       └── vlds / vld
│
├── LoadAlign<uint32_t, POST, DIST_DS>(maskReg, addr, stride)  (datacopy_intf_impl.h:433-435)
│   └── DataCopyImpl<T, postMode, dist>  (dav_3510/datacopy_impl.h:215-224)
│       └── plds
│
├── StoreAlign<T, dist>(addr, vreg, mask)  (datacopy_intf_impl.h:116-215)
│   └── DataCopyImpl<T, dist, U>  (dav_3510/datacopy_store_impl.h:46-325)
│       └── vsts / vst / vsstb
│
├── StoreUnAlign<T, postMode>(addr, vreg, ureg, count)  (datacopy_intf_impl.h:290-350)
│   └── DataCopyUnAlignImpl<T, postMode, U>  (dav_3510/datacopy_store_impl.h:328-373)
│       └── vstus
│
├── StoreUnAlignPost<T, postMode>(addr, ureg, offset)  (datacopy_intf_impl.h:301-361)
│   └── DataCopyUnAlignPostImpl<T, postMode>  (dav_3510/datacopy_store_impl.h:375-399)
│       └── vstas
│
├── Duplicate<T, U, S>(dstReg, scalarValue)  (vec_duplicate_intf_impl.h:38-40)
│   └── DuplicateImpl<T, U, S>  (dav_3510/vec_duplicate_impl.h:63-101)
│       └── vbr / vdup
│
├── Ln<T, mode, U>(dstReg, srcReg, mask)  (vec_unary_intf_impl.h:68-71)
│   └── LogImpl<T, mode, U>  (dav_3510/vec_unary_impl.h:517-547)
│       └── vln
│
├── Muls<T, U, mode, S>(dstReg, srcReg, scalar, mask)  (vec_binary_scalar_intf_impl.h:43-47)
│   └── MulsImpl<T, U, mode, S>  (dav_3510/vec_binary_scalar_impl.h:139-191)
│       └── vmuls
│
├── Select<T, U>(dstReg, srcReg0, srcReg1, mask)  (vec_cmpsel_intf_impl.h:54-58)
│   └── SelectImpl<T, U>  (dav_3510/vec_cmpsel_impl.h:191-225)
│       └── vsel
│
├── Max<T, mode, U>(dstReg, srcReg0, srcReg1, mask)  (vec_binary_intf_impl.h:64-68)
│   └── MaxImpl<T, mode, U>  (dav_3510/vec_binary_impl.h:1460-1476)
│       └── MaxOperator → vmax
│
├── ExpSub<T, U, layout, mode, S, V>(dstReg, srcReg0, srcReg1, mask)  (vec_fused_intf_impl.h:66-70)
│   └── FusedExpSubImpl<T, U, layout, mode, S, V>  (dav_3510/vec_fused_impl.h:78-99)
│       └── vexpdif
│
├── Sub<T, mode, U>(dstReg, srcReg0, srcReg1, mask)  (vec_binary_intf_impl.h:46-50)
│   └── SubImpl<T, mode, U>  (dav_3510/vec_binary_impl.h:226-256)
│       └── vsub
│
├── Add<T, mode, U>(dstReg, srcReg0, srcReg1, mask)  (vec_binary_intf_impl.h:40-44)
│   └── AddImpl<T, mode, U>  (dav_3510/vec_binary_impl.h:144-161)
│       └── AddOperator → vadd
│
├── Mul<T, mode, U>(dstReg, srcReg0, srcReg1, mask)  (vec_binary_intf_impl.h:52-56)
│   └── MulImpl<T, mode, U>  (dav_3510/vec_binary_impl.h:347-377)
│       └── vmul
│
├── Or<T, mode, U>(dstReg, srcReg0, srcReg1, mask)  (vec_binary_intf_impl.h:94-98)
│   └── OrImpl<T, mode, U>  (dav_3510/vec_binary_impl.h:1741-1772)
│       └── vor
│
├── Cast<T2, T, trait>(dstReg, srcReg, mask)  (vec_vconv_intf_impl.h:45-46)
│   └── CastImpl<T, U, trait, S, V>  (dav_3510/vec_vconv_impl.h:501-572)
│       └── CastImpl (含RoundMode重载) → vcvt
│
├── Reduce<type, T, U, mode, S, V>(dstReg, srcReg, mask)  (vec_reduce_intf_impl.h:57-65)
│   ├── ReduceMaxImpl<T, mode, U>  (dav_3510/vec_reduce_impl.h:99-127)
│   │   └── vcmax
│   └── ReduceSumImpl<T, U, mode, S, V>  (dav_3510/vec_reduce_impl.h:33-65)
│       └── vcadd
│
├── Gather<T, U, S, V>(dstReg, srcReg, indexReg)  (gather_mask_intf_impl.h:72-74)
│   └── GatherImpl<T, U, S, V>  (dav_3510/gather_mask_impl.h:69-91)
│       └── vselr
│
└── LocalMemBar<src, dst>()  (membar_intf_impl.h:46-48)
    └── LocalMemBarImpl<src, dst>  (dav_3510/membar_impl.h:31-71)
        └── mem_bar
```

### 5.2 文件路径与行号速查表

| 高阶API | 接口层文件 | 接口层行号 | 底层实现文件 | 底层实现行号 | 底层内联调用 |
|---------|-----------|-----------|------------|------------|------------|
| CreateMask | `asc-devkit/.../reg_compute/kernel_reg_compute_maskreg_intf_impl.h` | L43-47 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_maskreg_impl.h` | L83-109 | `pset_b32/pset_b16/pset_b8` |
| UpdateMask | 同上 | L37-41 | 同上 | L27-62 | `plt_b32/plt_b16/plt_b8` |
| LoadAlign(向量) | `asc-devkit/.../reg_compute/kernel_reg_compute_datacopy_intf_impl.h` | L48-106 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_datacopy_load_impl.h` | L47-150+ | `vlds/vld` |
| LoadAlign(MaskReg) | 同上 | L410-435 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_datacopy_impl.h` | L194-224 | `pld/plds` |
| StoreAlign(向量) | `asc-devkit/.../reg_compute/kernel_reg_compute_datacopy_intf_impl.h` | L116-215 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_datacopy_store_impl.h` | L46-325 | `vsts/vst/vsstb` |
| StoreUnAlign | 同上 | L290-350 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_datacopy_store_impl.h` | L328-373 | `vstus` |
| StoreUnAlignPost | 同上 | L301-361 | 同上 | L375-399 | `vstas` |
| Duplicate | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_duplicate_intf_impl.h` | L38-40 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_duplicate_impl.h` | L63-101 | `vbr/vdup` |
| Ln | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_unary_intf_impl.h` | L68-71 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_unary_impl.h` | L517-547 | `vln` |
| Muls | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_binary_scalar_intf_impl.h` | L43-47 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_binary_scalar_impl.h` | L139-191 | `vmuls` |
| Select | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_cmpsel_intf_impl.h` | L54-58 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_cmpsel_impl.h` | L191-225 | `vsel` |
| Max | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_binary_intf_impl.h` | L64-68 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h` | L1460-1476 | `vmax` |
| ExpSub | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_fused_intf_impl.h` | L66-70 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_fused_impl.h` | L78-99 | `vexpdif` |
| Sub | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_binary_intf_impl.h` | L46-50 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h` | L226-256 | `vsub` |
| Add | 同上 | L40-44 | 同上 | L144-161 | `vadd` |
| Mul | 同上 | L52-56 | 同上 | L347-377 | `vmul` |
| Or | 同上 | L94-98 | 同上 | L1741-1772 | `vor` |
| Cast | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_vconv_intf_impl.h` | L45-46 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_vconv_impl.h` | L501-572 | `vcvt` |
| Reduce(MAX) | `asc-devkit/.../reg_compute/kernel_reg_compute_vec_reduce_intf_impl.h` | L57-65 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_vec_reduce_impl.h` | L99-127 | `vcmax` |
| Reduce(SUM) | 同上 | L57-65 | 同上 | L33-65 | `vcadd` |
| Gather | `asc-devkit/.../reg_compute/kernel_reg_compute_gather_mask_intf_impl.h` | L72-74 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_gather_mask_impl.h` | L69-91 | `vselr` |
| LocalMemBar | `asc-devkit/.../reg_compute/kernel_reg_compute_membar_intf_impl.h` | L46-48 | `asc-devkit/.../reg_compute/dav_3510/kernel_reg_compute_membar_impl.h` | L31-71 | `mem_bar` |

> **注**: 表中 `asc-devkit/.../reg_compute/` 为 `/Users/cc/Downloads/ops-transformer/asc-devkit/impl/basic_api/reg_compute/` 的缩写。

---

## 6. 架构分发机制说明

所有接口层文件均通过 `__NPU_ARCH__` 宏进行平台分发，默认（else分支）指向 dav_3510 实现：

```cpp
#if __NPU_ARCH__ == 3003
#include "../../basic_api/reg_compute/dav_l300/kernel_reg_compute_xxx_impl.h"
#elif __NPU_ARCH__ == 3113
#include "../../basic_api/reg_compute/dav_l311/kernel_reg_compute_xxx_impl.h"
#elif __NPU_ARCH__ == 5102
#include "../../basic_api/reg_compute/dav_m510/kernel_reg_compute_xxx_impl.h"
#else
#include "../../basic_api/reg_compute/dav_3510/kernel_reg_compute_xxx_impl.h"  // ← 默认
#endif
```

本文档中所有底层实现均定位在 **dav_3510** 目录中，对应 Ascend 910B 系列处理器架构。

---

## 7. 底层内联函数汇总

| 底层内联函数 | 功能 | 对应高阶API |
|------------|------|-----------|
| `pset_b32/pset_b16/pset_b8` | 设置谓词寄存器（全1/全0/特定模式） | CreateMask |
| `plt_b32/plt_b16/plt_b8` | 生成尾部谓词（按元素数量） | UpdateMask |
| `vlds` | 向量对齐加载（带分布模式） | LoadAlign (向量) |
| `vld` | 向量加载（带地址寄存器偏移） | LoadAlign (向量+AddrReg) |
| `plds` | 谓词寄存器加载（带后置更新） | LoadAlign (MaskReg) |
| `vsts` | 向量对齐存储（带分布模式） | StoreAlign |
| `vst` | 向量存储（带地址寄存器偏移） | StoreAlign (AddrReg) |
| `vsstb` | 向量数据块存储（带blockStride/repeatStride） | StoreAlign (DATA_BLOCK_COPY) |
| `vstus` | 向量非对齐存储（带后置更新） | StoreUnAlign |
| `vstas` | 向量非对齐存储后置刷新 | StoreUnAlignPost |
| `vbr` | 向量标量广播（无mask） | Duplicate (无mask) |
| `vdup` | 向量标量广播（有mask） | Duplicate (有mask) |
| `vln` | 向量自然对数 | Ln |
| `vmuls` | 向量乘标量 | Muls |
| `vsel` | 向量条件选择 | Select |
| `vmax` | 向量逐元素取最大值 | Max |
| `vexpdif` | 向量 exp(src0 - src1) | ExpSub |
| `vsub` | 向量逐元素减法 | Sub |
| `vadd` | 向量逐元素加法 | Add |
| `vmul` | 向量逐元素乘法 | Mul |
| `vor` | 向量逐位或 | Or |
| `vcvt` | 向量类型转换（含舍入/饱和模式） | Cast |
| `vcmax` | 向量规约取最大值 | Reduce<MAX> |
| `vcadd` | 向量规约求和 | Reduce<SUM> |
| `vselr` | 向量按索引收集 | Gather |
| `mem_bar` | 内存屏障 | LocalMemBar |
