# CastScale + AntiquantVFFp8D448 SIMD VF 融合方案

## 1. 现状分析

### 当前 `DequantKv` 调用链

文件: `kv_quant_sparse_attn_sharedkv_scfa_block_vector.h`

```
DequantKv (line 433)
  ├── CastScale (line 441)          → 从输入 tensor 提取 fp8 scale → 转 fp32 → 写 dequantScaleBuff
  └── AntiquantVFFp8D448 (line 442) → 读 dequantScaleBuff fp32 scale + 读输入 fp8 KV → 反量化 → 写输出
```

### 数据布局（每行 640B）

| 区域 | 大小 | 说明 |
|------|------|------|
| nope | 448B | 非 rope 的 KV 数据，需要反量化 |
| rope | 64B | rope 数据，保持原始格式通过 |
| scale | 128B | 64 个 fp8_e8m0_t scale 值，带 4 路 interleave |
| **合计** | **640B** | `dSizeVInput` / `combineDim` |

- `CastScale` 从每行偏移 576B 处用 `DIST_UNPACK4_B8` 加载 scale 数据
- `AntiquantVFFp8D448` 从每行偏移 128B 处用 `DIST_UNPACK4_B8` 加载 nope KV 数据

### 关键中间 buffer

- `dequantScaleBuff`: 64×16×2×sizeof(float) = **8192 字节** UB (line 968)
- `stage0OutBuf[0/1]`: 640×(16+1)×sizeof(Q_T) 字节 UB

---

## 2. 融合策略

### 核心思路

将 `CastScaleImpl` 的 fp8_e8m0_t → bf16 → fp32 逻辑内联到 `AntiquantVFImplFp8D448` 中，消除中间 buffer。

### 融合后的单次循环数据流

```
                    ┌─ Load fp8 KV (DIST_UNPACK4_B8) ──→ Cast fp8→fp32 ──┐
输入 tensor ────────┤                                                      ├──→ Mul ──→ Cast fp32→Q_T ──→ DeIntlv ──→ Store
                    └─ Load fp8 scale (DIST_UNPACK4_B8) ──→ Cast fp8→bf16 ──→ Cast bf16→fp32 ──┘
```

所有中间数据保持在 VF 寄存器内，不经过 UB 访存。

### 收益总览

| 项目 | 当前 | 融合后 |
|------|------|--------|
| **UB 中间 buffer** | `dequantScaleBuff`: 64×16×2×4 = 8192 B | 不需要，节省 8KB UB |
| **VF 调用次数** | 2 次 (`CastScaleImpl` + `AntiquantVFImplFp8D448`) | 1 次（融合后的反量化） |
| **UB 访存** | CastScale Store + AntiquantVF Load（各 dealRow×256B） | 去掉这两次，改为直接从输入 tensor 读 fp8（带宽减半） |
| **DequantKv 代码** | 4 行（初始化 buffer + 两次 VF 调用） | 2 行（初始化 + 单次 VF 调用） |

---

## 3. 详细实现步骤

### 步骤 1：修改 `AntiquantVFImplFp8D448` 函数签名

**当前签名** (line 329-331):

```cpp
template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ Q_T* ubKRopeNzAddr, __ubuf__ int8_t* ubSrcAddr,
    __ubuf__ Q_T* ubDstAddr, __ubuf__ float* ubScaleSrcAddr, __ubuf__ int8_t* ubKRopeAddr,
    uint32_t dealRowCount)
```

**修改为**:

```cpp
template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ Q_T* ubKRopeNzAddr, __ubuf__ int8_t* ubSrcAddr,
    __ubuf__ Q_T* ubDstAddr, __ubuf__ int8_t* ubRawScaleAddr, __ubuf__ int8_t* ubKRopeAddr,
    uint32_t dealRowCount)
```

变更：`ubScaleSrcAddr` 类型从 `__ubuf__ float*` 变为 `__ubuf__ int8_t*`，指向输入 tensor 中原始 fp8 scale 区域的起始位置。

### 步骤 2：在 `AntiquantVFImplFp8D448` 内联 scale 转换

**删除的代码**（两个 DIST_BRC_B32 float 加载，lines 373-376）:

```cpp
MicroAPI::LoadAlign<float, ..., DIST_BRC_B32>(vScale0, ubScaleSrcAddrTemp, 1);
MicroAPI::LoadAlign<float, ..., DIST_BRC_B32>(vScale1, ubScaleSrcAddrTemp, tileSize - 1);
```

**替换为**（fp8 scale 加载 + 转换）:

需要新增的寄存器声明:

```cpp
MicroAPI::RegTensor<fp8_e8m0_t> vRawScale0;   // 原始 fp8 scale
MicroAPI::RegTensor<fp8_e8m0_t> vRawScale1;
MicroAPI::RegTensor<bfloat16_t> vScaleBf16_0;  // 中间 bf16
MicroAPI::RegTensor<bfloat16_t> vScaleBf16_1;
// vScale0, vScale1 保留为 MicroAPI::RegTensor<float> (已有)
```

新增的加载和转换逻辑:

```cpp
const uint32_t rowStride = 640;

// 加载原始 fp8_e8m0_t scale 数据 (使用与 CastScaleImpl 相同的 DIST_UNPACK4_B8 模式)
MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
    MicroAPI::LoadDist::DIST_UNPACK4_B8>(
    (MicroAPI::RegTensor<int8_t>&)vRawScale0, ubScaleSrcAddrTemp, tileSize);  // 64B
MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
    MicroAPI::LoadDist::DIST_UNPACK4_B8>(
    (MicroAPI::RegTensor<int8_t>&)vRawScale1, ubScaleSrcAddrTemp, rowStride - tileSize);  // 576B

// fp8_e8m0_t → bf16 → fp32 (与 CastScaleImpl 完全相同的转换链)
MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScaleBf16_0, vRawScale0, kvTypeMaskAll);
MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScaleBf16_1, vRawScale1, kvTypeMaskAll);
MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScale0, vScaleBf16_0, fp32MaskAll);
MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScale1, vScaleBf16_1, fp32MaskAll);
```

**关键设计决策 — 行跨度**：`CastScaleImpl` 每次从 scale 区域加载 640B（正是一整行的大小）。在融合版本中，`ubScaleSrcAddrTemp` 以 640B 为行跨度加载（与 `CastScaleImpl` 一致）。`ubScaleSrcAddrTemp` 初始化为 `ubRawScaleAddr + j * scaleNumPerLoop`（外层 j 循环偏移）。

### 步骤 3：修改 `AntiquantVFFp8D448` wrapper

**当前** (line 420-431):

```cpp
template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& kRopeUbNz, LocalTensor<Q_T>& outputUb,
    LocalTensor<KV_T>& inputUb, LocalTensor<float>& scaleUb, LocalTensor<int8_t>& kRopeUb,
    uint32_t dealRowCount)
{
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb[64 * sizeof(Q_T)].GetPhyAddr());
    __ubuf__ int8_t* ubKRopeAddr = (__ubuf__ int8_t*)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ Q_T* ubKRopeNzAddr = (__ubuf__ Q_T*)(kRopeUbNz.GetPhyAddr());
    __ubuf__ float* ubScaleAddr = (__ubuf__ float*)(scaleUb.GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr, dealRowCount);
}
```

**修改为**:

```cpp
template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& kRopeUbNz, LocalTensor<Q_T>& outputUb,
    LocalTensor<KV_T>& inputUb, LocalTensor<int8_t>& kRopeUb,
    uint32_t dealRowCount)
{
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb[64 * sizeof(Q_T)].GetPhyAddr());
    __ubuf__ int8_t* ubKRopeAddr = (__ubuf__ int8_t*)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ Q_T* ubKRopeNzAddr = (__ubuf__ Q_T*)(kRopeUbNz.GetPhyAddr());
    // 直接指向输入 tensor 中原始 fp8 scale 区域
    __ubuf__ int8_t* ubRawScaleAddr = (__ubuf__ int8_t*)(inputUb[448 + 64 * 2].GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubRawScaleAddr, ubKRopeAddr, dealRowCount);
}
```

变更：
- 移除 `LocalTensor<float>& scaleUb` 参数
- `ubScaleAddr` 改为 `ubRawScaleAddr`，使用与 `CastScale` 相同的偏移 `448 + 64 * 2`

### 步骤 4：简化 `DequantKv`

**当前** (line 433-443):

```cpp
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
    LocalTensor<KV_T> srcTensor, int64_t dealRow, int64_t s2ProcessBaseSize, ConstInfo &constInfo)
{
    LocalTensor<float> floatScale = dequantScaleBuff.Get<float>();
    LocalTensor<int8_t> kRopeUb = srcTensor.template ReinterpretCast<int8_t>();
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (16 + 1)];
    CastScale<Q_T, KV_T>(floatScale, srcTensor, dealRow);
    AntiquantVFFp8D448<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, floatScale, kRopeUb, dealRow);
}
```

**修改为**:

```cpp
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
    LocalTensor<KV_T> srcTensor, int64_t dealRow, int64_t s2ProcessBaseSize, ConstInfo &constInfo)
{
    LocalTensor<int8_t> kRopeUb = srcTensor.template ReinterpretCast<int8_t>();
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (16 + 1)];
    AntiquantVFFp8D448<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, kRopeUb, dealRow);
}
```

变更：
- 移除 `LocalTensor<float> floatScale = dequantScaleBuff.Get<float>()`
- 移除 `CastScale` 调用
- `AntiquantVFFp8D448` 调用去掉 `floatScale` 参数

### 步骤 5：删除 `CastScale` 和 `CastScaleImpl`

删除以下代码段：
- `CastScaleImpl` 函数 (line 293-317)
- `CastScale` 函数 (line 319-326)

### 步骤 6：删除 `dequantScaleBuff` 及其初始化

在类声明中 (line 150):

```cpp
// 删除
TBuf<> dequantScaleBuff;
```

在 `InitLocalBuffer` 中 (line 968):

```cpp
// 删除
pipe->InitBuffer(dequantScaleBuff, 64 * 16 * 2 * sizeof(float));
```

---

## 4. 寄存器压力评估

### 当前 `AntiquantVFImplFp8D448` 寄存器使用（Atlas 350, VL=256B）

| 寄存器 | 类型 | 数量 |
|--------|------|------|
| vKvData0/1 | RegTensor\<KV_T\> (int8) | 2 |
| vScale0/1 | RegTensor\<float\> | 2 |
| vCastFp32Res0/1 | RegTensor\<float\> | 2 |
| vMulRes0/1 | RegTensor\<float\> | 2 |
| vCastRes0/1 | RegTensor\<Q_T\> | 2 |
| vCastResPack0/1 | RegTensor\<Q_T\> | 2 |
| vKvRope | RegTensor\<int8\> | 1 |
| MaskReg × 4 | - | 4 |
| **当前合计** | | **~17** |

### 融合后增加

| 寄存器 | 类型 | 数量 |
|--------|------|------|
| vRawScale0/1 | RegTensor\<fp8_e8m0_t\> | +2 |
| vScaleBf16_0/1 | RegTensor\<bfloat16_t\> | +2 |
| vScale0/1 (保留) | RegTensor\<float\> | 0 (复用) |
| **融合后合计** | | **~21** |

### 评估

Atlas 350 VF 寄存器文件通常有 32 个向量寄存器，21 个的用量在安全范围内。且因为消除了独立的 `CastScaleImpl` VF 调用帧（~6 个寄存器），总体峰值用量变化不大。

**降低寄存器压力的可选优化**：如果寄存器紧张，可以将 `vScaleBf16_0/1` 复用 `vCastFp32Res0/1` 的存储（在 KV Cast 之前先做 scale Cast），节省 2 个寄存器。

---

## 5. 尾部处理对齐

`AntiquantVFImplFp8D448` 的尾部循环 (line 394-416) 处理最后 64 个 nope 元素。

**当前尾部** (line 405):

```cpp
MicroAPI::LoadAlign<float, ..., DIST_BRC_B32>(vScale0, ubScaleSrcAddrTemp, tileSize);
```

**融合后尾部修改为**:

```cpp
MicroAPI::LoadAlign<int8_t, ..., DIST_UNPACK4_B8>(
    (MicroAPI::RegTensor<int8_t>&)vRawScale0, ubScaleSrcAddrTemp, tileSize);
MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScaleBf16_0, vRawScale0, kvTypeMaskAll);
MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScale0, vScaleBf16_0, fp32MaskAll);
```

---

## 6. 风险与验证

| 风险 | 缓解措施 |
|------|---------|
| **DIST_UNPACK4_B8 的行跨度对齐** — scale 区域在行内的确切偏移和 unpack 后数据位置可能因 interleave 模式而异 | 参考 `CastScaleImpl` 已验证的 640B 加载模式，使用相同参数；实现后跑 correctness 对比 |
| **寄存器溢出** — 增加 4 个寄存器可能触发 spill | 若出现，复用 vCastFp32Res 作为 scale bf16 中间存储 |
| **`ubScaleSrcAddrTemp` 的行内偏移** — 外层 j 循环中 `+ j * scaleNumPerLoop` 偏移量在 raw fp8 地址上的语义可能与 fp32 buffer 不同 | `scaleNumPerLoop` 保持 2（与每 128 个 KV 元素对应的 scale 数量一致），但地址计算需从 float 偏移转为 int8 偏移 |
| **类型转换路径验证** — fp8_e8m0_t→bf16→fp32 是否在融合后的寄存器上下文中有不同行为 | 使用与 `CastScaleImpl` 相同的 `castTraitFp8_1` trait，已验证的转换路径 |

---

## 7. 测试验证方案

1. **单元级正确性**：用相同输入分别跑融合前/后的 `DequantKv`，逐元素对比输出
2. **集成测试**：跑完整的 SCFA attention 端到端用例，对比最终 attention output
3. **性能对比**：在 Atlas 350 上测量 `ProcessNotSparseKv` 的 cycle 数，关注：
   - VF 指令数变化
   - UB 带宽利用率
   - 整体 latency
4. **UB 空间检查**：确认释放的 8KB 不会导致其他 buffer 超限

---

## 8. 涉及文件清单

| 文件 | 修改内容 |
|------|---------|
| `kv_quant_sparse_attn_sharedkv_scfa_block_vector.h` | 删除 `CastScaleImpl`、`CastScale`、`dequantScaleBuff` 声明/初始化；修改 `DequantKv`、`AntiquantVFFp8D448` wrapper；修改 `AntiquantVFImplFp8D448` 的 scale 加载逻辑 |
| 无需修改其他文件 | `AntiquantVFFp8D448` 的调用者只有 `DequantKv`，无其他调用点 |

---

## 9. 可选后续优化

如果融合验证成功，可以考虑将 `AntiquantVFImplFp8D448` 中的 KV 数据加载和 scale 加载进一步合并——因为两者都使用 `DIST_UNPACK4_B8` 从同一 640B 行加载，理论上可以通过一次 640B 加载同时获取 KV 和 scale（通过解包后的寄存器 lane 分配）。
