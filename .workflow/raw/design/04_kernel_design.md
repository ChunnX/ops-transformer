# 04 Kernel 实现方案 — GMM A16W4 PerGroup ND

> 本文档描述 A16W4 INT4 weight ND pergroup 数据流的 kernel 方案。整体 controller、basic block、L1/UB 分配、cube 计算流程按 A16W8 ND basic-controller 方案复用；新增实现集中在 INT4 pergroup VF 反量化和 pergroup 地址偏移。
> 相关文档：[01 交付件清单](./01_deliverables.md)、[02 与硬件无关方案](./02_platform_independent_design.md)、[03 Tiling 方案](./03_tiling_design.md)

---

## 1. Kernel 入口与模板分发

入口位于 `gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp` 的 `V310_GMM_ANTI_QUANT` 分支，A16W4 pergroup ND 新增 4 个 basic-controller 分支：

```text
2 个 transB 状态 × 2 个 bias 状态 = 4
```

分支共同特征：

| 模板参数 | 取值 |
|----------|------|
| `W_TYPE` | `WQGMM_ND` |
| `C_QUANT_TYPE` | `WQGMM_NONE` |
| `W_QUANT_TYPE` | `WQGMM_PER_GROUP` |
| `WQ_A_TRANS` | `WQGMM_NO_TRANS` |
| `WQ_B_TRANS` | `WQGMM_NO_TRANS` / `WQGMM_TRANS` |
| `TEMPLATE_CUSTOM_SC` | `WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4` |
| `ALGORITHM_SUB_CATEGORY` | `WQGMM_N_FIRST_BASIC_BLOCK` |
| `ALGORITHM_CATEGORY` | `WQGMM_VECTOR_ANTIQUANT` |

调用形式复用：

```cpp
INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(
    GMMWeightQuantBasicController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
```

`VEC_ANTIQUANT_CONFIG_3` 保持现有值：

```cpp
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
// ubMte2BufferNum = 4
// ubMte2InnerSize = 256
```

---

## 2. Controller 层适配

文件：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h`

### 2.1 修改点总览

| 函数 | 设计结论 |
|------|----------|
| `Init()` | 复用，`groupSize` 已通过 `wqmmBasicBlock_.Init(gmmBaseTiling_->groupSize, ...)` 传入 |
| `Process()` | 复用 A16W8 M/N 二维分块和 round-robin 分核 |
| `SetMKN()` | 复用，M/K/N 获取逻辑与 pergroup 无关 |
| `SetGmAddr()` | 需要区分 pergroup scale 偏移，并将 bias 偏移独立出来 |
| `GetSplitValueFromGroupList()` | 复用 |

### 2.2 SetGmAddr 偏移

现有 A16W8 perchannel：

```cpp
antiquantParamsBaseOffset += offsetParam.nSize;
```

A16W4 pergroup：

```cpp
uint64_t groupNumK = offsetParam.kSize / gmmBaseTiling_->groupSize;
antiquantParamsBaseOffset += groupNumK * offsetParam.nSize;
```

原因：

```text
perchannel scale: [N]       每 group 占 N 个元素
pergroup scale:   [G, N]    每 group 占 (K/groupSize) * N 个元素
```

### 2.3 Bias 偏移独立

单单单场景下，bias shape 为 `[E,N]`，scale shape 为 `[E,G,N]`。因此 bias 不能继续复用 `antiquantParamsBaseOffset`。

设计：

```cpp
uint64_t biasBaseOffset = 0;

// singleWeight == 1
biasGm = GetTensorAddr<biasType>(0, biasGm_) + biasBaseOffset;

// 每个 group 结束后
biasBaseOffset += offsetParam.nSize;
```

完整偏移更新：

```cpp
xBaseOffset += offsetParam.mSize * offsetParam.kSize;

if constexpr (IsSameType<wType, int4b_t>::value) {
    weightBaseOffset += (offsetParam.nSize * offsetParam.kSize) >> 1;
} else {
    weightBaseOffset += offsetParam.nSize * offsetParam.kSize;
}

if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    antiquantParamsBaseOffset +=
        (offsetParam.kSize / gmmBaseTiling_->groupSize) * offsetParam.nSize;
} else {
    antiquantParamsBaseOffset += offsetParam.nSize;
}

biasBaseOffset += offsetParam.nSize;
yBaseOffset += offsetParam.mSize * offsetParam.nSize;
```

`weightBaseOffset >> 1` 的 INT4 packed 字节偏移与 perchannel/pergroup 无关，保留现有逻辑。

---

## 3. WqmmConfig 与 BasicBlock 分发

文件：
- `basic_block_config.h`
- `weight_quant_basic_block.h`

### 3.1 WqmmConfig

A16W4 pergroup ND 需要覆盖转置/非转置。本场景不支持 antiquantOffset，因此 `hasAntiQuantOffset` 固定为 `false`。可以新增命名常量，也可以在 `grouped_matmul_apt.cpp` 分支内局部定义 `static constexpr WqmmConfig`。

```cpp
// B 非转置
static constexpr WqmmConfig A16W4_ND_PG = {
    false, false, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND};

// B 转置
static constexpr WqmmConfig A16W4_ND_TRANS_PG = {
    false, true, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND};
```

### 3.2 BasicBlock 分发

`ComputeBasicBlock()` 的分发逻辑不需要改变：

```cpp
if ASCEND_IS_AIV {
    if constexpr ((wqmmConfig.weightFormat != CubeFormat::NZ && !wqmmConfig.bTrans) ||
                  (wqmmConfig.weightFormat == CubeFormat::NZ && wqmmConfig.bTrans)) {
        ComputeBasicBlockAivNdKnNzNk(offsetParam);
    } else {
        ComputeBasicBlockAivNdNkNzKn(offsetParam);
    }
} else {
    ComputeBasicBlockAic(offsetParam);
}
```

对应关系：

| `bTrans` | AIV 路径 | A16W4 VF 参考 |
|----------|----------|---------------|
| false | `ComputeBasicBlockAivNdKnNzNk` | ops-nn `AntiquantComputeKNPerGroup` |
| true | `ComputeBasicBlockAivNdNkNzKn` | ops-nn `AntiquantComputeW4NKPerGroup` |

---

## 4. L1/UB 空间划分

### 4.1 L1 分配

A16W4 pergroup L1 分配完全复用 A16W8 ND basic block。L1 中存放的是反量化后的 `xType` weight，因此 A16W8 与 A16W4 在 L1 上的数据类型和大小一致。

```cpp
weightL1Space = matmulTiling->baseN * matmulTiling->stepKb * matmulTiling->baseK;
// 256 * 4 * 64 = 65536 xType elements = 128KB when xType is half/bfloat16
```

无 bias：

```text
L1 512KB:
| WeightL1_P0 128KB | AL1_P0 128KB | AL1_P1 128KB | WeightL1_P1 128KB |
```

有 bias：

```text
L1 512KB:
| WeightL1_P0 128KB | Bias_P0 4KB | AL1_P0 124KB |
| AL1_P1 124KB | Bias_P1 4KB | WeightL1_P1 128KB |
```

有 bias 时 host tiling 将 `baseM` 从 256 调整为 240：

```text
A tile = ceilAlign(240,16) * 256 * 2B = 120KB
AL1 single buffer = 124KB
```

因此 A tile 能放入 AL1，L1 分配闭合，无需新增 bias 专用切分策略。

### 4.2 UB 分配

A16W4 pergroup 复用 A16W8 ND 的 `GetNdBufferInfo()`，不新增专用 pergroup buffer 配置。

```cpp
// 当前 A16W8 ND perchannel 配置
{
    .ubWeightOutputHighBitBufferNum = 2,       // 双缓冲
    .weightInputLowbitUbTotalSize = 174KB,     // weight 输入（int4 原始数据）
    .highBitDataUbTotalSize = 66KB,            // 反量化后 fp16 weight
    .antiQuantScaleUbTotalSize = 4KB,          // ★ perchannel scale（仅 1 行）
    .antiQuantScaleAfterCastUbTotalSize = 0,
    .antiQuantOffsetUbTotalSize = 4KB,         // perchannel offset（A16W4 pergroup 不使用，但保留空间分配）
    .weightInputLowBitUbSingleBufferSize = 174KB / ubMte2BufferNum,
    .antiQuantScaleUbSingleBufferSize = 4KB / ubMte2BufferNum,
    .antiQuantOffsetUbSingleBufferSize = 4KB / ubMte2BufferNum,
    .highBitDataUbSingleBufferSize = 66KB / 2,
};
```

具体大小：

| Buffer | 总大小 | 缓冲数 | 单缓冲 | 用途 |
|--------|:------:|:------:|:------:|------|
| `weightInputLowBit` | 174KB | 4 | 43.5KB | INT4 packed weight 输入 |
| `highBitData` | 66KB | 2 | 33KB | VF 输出的 FP16/BF16 weight |
| `antiQuantScale` | 4KB | 4 | 1KB | 当前 VF tile 的 scale |
| `antiQuantOffset` | 4KB | 4 | 1KB | 保留空间，A16W4 pergroup 不使用 |

**UB 空间分配逻辑不变**：复用 `GetNdBufferInfo()`，`antiQuantOffset` 的 4KB 保留。A16W4 pergroup 运行时不使用 offset buffer，但不修改分配代码。

### 4.3 UB 容量与 VF tile

`GetVfConfig()` 对 ND 场景的 VF 标准块为：

| 场景 | `vfNStandardLen` | `vfKStandardLen` |
|------|:----------------:|:----------------:|
| `bTrans=false` | 256 | 64 |
| `bTrans=true` | 64 | 256 |

最小 `groupSize=32` 时，单次 VF tile 的 scale 最大需求：

```text
bTrans=false: ceil(64 / 32) * 256 * 2B = 1024B = 1KB
bTrans=true:  ceil(256 / 32) * 64 * 2B = 1024B = 1KB
```

因此 scale 单缓冲 1KB 足够。A16W4 pergroup 的 scale 搬运和地址组织必须按当前 VF tile 对齐，而不是按完整 `kbL1Size × nL1Size` 一次性展开到单个 scale buffer。

weight 和 high-bit 输出容量：

```text
单次 VF high-bit 输出最大 = 256 * 64 * 2B = 32KB
highBitData 单缓冲 = 33KB

INT4 packed 输入最大低于 weightInputLowBit 单缓冲 43.5KB
```

---

## 5. Vec 反量化实现

文件：
- `basic_block_vf_nd.h`
- `weight_quant_vec_compute.h`
- 参考：`~/Code/ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h`

### 5.1 迁移边界

A16W4 pergroup 只在 VF 将 INT4 weight 反量化为 `xType` 时参考 ops-nn wqbmmv2。其余流程复用 GMM A16W8：
- GM 到 UB 的 weight 搬运框架复用。
- UB buffer 分配复用。
- AIV/AIC 同步复用。
- UB 到 L1 搬运复用。
- Cube matmul 复用。

### 5.2 非转置：KN 路径

`bTrans=false` 时，GMM 进入：

```text
ComputeBasicBlockAivNdKnNzNk // B 转置 → N 为内轴
  -> CopyGmToUb
  -> WeightAntiQuantCompute
  -> AntiQuantProcessNd
  -> A16W4 KN pergroup VF
```

参考 ops-nn：

```cpp
AntiquantComputeKNPerGroup(int32_t bubNLen, int32_t bubKLen)
```

该函数构造 `ParamsKN` 后调用：

```cpp
AscendC::VF_CALL<AntiquantPerGroupKN<xType, wType, false>>(param);
```

核心循环：

```text
for nLoop:
  for group:
    load scale for N vector
    for k in group:
      load int4 weight
      unpack/cast to xType
      multiply scale
      store high-bit weight
```

特点：
- groupSize 作为运行时参数进入 `ParamsKN`。
- `groupTail` 逻辑可复用，但本规格要求 `K % groupSize == 0`，正常路径不产生不完整 group。
- 支持 `groupSize=32/64/128/256`。

### 5.3 转置：NK 路径

`bTrans=true` 时，GMM 进入：

```text
ComputeBasicBlockAivNdNkNzKn
  -> CopyGmToUb
  -> WeightAntiQuantCompute
  -> AntiQuantProcessNd
  -> A16W4 NK pergroup VF
```

参考 ops-nn：

```cpp
AntiquantComputeW4NKPerGroup(int32_t bubNLen, int32_t bubKLen)
```

该函数按 `groupSize` 分发：

```cpp
if (groupSize == 32) {
    VF_CALL<AntiquantW4Pergroup32NK<...>>(params);
} else if (groupSize == 64) {
    VF_CALL<AntiquantW4Pergroup64NK<...>>(params);
} else if (groupSize == 128) {
    VF_CALL<AntiquantW4Pergroup128NK<...>>(params);
} else if (groupSize == 256) {
    VF_CALL<AntiquantW4Pergroup256NK<...>>(params);
}
```

本数据流只支持 32/64/128/256，因此不需要迁移 ops-nn 中 odd 或 greater-than-128 的泛化分支。

### 5.4 INT4 反量化步骤

每个 VF tile 的核心操作：

```text
1. 从 UB 读取 INT4 packed weight
2. unpack INT4
3. cast INT4 -> xType(FP16/BF16)
4. weight *= scale
5. 将 xType weight 写入 UB highBitData
6. 由现有 MTE3 流程搬到 L1 WeightL1
```

本场景 `hasAntiQuantOffset` 固定为 `false`，VF 函数模板参数中该位始终为 `false`。

### 5.5 scale UB 布局

VF 看到的 scale shape 统一为当前 tile 的 `[G_tile, N_tile]`：

```text
G_tile = vfKRealLen / groupSize    // K % groupSize == 0
N_tile = vfNRealLen
```

最大数据量为 1KB，见 §4.3。

KN 路径使用 N 方向向量加载，NK 路径使用 per group 广播加载，分别与 ops-nn `AntiquantComputeKNPerGroup` 和 `AntiquantComputeW4NKPerGroup` 对齐。

---

## 6. Cube 计算

文件：`weight_quant_cube_compute.h`

Cube 侧完全复用 A16W8：

| 函数 | 结论 |
|------|------|
| `ComputeBasicBlockAic()` | 复用 |
| `CopyAAndBiasGmToL1()` | 复用 |
| `LaunchMatmul()` | 复用 |
| `GetTensorC()` | 复用 |
| `WaitMTE1ToMTE2/SetMTE1ToMTE2` | 复用 |

原因：Cube 只消费 L1 中已反量化为 `xType` 的 weight，不感知 weight 原始是 A16W8 还是 A16W4，也不感知 perchannel/pergroup。

---

## 7. 修改点汇总

### 7.1 需要修改

| 文件 | 修改内容 |
|------|----------|
| `grouped_matmul_apt.cpp` | 新增 A16W4 pergroup ND 的 4 个 basic-controller 分支 |
| `weight_quant_tiling_key.h` | 新增对应 tiling key 组合 |
| `grouped_weight_quant_batch_matmul_tiling.cpp` | A16W4 pergroup shape/groupSize/tiling key 校验与生成 |
| `grouped_matmul_weight_quant_basic_controller.h` | `SetGmAddr` 中 pergroup scale 偏移和 bias 独立偏移 |
| `basic_block_config.h` | 新增或局部定义 A16W4 ND pergroup `WqmmConfig` |
| `weight_quant_vec_compute.h` | 在 ND + PER_GROUP + INT4 下分发到 A16W4 pergroup VF，并让 scale UB 布局与 VF tile 对齐 |
| `basic_block_vf_nd.h` | 新增 KN/NK A16W4 pergroup VF 实现 |
| `grouped_matmul_def.cpp` | config950 补充 INT4 ND 组合 |

### 7.2 不修改

| 文件/模块 | 结论 |
|-----------|------|
| `grouped_matmul_tiling_data_apt.h` | `groupSize` 字段已存在 |
| `weight_quant_basic_block.h` | BasicBlock AIV/AIC 分发复用 |
| `weight_quant_cube_compute.h` | Cube 计算复用 |
| `GetNdBufferInfo()` | A16W4 pergroup 复用 A16W8 ND UB buffer |
| L1 layout | A16W8 ND layout 复用 |

### 7.3 兼容性保护

| 保护点 | 方案 |
|--------|------|
| A16W8 perchannel | `WqmmConfig.antiQuantType == PER_CHANNEL` 原路径不变 |
| A16W4 pergroup | 新 tiling key + `PER_GROUP` 编译期分支隔离 |
| 有/无 bias | tiling key 选择 + runtime `hasBias` 传入 BasicBlock |
| A2/A3 | 仅 arch35 路径适配，不触碰 arch30 |
