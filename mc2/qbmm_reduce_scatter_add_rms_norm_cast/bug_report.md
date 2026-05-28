# Bug Report: MXFP8 Matmul SIGSEGV on Random Data (256×2560×5120)

## Summary

`qbmm_reduce_scatter_add_rms_norm_cast` 算子的 matmul 部分在大 shape (256, 2560, 5120) 下，使用随机 FP8 输入数据时 SIGSEGV 崩溃，但 fix-mode 数据（A=1.0/2.0, B=1.0/2.0）正常。小 shape (128, 256, 512) 随机数据正常。

## Root Cause

### Bug 1: `dbL0c` 使用了物理 L0C 大小 (1MB) 而非逻辑地址空间 (256KB)

**文件**: `quant_matmul_mx_tiling_swat.h:244`, `quant_matmul_mx_tiling_a_full_load.h:269`

```
runInfo_.dbL0c = baseM * baseN * 4 * 2 <= platformInfo_.l0cSize ? 2 : 1;
                                      ^^^^^^^^^^^^^^^^^^
                                      physical size = 1MB (wrong!)
```

`platformInfo_.l0cSize` 返回物理 L0C 大小 (1MB)，但 AscendC 的 `MakeL0CmemPtr` 使用逻辑地址空间 (256KB)。当 tile 用满 256KB 时，dbL0c=2 的乒乓 offset 会导致 L0C 越界。

### Bug 2: FP8 ND→NZ DMA 硬件缺陷（主要根因）

**触发条件**: 当 A 矩阵 NZ tile 尺寸 `baseM × (stepK × baseK) >= 65536` 时，行内元素非均匀的 FP8 ND→NZ DMA 触发硬件异常。

**文件**: `quant_matmul_mx_tiling_swat.h` CalStepKs

**为什么 fix mode 不崩？** Fix mode 行内值完全均匀（全 1.0 或 2.0），DMA 只需广播，不走真正的 interleave 路径。

**为什么小 case (128, 256, 512) 不崩？** `baseM=64, kL1=256` → NZ tile 仅 `64×256=16384` elements，远小于 L0A 容量。

**为什么大 case (256, 2560, 5120) 崩？** `baseM=256, stepK=2 → kL1=256` → NZ tile `256×256=65536` = 刚好填满 L0A，在这个临界尺寸下 ND→NZ DMA 硬件 bug 触发。

## Debugging Steps

1. **确认崩溃模式**: fix mode (1.0/2.0) 通过，random mode SIGSEGV
2. **隔离数据源**: Random A + Fix B → 崩；Fix A + Random B → 正常 → 问题在 A 侧
3. **隔离数据模式**: A row-constant（行内均匀）→ 正常；A ramp（行内非均匀）→ 崩
4. **排除 padding**: 禁用 A 侧 PadMxKAL1 → 仍崩 → 不是 padding 问题
5. **排除 L0C 乒乓**: 关闭 dbL0c → 不崩但 timeout
6. **定位 tiling**: stepK=2 → kL1=256 × baseM=256 = 65536 命中触发条件

## Fix

### 文件 1: `quant_matmul_tiling_common.h`
```cpp
// 新增常量：逻辑 L0C 地址空间
constexpr uint64_t LOGICAL_L0C_SIZE = 256UL * 1024UL;  // 256KB
```

### 文件 2: `quant_matmul_mx_tiling_swat.h` (两处)

**dbL0c 修复**: 使用逻辑 L0C 大小
```cpp
runInfo_.dbL0c =
    baseM * baseN * DATA_SIZE_L0C * DB_SIZE <= LOGICAL_L0C_SIZE ? DB_SIZE : 1U;
```

**NZ DMA 约束** (CalStepKs 末尾):
```cpp
constexpr uint64_t L0A_FP8_CAPACITY = 64UL * 1024UL;
if (stepKa > 1UL && aDataType == mm::DataType::DT_FLOAT8_E4M3FN &&
    baseM * stepKa * baseK >= L0A_FP8_CAPACITY) {
    stepKa -= 1UL;
    stepKb = std::min(stepKa, stepKb);
    depthA1 = stepKa * DB_SIZE;
    depthB1 = stepKb * DB_SIZE;
}
```

### 文件 3: `quant_matmul_mx_tiling_a_full_load.h`
```cpp
// 同上 dbL0c 修复
runInfo_.dbL0c =
    baseM * baseN * DATA_SIZE_L0C * DB_SIZE <= LOGICAL_L0C_SIZE ? DB_SIZE : 1U;
```

## Verification

| Shape | Mode | Before | After |
|-------|------|--------|-------|
| 128×256×512 | random | PASS | PASS |
| 256×2560×5120 | fix | PASS | PASS |
| 256×2560×5120 | random | **SIGSEGV** | **PASS** |
