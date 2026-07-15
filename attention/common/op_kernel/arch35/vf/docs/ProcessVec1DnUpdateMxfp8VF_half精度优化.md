声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ProcessVec1DnUpdateMxfp8VF half 精度优化说明

## 1. 概述

本次修改针对 `vf_mul_sel_softmaxflashv2_cast_nz_dn.h` 中的 `ProcessVec1DnUpdateMxfp8VF` 函数（arch35 VF 算子，FlashAttention2 的 update 路径）进行精度下调（precision lowering）性能优化。

核心思路：将 max 归约、scale 缩放、`FusedExpSub` 等大批量逐元素计算从 `float` 下调到 `half` 精度执行，利用 Vector 单元 half 精度更高的吞吐能力提升性能；同时对精度敏感的 max 向上取整（ceiling）步骤保留 `float` 精度，保证 MxFP8 量化 scale 计算的数值正确性。

## 2. 修改背景与动机

`ProcessVec1DnUpdateMxfp8VF` 是 MxFP8/FP8 场景下 FlashAttention update 阶段的核心 VF 函数，主要完成以下计算：

1. 对输入 `input_x_local_UB` 做 mask select（atten 场景）与 4 路 max 归约；
2. 对归约得到的 max 做 `dScale * INV_LN2` → 向上取整 → `* LN2` 的 ceiling 量化，并与历史 global max 比较；
3. 基于 `FusedExpSub(x * dScale, max)` 计算 exp，并做 sum 归约用于后续 softmax 归一化；
4. 将 exp 结果按 ND2NZ 排布写出为 FP8/BF16/FP16。

优化前，上述 step 1/3 中的逐元素运算（`LoadAlign` → `Max`/`Muls`/`FusedExpSub`）全部在 `float` 寄存器上进行。由于 arch35 Vector 单元 half 精度吞吐约为 float 的 2 倍，将这部分计算下沉到 half 可显著降低指令条数、提升算子性能。

## 3. 修改内容

### 3.1 寄存器类型调整

| 寄存器 | 修改前 | 修改后 | 用途 |
|--------|--------|--------|------|
| `vreg_x_f32_0` ~ `vreg_x_f32_7` | `RegTensor<float>` | `RegTensor<half>` | 输入数据的逐元素计算（scale/exp-sub） |
| `src0` ~ `src3` | `RegTensor<float>` | `RegTensor<half>` | mask select 前的输入数据 |
| `max0` ~ `max3` | `RegTensor<float>` | `RegTensor<half>` | 4 路 max 归约中间结果 |

### 3.2 新增寄存器

| 寄存器 | 类型 | 用途 |
|--------|------|------|
| `tmp_f32_0` ~ `tmp_f32_3` | `RegTensor<float>` | 从 UB 加载数据（保持 float 读取），随后 cast 到 half |
| `max0_f32` | `RegTensor<float>` | max ceiling 量化步骤的 float 工作寄存器 |
| `max0_half` | `RegTensor<half>` | ceiling 后回 cast 为 half，供 `FusedExpSub` 使用 |
| `vreg_x_max_f16_b` | `RegTensor<half>` | global max 的 half 版本，供 `FusedExpSub` 使用 |
| `vreg_min_half` | `RegTensor<half>` | `vreg_min` 的 half 版本，供 `Select` 使用 |

### 3.3 计算流程调整

**数据加载与精度转换**：

- 修改前：`LoadAlign(src, ub)` 直接以 `float` 加载并参与计算。
- 修改后：`LoadAlign(tmp_f32, ub)` 以 `float` 加载，再 `Cast<half, T>(src, tmp_f32)` 转为 half 参与后续计算。这样保留了对 UB 中 `T` 类型数据的一致读取，同时把计算下沉到 half。

**mask select 分支（hasAtten）**：

- `Select` 的数据源由 `vreg_min`（float）改为 `vreg_min_half`（half），与 half 类型的 `src*` 对齐。
- select 后的 store 需要 `T` 类型：先 `Cast<T, half>(tmp_f32, src)` 回 float，再 `StoreAlign<T>` 写回 UB。
- max 归约 `Max(max, max, src, preg_136)` 改用 `preg_136`（`UpdateMask<uint16_t>(128)`）掩码，匹配 half 寄存器 128 元素的宽度。

**max ceiling 量化步骤（精度保留点）**：

```text
max0(half) --Cast--> max0_f32(float)
  -> Muls(dScale) -> Muls(INV_LN2)
  -> Truncate<CAST_CEIL>
  -> Muls(LN2)
  -> Max(vreg_x_max_f32_b)          // 与历史 global max 比较，float
max0_f32 --Cast--> max0_half(half)  // 回 half 供后续 exp-sub
```

该步骤是 MxFP8 量化 scale 的关键计算，涉及 `INV_LN2 / LN2` 的对数域换底与向上取整，对精度敏感，因此保留在 `float` 执行。修改前该步骤直接在 `max0`（float）上完成；修改后通过 `max0_f32` 中转，保证计算内容不变。

**exp-sub 计算**：

- `FusedExpSub(vreg_x_exp, vreg_x_f32, max0_half, preg_134)` 的 `max0` 参数由 float 改为 `max0_half`，与 half 类型的 `vreg_x_f32_*` 对齐。
- `Muls(vreg_x_f32, vreg_x_f32, static_cast<half>(dScale), preg_136)` 的标量与掩码均调整为 half 语义。

**global max 更新（subLoop 分支）**：

- `FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f16_b, max0_half, preg_134)`：先将 `vreg_x_max_f32_b`（float）cast 为 `vreg_x_max_f16_b`（half），再与 `max0_half` 做 exp-sub，结果仍写回 `vreg_x_max_f32_b`（float）以保留 global max 的存储精度。

### 3.4 掩码调整

- half 类型的 `Max` 归约统一改用 `preg_136`（`UpdateMask<uint16_t>(128)`，对应 half 寄存器 128 元素）。
- 涉及 `T`（float）类型读写的 store 仍使用 `preg_108`（`CreateMask<uint16_t, ALL>`）。

## 4. 技术原理

```
UB(float/T) ──LoadAlign──> tmp_f32(float) ──Cast──> src/vreg_x_f32(half)
                                                            │
                                          ┌─────────────────┼──────────────────┐
                                          ▼                 ▼                  ▼
                                    Max 归约(half)    Muls*dScale(half)   FusedExpSub(half)
                                          │                                      │
                                          └──> max0(half) ──Cast──> max0_f32(float)
                                                                 Ceiling 量化(float)
                                                                 └──Cast──> max0_half(half)
```

- **性能收益**：`Max`、`Muls`、`FusedExpSub` 等指令在 half 下吞吐翻倍，且 half 寄存器单次处理元素数为 float 的 2 倍，指令条数相应减少。
- **精度保护**：仅 ceiling 量化（`dScale*INV_LN2` → 取整 → `*LN2`）与 global max 比较、`new_global_max`/`exp_max_fp32` 的存储保留 `float`，避免量化 scale 误差放大。

## 5. 精度与性能影响

- **精度**：max 归约与 exp-sub 下沉至 half 会引入有限的舍入误差；但 ceiling 量化 scale 在 float 下完成，MxFP8 量化边界的正确性不受影响。需结合端到端精度用例验证（建议覆盖 FP8_E5M2 / FP8_E4M3 / HiFloat8 / BF16 / FP16 各 `T2` 分支）。
- **性能**：预期 update 阶段 Vector 计算耗时下降，整体 FA 算子在 MxFP8 场景下获得性能收益。

## 6. 涉及文件

- `attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h`（`ProcessVec1DnUpdateMxfp8VF`，约 1203-1670 行）

## 7. 验证建议

1. 功能：覆盖 `T2` 为 `fp8_e5m2_t` / `fp8_e4m3fn_t` / `hifloat8_t` / `bfloat16_t` / `half` 的各分支，以及 `hasAtten=true/false`、`subLoop=0/>0` 的组合。
2. 精度：与优化前实现对齐 `new_global_max`、`new_global_sum`、`exp_max_fp32`、`x_exp` 输出，关注 max ceiling 边界与 sum 归约误差。
3. 性能：对比 update 阶段耗时，确认 half 下沉带来正向收益。
