# MixedQuantSparseFlashMla KV 的 D 维度组成分析

## 逻辑维度 D = 512

| 部分 | 元素数 | 数据类型 | 占内存 | 说明 |
|------|--------|---------|--------|------|
| nope | 448 | FP8_E4M3FN (1B) 或 HiF8 (1B) | 448 B | per-token-group 量化, group_size=64 |
| rope | 64 | BF16 (2B) | 128 B | 不量化, 与 Q 一致 |

> `nope_head_dim = D - rope_head_dim = 512 - 64 = 448`
> 代码来源: `golden.py:964`, `util_regbase.h:134-135`

## 物理存储维度 dSizeVInput = 640

KV 在内存中的实际存储维度为 640 字节, 由 rope + nope + scale + pad 组成, 按 128 字节对齐。

### quant_mode=1 (D=608, d_aligned_128=640)

布局: `block_size*(rope[64*2] + nope[448] + scale[448/64*2] + pad[18])`

| 部分 | 元素数 | 数据类型 | 占内存 | 说明 |
|------|--------|---------|--------|------|
| rope | 64 (x2=128B) | BF16 | 128 B | 非量化, 存储在最前 |
| nope | 448 | FP8_E4M3FN | 448 B | per-token-group 量化 |
| scale | 7 (448/64) x2=14B | E8M0, 按 BF16 存储 | 14 B | nope 的反量化 scale |
| pad | 18 | 填充 | 18 B | 对齐至 128 字节边界 |
| **合计** | - | - | **608 B** | 对齐至 640 B |

> scale 计算: `quant_scale_head_dim = ceil(448/64) = 7`, 每个 scale 为 E8M0(1B), 按 BF16 存储 (2B/组), 故 scale 占 7x2=14 字节
> pad 计算: `pad_d = 640 - 128 - 448 - 14 = 18`
> 代码来源: `golden.py:965-969`, `README.md:64`

### quant_mode=2 (D=584, d_aligned_128=640)

布局: `block_size*(nope+rope) + block_size*(scale+pad[1])`

| 部分 | 元素数 | 数据类型 | 占内存 | 说明 |
|------|--------|---------|--------|------|
| nope | 448 | FP8_E4M3FN | 448 B | per-token-group 量化 |
| rope | 64 (x2=128B) | BF16 | 128 B | 非量化 |
| scale | 7 (448/64) | E8M0 | 7 B | nope 的反量化 scale, 类型为 e8m0 |
| pad | 1 | 填充 | 1 B | 对齐 padding |
| **合计** | - | - | **584 B** | 对齐至 640 B |

> scale 计算同上, 但 scale 类型为 e8m0 (1B/元素), 故 scale 占 7 字节
> pad 计算: `pad_d = 640 - 448 - 128 - 7 = 1` (实际上是整体对齐后的余数)

## 代码关键位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `util_regbase.h` | 132-135 | `dSizeV=512`, `dSizeVInput=640=rope+nope+scale+pad`, `dSizeNope=448`, `dSizeRope=64` |
| `scfa_block_vector.h` | 276 | `combineBytes=640` (KV 单行搬运字节数) |
| `scfa_block_vector.h` | 379 | `ubScaleAddr=inputUb[448+64*2]` (scale 在 rope+nope 之后) |
| `scfa_block_vector.h` | 412 | `nopeDim=448`, `kvNumPerLoop=128`, `scaleNumPerLoop=2` |
| `check_feature.cpp` | 69-74 | 强制校验 `dSizeV==512`, `dSizeVInput==640` |
| `golden.py` | 964-969 | `nope_head_dim=448`, `quant_scale_head_dim=7`, `d_aligned_128=640`, `pad_d=18` |
| `README.md` | 63-65 | 两种 quant_mode 的 D 值和布局描述 |
| `tiling.cpp` | 486-487 | `dSizeV=512`, `dSizeVInput=dSizeKV_` (从 KV tensor shape 获取) |

## 内存占用对比

| quant_mode | 逻辑 D | 物理 D (dSizeVInput) | 每行 KV 内存 | 对齐后每行 |
|------------|--------|---------------------|-------------|-----------|
| 1 | 608 | 640 | 608 B | 640 B |
| 2 | 584 | 640 | 584 B | 640 B |

两种 quant_mode 物理存储维度均为 640 字节 (128 字节对齐), 区别在于 scale 的数据类型和 pad 的大小不同。

> 注意: 当前实现仅支持 D=512 (Q) / dSizeVInput=640 (KV), 不泛化, 见 `check_feature.cpp:69-74`