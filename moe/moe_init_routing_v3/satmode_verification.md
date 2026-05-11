# moe_init_routing_v3 arch35 INT8量化 CastTrait 饱和模式验证

## 修改范围

3 个文件的 `castTraitF32ToF16` 和 `castTraitF16ToI8` 从 `SatMode::NO_SAT` 改为 `SatMode::SAT`：

| 文件 | TilingKey | 路径说明 |
|------|-----------|----------|
| `op_kernel/arch35/moe_v3_gather_dynamic_quant.h` | 1020000/1021000/1120000/1121000 | 动态量化非全载 |
| `op_kernel/arch35/moe_v3_full_load_dynamic_quant.h` | 220000 | 动态量化全载 |
| `op_kernel/arch35/moe_v3_full_load_static_quant.h` | 210000 | 静态量化全载 |

---

## 验证用例

### 用例 1：动态量化全载路径 (TilingKey 220000)

```
quant_mode       = 1 (DYNAMIC_QUANT)
DTYPE_X          = bf16 或 fp16
DTYPE_EXPANDED_X = int8
n (batch_size)   = 1
cols             = 256
topK             = 8
expert_num       = 4
expert_range     = [0, 4)
row_idx_type     = 0 (GATHER)
```

### 用例 2：动态量化非全载路径 (TilingKey 1020000)

```
quant_mode       = 1 (DYNAMIC_QUANT)
DTYPE_X          = bf16 或 fp16
DTYPE_EXPANDED_X = int8
n (batch_size)   = 1024
cols             = 4096
topK             = 8
expert_num       = 64
expert_range     = [0, 64)
row_idx_type     = 0 (GATHER)
```

### 用例 3：静态量化全载路径 (TilingKey 210000)

```
quant_mode       = 0 (STATIC_QUANT)
DTYPE_X          = bf16 或 fp16
DTYPE_EXPANDED_X = int8
n (batch_size)   = 1
cols             = 128
topK             = 8
expert_num       = 4
expert_range     = [0, 4)
row_idx_type     = 0 (GATHER)
scale            = [1.0]
offset           = [0.0]
```

---

## 边界验证：输入含 INF 的场景

构造 BF16 输入 x 中部分值为 BF16 INF（0x7F80），验证：

1. 动态量化路径：scale 计算正常（max(|x|) = INF），cast 到 INT8 时饱和到 127/-128，不会产生 INF/NAN
2. 静态量化路径：即使 scale=inf 的输入经过乘加后产生 INF，cast 到 INT8 时饱和到 127/-128

对比项：修改前 `NO_SAT` 模式下相同输入可能产生非预期的 wrap-around 值。
