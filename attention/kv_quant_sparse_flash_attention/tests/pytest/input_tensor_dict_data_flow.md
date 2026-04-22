# input_tensor_dict 数据流程分析文档

## 一、总体流程概览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          input_tensor_dict 数据流程                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   │  参数解析   │───▶│  数据生成   │───▶│  CPU预处理  │───▶│  NPU调用    │   │
│   │  (utils)    │    │(generate_   │    │  (golden)   │    │ (process)   │   │
│   │             │    │tensor_data) │    │             │    │             │   │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│         │                   │                   │                   │         │
│         ▼                   ▼                   ▼                   ▼         │
│   params字典          numpy数组          BNSD格式           torch.Tensor       │
│   (shape_input,       (FP16/BF16/        + KV拼接           (NPU设备)          │
│    dtype_input,       FP8等)             + PA block         + dtype转换        │
│    range_input)                          table处理)                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、详细流程分阶段说明

### 阶段1：参数解析与格式转换

#### 核心函数
| 文件名 | 函数名 | 行号 | 作用 |
|--------|--------|------|------|
| `utils.py` | `convert_param_combination_to_cs_format()` | 207-384 | 将pytest参数组合转换为标准params字典 |
| `utils.py` | `combin_params()` | 129-180 | 参数组合生成 |

#### 数据结构构建
```python
# utils.py:264-355 关键数据结构构建
params = {
    "shape_input": {
        'query': [B, S1, N1, D],
        'key': [kv_dim0, S2_storage, N2, D],
        'value': [...],
        'sparse_indices': [B, S1, N2, sparse_blockcount],
        'block_table': [B, block_num],
        'query_cache': [B, S1, N1, D + rope_head_dim],
        'key_cache': [block_num_sum, block_size, N2, D + rope_head_dim * 2 + D // tile_size * 4],
        'value_cache': [...],
        'query_rope': [B, S1, N1, rope_head_dim],
        'key_rope': [B, S2, N2, rope_head_dim],
        'dequant_scale': [B, S2, N2, D // tile_size],
        'v_dequant_scale': [B, S2, N2, D // tile_size],
    },
    "dtype_input": {
        'query': q_dtype_str,      # "bf16" 或 "fp16"
        'key': kv_dtype_str,       # "float8_e4m3fn" 或 "hifloat8"
        'value': kv_dtype_str,
        'sparse_indices': "int32",
        'block_table': "int32",
        'dequant_scale': "fp32",
        ...
    },
    "range_input": {
        'query': [2, 10],
        'key': [-100, 100.0],
        'value': [-100.0, 100.0],
        ...
    }
}
```

#### 关键处理逻辑
1. **layout判断** (`utils.py:262-326`)：
   - 根据 `layout_query` (BSND/TND) 和 `layout_kv` (PA_BSND/TND/BSND) 确定各tensor的shape
   - PA_BSND模式需要额外处理 block_table 和 block_num

2. **dtype映射** (`generate_tensor_data.py:138-192`, `generate_tensor_data.py:195-216`)：
   - `get_np_dtype()` - numpy dtype映射
   - `get_torch_dtype()` - torch dtype映射

---

### 阶段2：Tensor数据生成

#### 核心函数调用链
```
utils.py:generate_input_tensors() [183-192]
    │
    ├── 遍历 shape_keys: ['query', 'key', 'value', 'sparse_indices', ...]
    │
    └── generate_tensor_data.py:gen_tensor_data() [235-286]
            │
            ├── 判断是否固定值: params[f"tensor_data_{index}"]
            │       └── 直接 reshape + astype
            │
            ├── 非固定值:
            │       ├── gen_boundary_tensor_data() [502-531] - 边界值场景
            │       │       ├── "null" → 空数组
            │       │       ├── "nan" → np.full(shape, np.NaN)
            │       │       ├── "inf" → np.full(shape, np.Inf)
            │       │       └── 单值 → np.full(shape, value)
            │       │
            │       └── gen_nonbound_tensor_data() [534-604] - 随机值场景
            │               ├── 处理 "-inf"/"inf" 特殊范围
            │               ├── gen_uniform_data() [608-626] - 均匀分布随机
            │               └── gen_normal_data() [629-643] - 正态分布随机
            │
            └── 特殊dtype处理:
                    ├── hifloat8: trans_np_float_tensor_to_hifuint8() [289-306]
                    ├── float8_e8m0: 位操作转换
                    ├── uint1: packbits压缩
                    └── fp4_e2m1_as_fp32: 特定值选择
```

#### 重点：特殊精度类型转换

| dtype | 处理方式 | 文件位置 |
|-------|----------|----------|
| hifloat8 | `trans_np_float_tensor_to_hifuint8()` FP32→hif8转换 | `generate_tensor_data.py:289-306` |
| float8_e4m3fn | 使用 `ml_dtypes.float8_e4m3fn` | `generate_tensor_data.py:185-187` |
| float8_e8m0 | 位操作: `<<1 >>24` | `generate_tensor_data.py:267-272` |
| uint1 | `np.packbits()` 布尔压缩 | `generate_tensor_data.py:279-281` |
| bf16 | `tf.bfloat16.as_numpy_dtype` | `generate_tensor_data.py:145-146` |

#### 数据转换工具 (`tools.py:50-85`)
```python
# tools.py:convert_tensor_data() - 数据格式转换核心函数
def convert_tensor_data(data_pool, data_path, type_str, params=None):
    # 根据dtype_str转换为对应的numpy类型
    # 处理int4 pack、float8转换等
```

---

### 阶段3：CPU Golden计算预处理

#### 调用入口
```python
# test_kv_quant_sparse_flash_attention_single.py:33-34
input_tensor_dict = utils.generate_input_tensors(params)
cpu_result, _, _ = utils.compute_cpu(input_tensor_dict, params)

# utils.py:195-204
def compute_cpu(input_tensor_dict, params):
    # FP16/BF16转FP32提高精度
    for key in input_tensor_dict:
        if dtype in [np.float16, bfloat16]:
            input_tensor_dict[key] = input_tensor_dict[key].astype(np.float32)
    out_data = kv_quant_sparse_flash_attention_golden.compute_golden(input_tensor_dict, params)
```

#### CPU预处理详细流程

```
kv_quant_sparse_flash_attention_golden.py:compute_golden() [752-942]
    │
    ├── get_param_fus() [614-718]
    │       └── 从input_tensor_dict和params构建fa_param字典
    │
    ├── preprocessing() [452-490]
    │       ├── concat_tensor(): Q与Q_Rope拼接
    │       ├── concat_tensor(): K与K_Rope拼接
    │       └── hifloat8转FP32: trans_np_hifuint8_tensor_to_float32()
    │
    ├── _n_trans_shape_to_bnsd() [493-536]
    │       ├── BSND → BNSD: transpose(0,2,1,3)
    │       ├── TND → BNSD: 按actual_seq重组
    │       └── 处理sparse_indices特殊转换
    │
    ├── sparse_indices生成 [855-876]
    │       └── 根据sparse_mode生成有效的block索引
    │
    └── PA模式分支:
            ├── PA_BSND: deepseek_ds_pa_preprocessing() [1057-1189]
            │       ├── 生成block_table
            │       ├── K/V按block重组到k_cache/v_cache
            │       └── K_Rope和dequant_scale按block重组
            │
            └── 非PA: kv_concat_nopa_preprocessing() [721-750]
                    └── K + K_Rope + dequant_scale 直接concat
```

#### 关键预处理函数详解

**1. preprocessing() - Q/K拼接** (`kv_quant_sparse_flash_attention_golden.py:452-490`)
```python
# Q拼接: query_base + query_rope
q_new_tensor = concat_tensor(q_tensor, q_rope_tensor, numHeads)

# K拼接: key_base + key_rope + antiquant_scale
# hifloat8需要先转FP32
key = trans_np_hifuint8_tensor_to_float32(k_tensor)  # 若为hif8
k_new_tensor = concat_tensor(key, k_rope_tensor, antiquant_scale)
```

**2. _n_trans_shape_to_bnsd() - Layout转换** (`kv_quant_sparse_flash_attention_golden.py:493-536`)
```python
# BSND → BNSD: [B,S,N,D] → [B,N,S,D]
tensor = tensor.transpose(0, 2, 1, 3)

# TND → BNSD: [T,N,D] → [B,N,S,D]
# 按actual_seq_list将每个batch的有效token展开
for b in range(B):
    new_tensor[b, n, 0:act_s, :] = tensor[t_start:t_end, n, :]
```

**3. deepseek_ds_pa_preprocessing() - PA模式KV Cache构建** (`kv_quant_sparse_flash_attention_golden.py:1057-1189`)
```python
# 核心逻辑：按block_table将KV数据填充到k_cache/v_cache
for b in range(B):
    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
        if kv_cache_blk_id == -1: continue
        # 将BSND格式的KV按block映射到cache
        k_cache[kv_cache_blk_id, 0:blockSize, n, :] = k_tensor_bsnd[b, block_offset:block_offset+blockSize, n, :]

# 最终concat: k_cache + k_rope_cache + dequant_scale_cache
k_cache_concat = np.concatenate((k_cache, k_rope_cache, dequant_scale_cache), axis=-1)
```

---

### 阶段4：NPU调用流程

#### 调用入口
```python
# test_kv_quant_sparse_flash_attention_single.py:35
npu_result = kv_quant_sparse_flash_attention_process.call_npu(input_tensor_dict, params)
```

#### NPU调用详细流程

```
kv_quant_sparse_flash_attention_process.py:call_npu() [151-193]
    │
    ├── 数据预处理:
    │       ├── sparse_indices/block_table转为numpy (如有tensor)
    │       └── uint64 → int64
    │       └── bfloat16: numpy → torch.from_numpy → to(torch.bfloat16)
    │       └── float8_e5m2/e4m3fn: numpy → torch.float8_xxx
    │
    ├── dtype统一转换:
    │       ├── query → torch.bfloat16
    │       ├── query_rope → torch.bfloat16
    │       └── key_rope → torch.bfloat16
    │
    ├── 设备迁移:
    │       └── torch_tensor_dict[key].to('npu')
    │
    └── call_npu_op() [42-126]
            └── torch_npu.npu_kv_quant_sparse_flash_attention(...)
```

#### NPU数据转换矩阵

| 输入dtype | numpy → torch转换 | 行号 |
|-----------|-------------------|------|
| uint64 | astype(np.int64) → torch.from_numpy | 165-169 |
| bfloat16 | astype(np.float32) → torch.bfloat16 | 171-173 |
| float8_e5m2 | astype(np.float32) → torch.float8_e5m2 | 175-177 |
| float8_e4m3fn | astype(np.float32) → torch.float8_e4m3fn | 179-181 |
| 其他 | torch.from_numpy() | 183-187 |

#### NPU算子调用参数映射 (`kv_quant_sparse_flash_attention_process.py:42-126`)
```python
torch_npu.npu_kv_quant_sparse_flash_attention(
    q_tensor,              # query_cache (已拼接query+query_rope)
    k_tensor,              # key_cache (已拼接key+key_rope+dequant_scale)
    v_tensor,              # value_cache
    sparse_indices,        # 稀疏索引
    scale_value,           # 缩放因子
    sparse_block_size,     # 稀疏块大小
    key_quant_mode,        # Key量化模式
    value_quant_mode,      # Value量化模式
    key_dequant_scale,     # Key反量化scale
    value_dequant_scale,   # Value反量化scale
    block_table,           # PA模式的block表
    actual_seq_lengths_query,  # 实际query序列长度
    actual_seq_lengths_kv,     # 实际KV序列长度
    layout_query,          # Query布局
    layout_kv,             # KV布局
    sparse_mode,           # 稀疏模式
    attention_mode,        # 注意力模式
    quant_scale_repo_mode, # 量化scale存储模式
    tile_size,             # Tile大小
    rope_head_dim,         # RoPE维度
    key_dtype,             # Key dtype (hifloat8/float8_e4m3fn)
    value_dtype,           # Value dtype
    pre_tokens,            # 前token数
    next_tokens,           # 后token数
)
```

---

## 三、数据流关键节点汇总

### 1. input_tensor_dict 数据结构演变

```
阶段                          数据结构                      主要变化
─────────────────────────────────────────────────────────────────────────────
生成后       numpy.ndarray          各dtype原始数据
                │ shape: params['shape_input'][key]
                │ dtype: np.float16/bf16/float8/int32等

CPU预处理前  numpy.ndarray          FP16/BF16统一转FP32
                │ (compute_cpu入口处)
                │ utils.py:198-199

CPU预处理后  numpy.ndarray          BNSD布局
                │ Q/K已拼接Rope
                │ PA模式已构建block_table和KV cache

NPU调用前    torch.Tensor          torch格式
                │ dtype: torch.bfloat16/float8_e4m3fn
                │ kv_quant_sparse_flash_attention_process.py:164-192

NPU设备上    torch.Tensor (npu)    NPU设备tensor
                │ .to('npu')
                │ kv_quant_sparse_flash_attention_process.py:192
```

### 2. 核心数据转换函数索引表

| 功能 | 函数 | 文件 | 行号 |
|------|------|------|------|
| numpy dtype获取 | `get_np_dtype()` | `generate_tensor_data.py` | 138-192 |
| torch dtype获取 | `get_torch_dtype()` | `generate_tensor_data.py` | 195-216 |
| tensor数据转换 | `convert_tensor_data()` | `tools.py` | 50-85 |
| hif8转FP32 | `trans_np_hifuint8_tensor_to_float32()` | `kv_quant_sparse_flash_attention_golden.py` | 305-314 |
| tensor拼接 | `concat_tensor()` | `kv_quant_sparse_flash_attention_golden.py` | 409-449 |
| Layout转换 | `_n_trans_shape_to_bnsd()` | `kv_quant_sparse_flash_attention_golden.py` | 493-536 |
| numpy→torch | `call_npu()` 内部处理 | `kv_quant_sparse_flash_attention_process.py` | 164-192 |
| NPU设备迁移 | `to('npu')` | `kv_quant_sparse_flash_attention_process.py` | 192 |

---

## 四、特殊处理场景

### 1. PA_BSND模式
**文件**: `kv_quant_sparse_flash_attention_golden.py:894-936`, `1057-1189`

```
特点:
├── KV采用block形式存储 (key_cache/value_cache)
├── 需要block_table映射逻辑block到物理block
├── K/V/Rope/dequant_scale都需要按block重组
└── 最终concat: [K_base(512) | K_rope(64) | dequant_scale(4)]
```

### 2. TND模式
**文件**: `kv_quant_sparse_flash_attention_golden.py:515-534`, `564-587`

```
特点:
├── Tensor维度为 [T, N, D] (T=各batch token总和)
├── 需按actual_seq_lengths展开为BSND/BNSD格式
├── CPU计算时转为BNSD: [B, N, S, D]
└── 输出需还原为TND格式
```

### 3. hifloat8精度
**文件**: `generate_tensor_data.py:289-306`, `kv_quant_sparse_flash_attention_golden.py:305-406`

```
特点:
├── 生成时: FP32 → hifuint8 (trans_np_float_tensor_to_hifuint8)
├── CPU计算时: hifuint8 → FP32 (trans_np_hifuint8_tensor_to_float32)
├── NPU调用时: torch_npu.hifloat8 dtype
└── 支持多种舍入模式: hybrid/TA/SSR
```

---

## 五、流程图（ASCII）

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    pytest 测试入口                           │
                    │  test_kv_quant_sparse_flash_attention_single.py:42          │
                    └──────────────────────────┬──────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 阶段1: 参数解析                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐  │
│ │ utils.convert_param_combination_to_cs_format()                              │  │
│ │ ├── 解析 layout_query, layout_kv                                             │  │
│ │ ├── 构建 shape_input (根据layout确定各tensor shape)                          │  │
│ │ ├── 构建 dtype_input (q_dtype, kv_dtype等)                                   │  │
│ │ └── 构建 range_input (数据生成范围)                                           │  │
│ └─────────────────────────────────────────────────────────────────────────────┘  │
│ 输出: params字典                                                                  │
└──────────────────────────────────────────────────────────────────┬───────────────┘
                                                                   │
                                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 阶段2: Tensor生成                                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────────┐  │
│ │ utils.generate_input_tensors()                                              │  │
│ │ for key in shape_keys:                                                       │  │
│ │     generate_tensor_data.gen_tensor_data(params, key)                        │  │
│ │         ├── gen_boundary_tensor_data() - 边界值(null/nan/inf)               │  │
│ │         ├── gen_nonbound_tensor_data() - 随机值                              │  │
│ │         │       └── gen_uniform_data() / gen_normal_data()                   │  │
│ │         └── 特殊dtype转换:                                                    │  │
│ │             ├── hifloat8: trans_np_float_tensor_to_hifuint8()               │  │
│ │             ├── float8_e8m0: 位操作                                          │  │
│ │             └── uint1: packbits                                              │  │
│ │     tools.convert_tensor_data() - 格式标准化                                 │  │
│ └─────────────────────────────────────────────────────────────────────────────┘  │
│ 输出: input_tensor_dict = {key: numpy.ndarray}                                   │
└──────────────────────────────────────────────────────────┬───────────────────────┘
                                                           │
                              ┌────────────────────────────┴────────────────────────┐
                              │                                                     │
                              ▼                                                     ▼
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│ 阶段3: CPU Golden计算                            │   │ 阶段4: NPU调用                                  │
│ ┌─────────────────────────────────────────────┐ │   │ ┌─────────────────────────────────────────────┐ │
│ │ utils.compute_cpu()                         │ │   │ │ kv_quant_sparse_flash_attention_process.    │ │
│ │ ├── FP16/BF16 → FP32 (精度提升)             │ │   │ │ call_npu()                                   │ │
│ │ └── compute_golden()                        │ │   │ │ ├── numpy → torch转换                        │ │
│ │     ├── get_param_fus() - 参数获取          │ │   │ │ │   ├── uint64 → int64                        │ │
│ │     ├── preprocessing() - Q/K拼接           │ │   │ │ │   ├── bf16: numpy→torch.bf16               │ │
│ │     │   ├── concat Q + Q_Rope              │ │   │ │ │   └── float8: numpy→torch.float8            │ │
│ │     │   └── concat K + K_Rope              │ │   │ │ ├── dtype统一: query→bf16                    │ │
│ │     │       + dequant_scale                │ │   │ │ ├── .to('npu') 设备迁移                      │ │
│ │     │   └── hifloat8 → FP32转换             │ │   │ │ └── call_npu_op()                            │ │
│ │     ├── _n_trans_shape_to_bnsd()            │ │   │ │     └── torch_npu.npu_kv_quant_sparse_       │ │
│ │     │   ├── BSND → BNSD                    │ │   │ │         flash_attention(...)                 │ │
│ │     │   └── TND → BNSD                     │ │   │ └─────────────────────────────────────────────┘ │
│ │     ├── sparse_indices生成                   │ │   │ 输出: torch.Tensor (npu设备)                   │
│ │     └── PA分支:                              │ │   └─────────────────────────────────────────────────┘
│ │         ├── deepseek_ds_pa_preprocessing() │ │   │
│ │         │   ├── block_table生成            │ │   │
│ │         │   ├── KV按block重组              │ │   │
│ │         │   └── k_cache/v_cache构建        │ │   │
│ │         └── kv_concat_nopa_preprocessing() │ │   │
│ │     └── _t_increattention_bnsd()            │ │   │
│ │         └── Attention计算                   │ │   │
│ └─────────────────────────────────────────────┘ │   │
│ 输出: cpu_result (torch.Tensor)                  │   │
└─────────────────────────────────────────────────┘   └──────────────────────────────────────────────────┘
                              │                                                     │
                              └────────────────────────────┬────────────────────────┘
                                                           │
                                                           ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    结果对比                                   │
                    │  result_compare_method.check_result(cpu_result, npu_result) │
                    └─────────────────────────────────────────────────────────────┘
```

---

## 六、重点文件和函数速查表

| 功能模块 | 文件 | 核心函数 | 行号范围 |
|----------|------|----------|----------|
| **参数转换** | `utils.py` | `convert_param_combination_to_cs_format()` | 207-384 |
| **数据生成** | `generate_tensor_data.py` | `gen_tensor_data()` | 235-286 |
| **边界值生成** | `generate_tensor_data.py` | `gen_boundary_tensor_data()` | 502-531 |
| **随机值生成** | `generate_tensor_data.py` | `gen_nonbound_tensor_data()` | 534-604 |
| **数据转换** | `tools.py` | `convert_tensor_data()` | 50-85 |
| **hif8转换** | `generate_tensor_data.py` | `trans_np_float_tensor_to_hifuint8()` | 289-306 |
| **CPU入口** | `utils.py` | `compute_cpu()` | 195-204 |
| **Golden计算** | `kv_quant_sparse_flash_attention_golden.py` | `compute_golden()` | 752-942 |
| **参数获取** | `kv_quant_sparse_flash_attention_golden.py` | `get_param_fus()` | 614-718 |
| **预处理拼接** | `kv_quant_sparse_flash_attention_golden.py` | `preprocessing()` | 452-490 |
| **Layout转换** | `kv_quant_sparse_flash_attention_golden.py` | `_n_trans_shape_to_bnsd()` | 493-536 |
| **PA预处理** | `kv_quant_sparse_flash_attention_golden.py` | `deepseek_ds_pa_preprocessing()` | 1057-1189 |
| **NPU入口** | `kv_quant_sparse_flash_attention_process.py` | `call_npu()` | 151-193 |
| **NPU算子** | `kv_quant_sparse_flash_attention_process.py` | `call_npu_op()` | 42-126 |

---

## 七、总结

### 核心流程要点

1. **数据生成阶段**：根据params中的shape_input、dtype_input、range_input生成numpy数组，特殊dtype需要额外转换处理。

2. **CPU预处理阶段**：
   - 统一提升精度（FP16/BF16→FP32）
   - Q/K与Rope拼接
   - Layout转换（BSND/TND→BNSD）
   - PA模式构建block_table和KV cache

3. **NPU调用阶段**：
   - numpy→torch格式转换
   - dtype统一为torch.bfloat16
   - 迁移到NPU设备
   - 调用torch_npu算子

### 关键设计决策

| 决策点 | 原因 | 实现 |
|--------|------|------|
| CPU用FP32计算 | 提高精度避免FP16累积误差 | `utils.py:198-199` |
| hifloat8双向转换 | CPU不支持hif8原生计算 | 生成时转hif8，计算时转FP32 |
| TND→BNSD转换 | CPU计算需标准维度顺序 | 按actual_seq展开 |
| PA block_table | KV Cache按block存储提高效率 | block_table映射逻辑block到物理block |