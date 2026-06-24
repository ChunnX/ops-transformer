# MXFP8 Golden 代码修改与优化文档

**来源提交**: `43bb5362a4e1299a415b038ebf91474be9be8a28` (mxfp8 pytest)
**涉及文件**: `quant_lightning_indexer_golden.py` 及相关测试文件

---

## 一、MXFP8 功能新增

### 1.1 E8M0 与 Float 互转工具函数

新增两个工具函数，用于 MXFP8 scale 的格式转换：

```python
def float_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """float -> E8M0: E = clamp(floor(log2(|x|)) + 127, 1, 254), 零值 E=0"""
    x_abs = x.float().abs()
    zero_mask = (x_abs == 0)
    x_safe = torch.where(zero_mask, torch.ones_like(x_abs), x_abs)
    biased_exp = torch.clamp(torch.floor(torch.log2(x_safe)) + 127.0, 1.0, 254.0)
    biased_exp = torch.where(zero_mask, torch.zeros_like(biased_exp), biased_exp)
    return biased_exp.to(torch.uint8).view(torch.float8_e8m0fnu)


def e8m0_to_float(x: torch.Tensor) -> torch.Tensor:
    """E8M0 -> float: E=0 -> 0.0, E=1~254 -> 2^(E-127), E=255 -> NaN"""
    raw = x.view(torch.uint8).float()
    result = torch.pow(2.0, raw - 127.0)
    result = torch.where(raw == 0, torch.zeros_like(result), result)
    result = torch.where(raw == 255, torch.full_like(result, float('nan')), result)
    return result
```

### 1.2 MXFP8 计算路径 `cal_atten_per_batch_mxfp8`

新增 MXFP8 专用的 per-batch 计算函数，核心差异：
- **Scale 类型**: 使用 `e8m0_to_float` 将 E8M0 scale 转为 float
- **Scale 应用时机**: 在 QK 矩阵乘之前先乘 scale（`cur_q = cur_q * cur_q_scale`）
- **无 kScale 后乘**: 不像 fp8 路径在 reduce_sum 后再乘 kScale

```python
def cal_atten_per_batch_mxfp8(self, b_idx):
    cur_q = self.cur_q.to(dtype=torch.float32)
    cur_k = self.cur_k.to(dtype=torch.float32)
    cur_wt = self.cur_wt.to(dtype=torch.float32)
    cur_q_scale = e8m0_to_float(self.cur_q_scale)
    cur_k_scale = e8m0_to_float(self.cur_k_scale)

    # 先乘 scale 再做矩阵乘
    cur_q = cur_q * cur_q_scale
    cur_k = cur_k * cur_k_scale

    qk_bmm_res = torch.bmm(
        cur_q.squeeze(0),
        cur_k.permute(0, 1, 3, 2).squeeze(0)
    ).unsqueeze(0)

    qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)

    brc_vmul = torch.bmm(
        cur_wt.permute(0, 2, 3, 1).to(dtype=torch.float32).squeeze(0),
        qk_relu_out.permute(0, 2, 1, 3).to(dtype=torch.float32).squeeze(0)
    ).unsqueeze(0)
    # ... 后续 reduce_sum + topk 逻辑与 fp8 路径一致
```

### 1.3 `cal_atten_bnsd` 中的 MXFP8 分支

```python
elif self.qk_dtype == torch.float8_e4m3fn:
    if self.mxfp8_flag == 0:
        y[...], y_value[...] = self.cal_atten_per_batch_fp8(b_idx)
    else:
        y[...], y_value[...] = self.cal_atten_per_batch_mxfp8(b_idx)
```

### 1.4 MXFP8 Scale Shape 处理

MXFP8 的 scale 维度为 `[B, S, N, D/32]`（per-block scale），需要额外处理：

```python
# 在 qli_output_single 中
if dequant_dtype == torch.float8_e8m0fnu:
    mxfp8_flag = 1
    query_dequant_scale_shape.append(head_dim // 32)  # 增加 D/32 维度
    key_dequant_scale_shape.append(head_dim // 32)
```

### 1.5 D 轴广播 `broadcast_d_axis`

MXFP8 scale 的 D 维度为 D/32，需要广播到完整 D 维度才能与 Q/K 做乘法：

```python
def broadcast_d_axis(self, temp_tensor):
    dg = 32
    return temp_tensor.repeat_interleave(dg, dim=-1)
```

在 `forward` 中调用：

```python
if mxfp8_flag == 1:
    q_scale_bnsd_tensor = self.broadcast_d_axis(q_scale_bnsd_tensor)
    k_scale_bnsd_tensor = self.broadcast_d_axis(k_scale_bnsd_tensor)
```

### 1.6 MXFP8 数据生成

```python
if mxfp8_flag == 1:
    query_dequant_scale = float_to_e8m0(
        torch.tensor(np.random.uniform(q_scale_datarange[0], q_scale_datarange[1],
                                       query_dequant_scale_shape))
    ).cpu()
    key_dequant_scale = float_to_e8m0(
        torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1],
                                       key_dequant_scale_shape))
    ).cpu()
```

### 1.7 MXFP8 下 kScale 的 N 轴广播

MXFP8 场景下 k_head_num 广播时，kScale 也需要广播：

```python
if q_head_num != k_head_num:
    k_bnsd_tensor = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor)
    if mxfp8_flag == 1:
        k_scale_bnsd_tensor = self.broadcast_n_axis(q_head_num, k_head_num, k_scale_bnsd_tensor)
```

### 1.8 PA_BBND 场景下 MXFP8 kScale 处理

PA 场景构建 key_dequant_scale 时，MXFP8 需要额外处理 D/32 维度：

```python
key_dequant_scale_expand_shape = [batch_size, k_head_num, k_max_block_num_per_batch * block_size]
key_dequant_scale_shape = [block_num, block_size, k_head_num]
if mxfp8_flag == 1:
    key_dequant_scale_expand_shape.append(head_dim // 32)
    key_dequant_scale_shape.append(head_dim // 32)

# 构建时
if mxfp8_flag == 1:
    key_dequant_scale[cur_block_id, :, i_n, :] = \
        key_dequant_scale_expand[i_batch, i_n, block_start_pos:block_start_pos+block_size, :]
else:
    key_dequant_scale[cur_block_id, :, i_n] = \
        key_dequant_scale_expand[i_batch, i_n, block_start_pos:block_start_pos+block_size]
```

---

## 二、代码优化（重构）

### 2.1 `trans_shape_to_bnsd` 简化

**修改前**: 需要传入 `shape` 参数和 `is_weights` 标志，返回 `(tensor, shape)` 元组

**修改后**: 直接从 `tensor.shape` 获取维度，用 `len(tensor.shape)` 判断是否为 weight，只返回 tensor

```python
# 修改前
def trans_shape_to_bnsd(self, tensor, shape, layout, headnums=None, act_seq=None, is_weights=False, tensor_name=None):
    if layout in ["BSND"]:
        B = shape[0]; S = shape[1]; N = shape[2]; D = 1
        if is_weights:
            tensor = torch.unsqueeze(tensor, dim=-1)
        else:
            D = shape[3]
        ...
        return tensor, [B, N, S, D]

# 修改后
def trans_shape_to_bnsd(self, tensor, layout, headnums=None, act_seq=None, tensor_name=None):
    if layout == "BSND":
        B = tensor.shape[0]; S = tensor.shape[1]; N = tensor.shape[2]; D = 1
        if len(tensor.shape) == 3:
            tensor = torch.unsqueeze(tensor, dim=-1)
        else:
            D = tensor.shape[3]
        ...
        return tensor
```

**删除的分支**: `BSN`、`TN` 布局分支（不再需要，统一用 `len(tensor.shape)` 判断）

**新增分支**: `PA_BBND` 布局

```python
elif layout == "PA_BBND":
    if len(tensor.shape) == 3:
        tensor = torch.unsqueeze(tensor, dim=-1)
    return tensor
```

### 2.2 移除冗余 shape 成员变量

**修改前**: 维护 `q_bnsd_shape`、`k_bnsd_shape`、`wt_bnsd_shape` 等成员变量

**修改后**: 直接使用 `.shape` 属性

```python
# 修改前
self.q_bnsd_shape = q_bnsd_shape
m_shape_std = [q_bnsd_shape[2], k_bnsd_shape[2]]

# 修改后
m_shape_std = [q_bnsd_tensor.shape[2], k_bnsd_tensor.shape[2]]
```

### 2.3 `broadcast_n_axis` 简化

移除 `input_dtype` 参数，不再返回 shape：

```python
# 修改前
def broadcast_n_axis(self, n1, n2, temp_tensor, input_dtype):
    ...
    return modify_tensor, modify_tensor.shape

# 修改后
def broadcast_n_axis(self, n1, n2, temp_tensor):
    ...
    return modify_tensor
```

### 2.4 `forward` 接口变更

新增 `mxfp8_flag` 参数，移除 `q_shape` 相关逻辑：

```python
# 修改前
def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale,
            actual_seq_lengths_query, actual_seq_lengths_key, block_table):

# 修改后
def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale,
            actual_seq_lengths_query, actual_seq_lengths_key, block_table, mxfp8_flag):
```

### 2.5 `__init__` 接口变更

新增 `weight_dtype` 和 `cmp_ratio` 参数，移除 `q_shape`/`k_shape`/`w_shape` 成员变量：

```python
# 修改前
def __init__(self, batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num,
             head_dim, block_size, block_num, qk_dtype, dequant_dtype, actual_seq_dtype,
             act_seq_q, act_seq_k, query_quant_mode, key_quant_mode, layout_query,
             layout_key, sparse_count, sparse_mode):
    self.w_dtype = dequant_dtype
    # 内部计算 q_shape, k_shape, w_shape

# 修改后
def __init__(self, batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num,
             head_dim, block_size, block_num, qk_dtype, weight_dtype, dequant_dtype,
             actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode, key_quant_mode,
             layout_query, layout_key, sparse_count, sparse_mode, cmp_ratio):
    self.weight_dtype = weight_dtype
    self.cmp_ratio = cmp_ratio
    self.w_dtype = torch.bfloat16
    self.mxfp8_flag = 0
```

### 2.6 依赖变更

```python
# 修改前
import npu_ops_transformer

# 修改后
import torchair
import torch.nn as nn
from torchair.configs.compiler_config import CompilerConfig
```

### 2.7 NPU 调用简化

移除 metadata 构建，直接调用 `torch_npu.npu_quant_lightning_indexer`：

```python
# 修改前
metadata = npu_ops_transformer.ops.npu_quant_lightning_indexer_metadata(...)
npu_result = npu_ops_transformer.ops.npu_quant_lightning_indexer_v2(
    query, key, weights, ..., metadata=metadata, ...)

# 修改后
npu_result = torch_npu.npu_quant_lightning_indexer(
    query, key, weights, ..., sparse_mode=sparse_mode)
```

### 2.8 结果排序对齐

CPU 和 NPU 结果都先排序再比较，消除 topk 顺序差异：

```python
cpu_result, _ = torch.sort(cpu_result)
npu_result, _ = torch.sort(npu_result)
```

---

## 三、cmp_ratio（压缩比）支持

### 3.1 概述

`cmp_ratio` 用于支持 Key 序列压缩场景，影响：
- `actualSeqLengths_k` 的有效长度 = `actualSeqLengths_k / cmp_ratio`
- Mask 生成中的有效位置计算
- PA 场景下 block 数量计算

### 3.2 Mask 生成中的 cmp_ratio

```python
def create_mask(self, m_shape, act_k, S1):
    cmp_ratio = self.cmp_ratio
    tmp_pos_orig = act_k - S1
    for i in range(S1):
        if ((tmp_pos_orig + i + 1) / cmp_ratio) < 0:
            atten_masks[i, :] = 1
        else:
            atten_masks[i, math.floor((tmp_pos_orig + i + 1) / cmp_ratio):] = 1
```

### 3.3 `create_mask_right_down` 中的 cmp_ratio

```python
S2 = math.floor(actualSeqLengthsK[i] / cmp_ratio)
```

### 3.4 `cal_atten_bnsd` 中的 cmp_ratio

```python
curr_actualSeq_k = math.floor(actualSeqLengths_k[b_idx] / cmp_ratio)
out_shape_bnss[-1] = math.floor(max(actualSeqLengths_k) / cmp_ratio)
```

### 3.5 PA 场景中的 cmp_ratio

```python
k_max_s2 = math.floor(max(act_seq_k) / cmp_ratio)
cur_cmp_act_k = math.floor(cur_act_k / cmp_ratio)
```

### 3.6 `qli_output_single` 中的 cmp_ratio

```python
actual_seq_lengths_key = torch.tensor(
    np.random.uniform(k_seq * cmp_ratio, k_seq * cmp_ratio, batch_size)
).to(actual_seq_dtype).cpu()

k_bnsd_tensor = self.trans_shape_to_bnsd(
    key, layout_key, k_head_num,
    torch.floor(actualSeqLengths_k / cmp_ratio).to(actual_seq_dtype))
```

---

## 四、测试基础设施变更

### 4.1 新增参数

| 参数 | 位置 | 说明 |
|------|------|------|
| `weight_dtype` | params[11] | weight 数据类型（如 `torch.bfloat16`） |
| `cmp_ratio` | params[27] | Key 压缩比（1/2/4/8/16/32/64/128） |

### 4.2 新增测试参数集

```python
TEST_PARAMS = {
    "fp8_1":    { "dequant_dtype": [torch.float32],       "sparse_count": [2048], ... },
    "mxfp8_1":  { "dequant_dtype": [torch.float8_e8m0fnu], "sparse_count": [2048], ... },
    "fp8_2":    { "dequant_dtype": [torch.float32],       "sparse_count": [128],  ... },
    "mxfp8_2":  { "dequant_dtype": [torch.float8_e8m0fnu], "sparse_count": [128],  ... },
    "mxfp8_3":  { "dequant_dtype": [torch.float8_e8m0fnu], "query_datarange": [[1,1]], ... },  # 全1
    "mxfp8_4":  { "dequant_dtype": [torch.float8_e8m0fnu], "query_datarange": [[-1,1]], ... }, # 非全1
}
```

### 4.3 `result_compare_method.py` 变更

- 新增 `compare_topk_invalid` 函数：检查 invalid 部分是否全为 -1
- `check_result` 新增 `weight_dtype` 和 `cmp_ratio` 参数解包
- TND 场景 `act_seq_q` 的前缀和处理移到更早位置

### 4.4 `test_quant_lightning_indexer_single.py` 变更

- 参数组合生成从循环内移到循环外（`all_param_combinations` 全局收集）
- 新增 `cmp_ratio` 到 `param_names` 和 `param_values`
- 添加 `assert` 断言确保精度校验失败时测试失败

### 4.5 `test_run.sh` 变更

- 添加 `set -euo pipefail` 增强错误处理
- 移除末尾 `exit 0`

### 4.6 `quant_lightning_indexer_pt_save.py` 变更

- `q_t_size`/`k_t_size` 添加 `int()` 转换
- TND 场景 `act_seq_k` 前缀和转换提前
- 修复 `k_datarange` -> `key_datarange` 变量名错误

---

## 五、V2 Golden 代码对齐整改（从 V1 mxpytest 回合）

> 以下修改将 V2 `quant_lightning_indexer_v2_golden.py` 对齐到 V1 mxpytest commit 中的重构和 bugfix。

### 5.1 Shape 推导整改：删除冗余 shape 成员变量

**问题**: V2 的 `__init__` 中仍维护 `self.q_shape`/`self.k_shape`/`self.w_shape` 成员变量，`forward()` 中维护 `q_bnsd_shape`/`k_bnsd_shape` 等，全部冗余。

**修改**:

#### `__init__` 中删除 shape 成员变量

```python
# 修改前（删除以下代码）
if layout_query == "BSND":
    self.q_shape = [batch_size, q_seq, q_head_num, head_dim]
    self.w_shape = [batch_size, q_seq, q_head_num]
    self.q_tnd_flag = 0
elif layout_query == "TND":
    self.q_shape = [q_t_size, q_head_num, head_dim]
    self.w_shape = [q_t_size, q_head_num]
    self.q_tnd_flag = 1

if layout_key == "BSND":
    self.k_shape = [batch_size, k_seq, k_head_num, head_dim]
elif layout_key == "TND":
    self.k_shape = [k_t_size, k_head_num, head_dim]

# 修改后：全部删除，w_dtype 改为固定 bfloat16
self.w_dtype = torch.bfloat16  # 原来是 self.w_dtype = dequant_dtype
```

#### `forward()` 中删除 shape 局部变量和 bnsd_shape 成员赋值

```python
# 修改前
q_shape = self.q_shape
w_shape = self.w_shape
q_scale_shape = [batch_size, q_seq, q_head_num]  # 或 [q_t_size, q_head_num]
layout_key_scale = "BSN"  # 或 "TN" 或 layout_key
k_shape = self.k_shape
k_scale_shape = [batch_size, k_seq, k_head_num]  # 或 [k_t_size, k_head_num] 等

# 修改后：全部删除，不再维护这些局部变量
```

```python
# 修改前（forward 末尾）
self.q_bnsd_shape = q_bnsd_shape
self.k_bnsd_shape = k_bnsd_shape
self.wt_bnsd_shape = wt_bnsd_shape
self.q_scale_bnsd_shape = q_scale_bnsd_shape
self.k_scale_bnsd_shape = k_scale_bnsd_shape
m_shape_std = [q_bnsd_shape[2], k_bnsd_shape[2]]
batch = q_bnsd_shape[0]

# 修改后：删除所有 *_shape 成员赋值，直接用 .shape
self.q_bnsd_tensor = q_bnsd_tensor
self.k_bnsd_tensor = k_bnsd_tensor
self.wt_bnsd_tensor = wt_bnsd_tensor
self.q_scale_bnsd_tensor = q_scale_bnsd_tensor
self.k_scale_bnsd_tensor = k_scale_bnsd_tensor
m_shape_std = [q_bnsd_tensor.shape[2], k_bnsd_tensor.shape[2]]
batch = q_bnsd_tensor.shape[0]
```

#### `cal_atten_bnsd()` 和 `forward()` 末尾的 `copy.deepcopy` 替换

```python
# 修改前（3 处）
out_shape_bnsd = copy.deepcopy(self.q_bnsd_shape)

# 修改后
out_shape_bnsd = list(q_bnsd_tensor.shape)
```

### 5.2 `trans_shape_to_bnsd` 函数整改

**问题**: V2 的 `trans_shape_to_bnsd` 仍使用旧签名（含 `shape`/`is_weights` 参数），返回 `(tensor, shape)` 元组，且包含已废弃的 `BSN`/`TN` 分支。

**修改**:

```python
# 修改前
def trans_shape_to_bnsd(self, tensor, shape, layout, headnums=None, act_seq=None,
                        is_weights=False, tensor_name=None):
    if layout in ["BSND"]:
        B = shape[0]; S = shape[1]; N = shape[2]; D = 1
        if is_weights:
            tensor = torch.unsqueeze(tensor, dim=-1)
        else:
            D = shape[3]
        ...
        return tensor, [B, N, S, D]
    elif layout == "BSN":       # ← 废弃分支
        ...
    elif layout == "TN":        # ← 废弃分支
        ...

# 修改后
def trans_shape_to_bnsd(self, tensor, layout, headnums=None, act_seq=None,
                        tensor_name=None):
    if layout == "BSND":
        B = tensor.shape[0]; S = tensor.shape[1]; N = tensor.shape[2]; D = 1
        if len(tensor.shape) == 3:          # 替代 is_weights
            tensor = torch.unsqueeze(tensor, dim=-1)
        else:
            D = tensor.shape[3]
        ...
        return tensor                       # 只返回 tensor
    elif layout == "TND":
        ...
        return new_tensor
    elif layout == "PA_BBND":              # ← 新增分支
        if len(tensor.shape) == 3:
            tensor = torch.unsqueeze(tensor, dim=-1)
        return tensor
    else:
        return tensor
```

**调用方同步更新**（`forward()` 中 5 处调用）:

```python
# 修改前
q_bnsd_tensor, q_bnsd_shape = self.trans_shape_to_bnsd(
    query, q_shape, layout_query, q_head_num, actualSeqLengths_q)
k_scale_bnsd_tensor, k_scale_bnsd_shape = self.trans_shape_to_bnsd(
    key_dequant_scale, k_scale_shape, layout_key_scale, k_head_num, ...)
wt_bnsd_tensor, wt_bnsd_shape = self.trans_shape_to_bnsd(
    weights, w_shape, layout_query, q_head_num, actualSeqLengths_q, is_weights)

# 修改后：去掉 shape 参数，去掉元组解包，key_scale 改用 layout_key
q_bnsd_tensor = self.trans_shape_to_bnsd(
    query, layout_query, q_head_num, actualSeqLengths_q)
q_scale_bnsd_tensor = self.trans_shape_to_bnsd(
    query_dequant_scale, layout_query, q_head_num, actualSeqLengths_q)
k_bnsd_tensor = self.trans_shape_to_bnsd(
    key, layout_key, k_head_num, torch.floor(actualSeqLengths_k).to(actual_seq_dtype))
k_scale_bnsd_tensor = self.trans_shape_to_bnsd(
    key_dequant_scale, layout_key, k_head_num, torch.floor(actualSeqLengths_k).to(actual_seq_dtype))
wt_bnsd_tensor = self.trans_shape_to_bnsd(
    weights, layout_query, q_head_num, actualSeqLengths_q)
```

### 5.3 `broadcast_n_axis` 简化

```python
# 修改前
def broadcast_n_axis(self, n1, n2, temp_tensor, input_dtype):
    ...
    return modify_tensor, modify_tensor.shape

# 修改后
def broadcast_n_axis(self, n1, n2, temp_tensor):
    ...
    return modify_tensor
```

调用方同步更新（去掉 `k_dtype` 参数和元组解包）:

```python
# 修改前
k_bnsd_tensor, k_bnsd_shape = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor, k_dtype)
k_scale_bnsd_tensor, k_scale_bnsd_shape = self.broadcast_n_axis(q_head_num, k_head_num, k_scale_bnsd_tensor, k_dtype)

# 修改后
k_bnsd_tensor = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor)
k_scale_bnsd_tensor = self.broadcast_n_axis(q_head_num, k_head_num, k_scale_bnsd_tensor)
```

### 5.4 `cal_atten_per_batch_fp8` — k_scale 添加 `.squeeze(-1)`

**问题**: fp8 路径中 `cur_k_scale` 未 squeeze 末尾维度，导致与 `reduce_sum` 相乘时广播不正确。

```python
# 修改前
cur_k_scale = self.cur_k_scale.to(dtype=torch.float32)

# 修改后
cur_k_scale = self.cur_k_scale.to(dtype=torch.float32).squeeze(-1)
```

### 5.5 `cal_atten_per_batch_int8` — k_scale 添加 `.unsqueeze(-1)`

**问题**: int8 路径中 `cur_k_scale` 缺少末尾维度，导致与 `reduce_sum` 相乘时维度不匹配。

```python
# 修改前
cur_k_scale = self.cur_k_scale.to(dtype=torch.float16)

# 修改后
cur_k_scale = self.cur_k_scale.to(dtype=torch.float16).unsqueeze(-1)
```

### 5.6 `create_mask_right_down` — mask 列表添加 numpy 转换

**问题**: `re_mask_batch` 中每个元素是 `torch.Tensor`，直接 `np.array(re_mask_batch, dtype=np.bool_)` 可能产生非预期的嵌套结构。

```python
# 修改前
re_mask_batch.append(atten_masks)
re_mask_np = np.array(re_mask_batch, dtype=np.bool_)

# 修改后：先逐个转为 numpy array，再整体转 bool
re_mask_batch.append(atten_masks)
re_mask_batch = [np.array(x) for x in re_mask_batch]
re_mask_np = np.array(re_mask_batch, dtype=np.bool_)
```

### 5.7 `qliv2_output_single` shape 推导整改

**问题**: V2 的 `qliv2_output_single` 函数中，shape 推导与数据生成混合在每个 layout 分支内，每个分支都重复写完整 shape，且 cpu golden forward 调用分散在各分支中。

**修改**: 将 shape 推导与数据生成分离，统一在分支外生成 tensor 和调用 forward。

#### Shape 推导与数据生成分离

```python
# 修改前：每个分支内直接写死 shape 生成 tensor
if layout_query == "BSND":
    if hifp8mode == 1:
        query = torch.tensor(np.random.uniform(..., (batch_size, q_seq, q_head_num, head_dim))).to(torch.float)
        query = trans_float_tensor_to_hifuint8(query, ...)
        ...
    elif mxfp8_flag == 1:
        query = torch.tensor(np.random.uniform(..., (batch_size, q_seq, q_head_num, head_dim))).to(qk_dtype)
        ...
    else:
        query = torch.tensor(np.random.uniform(..., (batch_size, q_seq, q_head_num, head_dim))).to(qk_dtype)
        ...
    weights = torch.tensor(np.random.uniform(..., (batch_size, q_seq, q_head_num))).to(weight_dtype)

# 修改后：先推导 shape，再统一生成
# 1. Shape 推导
if layout_query == "BSND":
    query_shape = [batch_size, q_seq, q_head_num, head_dim]
    query_dequant_scale_shape = [batch_size, q_seq, q_head_num]
    weights_shape = [batch_size, q_seq, q_head_num]
elif layout_query == "TND":
    query_shape = [q_t_size, q_head_num, head_dim]
    query_dequant_scale_shape = [q_t_size, q_head_num]
    weights_shape = [q_t_size, q_head_num]

# 2. MXFP8 scale shape 动态 append
if dequant_dtype == torch.float8_e8m0fnu:
    mxfp8_flag = 1
    query_dequant_scale_shape.append(head_dim // 32)
    key_dequant_scale_shape.append(head_dim // 32)

# 3. 统一数据生成（使用 shape 变量）
if hifp8mode == 1:
    query = torch.tensor(np.random.uniform(..., query_shape)).to(torch.float)
    query = trans_float_tensor_to_hifuint8(query, ...)
    ...
elif mxfp8_flag == 1:
    query = torch.tensor(np.random.uniform(..., query_shape)).to(qk_dtype)
    query_dequant_scale = float_to_e8m0(torch.tensor(np.random.uniform(..., query_dequant_scale_shape)))
    ...
else:
    query = torch.tensor(np.random.uniform(..., query_shape)).to(qk_dtype)
    ...
weights = torch.tensor(np.random.uniform(..., weights_shape)).to(weight_dtype)
```

#### PA_BBND 使用中间变量和动态 shape

```python
# 修改前：直接用 key_bnsd/key_dequant_scale_bns，shape 写死
key_bnsd = torch.tensor(np.random.uniform(..., (batch_size, k_head_num, k_max_s2, head_dim))).to(qk_dtype)
key_dequant_scale_bns = torch.tensor(np.random.uniform(..., (batch_size, k_head_num, k_max_s2))).to(dequant_dtype)
...
key_expand[:,:,:k_max_s2,:] = key_bnsd
...
key_dequant_scale_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim // 32), dtype=dequant_dtype)
key_dequant_scale_expand[:,:,:k_max_s2,:] = key_dequant_scale_bns

# 修改后：使用 key_cpu/key_dequant_scale_cpu 中间变量，shape 动态推导
key_cpu = key
key_dequant_scale_cpu = key_dequant_scale
...
key_expand[:,:,:k_max_s2,:] = key_cpu
...
key_dequant_scale_expand_shape = [batch_size, k_head_num, k_max_block_num_per_batch * block_size]
key_dequant_scale_block_shape = [block_num, block_size, k_head_num]
if mxfp8_flag == 1:
    key_dequant_scale_expand_shape.append(head_dim // 32)
    key_dequant_scale_block_shape.append(head_dim // 32)
key_dequant_scale_expand = torch.zeros(key_dequant_scale_expand_shape, dtype=dequant_dtype)
if mxfp8_flag == 1:
    key_dequant_scale_expand[:,:,:k_max_s2,:] = key_dequant_scale_cpu
else:
    key_dequant_scale_expand[:,:,:k_max_s2] = key_dequant_scale_cpu
```

#### cpu golden forward 统一调用

```python
# 修改前：每个 layout 分支内各自调用 forward
if layout_key == "BSND":
    ...
    cpu_result, topk_value = test_qliv2.forward(query, key, weights, ...)
elif layout_key == "TND":
    ...
    cpu_result, topk_value = test_qliv2.forward(query, key, weights, ...)
elif layout_key == "PA_BBND":
    ...
    cpu_result, topk_value = test_qliv2.forward(query, key_bnsd, weights, ...)

# 修改后：统一在分支外调用，传入 key_cpu/key_dequant_scale_cpu
# cpu golden 生成
cpu_result, topk_value = test_qliv2.forward(
    query, key_cpu, weights, query_dequant_scale, key_dequant_scale_cpu,
    actual_seq_lengths_query, actual_seq_lengths_key, block_table, mxfp8_flag)
cpu_result, _ = torch.sort(cpu_result)
```

---

## 六、关键设计决策总结

| 决策 | 说明 |
|------|------|
| `mxfp8_flag` 通过 `dequant_dtype == torch.float8_e8m0fnu` 判断 | 不依赖 Q_T 类型，只看 scale 类型 |
| MXFP8 scale 先乘再做矩阵乘 | 与 fp8 路径（先做矩阵乘再乘 scale）不同 |
| `broadcast_d_axis` 将 D/32 广播到 D | 使 scale 能与 Q/K 直接做元素乘 |
| 移除 `npu_ops_transformer` 依赖 | 改用 `torch_npu` 直调，简化接口 |
| CPU/NPU 结果都先排序 | 消除 topk 同值时顺序不确定的问题 |
| `cmp_ratio` 全局传递 | 支持 Key 序列压缩场景的 golden 计算 |
| 删除所有 shape 成员变量 | 统一用 `tensor.shape` 获取维度，消除冗余 |
| `trans_shape_to_bnsd` 去 shape 参数 | 从 tensor 自身推导维度，简化调用链 |
| fp8 的 k_scale `.squeeze(-1)` | 消除末尾多余维度，确保广播正确 |
| int8 的 k_scale `.unsqueeze(-1)` | 补齐末尾维度，确保维度匹配 |
| `qliv2_output_single` shape 推导分离 | 先推导 shape 再统一生成 tensor，消除分支内重复代码 |
