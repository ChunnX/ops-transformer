# kv_quant_sparse_attn_sharedkv pytest 测试工程修改方案

**文档版本**: v1.0
**创建日期**: 2026-05-19
**作者**: CANNBot

---

## 1. 概述

### 1.1 目标
更新 pytest 测试工程，使其能够正确测试当前算子的最新参数定义，包括：
- 新增参数支持：`seqused_ori_kv`、`seqused_cmp_kv`、`cmp_residual_kv`
- TND layout 支持：`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`
- 废弃参数清理：删除已废弃的 `seqused_kv` 参数

### 1.2 当前测试工程状态
测试工程位于 `tests/pytest/` 目录，主要文件：
- `test_kv_quant_sparse_attn_sharedkv_single.py`: 单例测试入口
- `kv_quant_sparse_attn_sharedkv_paramset.py`: 参数配置
- `kv_quant_sparse_attn_sharedkv_golden.py`: CPU golden 计算实现
- `batch/kv_quant_sparse_attn_sharedkv_process.py`: NPU 调用逻辑
- `utils.py`: 工具函数

### 1.3 当前支持的测试场景
- **layout_q**: TND, BSND
- **layout_kv**: PA_ND（仅支持 PageAttention）
- **template_run_mode**: SWA, CFA, SCFA
- **kv_quant_mode**: 1 (FP8_E4M3FN), 10 (HIFLOAT8)

---

## 2. 参数变动分析

### 2.1 已删除参数
| 参数名 | 原用途 | 删除原因 |
|--------|--------|----------|
| `seqused_kv` | 表示 ori_kv 的存储长度 | 已由 `seqused_ori_kv` 替代，语义更清晰 |

**影响范围**：
- 测试用例参数配置（paramset.py）
- 测试数据生成逻辑（golden.py）
- NPU 调用逻辑（process.py）
- 结果保存逻辑（utils.py）

### 2.2 新增参数

#### 2.2.1 真实使用长度参数
| 参数名 | 类型 | 维度 | 含义 | 默认值 |
|--------|------|------|------|--------|
| `seqused_ori_kv` | int32 | [B] | ori_kv 每个batch的真实使用长度 | 使用 `seqused_kv` 旧逻辑 |
| `seqused_cmp_kv` | int32 | [B] | cmp_kv 每个batch的真实使用长度 | floor(seqused_ori_kv / cmp_ratio) |

**语义说明**：
```
存储长度（Storage Length）：
- tensor 实际分配的内存大小
- 当前由 block_table + block_size 决定

真实使用长度（Actual Usage Length）：
- 参与 attention 计算的有效 token 数
- 用于 mask 计算、循环边界判断

典型场景：
- Prefill阶段：存储长度 = 真实使用长度
- Decode阶段：存储长度 > 真实使用长度（KV cache 累积）
```

#### 2.2.2 余数参数
| 参数名 | 类型 | 维度 | 含义 | 是否可选 |
|--------|------|------|------|----------|
| `cmp_residual_kv` | int32 | [B] | cmp_mask_mode=3且cmp_ratio≠1时，每个batch的余数 | 强制（特定场景） |

**触发条件**：`cmp_mask_mode=3` 且 `cmp_ratio≠1`

**用途**：解决 mask 计算中的余数问题，防止越界访问

**公式关系**：
```
seqused_ori_kv[i] = seqused_cmp_kv[i] * cmp_ratio + cmp_residual_kv[i]
cmp_residual_kv[i] = seqused_ori_kv[i] % cmp_ratio
```

#### 2.2.3 TND layout 参数
| 参数名 | 类型 | 维度 | 含义 | 是否可选 |
|--------|------|------|------|----------|
| `cu_seqlens_ori_kv` | int32 | [B+1] | ori_kv 的累加和（TND layout） | 可选（仅TND场景） |
| `cu_seqlens_cmp_kv` | int32 | [B+1] | cmp_kv 的累加和（TND layout） | 可选（仅TND场景） |

**语义说明**：
```
cu_seqlens_ori_kv[i] = 前 i 个 batch 中 ori_kv 有效 token 数的累加和
cu_seqlens_ori_kv[0] = 0
cu_seqlens_ori_kv[B] = 所有 batch 中 ori_kv 有效 token 总数

示例：
B=2, seqused_ori_kv=[5, 10]
cu_seqlens_ori_kv = [0, 5, 15]
```

---

## 3. TND Layout 支持分析

### 3.1 TND Layout 定义
**TND (Token-Num-Dim)**：将所有 batch 的 token 拼接成一个连续张量

```
BSND layout: [B, S, N, D]  → 4维，batch维度显式
TND layout: [T, N, D]      → 3维，batch维度隐式，T = Σ(S_i)
```

**适用场景**：
- Prefill 阶段（变长序列）
- 需要紧凑存储，避免 padding

### 3.2 算子当前支持的 layout 组合

| layout_q | layout_kv | cu_seqlens_q | cu_seqlens_ori_kv | cu_seqlens_cmp_kv | seqused_ori_kv | seqused_cmp_kv |
|----------|-----------|--------------|-------------------|-------------------|----------------|----------------|
| BSND | PA_ND | None | None | None | 可选 | 可选 |
| TND | PA_ND | 必传 | None | None | **必传** | 可选 |
| TND | TND | 必传 | 必传 | 必传 | 可选 | 可选 |

**关键约束**：
- `layout_q="TND"` 时，`cu_seqlens_q` **必须传入**
- `layout_kv="PA_ND"` 时，`cu_seqlens_ori_kv/cu_seqlens_cmp_kv` 在 CheckUnrequiredParaExistence() 强制为 nullptr
- `layout_kv="TND"` 时，`cu_seqlens_ori_kv/cu_seqlens_cmp_kv` **必须传入**
- `seqused_ori_kv` 在 `PA_ND` layout 下可选，在 TND layout 下可选（需确认）

### 3.3 当前测试工程的 layout 支持情况

**现状**：
- 只测试 `layout_q="TND" + layout_kv="PA_ND"` 组合
- 未测试 `layout_kv="TND"` 场景
- 未传入 `cu_seqlens_ori_kv/cu_seqlens_cmp_kv`

**问题**：
- metadata 调用时传入空 tensor，未真正测试 TND KV 功能
- 无法验证累加和参数的正确性

---

## 4. 改动点分析

### 4.1 参数配置层（paramset.py）

#### 改动点1：删除废弃参数
```python
# 删除
"seqused_kv": [None],

# 新增
"seqused_ori_kv": [None],
"seqused_cmp_kv": [None],
"cmp_residual_kv": [None],
"cu_seqlens_ori_kv": [None],
"cu_seqlens_cmp_kv": [None],
```

#### 改动点2：新增参数组合
需要新增测试场景配置：
- `layout_kv="TND"` 场景
- `cmp_residual_kv` 强制传入场景（cmp_mask_mode=3, cmp_ratio≠1）
- `seqused_ori_kv ≠ seqused_kv` 场景（真实使用长度 ≠ 存储长度）

### 4.2 测试入口层（test_kv_quant_sparse_attn_sharedkv_single.py）

#### 改动点3：参数列表更新
```python
# 当前（第43-48行）
param_names = [
    ..., "cu_seqlens_q", "seqused_kv", ...  # 需删除 seqused_kv
]

# 修改为
param_names = [
    ..., "cu_seqlens_q", "seqused_ori_kv", "seqused_cmp_kv", 
    "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "cmp_residual_kv", ...
]
```

#### 改动点4：参数值列表更新
```python
# 当前（第49-81行）
param_values = [
    ..., locals()["param_cu_seqlens_q"], locals()["param_seqused_kv"], ...
]

# 修改为
param_values = [
    ..., locals()["param_cu_seqlens_q"], locals()["param_seqused_ori_kv"],
    locals()["param_seqused_cmp_kv"], locals()["param_cu_seqlens_ori_kv"],
    locals()["param_cu_seqlens_cmp_kv"], locals()["param_cmp_residual_kv"], ...
]
```

#### 改动点5：参数生成逻辑（sas函数）
```python
# 当前（第109-162行）
cu_seqlens_q = param_combinations['cu_seqlens_q']
seqused_kv = param_combinations['seqused_kv']

# 需修改为
seqused_ori_kv = param_combinations['seqused_ori_kv']
seqused_cmp_kv = param_combinations['seqused_cmp_kv']
cu_seqlens_ori_kv = param_combinations['cu_seqlens_ori_kv']
cu_seqlens_cmp_kv = param_combinations['cu_seqlens_cmp_kv']
cmp_residual_kv = param_combinations['cmp_residual_kv']
```

#### 改动点6：actLen生成逻辑适配
```python
# 当前逻辑（第135-162行）生成 seqused_kv
# 需修改为生成 seqused_ori_kv（兼容旧逻辑）
if seqused_ori_kv == None:
    if layout_q == "TND":
        # 生成 ori_kv 的真实使用长度
        ...
    seqused_ori_kv = KVS

# 新增 seqused_cmp_kv 生成逻辑
if seqused_cmp_kv == None:
    seqused_cmp_kv = [math.floor(s / cmp_ratio) for s in seqused_ori_kv]

# 新增 cmp_residual_kv 生成逻辑
if cmp_mask_mode == 3 and cmp_ratio != 1:
    if cmp_residual_kv == None:
        cmp_residual_kv = [s % cmp_ratio for s in seqused_ori_kv]
```

### 4.3 Golden计算层（golden.py）

#### 改动点7：GeneralizedSFAQuant 类初始化
```python
# 当前（第96-127行）
def __init__(self, layout_q, layout_kv, ..., seqused_kv, ...):
    self.seqused_kv = seqused_kv

# 修改为
def __init__(self, layout_q, layout_kv, ..., seqused_ori_kv, seqused_cmp_kv, 
             cu_seqlens_ori_kv, cu_seqlens_cmp_kv, cmp_residual_kv, ...):
    self.seqused_ori_kv = seqused_ori_kv
    self.seqused_cmp_kv = seqused_cmp_kv
    self.cu_seqlens_ori_kv = cu_seqlens_ori_kv
    self.cu_seqlens_cmp_kv = cu_seqlens_cmp_kv
    self.cmp_residual_kv = cmp_residual_kv
```

#### 改动点8：calculate_by_bnsd 函数
```python
# 当前（第129-235行）使用 seqused_kv
# 需修改为使用 seqused_ori_kv 和 seqused_cmp_kv
for i_B in range(B):
    cur_ori_act_kv = seqused_ori_kv[i_B]  # 修改
    cur_cmp_act_kv = seqused_cmp_kv[i_B]  # 新增
    
    # cmp mask 计算需考虑余数
    if self.cmp_mask_mode == 3 and self.cmp_ratio != 1:
        residual = self.cmp_residual_kv[i_B]
        threshold = self.calculate_threshold_with_residual(cur_ori_act_kv, cur_act_q, i_S1, residual)
```

#### 改动点9：forward 函数参数传递
```python
# 当前（第339-354行）
def forward(self, q, ori_k_bnsd, cmp_k_bnsd, ..., seqused_kv, sinks):

# 修改为
def forward(self, q, ori_k_bnsd, cmp_k_bnsd, ..., seqused_ori_kv, seqused_cmp_kv, 
            cu_seqlens_ori_kv, cu_seqlens_cmp_kv, cmp_residual_kv, sinks):
```

#### 改动点10：generate_and_save_testdata 函数
```python
# 当前（第881-1044行）
def generate_and_save_testdata(params, ...):
    ..., cu_seqlens_q, seqused_kv, ... = params
    
    # 需修改参数解包
    ..., cu_seqlens_q, seqused_ori_kv, seqused_cmp_kv, 
    cu_seqlens_ori_kv, cu_seqlens_cmp_kv, cmp_residual_kv, ... = params
    
    # 需修改 metadata_input 结构
    metadata_input = {
        'seqused_ori_kv': seqused_ori_kv,    # 新增
        'seqused_cmp_kv': seqused_cmp_kv,    # 新增
        'cu_seqlens_ori_kv': cu_seqlens_ori_kv,  # 新增
        'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,  # 新增
        'cmp_residual_kv': cmp_residual_kv,  # 新增
        ...
    }
    
    # 需修改 input 结构
    input = {
        'seqused_ori_kv': seqused_ori_kv,    # 新增（替代 seqused_kv）
        'seqused_cmp_kv': seqused_cmp_kv,    # 新增
        'cu_seqlens_ori_kv': cu_seqlens_ori_kv,  # 新增
        'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,  # 新增
        'cmp_residual_kv': cmp_residual_kv,  # 新增
        ...
    }
```

#### 改动点11：gen_ori_kv/gen_cmp_kv 函数
```python
# 当前使用 seqused_kv 计算最大长度
# 需修改为使用 seqused_ori_kv/seqused_cmp_kv
ori_max_s2 = max(seqused_ori_kv)  # 修改
cmp_max_s2 = max(seqused_cmp_kv)  # 修改
```

### 4.4 NPU调用层（batch/kv_quant_sparse_attn_sharedkv_process.py）

#### 改动点12：metadata 调用更新
```python
# 当前（第164-187行）
metadata = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata(
    ...
    cu_seqlens_ori_kv = torch.tensor([]).npu(),  # 未真正传入
    cu_seqlens_cmp_kv = torch.tensor([]).npu(),  # 未真正传入
    seqused_kv = metadata_input['seqused_kv'].npu() if input['seqused_kv'] is not None else torch.tensor([]).npu(),
    ...
)

# 修改为
metadata = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata(
    ...
    cu_seqlens_ori_kv = metadata_input['cu_seqlens_ori_kv'].npu() if input['cu_seqlens_ori_kv'] is not None else torch.tensor([]).npu(),
    cu_seqlens_cmp_kv = metadata_input['cu_seqlens_cmp_kv'].npu() if input['cu_seqlens_cmp_kv'] is not None else torch.tensor([]).npu(),
    seqused_ori_kv = metadata_input['seqused_ori_kv'].npu() if input['seqused_ori_kv'] is not None else torch.tensor([]).npu(),
    seqused_cmp_kv = metadata_input['seqused_cmp_kv'].npu() if input['seqused_cmp_kv'] is not None else torch.tensor([]).npu(),
    cmp_residual_kv = metadata_input['cmp_residual_kv'].npu() if input['cmp_residual_kv'] is not None else torch.tensor([]).npu(),
    ...
)
```

#### 改动点13：算子调用更新
```python
# 当前（第192-213行）
npu_result, _ = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(
    ...
    seqused_kv=input['seqused_kv'].npu() if input['seqused_kv'] is not None else None,
    ...
)

# 修改为
npu_result, _ = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(
    ...
    seqused_ori_kv=input['seqused_ori_kv'].npu() if input['seqused_ori_kv'] is not None else None,
    seqused_cmp_kv=input['seqused_cmp_kv'].npu() if input['seqused_cmp_kv'] is not None else None,
    cu_seqlens_ori_kv=input['cu_seqlens_ori_kv'].npu() if input['cu_seqlens_ori_kv'] is not None else None,
    cu_seqlens_cmp_kv=input['cu_seqlens_cmp_kv'].npu() if input['cu_seqlens_cmp_kv'] is not None else None,
    cmp_residual_kv=input['cmp_residual_kv'].npu() if input['cmp_residual_kv'] is not None else None,
    ...
)
```

### 4.5 工具函数层（utils.py）

#### 改动点14：save_result 函数更新
```python
# 当前（第70-128行）
row_data = {
    ...
    "seqused_kv": params[18],  # 需删除
    ...
}

# 修改为
row_data = {
    ...
    "seqused_ori_kv": params[18],       # 新增
    "seqused_cmp_kv": params[19],       # 新增
    "cu_seqlens_ori_kv": params[20],    # 新增
    "cu_seqlens_cmp_kv": params[21],    # 新增
    "cmp_residual_kv": params[22],      # 新增
    ...
}
```

#### 改动点15：load_excel_test_cases 函数更新
```python
# 当前（第45-49行）
required_columns = [
    ..., "cu_seqlens_q", "seqused_kv", ...  # 需删除 seqused_kv
]

# 修改为
required_columns = [
    ..., "cu_seqlens_q", "seqused_ori_kv", "seqused_cmp_kv",
    "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "cmp_residual_kv", ...
]
```

---

## 5. 修改方案步骤拆解

### 步骤1：删除废弃参数（基础清理）
**目标**：删除所有 `seqused_kv` 相关代码

**修改文件**：
1. `kv_quant_sparse_attn_sharedkv_paramset.py`
   - 删除 `"seqused_kv": [None]` 配置项
   
2. `test_kv_quant_sparse_attn_sharedkv_single.py`
   - 从 `param_names` 列表删除 `"seqused_kv"`
   - 从 `param_values` 列表删除对应项
   - 删除 `sas()` 函数中的 `seqused_kv = param_combinations['seqused_kv']`

3. `utils.py`
   - 从 `save_result()` 的 `row_data` 删除 `"seqused_kv": params[18]`
   - 从 `load_excel_test_cases()` 的 `required_columns` 删除 `"seqused_kv"`

**验证方法**：
- 运行现有测试用例，确保删除后不报错（兼容性检查）

---

### 步骤2：新增参数定义（参数扩展）
**目标**：在参数配置层新增5个参数

**修改文件**：
1. `kv_quant_sparse_attn_sharedkv_paramset.py`
   ```python
   TEST_PARAMS = {
       "decode_first": {
           ...
           "seqused_ori_kv": [None],       # 新增
           "seqused_cmp_kv": [None],       # 新增
           "cu_seqlens_ori_kv": [None],    # 新增
           "cu_seqlens_cmp_kv": [None],    # 新增
           "cmp_residual_kv": [None],      # 新增
           ...
       },
       ...
   }
   ```

2. `test_kv_quant_sparse_attn_sharedkv_single.py`
   - 在 `param_names` 列表新增5个参数名（在 `cu_seqlens_q` 之后）
   - 在 `param_values` 列表新增5个参数值
   - 在 `sas()` 函数新增参数获取逻辑

**验证方法**：
- 运行测试，检查参数是否正确传递

---

### 步骤3：Golden计算逻辑适配（CPU端）
**目标**：修改 CPU golden 计算逻辑，使用新参数

**修改文件**：
1. `kv_quant_sparse_attn_sharedkv_golden.py`
   - `GeneralizedSFAQuant.__init__()`：新增5个参数接收
   - `calculate_by_bnsd()`：使用 `seqused_ori_kv` 和 `seqused_cmp_kv`
   - `forward()`：新增参数传递
   - `generate_and_save_testdata()`：参数解包更新、metadata_input/input 结构更新
   - `gen_ori_kv()`/`gen_cmp_kv()`：使用新参数计算最大长度

**关键改动**：
```python
# generate_and_save_testdata() 参数解包
Testcase_Name, layout_q, layout_kv, ..., B, S1, T1, ..., cu_seqlens_q, 
seqused_ori_kv, seqused_cmp_kv, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, 
cmp_residual_kv, softmax_scale, ... = params

# metadata_input 结构
metadata_input = {
    ...
    'seqused_ori_kv': seqused_ori_kv,
    'seqused_cmp_kv': seqused_cmp_kv,
    'cu_seqlens_ori_kv': cu_seqlens_ori_kv,
    'cu_seqlens_cmp_kv': cu_seqlens_cmp_kv,
    'cmp_residual_kv': cmp_residual_kv,
    ...
}
```

**验证方法**：
- 对比 CPU golden 计算结果与旧逻辑是否一致（兼容性）
- 验证新参数场景的正确性

---

### 步骤4：NPU调用逻辑适配（Kernel端）
**目标**：修改 NPU 调用逻辑，传入新参数

**修改文件**：
1. `batch/kv_quant_sparse_attn_sharedkv_process.py`
   - `test_sas_quant_process_ci()`：
     - metadata 调用新增5个参数
     - 算子调用新增5个参数
   
   - `test_sas_quant_process_graph()`：
     - 同样修改 metadata 和算子调用

**关键改动**：
```python
# metadata 调用
cu_seqlens_ori_kv = metadata_input['cu_seqlens_ori_kv'].npu() if input['cu_seqlens_ori_kv'] is not None else torch.tensor([]).npu(),
cu_seqlens_cmp_kv = metadata_input['cu_seqlens_cmp_kv'].npu() if input['cu_seqlens_cmp_kv'] is not None else torch.tensor([]).npu(),
seqused_ori_kv = metadata_input['seqused_ori_kv'].npu() if input['seqused_ori_kv'] is not None else torch.tensor([]).npu(),
seqused_cmp_kv = metadata_input['seqused_cmp_kv'].npu() if input['seqused_cmp_kv'] is not None else torch.tensor([]).npu(),
cmp_residual_kv = metadata_input['cmp_residual_kv'].npu() if input['cmp_residual_kv'] is not None else torch.tensor([]).npu(),

# 算子调用
seqused_ori_kv=input['seqused_ori_kv'].npu() if input['seqused_ori_kv'] is not None else None,
seqused_cmp_kv=input['seqused_cmp_kv'].npu() if input['seqused_cmp_kv'] is not None else None,
cu_seqlens_ori_kv=input['cu_seqlens_ori_kv'].npu() if input['cu_seqlens_ori_kv'] is not None else None,
cu_seqlens_cmp_kv=input['cu_seqlens_cmp_kv'].npu() if input['cu_seqlens_cmp_kv'] is not None else None,
cmp_residual_kv=input['cmp_residual_kv'].npu() if input['cmp_residual_kv'] is not None else None,
```

**验证方法**：
- 运行测试，检查 NPU 计算是否正确
- 对比 CPU 和 NPU 结果

---

### 步骤5：参数生成逻辑适配（自动生成）
**目标**：在测试用例中自动生成新参数（当用户未提供时）

**修改文件**：
1. `test_kv_quant_sparse_attn_sharedkv_single.py` 的 `sas()` 函数
   ```python
   # 当 seqused_ori_kv == None 时，生成默认值（兼容旧逻辑）
   if seqused_ori_kv == None:
       if layout_q == "TND":
           # 根据 cu_seqlens_q 和 actlen_mode 生成
           ...
       seqused_ori_kv = KVS
   
   # 当 seqused_cmp_kv == None 时，根据 cmp_ratio 计算
   if seqused_cmp_kv == None:
       seqused_cmp_kv = [math.floor(s / cmp_ratio) for s in seqused_ori_kv]
   
   # 当 cmp_residual_kv == None 时，根据公式计算
   if cmp_mask_mode == 3 and cmp_ratio != 1:
       if cmp_residual_kv == None:
           cmp_residual_kv = [s % cmp_ratio for s in seqused_ori_kv]
   
   # 当 cu_seqlens_ori_kv == None 且 layout_kv == "TND" 时，生成累加和
   if cu_seqlens_ori_kv == None and layout_kv == "TND":
       cu_seqlens_ori_kv = [0]
       for s in seqused_ori_kv:
           cu_seqlens_ori_kv.append(cu_seqlens_ori_kv[-1] + s)
   
   # 同样处理 cu_seqlens_cmp_kv
   ```

**验证方法**：
- 测试 None 参数场景，验证默认值生成正确

---

### 步骤6：新增测试场景配置（功能扩展）
**目标**：新增测试场景，覆盖新参数功能

**新增场景**：
1. **场景1：强制余数测试**
   ```python
   "cmp_residual_force": {
       "cmp_mask_mode": [3],
       "cmp_ratio": [4],  # ≠1
       "cmp_residual_kv": [None],  # 自动生成，验证强制传入逻辑
       ...
   }
   ```

2. **场景2：真实使用长度测试**
   ```python
   "actual_seq_diff": {
       "seqused_ori_kv": [[5, 10]],  # ≠ 存储长度
       "seqused_cmp_kv": [[1, 2]],   # 手动指定
       ...
   }
   ```

3. **场景3：TND KV layout测试**
   ```python
   "tnd_kv_layout": {
       "layout_q": ["TND"],
       "layout_kv": ["TND"],  # 新增
       "cu_seqlens_ori_kv": [None],  # 自动生成累加和
       "cu_seqlens_cmp_kv": [None],
       ...
   }
   ```

**验证方法**：
- 运行新增测试场景，检查正确性

---

### 步骤7：工具函数更新（结果保存）
**目标**：更新结果保存和加载逻辑

**修改文件**：
1. `utils.py`
   - `save_result()`：更新 `row_data` 字段名
   - `load_excel_test_cases()`：更新 `required_columns`

**注意**：
- 需要调整参数索引（params 列表位置）
- 确保字段名与 Excel 列名一致

---

### 步骤8：全面测试与验证
**目标**：验证所有修改的正确性

**测试步骤**：
1. **兼容性测试**：
   - 运行旧测试用例（参数为 None），验证默认值逻辑正确
   
2. **功能测试**：
   - 运行新增场景，验证新参数功能正确
   
3. **精度测试**：
   - 对比 CPU golden 和 NPU 结果，验证精度达标

4. **边界测试**：
   - 测试 cmp_residual_kv 强制传入场景
   - 测试 TND layout 累加和正确性

---

## 6. 改动文件清单

| 序号 | 文件路径 | 改动类型 | 改动内容 |
|------|----------|----------|----------|
| 1 | `kv_quant_sparse_attn_sharedkv_paramset.py` | 删除+新增 | 删除 seqused_kv，新增5个参数 |
| 2 | `test_kv_quant_sparse_attn_sharedkv_single.py` | 删除+新增+修改 | 参数列表更新、参数生成逻辑 |
| 3 | `kv_quant_sparse_attn_sharedkv_golden.py` | 修改 | 类初始化、计算逻辑、数据生成 |
| 4 | `batch/kv_quant_sparse_attn_sharedkv_process.py` | 修改 | metadata调用、算子调用 |
| 5 | `utils.py` | 删除+新增 | 结果保存、Excel加载 |
| 6 | `check_valid_param.py` | 无改动 | 参数校验逻辑已在算子侧实现 |

---

## 7. 风险点与注意事项

### 7.1 向后兼容性
**风险**：删除 seqused_kv 后，旧测试用例可能报错

**缓解措施**：
- 保留参数生成逻辑（当 seqused_ori_kv=None 时自动生成）
- 充分测试兼容性场景

### 7.2 参数索引调整
**风险**：params 列表索引调整可能导致保存/加载错误

**缓解措施**：
- 同步更新所有 params 索引引用
- 使用明确的变量名而非索引访问（如 params[18]）

### 7.3 TND layout 累加和正确性
**风险**：cu_seqlens_ori_kv/cu_seqlens_cmp_kv 计算错误

**缓解措施**：
- 明确公式：cu_seqlens[i+1] = cu_seqlens[i] + seqused[i]
- 验证累加和单调递增

### 7.4 cmp_residual_kv 强制场景
**风险**：cmp_mask_mode=3 且 cmp_ratio≠1 时未传入 cmp_residual_kv，算子报错

**缓解措施**：
- 在测试用例生成逻辑中自动计算并传入
- 添加明确的错误提示

---

## 8. 实施时间估算

| 步骤 | 预估时间 | 说明 |
|------|----------|------|
| 步骤1：删除废弃参数 | 0.5天 | 基础清理，快速完成 |
| 步骤2：新增参数定义 | 0.5天 | 参数配置扩展 |
| 步骤3：Golden计算适配 | 1天 | CPU逻辑修改，需仔细调试 |
| 步骤4：NPU调用适配 | 1天 | Kernel调用修改 |
| 步骤5：参数生成逻辑 | 0.5天 | 自动生成逻辑 |
| 步骤6：新增测试场景 | 0.5天 | 功能扩展 |
| 步骤7：工具函数更新 | 0.5天 | 结果保存/加载 |
| 步骤8：全面测试验证 | 1天 | 测试与验证 |
| **总计** | **5天** | 完整修改周期 |

---

## 9. 附录

### 9.1 参数索引对照表（修改前后）

**修改前**：
```
params[17]: cu_seqlens_q
params[18]: seqused_kv（删除）
params[19]: softmax_scale
...
```

**修改后**：
```
params[17]: cu_seqlens_q
params[18]: seqused_ori_kv（新增）
params[19]: seqused_cmp_kv（新增）
params[20]: cu_seqlens_ori_kv（新增）
params[21]: cu_seqlens_cmp_kv（新增）
params[22]: cmp_residual_kv（新增）
params[23]: softmax_scale（原索引19）
...
```

### 9.2 TND layout 累加和生成示例代码
```python
def generate_cu_seqlens(seqused):
    """
    从 seqused 生成 cu_seqlens
    seqused: [B] 每个batch的有效长度
    cu_seqlens: [B+1] 累加和，cu_seqlens[0]=0
    """
    cu_seqlens = [0]
    for s in seqused:
        cu_seqlens.append(cu_seqlens[-1] + s)
    return cu_seqlens

# 示例
seqused_ori_kv = [5, 10, 8]
cu_seqlens_ori_kv = generate_cu_seqlens(seqused_ori_kv)
# 结果：[0, 5, 15, 23]
```

### 9.3 cmp_residual_kv 计算示例代码
```python
def calculate_cmp_residual(seqused_ori_kv, cmp_ratio):
    """
    计算 cmp_residual_kv
    公式：residual = seqused_ori_kv % cmp_ratio
    """
    return [s % cmp_ratio for s in seqused_ori_kv]

# 示例
seqused_ori_kv = [10, 8, 12]
cmp_ratio = 4
cmp_residual_kv = calculate_cmp_residual(seqused_ori_kv, cmp_ratio)
# 结果：[2, 0, 0]  (10%4=2, 8%4=0, 12%4=0)
```

---

**文档结束**