# SFA (Sparse Flash Attention) 快速学习指南

## 一、代码架构概览

```
pytest/
├── sparse_flash_attention_golden.py   # CPU golden 实现（核心）
├── generate_tensor_data.py            # Tensor 数据生成器
├── utils.py                           # 参数解析/执行入口
├── sparse_flash_attention_paramset.py # 参数配置模板
├── test_run.sh                        # 执行脚本（三种模式）
├── test_sparse_flash_attention_single.py  # 单用例测试
└── batch/
    └── sparse_flash_attention_process.py  # NPU 接口调用
```

## 二、核心数据流

```
Excel/Paramset → utils.convert_param_combination_to_cs_format() 
    → sparse_flash_attention_golden.generate_input_tensors() 
    → CPU Golden compute_cpu() 
    → NPU sparse_flash_attention_process.call_npu() 
    → result_compare_method.check_result()
```

## 三、关键概念与数据转换

### 1. Layout 格式（内存布局）

| Layout | Shape含义 | 适用场景 |
|--------|----------|---------|
| `BSND` | [Batch, Seq, NumHeads, Dim] | 标准推理 |
| `TND` | [TotalTokens, NumHeads, Dim] | 变长序列压缩 |
| `PA_BSND` | PageAttention格式 | KV Cache分页管理 |

### 2. 核心参数映射 (`utils.py:188-345`)

```python
# 参数转换关键逻辑
params = {
    "shape_input": {
        'query': [B, S1, N1, D],           # Q张量形状
        'key': [B, S2, N2, D],             # K张量形状
        'sparse_indices': [B, S1, N2, K],  # 稀疏索引（K=sparse_blockcount）
        'query_rope': [B, S1, N1, rope_head_dim],  # Q的RoPE部分
        'key_rope': [B, S2, N2, rope_head_dim],    # K的RoPE部分
    },
    "actual_seq_q": [S1_1, S1_2, ...],     # 每batch实际Q长度
    "actual_seq_kv": [S2_1, S2_2, ...],    # 每batch实际KV长度
}
```

### 3. Layout转换核心函数 (`golden.py:89-179`)

```python
# BSH/BSND → BNSD（统一计算格式）
_n_trans_shape_to_bnsd(tensor, shape, layout, headnums, act_seq)

# BNSD → 目标Layout输出
trans_bnsd_to_layout(tensor, shape, layout, act_q)
```

**TND格式特殊处理**：
- 输入：`[T, N, D]` (T=∑actual_seq_q)
- 转换：按 `actual_seq_q` 展开 → `[B, N, S, D]`

## 四、Golden实现核心逻辑 (`golden.py:535-583`)

```python
def _t_increattention_bnsd(fa_param):
    # 稀疏Flash Attention核心计算
    for batch in range(batch_size):
        for n2Idx in range(numKeyValueHeads):  # KV头
            for s1Idx in range(curr_actualSeq_q):  # Q序列位置
                # 1. 根据sparse_indices收集稀疏KV块
                emptyFlag, k_sparse, v_sparse = gatherKV(...)
                
                # 2. 注意力计算
                bmm1Res = Q @ K.T           # [g, len_sparse]
                scaleRes = bmm1Res * scaleValue
                softmax_res = softmax(scaleRes)
                bmm2Res = softmax @ V       # [g, D]
                
                # 3. 写入结果
                y[batch, n2Idx*g:(n2Idx+1)*g, s1Idx, :] = bmm2Res
```

**GQA（Grouped Query Attention）**：
- `g = N1 // N2`（Query头数 / KV头数）
- 每个KV头对应 `g` 个Query头

## 五、稀疏索引机制 (`golden.py:390-421`)

```python
# sparse_indices: [B, N2, S1, K]
# 每个位置记录K个有效KV块索引
sparse_indices[b, n, s, :] = [block_idx_0, block_idx_1, ..., -1]

# sparse_mode:
#   0: 全量注意力（threshold = act_seqlen_kv）
#   3: causal注意力（threshold = act_seqlen_kv - act_seqlen_q + s + 1）
```

## 六、PA (PageAttention) 机制

### 1. Key_Cache形状定义 (`utils.py:222-238`)

```python
# 第222-224行：计算 block_num_sum（所有batch的物理块总数）
if layout_kv == "PA_BSND":
    block_num_sum = 0
    for length in actual_seq_kv:
        block_num_sum = block_num_sum + math.ceil(length / block_size)

# 第238-239行：PA_BSND模式下定义key_cache/value_cache形状
shape_input = {
    'key_cache': [block_num_sum, block_size, N2, D + rope_head_dim],
    'value_cache': [block_num_sum, block_size, N2, D + rope_head_dim],
}
```

**含义解析**：
- `block_num_sum` = 所有batch的KV序列需要的物理块总数
- `[block_num_sum, block_size, N2, D+rope_head_dim]` = PageAttention的KV cache物理存储格式
- `D + rope_head_dim` = base维度 + RoPE维度（拼接后的完整KV）

### 2. CPU与NPU的数据使用差异

| 使用方 | 使用的Tensor | 格式 |
|--------|-------------|------|
| **CPU Golden** | `k_bnsd_tensor`, `v_bnsd_tensor` | `[B, N2, S2, D]` (BNSD格式) |
| **NPU调用** | `key_cache`, `value_cache` | `[block_num_sum, block_size, N2, D+rope_head_dim]` (PA格式) |

**证据**：

1. **CPU计算** (`golden.py:550-552`)：
```python
k_bnsd_tensor = fa_param['k_bnsd_tensor'].float()  # 使用BNSD格式
v_bnsd_tensor = k_bnsd_tensor[..., :512].clone()
```

2. **NPU调用** (`process.py:39`)：
```python
key=torch_tensor_dict.get("key_cache"),  # 使用PA格式
```

3. **数据生成** (`golden.py:632`)：`key_cache` 仅在 `kv_pa_preprocessing()` 中生成并放入 `input_tensor_dict`，供NPU使用。

### 3. BlockTable与Cache转换 (`golden.py:423-455, 586-641`)

```python
# block_table: [B, block_num] → 映射batch到物理块
# key_cache: [block_num_sum, block_size, N2, D+rope_head_dim]
# 通过block_table索引获取KV cache
```

## 七、执行流程

```bash
# 单用例调试
bash test_run.sh single -P sparse_flash_attention_paramset

# 批量生成pt文件
bash test_run.sh batch_save -E excel/example.xlsx -S Sheet1

# 批量执行NPU测试
bash test_run.sh batch_exec -P ./pt_files/
```

## 八、快速上手步骤

1. **理解参数**：阅读 `sparse_flash_attention_paramset.py` 的5个典型配置
2. **追踪数据流**：从 `utils.convert_param_combination_to_cs_format()` 开始
3. **理解Golden**：重点看 `_t_increattention_bnsd()` 的循环结构
4. **调试单用例**：修改 paramset 后执行 `bash test_run.sh single`

## 九、关键修改点总结

| Commit | 核心修改 |
|--------|---------|
| `32946ef9` | 重构Golden实现，新增 `generate_tensor_data.py`，完善参数转换 |
| `fadd67ca` | 简化Golden移除冗余，添加README和Excel示例 |
| `5ed2ae77` | 完善 `test_run.sh` 三模式支持，增强文档 |

---

**建议学习路径**：paramset → utils.convert → golden.generate_input_tensors → golden._t_increattention_bnsd → process.call_npu