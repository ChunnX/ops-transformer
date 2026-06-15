# FIA FullQuant MXFP8 测试用例执行指南

## 1. 环境准备

### 1.1 安装 CANN / PyTorch + torch-npu

参考根目录 README.md，或使用项目根目录下的安装脚本：

```bash
bash install_cann_950.sh    # 安装 CANN 9.1.0 + 950 ops
bash install_torch_npu.sh   # 安装 PyTorch + torch-npu
```

### 1.2 每次执行前加载环境

```bash
source /home/user/Ascend/cann/set_env.sh
conda activate your-env-name
pip install pytest
cd /home/user/code/ops-transformer/attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test
```

---

## 2. 执行测试

### 2.1 运行全部用例

```bash
pytest test_fia_fullquant_mxfp8.py -v
```

### 2.2 运行指定用例（-k 关键字过滤）

```bash
# 精确指定单个用例
pytest test_fia_fullquant_mxfp8.py -v -k "PA_NZ_QS1_KVS512_Nq1_Nkv1_D128_SP3_Decode"

# 按 Prefill / Decode 模式
pytest test_fia_fullquant_mxfp8.py -v -k "Prefill"
pytest test_fia_fullquant_mxfp8.py -v -k "Decode"

# 按 layout 类型
pytest test_fia_fullquant_mxfp8.py -v -k "PA_NZ"
pytest test_fia_fullquant_mxfp8.py -v -k "PA_BNBD"

# 组合过滤
pytest test_fia_fullquant_mxfp8.py -v -k "PA_NZ and Prefill"

# 排除某些用例
pytest test_fia_fullquant_mxfp8.py -v -k "not Decode"
```

### 2.3 常用选项

| 选项 | 作用 |
|------|------|
| `-v` | 详细输出，显示每个用例名 |
| `-s` | 不截断 stdout/stderr |
| `-x` | 遇到失败立即停止 |
| `--tb=long` | 失败时显示完整堆栈 |
| `-m ci` | 只运行 CI 标记的用例 |

---

## 3. 用例命名规则

格式：`{LAYOUT}_{QS长度}_{KVS长度}_Nq{Q头数}_Nkv{KV头数}_D{维度}_SP{稀疏模式}[_{模式}]`

| 字段 | 含义 | 示例 |
|------|------|------|
| `PA_NZ` | kv_cache_layout = PA_NZ | `PA_NZ_QS64_KVS64_...` |
| `PA_BNBD` | kv_cache_layout = BnNBsD | `PA_BNBD_QS48_KVS64_...` |
| `QS` | Q 序列长度 (actual_seq_q) | `QS128` = Q 长 128 |
| `KVS` | KV 序列长度 (actual_seq_kv) | `KVS512` = KV 长 512 |
| `Nq` | Q 头数 | `Nq80` |
| `Nkv` | KV 头数 | `Nkv8` |
| `D` | Head 维度 | `D128` |
| `SP3` | sparse_mode = 3 | |
| `Prefill` | q_scale_layout = TND | 长序列 prefill 阶段 |
| `Decode` | q_scale_layout = N2TGD | 短序列 decode 阶段 |
| (无后缀) | q_scale_layout = AUTO | 自动选择 |

---

## 4. 新增用例

在 `fia_fullquant_mxfp8_paramset.py` 的 `TEST_PARAMS` 字典中添加新条目：

```python
"PA_NZ_QS256_KVS512_Nq4_Nkv1_D128_SP3_Prefill": {
    "B": [1],
    "N_q": [4],
    "N_kv": [1],
    "D": [128],
    "actual_seq_q": [[256]],
    "actual_seq_kv": [[512]],
    "enable_pa": [True],
    "kv_cache_layout": ["PA_NZ"],
    "block_size": [512],
    "sparse_mode": [3],
    "q_scale_layout": ["TND"],
    "p_scale": [1.0],
},
```

### 参数说明

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `B` | 正整数 | batch size |
| `N_q` | 正整数 | Q 头数 |
| `N_kv` | 正整数 | KV 头数（N_q 需能被 N_kv 整除） |
| `D` | 128 | Head 维度 |
| `actual_seq_q` | 列表的列表 | Q 序列长度 |
| `actual_seq_kv` | 列表的列表 | KV 序列长度 |
| `enable_pa` | True / False | 是否启用 PageAttention |
| `kv_cache_layout` | `"PA_NZ"` / `"BnNBsD"` | KV cache 内存布局 |
| `block_size` | 512 | PageAttention block 大小 |
| `sparse_mode` | 3 | 稀疏模式 |
| `q_scale_layout` | `"AUTO"` / `"TND"` / `"N2TGD"` | Q scale 布局 |
| `p_scale` | 1.0 | P scale 因子 |

### q_scale_layout 选择

| 场景 | 值 | 用例名后缀 |
|------|-----|-----------|
| 自动选择 | `"AUTO"` | 无 |
| Prefill 阶段 | `"TND"` | `Prefill` |
| Decode 阶段 | `"N2TGD"` | `Decode` |

---

## 5. 文件结构

```
fia_fullquant_mxfp8_test/
├── pytest.ini                       # pytest 配置（自定义标记）
├── fia_fullquant_mxfp8_paramset.py  # 参数定义（唯一入口）
├── fia_fullquant_mxfp8_golden.py    # CPU golden + NPU 调用
├── result_compare_method.py         # 精度对比
└── test_fia_fullquant_mxfp8.py      # pytest 测试入口
```

---

## 6. 常见问题

| 报错 | 原因 | 解决 |
|------|------|------|
| `Unsupported Q scale layout` | q_scale_layout 值不合法 | 只支持 `"AUTO"` / `"TND"` / `"N2TGD"` |
| `libhccl.so not found` | 未 source CANN 环境变量 | `source /home/user/Ascend/cann/set_env.sh` |
| `No module named 'torch_npu'` | 未安装 torch-npu | `bash install_torch_npu.sh` |
