# MoE 算子 TTK 测试适配方案

## 1. 目标

为 `/workspace/log/ttk/st` 目录下除 `moe_init_routing_v3` 外的所有算子生成 TTK 测试用例和 TestSpec 插件，放置到 `/workspace/ops-transformer_3876/moe` 对应算子目录下，并全部验证通过。

## 2. 算子清单与分类

### 2.1 已完成（3 个算子）

| 算子 | spec.py 位置 | ACLNN CSV | Kernel CSV | 状态 |
|------|-------------|-----------|------------|------|
| moe_gating_top_k | `moe_gating_top_k/tests/assets/spec.py` | 38 cases ✓ | 829 cases ✓ | 已验证 |
| moe_gating_top_k_v2 | 同上（共用 spec.py） | 32 cases ✓ | — | 已验证 |
| moe_gating_top_k_softmax_v2 | `moe_gating_top_k_softmax_v2/tests/assets/spec.py` | 232 cases ✓ | 832 cases ✓ | 已验证 |

### 2.2 待适配算子（16 个，分 3 类）

#### A 类：有 ACLNN 标杆 CSV + 有 aclnn API（9 个）

| 算子 | 标杆 ACLNN CSV | 用例数 | 标杆 Kernel CSV | 用例数 | 有 aclnn API |
|------|---------------|--------|-----------------|--------|-------------|
| moe_finalize_routing_v2 | `aclnn_moe_finalize_routing_v2_grad`* | 94 | `moe_finalize_routing_v2` | 112 | V2/V3/V4 |
| moe_finalize_routing_v2_grad | `aclnn_moe_finalize_routing_v2_grad` | 94 | `moe_finalize_routing_v2_grad` | 198 | Yes |
| moe_init_routing | `aclnn_moe_init_routing` | 10 | `moe_init_routing` | 203 | No** |
| moe_init_routing_v2 | `aclnn_moe_init_routing_v2` | 27 | `moe_init_routing_v2` | 200 | Yes |
| moe_init_routing_v2_grad | `aclnn_moe_init_routing_v2_grad` | 116 | `moe_init_routing_v2_grad` | 144 | Yes |
| moe_token_permute | `aclnn_moe_token_permute` | 4 | — | — | V1/V2 |
| moe_token_permute_grad | `aclnn_moe_token_permute_grad` | 4 | — | — | Yes |
| moe_token_unpermute | `aclnn_moe_token_unpermute` | 2 | — | — | Yes |
| moe_token_unpermute_grad | `aclnn_moe_token_unpermute_grad` | 4 | — | — | Yes |

> *`aclnn_moe_finalize_routing_v2_grad` CSV 中的 api_name 实际为 `aclnnMoeFinalizeRoutingV2Grad`，不是 V2 forward。需确认标杆是否也有 forward 的 ACLNN CSV。若无，仅做 Kernel CSV。
> **`moe_init_routing` 无独立 aclnn API，但有 `aclnn_moe_init_routing` 标杆 CSV（10 cases）。需确认是否走 `aclnnMoeInitRouting`。

#### B 类：仅有 Kernel 标杆 CSV，无 ACLNN 标杆 CSV（4 个）

| 算子 | 标杆 Kernel CSV | 用例数 | 有 aclnn API |
|------|-----------------|--------|-------------|
| moe_compute_expert_tokens | `moe_compute_expert_tokens` | 159 | No |
| moe_re_routing | `moe_re_routing` | 578 | No |
| moe_init_routing_quant_v2 | (空文件) | 0 | No |

#### C 类：已完成但需在 softmax 目录下补充 v2 用例（1 个）

| 算子 | 说明 |
|------|------|
| moe_gating_top_k_softmax | 已有 spec.py，需补 v2 ACLNN CSV（已放入 32 cases） |

### 2.3 排除

- `moe_init_routing_v3`：明确排除
- `moe_init_routing_quant_v2`：标杆 CSV 为空文件，无可用用例

## 3. 标杆文件映射

| 标杆文件 | 用途 | 涵盖算子 |
|---------|------|---------|
| `log/ttk/api_golden.py` | ACLNN golden（torch） | aclnnMoeGatingTopK/V2, aclnnMoeGatingTopKSoftmaxV2, aclnnMoeInitRouting, aclnnMoeInitRoutingV2, aclnnMoeInitRoutingQuantV2, aclnnMoeTokenPermute/Grad, aclnnMoeTokenUnpermute/Grad |
| `log/ttk/api_input.py` | ACLNN customize_inputs（torch） | aclnnMoeTokenPermuteGrad, aclnnMoeTokenUnpermute, aclnnMoeTokenUnpermuteGrad, aclnnMoeInitRouting, aclnnMoeInitRoutingV2, aclnnMoeInitRoutingQuantV2 |
| `log/ttk/op_golden_moe.py` | Kernel golden（numpy） | moe_init_routing, moe_init_routing_v3, moe_init_routing_quant, moe_compute_expert_tokens, moe_gating_top_k_softmax, moe_gating_top_k_softmax_v2, moe_finalize_routing, moe_finalize_routing_v2, moe_init_routing_v2, moe_init_routing_quant_v2, moe_gating_top_k, moe_re_routing |
| `log/ttk/op_input_moe.py` | Kernel customize_inputs（numpy） | moe_init_routing, moe_init_routing_quant, moe_compute_expert_tokens, moe_init_routing_v2, moe_init_routing_quant_v2, moe_finalize_routing_v2, moe_re_routing |

## 4. 每个算子的适配方案

### 4.1 通用规则

1. **CSV 处理**：
   - ACLNN 模式 CSV（`api_name` 以 `aclnn` 开头）：直接从标杆 `aclnn_*.csv` 复制
   - Kernel 模式 CSV（`op_name` 非 aclnn）：从标杆 `*.csv`（无 aclnn 前缀）转换字段格式
   - Kernel CSV 字段映射：`stc_inputs`→`input_shapes`, `stc_input_dtypes`→`input_dtypes`, `stc_outputs`→`output_shapes`, `output_dtypes`→`output_dtypes`, `other_compilation_params`→`attributes`

2. **spec.py 结构**：
   - 每个算子目录下 `tests/assets/spec.py`
   - `__spec__` dict 注册 op_name 和/或 aclnn api_name
   - ACLNN golden 用 torch.Tensor，参数名与 `aclnnXXGetWorkspaceSize` 一致（去掉 workspaceSize 和 executor）
   - Kernel golden 用 numpy.ndarray，参数名与 def.cpp 输入一致
   - `tolerance` 默认 `stat_rel_err`（float32/float16/bfloat16）

3. **文件放置**：
   ```
   moe/<op_name>/tests/
   ├── assets/spec.py
   └── ut/
       ├── op_api/test_aclnn_<op_name>_arch35.csv     (ACLNN 模式)
       └── op_kernel/test_<op_name>_arch35.csv         (Kernel 模式)
   ```

4. **验证流程**：
   - `python3 -m ttk aclnn/kernel -i <csv> --validate --plugin <spec.py>`
   - `python3 -m ttk aclnn/kernel -i <csv> -o /tmp/result.csv --plugin <spec.py> --pc 1 --ti=0-N`
   - 结果统计：`Total:N PASS:N FAIL:0`

### 4.2 各算子详细方案

#### 4.2.1 moe_init_routing

- **标杆 golden**：`op_golden_moe.py:263`（kernel numpy），`api_golden.py:336`（aclnn torch）
- **标杆 input**：`op_input_moe.py:14`（kernel），`api_input.py:49`（aclnn）
- **接口**：kernel 模式（3 输入：x, row_idx, expert_idx；3 输出；1 attr: active_num）
- **spec.py**：
  - `MoeInitRoutingTestSpec`（kernel, numpy）：3 输入 + active_num
  - `AclnnMoeInitRoutingTestSpec`（aclnn, torch）：同上但参数名用 camelCase
- **golden 逻辑**：argsort expert_idx → gather x → expand_row_idx
- **input 定制**：row_idx 从 arange 生成转置
- **CSV**：kernel 203 cases + aclnn 10 cases

#### 4.2.2 moe_init_routing_v2

- **标杆 golden**：`op_golden_moe.py:1113`（kernel numpy），`api_golden.py:366`（aclnn torch）
- **标杆 input**：`op_input_moe.py:36`（kernel），`api_input.py:58`（aclnn）
- **接口**：2 输入（x, expert_idx）+ 4 输出（2 optional）+ 6 attrs
- **spec.py**：
  - `MoeInitRoutingV2TestSpec`（kernel, numpy）
  - `AclnnMoeInitRoutingV2TestSpec`（aclnn, torch）
- **golden 逻辑**：argsort expert_idx → expand → gather，支持 drop_pad_mode
- **input 定制**：expert_idx 随机生成
- **CSV**：kernel 200 cases + aclnn 27 cases

#### 4.2.3 moe_init_routing_v2_grad

- **标杆**：无独立 golden/input 在标杆文件中（需自行实现）
- **接口**：2 输入（grad_expanded_x, expanded_row_idx）+ 1 输出 + 3 attrs（top_k, drop_pad_mode, active_num）
- **spec.py**：
  - `AclnnMoeInitRoutingV2GradTestSpec`（aclnn, torch）
  - `MoeInitRoutingV2GradTestSpec`（kernel, numpy）
- **golden 逻辑**：反向 scatter-add，按 expanded_row_idx 将 grad 累加回原始位置
- **CSV**：aclnn 116 cases + kernel 144 cases

#### 4.2.4 moe_finalize_routing_v2

- **标杆 golden**：`op_golden_moe.py:1053`（kernel numpy）
- **标杆 input**：`op_input_moe.py:95`（kernel）
- **接口**：7-11 输入（含 optional x/a1/a2/v）+ 1 输出 + 5 attrs
- **spec.py**：
  - `MoeFinalizeRoutingV2TestSpec`（kernel, numpy）
  - `AclnnMoeFinalizeRoutingV2TestSpec`（aclnn, torch）
- **golden 逻辑**：遍历 tokens，按 row_idx gather + bias + scale 累加
- **input 定制**：expanded_row_idx 随机生成，expert_idx 随机
- **CSV**：kernel 112 cases

#### 4.2.5 moe_finalize_routing_v2_grad

- **标杆**：无独立 golden/input（需自行实现）
- **接口**：6 输入 + 2 输出 + 4 attrs
- **spec.py**：
  - `AclnnMoeFinalizeRoutingV2GradTestSpec`（aclnn, torch）
  - `MoeFinalizeRoutingV2GradTestSpec`（kernel, numpy）
- **golden 逻辑**：grad_y 按 expanded_row_idx scatter 到 grad_expanded_x，grad_scales = grad_y * expanded_x
- **CSV**：aclnn 94 cases + kernel 198 cases

#### 4.2.6 moe_compute_expert_tokens

- **标杆 golden**：`op_golden_moe.py:758`（kernel numpy）
- **标杆 input**：`op_input_moe.py:26`（kernel）
- **接口**：1 输入（sorted_experts）+ 1 输出 + 1 attr（num_experts）
- **spec.py**：`MoeComputeExpertTokensTestSpec`（kernel, numpy）
- **golden 逻辑**：二分查找每个 expert 的 token 数量
- **input 定制**：sorted_experts 先排序
- **CSV**：kernel 159 cases

#### 4.2.7 moe_re_routing

- **标杆 golden**：`op_golden_moe.py:1518`（kernel numpy）
- **标杆 input**：`op_input_moe.py:129`（kernel）
- **接口**：3 输入（tokens, expert_token_num_per_rank, per_token_scales[optional]）+ 4 输出 + 2 attrs
- **spec.py**：`MoeReRoutingTestSpec`（kernel, numpy）
- **golden 逻辑**：按 expert_token_num_per_rank 累积偏移，重排 tokens
- **input 定制**：expert_token_num_per_rank 随机生成（总和=token 数）
- **CSV**：kernel 578 cases

#### 4.2.8 moe_token_permute

- **标杆 golden**：`api_golden.py:6`（aclnn torch）
- **接口**：2 输入（tokens, indices）+ 2 输出 + 2 attrs（num_out_tokens, padded_mode）
- **spec.py**：`AclnnMoeTokenPermuteTestSpec`（aclnn, torch）
- **golden 逻辑**：按 indices scatter tokens 到 permute_tokens，生成 sorted_indices
- **CSV**：aclnn 4 cases

#### 4.2.9 moe_token_permute_grad

- **标杆 golden**：`api_golden.py:85`（aclnn torch）
- **标杆 input**：`api_input.py:6`（aclnn）
- **接口**：2 输入 + 1 输出 + 2 attrs（num_topk, padded_mode）
- **spec.py**：`AclnnMoeTokenPermuteGradTestSpec`（aclnn, torch）
- **golden 逻辑**：按 sorted_indices scatter_add grad 回原位置
- **CSV**：aclnn 4 cases

#### 4.2.10 moe_token_unpermute

- **标杆 golden**：`api_golden.py:165`（aclnn torch）
- **标杆 input**：`api_input.py:23`（aclnn）
- **接口**：3 输入（permuted_tokens, sorted_indices, probs[optional]）+ 1 输出 + 2 attrs
- **spec.py**：`AclnnMoeTokenUnpermuteTestSpec`（aclnn, torch）
- **golden 逻辑**：按 sorted_indices scatter_add + probs 乘法
- **CSV**：aclnn 2 cases

#### 4.2.11 moe_token_unpermute_grad

- **标杆 golden**：`api_golden.py:192`（aclnn torch）
- **标杆 input**：`api_input.py:37`（aclnn）
- **接口**：4 输入 + 2 输出 + 2 attrs
- **spec.py**：`AclnnMoeTokenUnpermuteGradTestSpec`（aclnn, torch）
- **golden 逻辑**：反向 scatter + probs grad
- **CSV**：aclnn 4 cases

## 5. 执行步骤

### Step 1: 编译安装

```bash
source /home/developer/Ascend/cann/set_env.sh
bash build.sh --pkg --soc=ascend950 \
  --ops=moe_init_routing,moe_init_routing_v2,moe_init_routing_v2_grad,\
moe_finalize_routing_v2,moe_finalize_routing_v2_grad,\
moe_compute_expert_tokens,moe_re_routing,\
moe_token_permute,moe_token_permute_grad,\
moe_token_unpermute,moe_token_unpermute_grad -j8
bash build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet
```

### Step 2: 逐算子适配

每个算子按以下子步骤：

1. 创建 `tests/assets/spec.py`（golden + tolerance）
2. 创建 `tests/ut/op_api/test_aclnn_<op>_arch35.csv`（ACLNN 模式，从标杆复制）
3. 创建 `tests/ut/op_kernel/test_<op>_arch35.csv`（Kernel 模式，从标杆转换）
4. `--validate` 验证 CSV 格式
5. 全量运行 `--ti=0-N`，确认 PASS

### Step 3: 验证

```bash
export ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.1.0/opp/vendors/custom_transformer
export LD_LIBRARY_PATH=.../custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
cd /workspace/ops-test-kit

# ACLNN
python3 -m ttk aclnn -i <csv> -o /tmp/result.csv --plugin <spec.py> --pc 1 --ti=0-N

# Kernel
python3 -m ttk kernel -i <csv> -o /tmp/result.csv --plugin <spec.py> --pc 1 --ti=0-N
```

## 6. 总用例数预估

| 类型 | 用例数 |
|------|--------|
| ACLNN 模式 | ~295 (38+32+232+10+27+116+94+4+4+2+4) |
| Kernel 模式 | ~3055 (829+832+203+200+144+112+198+159+578) |
| **合计** | **~3350** |

## 7. 风险点

1. **moe_finalize_routing_v2**：regbase 配置有 4 个额外 optional 输入（x, a1, a2, v），标杆 golden 中有 hif4 反量化逻辑较复杂，可能需要简化
2. **moe_init_routing_quant_v2**：标杆 CSV 为空，无法适配
3. **moe_re_routing**：regbase 配置支持多种量化 dtype（FP8/HIFP4 等），golden 可能需要对应处理
4. **无 aclnn API 的算子**（moe_compute_expert_tokens, moe_init_routing, moe_re_routing）：只能做 Kernel 模式测试
5. **golden 精度**：量化类算子（quant_v2, re_routing 的 FP8 路径）可能因 golden 与 kernel 量化方式不完全一致导致精度 FAIL，需逐用例调试
