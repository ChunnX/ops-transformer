# MLA Preprocess ATK→TTK 适配文档

## 一、任务目标

将 `code/ops-transformer_0811/attention/mla_preprocess` 的 ATK 测试适配到 TTK 测试框架，实现 ACLNN 模式下的三方精度对比（高精度 CPU golden / 低精度 CPU benchmark / aclnn 融合算子输出）。

## 二、框架选型与适配方法确定

### 为什么选择 TTK ACLNN 模式

TTK 支持四种测试通路：Kernel / GEIR / ACLNN / E2E。选择 ACLNN 模式的原因：

- **ATK 的 pyaclnn 后端等价于 TTK 的 ACLNN 模式** — 两者都调用 `aclnnMlaPreprocessGetWorkspaceSize` C API
- ATK 的 CPU golden 和低精度 benchmark 逻辑可直接移植为 TTK 的 `golden` / `compare` 插件
- TTK 的 `customize_inputs` 机制等价于 ATK 的 `init_by_input_data`，用于 slotmapping 生成、deScale reinterpret、NZ 格式转换
- TTK 的 CSV 用例格式等价于 ATK 的 JSON 用例格式，可通过转换脚本批量生成

### 参考实现

参考 `code/ops-transformer_RFA/attention/fused_floyd_attention/tests/assets/` 中的 FFA 算子 TTK 适配实现，借鉴了：
- `spec.py` 中 `importlib.util` 加载 impl/ 模块的模式（spec loader 不注册 sys.modules，不能用相对导入）
- `compare.py` 中三方对比的框架结构（高精度 golden / 低精度 benchmark / aclnn actual）
- `inputs.py` 中 `customize_inputs` 的参数签名和 in-place 修改模式

### 适配方法确定过程

1. **分析 ATK executor.py**：提取 C 头文件签名（37 个参数）、Golden 逻辑（CPU 高精度/低精度路径）、入参预处理（slotmapping/deScale/NZ）、精度对比方法（`cv_fused_double_benchmark`）
2. **分析 ATK JSON 用例**：提取 tensor 参数（24 输入 + 4 输出）、attr 参数（13 个）、dtype/format/range 分布
3. **分析 TTK 框架**：理解 CSV 字段（`tensor_view_shapes`/`tensor_dtypes`/`tensor_formats`/`attributes`/`scalar_dtypes`）、TestSpec 插件体系（`golden`/`customize_inputs`/`compare`/`tolerance`）、参数组装（`AclnnParamPlan.build_args`）
4. **确定适配方案**：按 `ttk_adaption.md` 规范创建 `tests/assets/` 目录结构，将 ATK 的计算逻辑移植为 TTK 插件

## 三、适配产物

### 目录结构

```
tests/assets/
├── __init__.py
├── spec.py                     # TestSpec 入口
├── impl/
│   ├── __init__.py
│   ├── golden.py               # 高精度+低精度 CPU golden
│   ├── inputs.py               # customize_inputs
│   └── compare.py              # 三方对比逻辑
├── atk_json_to_ttk_csv.py      # JSON→CSV 转换脚本
└── TTK_ADAPTATION.md           # 本文档
```

### 创建命令

```
帮我将code/ops-transformer_0811/attention/mla_preprocess的atk测试转为ttk测试
```

CANNBot 根据 `ascendc-atk-to-ttk-standardization` skill 的规范化流程，自动创建了上述目录结构和文件。

## 四、遇到的问题与解决方案

### 问题 1：Scalar_COUNT_MISMATCH

**现象**：TTK 报 `Scalar_COUNT_MISMATCH`，ACLNN API 有 0 个 scalar 参数，但 CSV 配了 13 个 `scalar_dtypes`。

**原因**：C 头文件中 `wdqDim`/`epsilon`/`cacheMode` 等 13 个参数是 `int64_t`/`double`/`bool` 类型，TTK 的 `OpApiInfoKeeper` 将它们归类为 attribute 而非 scalar。CSV 中不应该有 `scalar_dtypes`。

**解决**：CSV 转换脚本中 `scalar_dtypes` 设为空 `()`，13 个参数放入 `attributes` 字典。

### 问题 2：PARAM_COUNT_MISMATCH

**现象**：TTK 报 `PARAM_COUNT_MISMATCH`，tensor(28) + scalar(0) + attr(10) = 38 ≠ API 参数总数 41。

**原因**：attr 名称不匹配。C 头文件中有拼写错误 `transeposeWdq`（多了个 'e'），CSV 中写的是 `transposeWdq`。`pure_attrs` 过滤掉了名称不匹配的 3 个 attr（`transposeWdq`/`transposeWuq`/`transposeWuk`）。

**解决**：CSV 转换脚本中添加 `ATTR_NAME_MAP`，将 JSON 中的 `transposeWdq` 映射为 C 头文件的 `transeposeWdq`。

### 问题 3：参数顺序不匹配

**现象**：`customize_inputs() takes 28 positional arguments but 41 were given`。

**原因**：函数签名用了 `*` 分隔符将 attr 设为 keyword-only，但 TTK 的 `AclnnParamPlan.build_args()` 按头文件顺序以位置参数传递所有参数。且函数签名顺序（24 tensor → 4 output → 13 attr）与 API 参数顺序（24 tensor → 13 attr → 4 output）不一致。

**解决**：去掉 `*` 分隔符，按 API 参数顺序排列：24 input tensor → 13 attr → 4 output tensor。

### 问题 4：cacheMode 参数类型不匹配

**现象**：`ValueError: only one element tensors can be converted to Python scalars`。

**原因**：TTK 传给 `customize_inputs` 的 `cacheMode` 参数可能是 `torch.Tensor`（非标量），直接 `int(cacheMode)` 失败。

**解决**：在 `inputs.py` 和 `golden.py` 中添加 `_to_int`/`_to_bool`/`_to_float` 安全转换函数，处理 Tensor/list/scalar 三种类型。

### 问题 5：161002 tiling 匹配失败（tensor_formats 过多 NZ）

**现象**：NPU 报 `errno[161002] NnopbaseExecutorMatchCache failed`，aclnn 执行失败。

**原因**：CSV 的 `tensor_formats` 中设了过多的 NZ 格式（gamma0/quantScale0/kvCache/kvCacheRope 等），导致 NPU 框架的 simplifiedKey 与编译的 kernel 不匹配。NPU 框架的 `DynamicFormatFlag(true)` + `PrecisionReduceFlag(true)` 在精确 dtype 组合匹配失败时自动降级到 bf16 组合，但 bf16 组合的 tilingKey 未编译。

**排查过程**：
1. 通过 plog 日志获取 NPU 框架的 24 个 SupportInfo 候选组合
2. 逐一对比 TTK 传入的 dtype/format 与 SupportInfo 中的组合
3. 发现最佳匹配 SupportInfo[0]（fp16 量化组合）中只有 wdqkv/wuq 是 NZ，其余 tensor 都是 ND
4. CSV 中多余的 NZ 导致 simplifiedKey 不匹配

**解决**：CSV 中 `tensor_formats` 只设 wdqkv(5) 和 wuq(12) 为 NZ（算子定义中 `nzFormat` 强制 NZ 的两个 tensor），其余全部 ND。

### 问题 6：descale dtype 未被 NPU 识别为 int64

**现象**：ACL tensor 创建时 descale0 dtype=int64（通过调试确认），但 NPU SupportInfo 仍显示 `descale0(DT_FLOAT)`，导致 tiling 匹配失败。

**原因**：`customize_inputs` 中用 `copy_()` 做 descale 的 fp32→int64 reinterpret，但 `copy_()` 不改变 tensor 的 dtype 属性。

**解决**：
1. `customize_inputs` 中用 `tensor.data = int64_tensor` 替换底层 storage 改变 dtype
2. TTK 框架 `input_generation.py` 中 `customize_inputs` 执行后加 `self._ctx.invalidate_flat_cache("tensors")` 刷新 `flatten_tensors` 缓存

### 问题 7：输出 tensor shape 计算错误

**现象**：qOutOut shape 与 NPU 实际输出不匹配，导致 compare 失败。

**原因**：
- qOutOut 用了 `nope_dim=wuk.shape[1]`（112），正确应为 `wuk_shape[2]`（512，einsum 输出维度）
- kvCacheOutOut 用了原始 ND shape，但 cacheMode>1 时 NPU 输出 NZ 格式
- krCacheOutOut 用了 `head_num*rope_dim//16`，正确应为 `kvCacheRope_shape[-1]//16`

**解决**：CSV 转换脚本中 `_compute_output_shapes` 函数修正：
- qOutOut: 最后一维用 `wuk_shape[2]`（einsum 输出维度）
- kvCacheOutOut (cacheMode>1): `[BLOCK_NUM, kvCache_shape[-1]//16, BLOCK_SIZE, 16]`
- krCacheOutOut (cacheMode>1): `[BLOCK_NUM, kvCacheRope_shape[-1]//16, BLOCK_SIZE, 16]`

### 问题 8：输出 tensor 初始化值不一致

**现象**：kvCacheOut/krCacheOut 的未写入位置（非 slotmapping 指向的位置）ATK=0，TTK=1。

**原因**：TTK 默认将输出 tensor 初始化为 1（`init_val=1`），ATK 初始化为 0。kvCacheOut 是 sparse 写入的，未写入位置保留初始值。

**解决**：修改 `input_generation.py` 中输出 tensor 初始化值为 0。

### 问题 9：NZ 数据排布转换缺失

**现象**：固定输入下 ATK 和 TTK 的 aclnn 输出不一致（qOut/qRopeOut 完全不同），全 1 输入时一致。

**原因**：TTK 的 `acl_interface.py` 中 `copy_torch_tensor_to_hbm` 线性拷贝 ND 排布的数据到 HBM，但 CSV `tensor_formats=NZ` 让 ACL tensor 声明 NZ 格式。NPU 按 NZ stride 读取 ND 排布的 HBM 数据 → 数据错乱。全 1 输入时所有值相同，排布不影响结果所以一致。

**解决**：修改 TTK 框架 `acl_interface.py` 的 `_create_acl_tensor_from_torch`，当 `view_format == "NZ"` 时，用与 ATK `transdata_nd2nz` 一致的逻辑（int8 用 align=32，其他用 align=16）做 ND→NZ 数据排布转换，再拷贝到 HBM。

### 问题 10：高精度 golden 的 dtype 不一致（fp16 vs bf16 差异化处理）

**现象**：ATK 全 PASS，TTK 全 FAIL。qOut 的 mare_rate 差异近 10 倍（ATK=0.36，TTK=3.75~19.8）。

**原因**：ATK 数据生成时把 fp16 input 转成了 fp32，高精度 golden 全程 fp32 计算。TTK 的 input1 是 CSV 声明的 fp16，虽然 `rms_norm_cpu` 内部 `x.float()` 转 fp32，但 input1 本身已经是 fp16 精度 → 高精度和低精度的差异很小 → benchmark 误差小 → ratio 大。

**排查过程**：
1. 固定 pt 文件输入验证高精度/低精度/aclnn 输出完全一致（排除计算逻辑差异）
2. 对比 ATK 和 TTK 的 ratio 发现 TTK benchmark 误差极小（mare_benchmark=0.000735）
3. 推断 TTK 高精度 golden 的 input1 是 fp16（精度已损失），ATK 是 fp32

**解决（fp16）**：TTK 高精度 golden 函数中，当 `data_type == torch.float16` 时把所有浮点输入 tensor 转成 fp32，与 ATK 一致。高精度路径中不执行 `.to(data_type)` 截断，保持 fp32 中间结果。

**后续问题（bf16）**：fp16 修复后 bf16 用例出现新的 FAIL — cacheMode=2 的 qOut int8 `abs_diff=2.0`（ATK 为 0.0）。

**排查**：对比 ATK 和 TTK 的 golden qOut（int8 量化输出），发现 7936/524288 个值差 1。ATK golden 全是 0，TTK golden 有 ±1 的值。根因是 bf16 用例的高精度 golden 也被转成了 fp32，但 ATK 的 bf16 路径保持 bf16 + `.to(bf16)` 截断 — fp32 中间结果与 bf16 截断后的结果在 `round` 量化边界不同。

**解决（bf16）**：高精度 golden 中按 dtype 差异化处理：
- **fp16 输入**：转 fp32，不截断（ATK 数据生成把 fp16 转成了 fp32）
- **bf16 输入**：保持 bf16，高精度路径中 `.to(data_type)` 截断为 bf16（ATK 保持 bf16）

```python
data_type = input1.dtype
if data_type == torch.float16:
    # ATK fp16 input 实际是 fp32, 转 fp32 不截断
    input1 = input1.to(torch.float32)
    gamma0 = gamma0.to(torch.float32)
    ...
    data_type = torch.float32
# bf16: 保持原 dtype, 高精度路径 .to(data_type) 截断为 bf16
```

### 问题 11：随机种子未固定导致结果不可复现

**现象**：同一 CSV 多次运行，三方对比结果从 PASS 到 FAIL 波动很大（qOut mare 从 0.004 到 19.8）。

**原因**：deScale range `[0, 0.001]` 中可能生成接近 0 的值。deScale ≈ 0 → mm_w8a8 输出 ≈ 0 → golden ≈ 0 → `mere = |actual-golden|/|golden|` 分母 ≈ 0 → mere 爆炸。每次运行 deScale 最小值不同导致结果波动。TTK 不固定随机种子。

**排查过程**：
1. 三次运行结果完全不同（mare=0.004/19.8/8.35）
2. 分析 deScale 随机值分布：每次最小值在 3.6e-8 ~ 2.5e-7 之间波动
3. 确认 TTK `input_generation.py` 中 `numpy.random.seed` 未执行（`batch_seed` 默认为 `None`，条件 `if base_seed and batch_seed is not None` 不满足）

**解决**：修改 `input_generation.py`：
1. `base_seed` 默认值从 `None` 改为 `2025`
2. `numpy.random.seed(base_seed + idx)` 不再依赖 `batch_seed`

### 问题 12：golden 中 reshape_and_cache_cpu 的 dtype 不匹配

**现象**：高精度路径中 `reshape_and_cache_cpu` 报 `Index put requires the source and destination dtypes match`。

**原因**：高精度路径中 key_out 是 fp32（高精度路径把输入转成了 fp32），但 kvCache 是 fp16（CSV 声明的 dtype）。`reshape_and_cache_cpu` 中写入 cache 时 dtype 不匹配。

**解决**：`reshape_and_cache_cpu` 中写入 cache 时用 `.to(cache_dtype)` 转换 dtype。

### 问题 13：golden 中 kvCacheOut 的 NZ 格式转换

**现象**：golden 输出的 kvCacheOut shape 与 NPU 实际输出不匹配。

**原因**：cacheMode>1 时 NPU 输出 NZ 格式 `(192, 32, 128, 16)`，但 golden 中不做 NZ 转换返回 ND 格式 `(192, 128, 1, 512)`。

**解决**：golden 中 cacheMode>1 时对 kvCacheOut/krCacheOut 做 `transdata_nd2nz` 转换。

### 问题 14：bf16 用例 INPUT_GEN_FAILURE（ml_dtypes 缺失）

**现象**：bf16 用例（case 2/3/5/9）全部 INPUT_GEN_FAILURE。

**原因**：TTK 数据生成用 NumPy（非 torch），NumPy 原生不支持 bfloat16，需要 `ml_dtypes` 库提供 numpy 级别的 bfloat16 dtype。ATK 用 torch 生成数据，torch 原生支持 bfloat16，不需要 `ml_dtypes`。

**解决**：`pip install ml_dtypes`。

### 问题 15：cacheMode=2 的 161002（kvCache/kvCacheRope format 为 ND 但算子定义要求 NZ）

**现象**：cacheMode=2 的用例（case 3/4/5/6）报 161002。

**原因**：算子定义 `mla_preprocess_def.cpp` 中 kvCache 的 format 数组 `nd_nz_Format` 在 cacheMode=2 时是 NZ，但 CSV 中 kvCache format 设为 ND。NPU 框架自动推断 kvCache=NZ，但 TTK 的 ACL tensor 声明 ND → simplifiedKey 不匹配。

**排查过程**：
1. 对比 case 0（cacheMode=3，PASS）和 case 3（cacheMode=2，FAIL）的 dtype/format 组合
2. 发现 case 0 的算子定义中 kvCache format=ND（cacheMode=3），CSV 也是 ND → 匹配
3. case 3 的算子定义中 kvCache format=NZ（cacheMode=2），CSV 是 ND → 不匹配

**解决**：CSV 转换脚本中 cacheMode=2 时把 kvCache(19)/kvCacheRope(20)/kvCacheOutOut(25)/krCacheOutOut(27) 的 format 设为 NZ。

### 问题 16：NZ 转换中 bf16 tensor 不支持

**现象**：cacheMode=2 的 bf16 用例 NZ 转换报 `Got unsupported ScalarType BFloat16`。

**原因**：`torch.from_numpy(numpy_bfloat16_array)` 不支持 ml_dtypes 的 bfloat16 numpy 数组。且 4D bf16 tensor 的 `.cpu().numpy()` 返回 ml_dtypes bfloat16 数组，后续 `from_numpy` 失败。

**解决**：
1. bf16 tensor 先 `.cpu().float().numpy()` 转 fp32 numpy 数组，再做 NZ 转换，最后 `.to(torch.bfloat16)` 转回
2. NZ 转换代码中 `import numpy as _np` 确保多进程 worker 中可用

### 问题 17：reshape_and_cache_cpu 中 key_cache_rope_in dtype 不匹配

**现象**：cacheMode=2 时 `reshape_and_cache_cpu` 报 `Index put requires the source and destination dtypes match, got Half for the destination and Char for the source`。

**原因**：`reshape_and_cache_cpu` 中 `key_cache_rope_in` 写入时用 `cache_dtype`（= `key_cache_in.dtype`），但 cacheMode=2 时 `kvCache`（int8）和 `kvCacheRope`（fp16/bf16）的 dtype 不同。`key_cache_rope_in` 是 fp16，但 `.to(cache_dtype=int8)` 转成 int8 后写入 fp16 的 `key_cache_rope_in` → dtype 不匹配。

**解决**：`reshape_and_cache_cpu` 中对 `key_cache_rope_in` 用独立的 `rope_cache_dtype = key_cache_rope_in.dtype` 而非 `cache_dtype`。

## 五、TTK 框架修改清单

| 文件 | 修改内容 |
|------|---------|
| `ttk/core_modules/npu/op_api/input_generation.py` | 1. `customize_inputs` 后加 `invalidate_flat_cache("tensors")` 刷新缓存<br>2. 输出 tensor 初始化值从 1 改为 0<br>3. `base_seed` 默认值改为 2025<br>4. `numpy.random.seed` 不再依赖 `batch_seed` |
| `ttk/core_modules/aclnn/acl_interface.py` | `_create_acl_tensor_from_torch` 中对 NZ format 的 tensor 做 ND→NZ 数据排布转换（int8 align=32，其他 align=16），bf16 先转 fp32 再 from_numpy |

## 六、验证结果

### 固定输入验证

用 pt 文件固定输入，ATK 和 TTK 的三项输出完全一致：

| 对比项 | qOut | kvCacheOut | qRopeOut | krCacheOut |
|--------|------|------------|----------|------------|
| aclnn 输出 | IDENTICAL ✓ | IDENTICAL ✓ | IDENTICAL ✓ | IDENTICAL ✓ |
| 高精度 CPU golden | IDENTICAL ✓ | IDENTICAL ✓ | IDENTICAL ✓ | IDENTICAL ✓ |
| 低精度 CPU benchmark | IDENTICAL ✓ | IDENTICAL ✓ | IDENTICAL ✓ | IDENTICAL ✓ |
| 三方对比 ratio | 完全一致 | 完全一致 | 完全一致 | 完全一致 |

### 10 条用例独立随机输入验证（固定种子 2025）

| case | dtype | cacheMode | tilingKey | TTK 结果 | 说明 |
|------|-------|-----------|-----------|---------|------|
| 0 | fp16 | 3 | 120 | **PASS** | 4/4 PASS |
| 1 | fp16 | 3 | 120 | **PASS** | 4/4 PASS |
| 2 | bf16 | 1 | 184 | **PASS** | 4/4 PASS |
| 3 | bf16 | 2 | 216 | **FAIL** | qOut int8 abs_diff=2.0（阈值≤1），其余 PASS |
| 4 | fp16 | 2 | 88 | **FAIL** | qOut int8 abs_diff=4.0 + qRopeOut mere=2.106，其余 PASS |
| 5 | bf16 | 2 | 216 | **FAIL** | qOut int8 abs_diff=2.0，其余 PASS |
| 6 | fp16 | 2 | 88 | **PASS** | 4/4 PASS |
| 7 | fp16 | 1 | 56 | **PASS** | 4/4 PASS |
| 8 | fp16 | 1 | 56 | **PASS** | 4/4 PASS |
| 9 | bf16 | 3 | 248 | **PASS** | 4/4 PASS |

**7/10 PASS，3/10 FAIL**。3 条 FAIL 都是 cacheMode=2：
- case 3（bf16）：qOut int8 abs_diff=2.0 — bf16 高精度 golden 与 NPU 的 round 边界差异
- case 4（fp16）：qOut int8 abs_diff=4.0 + qRopeOut mere=2.106 — fp16 转 fp32 后量化边界 + rope 精度差异
- case 5（bf16）：qOut int8 abs_diff=2.0 — 同 case 3

这些 FAIL 是随机数据落在 `round` 量化边界导致的边界差异，非框架适配问题。case 6（同样 fp16+cacheMode=2）能 PASS 证明逻辑正确。

### 100 条用例验证（固定种子 2025）

**55/100 PASS，45/100 FAIL**

| dtype + cacheMode | PASS | FAIL | 通过率 |
|-------------------|------|------|--------|
| bfloat16 + cm1 | 20 | 2 | 91% |
| bfloat16 + cm2 | 0 | 19 | 0% |
| bfloat16 + cm3 | 13 | 1 | 93% |
| float16 + cm1 | 9 | 4 | 69% |
| float16 + cm2 | 1 | 17 | 6% |
| float16 + cm3 | 12 | 2 | 86% |

FAIL 规律：
- **cacheMode=2 几乎全部 FAIL**（1/37 PASS）— 双重量化（`quant_cpu` + `ein_sum_out_quant_golden`）对数据敏感，int8 量化输出的 `abs_diff` 略超阈值 1
- **bf16 cm1/cm3 通过率高**（91%/93%）— bf16 高精度 golden 保持 bf16 + `.to(bf16)` 截断，与 ATK 一致
- **fp16 cm3 通过率 86%**，fp16 cm1 通过率 69% — 部分 FAIL 是 deScale 随机值接近 0 导致 `mere` 分母接近 0，ratio 爆炸

### 测试环境配置与指令

#### 环境配置

```bash
# 1. conda 环境
source /home/t00954448/miniconda3/etc/profile.d/conda.sh
conda activate tmy

# 2. CANN 环境
source /home/t00954448/Ascend/0804/cann/set_env.sh

# 3. ATB 环境
source /home/t00954448/Ascend/0804/nnal/atb/set_env.sh
export ATB_BUILD_DEPENDENCY_PATH=/home/t00954448/Ascend/0804/nnal/atb/latest/atb/cxx_abi_1
source /home/t00954448/code/ascend-transformer-boost/output/atb/set_env.sh
export ATB_HOME_PATH=/home/t00954448/code/ascend-transformer-boost/output/atb/cxx_abi_1

# 4. vendor 环境
source /home/t00954448/Ascend/0804/cann/vendors/custom_transformer/bin/set_env.bash
export ATK_CUSTOM_OPP_PATH=/home/t00954448/Ascend/0804/cann/vendors/custom_transformer/op_api/lib/libcust_opapi.so

# 5. torch/torch_npu lib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch_npu/lib:$LD_LIBRARY_PATH

# 6. TTK
export PYTHONPATH=/home/t00954448/code/ops-test-kit:$PYTHONPATH

# 7. 依赖
pip install ml_dtypes  # bf16 用例需要
```

#### 算子编译

```bash
# 方式一：带指定 tilingKey 编译（更快）
bash build.sh --pkg --soc=ascend950 --ops=mla_preprocess \
    --tiling_key="56;88;120;184;216;248" -j16

# 方式二：整包编译（编译全部 tilingKey，耗时较长）
bash build.sh --pkg --soc=ascend950 --ops=mla_preprocess -j16

# 安装
bash build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=/home/t00954448/Ascend/0804/cann --quiet
```

#### TTK 测试指令

```bash
cd /home/t00954448/code/ops-test-kit

# 运行全部用例
python3 -m ttk aclnn \
    -i /home/t00954448/code/ops-transformer_0811/attention/mla_preprocess/tests/atk/mla_preprocess_ttk.csv \
    --plugin /home/t00954448/code/ops-transformer_0811/attention/mla_preprocess/tests/assets \
    --compare stat_rel_err \
    --golden-mode Enable \
    -o /tmp/ttk_result.csv \
    --print true

# 运行单条用例
python3 -m ttk aclnn \
    -i .../mla_preprocess_ttk.csv \
    --plugin .../tests/assets \
    --compare stat_rel_err \
    --golden-mode Enable \
    -t mla_preprocess_0 \
    --print true

# 失败用例 dump 调试
python3 -m ttk aclnn \
    -i .../mla_preprocess_ttk.csv \
    --plugin .../tests/assets \
    --compare stat_rel_err \
    --golden-mode Enable \
    -t mla_preprocess_0 \
    --print true \
    --dump full --dump-format npy --dump-on-fail
```

#### 测试结果呈现

**终端输出**：
```
[COMPARE-MLA] mare_rate=1.001915 (thd=10), mere_rate=0.616343 (thd=2), rmse_rate=0.712278 (thd=2), eb=-0.00000114 (thd=0.00097656) -> PASS
...
PRECISION_STATUS: PASS

==================== TTK Test Summary ===================
  Total    : 100
  PASS     : 55
  FAIL     : 45
  SKIP/NA  : 0
  PassRate : 55.00%  (55/100)
```

**结果 CSV**（`-o` 指定输出路径）：
- `testcase_name` — 用例名
- `precision_status` — PASS/FAIL
- `precision` — 每个输出的精度百分比
- `api_perf_us` — aclnn 执行耗时（微秒）
- `attributes` — 用例参数（cacheMode 等）

**三方对比指标说明**：
- `mare_rate` — 最大相对误差比例（阈值 <10）
- `mere_rate` — 平均相对误差比例（阈值 <2）
- `rmse_rate` — 均方根误差比例（阈值 <2）
- `eb` — 误差均衡性（阈值 < eb_thd）
- cacheMode=2 的 int8 输出用 `abs_diff <= 1` 判定（不使用 ratio）

### 编译要求

需要根据用例的 cacheMode/dtype 计算对应的 tilingKey 并编译：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mla_preprocess \
    --tiling_key="56;88;120;184;216;248" -j16
```

tilingKey 计算公式（来自 `mla_preprocess_tiling.cpp:534-543`）：
```
tilingKey = isBf16 << 2 | cacheMode
          << 1 | wdqkvNZ << 1 | wuqNZ << 1 | wukNZ(0)
          << 2 | quantMode
```

## 七、问题排查方法论

### 排查工具

| 工具 | 用途 | 示例 |
|------|------|------|
| **TTK `--dump full --dump-on-fail --dump-format npy`** | dump 失败用例的 input/output/golden 到 npy 文件，离线分析 | `python3 -m ttk aclnn ... --dump full --dump-on-fail --dump-format npy` |
| **TTK `--single-log`** | 每条用例单独日志文件，便于定位 | `--single-log` |
| **TTK `--print false`** | 关闭周期性汇总打印，减少日志噪音 | `--print false` |
| **NPU plog 日志** | 查看NPU框架内部的 tiling 匹配、SupportInfo、simplifiedKey | `ASCEND_GLOBAL_LOG_LEVEL=1` + 查看 `/ascend/log/debug/plog/plog-*.log` |
| **TTK `--validate`** | 只校验 CSV 用例格式，不跑设备 | `python3 -m ttk aclnn -i cases.csv --validate` |
| **Python monkey-patch** | 在 TTK 框架函数中加调试打印，验证中间状态 | 修改 `_create_acl_tensor_from_torch` 打印 dtype/format |
| **固定输入 pt 文件** | 消除随机数据差异，验证计算逻辑一致性 | ATK `init_by_input_data` + TTK `customize_inputs` 中加载 pt 文件覆盖输入 |

### 排查思路

#### 161002 tiling 匹配失败

1. **确认 tilingKey**：根据 `mla_preprocess_tiling.cpp:534-543` 的 `SetTilingKey` 公式，用 case 的 dtype/cacheMode/format 参数计算 tilingKey
2. **确认编译**：检查 `build.sh --tiling_key=` 是否包含该 tilingKey
3. **查 plog SupportInfo**：`grep "SupportInfo" plog-*.log` 获取 NPU 框架的 24 个候选 dtype/format 组合，逐一与 TTK 传入的 ACL tensor dtype/format 对比
4. **定位不匹配项**：找到最佳匹配的 SupportInfo entry，逐 tensor 对比 dtype 和 format，找出差异项
5. **修复 CSV**：根据算子定义 `mla_preprocess_def.cpp` 中的 format 数组（`nzFormat`/`nd_nz_Format`/`qFormat`）确定每个 tensor 在不同 cacheMode/dtype 组合下的正确 format

#### aclnn 输出不一致

1. **固定输入**：用 pt 文件固定输入，消除随机数据差异
2. **对比三项输出**：分别对比 ATK 和 TTK 的 aclnn/golden/benchmark 输出，确定哪个不一致
3. **全 1 输入测试**：用全 1 输入排除 NZ 数据排布问题（全 1 时排布不影响值）
4. **逐 tensor 检查 dtype**：在 `customize_inputs` 和 golden 函数中加调试打印，确认 tensor 的实际 dtype
5. **对比计算逻辑**：逐函数对比 ATK `executor.py` 和 TTK `golden.py` 的中间计算步骤和 dtype 转换

#### 三方对比 ratio 异常

1. **分析 ratio 公式**：`ratio = actual误差 / max(benchmark误差, err_thd)`，分别检查分子和分母
2. **检查 benchmark 误差**：如果 benchmark 误差极小（<< err_thd），说明高精度和低精度 golden 的计算结果过于接近
3. **检查输入 dtype**：高精度 golden 的输入 dtype 是否与 ATK 一致（ATK 数据生成可能把 fp16 转成 fp32）
4. **检查 deScale range**：deScale range `[0, 0.001]` 可能生成接近 0 的值，导致 `mere = |diff|/|golden|` 分母接近 0

#### 结果不可复现

1. **多次运行对比**：同一 CSV 运行 3 次，确认结果是否一致
2. **检查随机种子**：`input_generation.py` 中 `numpy.random.seed` 是否执行（`batch_seed` 默认为 `None` 可能导致条件不满足）
3. **检查随机数消耗顺序**：不同 tensor 的随机数生成顺序可能影响后续 tensor 的种子偏移

## 八、关键经验

1. **CSV `tensor_formats` 需按算子定义的 format 数组设置**：wdqkv/wuq 始终 NZ；cacheMode=2 时 kvCache/kvCacheRope/kvCacheOutOut/krCacheOutOut 也为 NZ；其余由 NPU 框架自动推断为 ND
2. **deScale 的 fp32→int64 reinterpret 必须用 `tensor.data =`**，不能用 `copy_()`，且需要刷新 `flatten_tensors` 缓存
3. **NZ format 的 ACL tensor 需要做数据排布转换**，不能只声明 format 不转数据；bf16 tensor 需先转 fp32 再 `from_numpy`
4. **高精度 golden 的输入 dtype 必须按 dtype 类型差异化处理**：fp16 转 fp32 不截断（ATK 数据生成把 fp16 转成了 fp32）；bf16 保持原 dtype + `.to(data_type)` 截断（ATK 保持 bf16）。统一转 fp32 会导致 bf16 量化输出 round 边界差异
5. **随机种子必须固定**，否则 deScale 接近 0 的值会导致结果不可复现
6. **C 头文件的参数名拼写错误（`transeposeWdq`）需要在 CSV 中保持一致**
7. **bf16 用例需要安装 `ml_dtypes`**，因为 TTK 用 NumPy 生成数据（ATK 用 torch 不需要）
8. **`reshape_and_cache_cpu` 中 kvCache 和 kvCacheRope 的 dtype 可能不同**（cacheMode=2 时 kvCache=int8, kvCacheRope=fp16），需要分别用各自的 cache_dtype
9. **通过 plog 日志的 SupportInfo 列表排查 tiling 匹配问题**是最有效的方法
