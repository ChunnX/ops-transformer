# QkvRmsNormRopeCacheWithKScale ACLNN 接口覆盖测试报告

## 1. 测试目标

根据 `qkv_rms_norm_rope_cache_with_kscale_tasks/01_operator_proto_and_formula.md` 中的接口原型、shape 符号和校验约束，从 ACLNN op_api 接口角度覆盖正常与异常入参场景。

本轮只覆盖 `GetWorkspaceSize` 阶段的接口校验，不覆盖 tiling/kernel 数值正确性。

## 2. 测试文件

| 文件 | 说明 |
|------|------|
| `attention/qkv_rms_norm_rope_cache_with_kscale/tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp` | CSV 驱动的 ACLNN op_api UT |
| `attention/qkv_rms_norm_rope_cache_with_kscale/tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.csv` | 用例矩阵，共 57 条 |

## 3. 覆盖范围

### 正常用例

- decode 典型形状：`T=128, Nq=16, Nk=Nv=2, D=128`
- prefill/multi-batch 静态形状：`T=64, Batch=4, Nq=8, Nk=Nv=4, D=128`

### 空指针校验

覆盖所有接口必选 tensor、当前场景强制非空的 optional tensor、属性数组和输出：

- `qkv`
- `q_gamma`
- `k_gamma`
- `cos_sin`
- `slot_mapping`
- `k_cache`
- `v_cache`
- `k_scale_cache`
- `cu_seq_len`
- `optionalQRotation`
- `optionalKRotation`
- `optionalVScale`
- `headNums`
- `qOut`
- `qScale`

### dtype 校验

覆盖以下 dtype 错误：

- `qkv` 非 `BF16`
- `q_gamma` / `k_gamma` 非 `FLOAT`
- `cos_sin` 非 `FLOAT`
- `slot_mapping` 非 `INT32`
- `k_cache` / `v_cache` 非 `FLOAT8_E4M3FN`
- `k_scale_cache` 非 `FLOAT`
- `cu_seq_len` 非 `INT32`
- `optionalQRotation` / `optionalKRotation` 非 `BF16`
- `optionalVScale` 非 `FLOAT`
- `qOut` 非 `FLOAT8_E4M3FN`
- `qScale` 非 `FLOAT`

### shape 与属性校验

覆盖以下异常：

- `qkv` rank 非 3
- `qkv.shape[1] != Nq + Nk + Nv`
- fast path `D != 128`
- `q_gamma.shape != [D]`
- `k_gamma.shape != [D]`
- `cos_sin.shape[1] != D`
- `slot_mapping.shape != [T]`
- `cu_seq_len` rank 非 1
- `cu_seq_len.shape[0] < 2`
- `optionalQRotation.shape != [D, D]`
- `optionalKRotation.shape != [D, D]`
- `optionalVScale.shape != [Nv]`
- `k_cache.shape[1] != Nk`
- `k_cache.shape[3] != D`
- `v_cache.shape[1] != Nv`
- `v_cache.shape[3] != D`
- `k_scale_cache.shape[1] != Nk`
- `k_scale_cache.shape[3] != 1`
- `k_scale_cache` 与 `k_cache` 的 `BlockNum/BlockSize` 不一致
- `v_cache` 与 `k_cache` 的 `BlockNum/BlockSize` 不一致
- `headNums` 长度非 3
- `headNums` 含非正值
- `Nv != Nk`
- `qOut.shape != [T, Nq, D]`
- `qScale.shape != [T, Nq]`
- 空 tensor：`qkv.shape[0] == 0`

## 4. 验证命令与结果

### CSV 列数检查

```bash
awk -F',' '{print NR, NF, $1}' attention/qkv_rms_norm_rope_cache_with_kscale/tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.csv
```

结果：所有行均为 57 列。

### 编译

```bash
bash build.sh -u --opapi --soc=ascend950 --ops=qkv_rms_norm_rope_cache_with_kscale -j8 --noexec
```

结果：`transformer_op_api_ut` 编译通过。

### 运行

```bash
ASCEND_SLOG_PRINT_TO_STDOUT=0 \
ASCEND_GLOBAL_LOG_LEVEL=3 \
BUILD_PATH=$(pwd)/build \
./build/tests/ut/framework_normal/op_api/transformer_op_api_ut
```

结果：

```text
[==========] Running 57 tests from 1 test suite.
[----------] 57 tests from QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_CSV/qkv_rms_norm_rope_cache_with_kscale_csv_test
[----------] 57 tests from QKV_RMS_NORM_ROPE_CACHE_WITH_KSCALE_CSV/qkv_rms_norm_rope_cache_with_kscale_csv_test (31 ms total)
[  PASSED  ] 57 tests.
```

## 5. 接口校验修复

本轮补充用例时发现 1 个接口校验缺口并修复：

| 问题 | 修复 |
|------|------|
| `v_cache` 的 `BlockNum/BlockSize` 未与 `k_cache` 做一致性校验 | 在 `ShapeCheck` 中新增 `vCacheShape.GetDim(0) != kCacheShape.GetDim(0) \|\| vCacheShape.GetDim(2) != kCacheShape.GetDim(2)` 校验，返回 `ACLNN_ERR_PARAM_INVALID` |

修复文件：`attention/qkv_rms_norm_rope_cache_with_kscale/op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp`

## 6. 说明与限制

- 本轮测试目标是 ACLNN op_api 接口校验，不验证 kernel 计算输出。
- `cu_seq_len` 内容值如 `cu_seq_len[0] == 0`、单调性、末尾等于 `T` 未在 ACLNN op_api 当前实现中读取校验，因此本轮不构造值级负例。
- `k_cache`、`v_cache`、`k_scale_cache` 是 inplace 参数。当前 UT 验证其作为输入传入且不暴露独立 cache out 参数；不执行真实 kernel 写回验证。
- 正常用例在无 device context 的 UT 环境中仍会打印 runtime context 相关错误日志，但 `GetWorkspaceSize` 返回码符合预期，测试全部通过。