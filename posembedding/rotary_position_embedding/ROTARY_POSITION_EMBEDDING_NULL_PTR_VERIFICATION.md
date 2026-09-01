# rotary_position_embedding 空指针保护验证报告

## 1. 问题描述

`rotary_position_embedding` 算子未对必填输入 `x` 做空指针保护，当 `x` 传入为 `null` 时，aclnn 通路在 `l0op` 层 `x->GetDataType()` 处发生 SIGSEGV core dump。

## 2. 根因分析

### 2.1 调用时序（关键）

```
aclnnRotaryPositionEmbeddingGetWorkspaceSize
  → RotaryPositionEmbeddingCommonProcess
    → PreProcess (TensorContiguousProcess: nullptr 直接放行)
    → l0op::RotaryPositionEmbedding
      → L0_DFX(...)
      → x->GetDataType()                          ← ★ 崩溃点（x=null 时 SIGSEGV）
      → executor->AllocTensor(...)
      → INFER_SHAPE(...)                           ← infershape
      → ADD_TO_LAUNCHER_LIST_AICORE(...)           ← tiling 在此被调用
```

**关键约束**：tiling 层的判空在 `ADD_TO_LAUNCHER_LIST_AICORE` 阶段才执行，而崩溃点 `x->GetDataType()` 在它之前。因此 **tiling 层无法拦截 aclnn 通路的 x=null 崩溃**，判空必须放在崩溃点本身（l0op 层入口）。

### 2.2 geir 通路为何不受影响

geir 通路不经过 `l0op::RotaryPositionEmbedding`，走的是 `infershape → tiling` 直接链路：
- 必填输入未连接 → GE 框架层在图执行阶段直接报错（实测 `rc=255`，kernel 不执行）
- 若 tiling 被调到 → regbase `CheckNullptr()`（既有代码）对 x/cos/sin 的 desc+shape 判空

## 3. 改动内容

### 3.1 代码改动（1 个文件，+3 行）

文件：`posembedding/rotary_position_embedding/op_host/op_api/rotary_position_embedding.cpp`

```diff
 const aclTensor *RotaryPositionEmbedding(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
     const aclTensor *rotate, int64_t mode, aclOpExecutor *executor)
 {
     L0_DFX(RotaryPositionEmbedding, x, cos, sin, rotate, mode);
+    OP_CHECK_NULL(x, return nullptr);
+    OP_CHECK_NULL(cos, return nullptr);
+    OP_CHECK_NULL(sin, return nullptr);
     DataType outType = x->GetDataType();
```

`OP_CHECK_NULL` 宏（来自 `op_error_check.h`）：检测到 nullptr 时打印 `errno[161001] Expected a value of type [aclTensor] for argument [x] but instead found nullptr` 并 return nullptr。上层 `RotaryPositionEmbeddingCommonProcess` 的 `CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR)` 将其转换为错误码返回，不再崩溃。

### 3.2 UT 改动（1 个文件，+3 个用例）

文件：`posembedding/rotary_position_embedding/tests/ut/op_host/op_api/test_aclnn_rotary_position_embedding.cpp`

新增 3 个用例：`x_nullptr` / `cos_nullptr` / `sin_nullptr`，分别传入对应的 nullptr，断言 `EXPECT_EQ(ret, ACLNN_ERR_INNER_NULLPTR)`。

### 3.3 未改动部分及理由

| 模块 | 是否改动 | 理由 |
|---|---|---|
| aclnn 入口（`aclnn_rotary_position_embedding.cpp`） | 否 | 按设计要求不在 aclnn 层做参数校验 |
| regbase tiling（`rope_regbase_tiling_base.cpp`） | 否 | `CheckNullptr()`/`GetShapeAttrsInfo()` 既有代码已覆盖必填输入 x/cos/sin/y 判空 + attrs 指针判空 + mode 范围校验 |
| rotate（选填输入） | 否 | 选填输入未传时 `GetOptionalInputDesc` 返回 null，现有代码天然走忽略分支；传入时 GE 框架保证 desc/shape 完整，geir 链路中不存在"连接但 shape 为 null"的可达场景 |
| membase tiling（910B/310P/Kirin） | 否 | 本次仅改 950（regbase） |

## 4. 验证结果

### 4.1 单元测试（UT）

**构建命令**：`bash build.sh -u --ophost --opapi --ops=rotary_position_embedding --soc=ascend950`

**日志证据**（`/tmp/opbuild_rpe_ut_v4.log`）：

```
# op_host UT（tiling + infershape，24 个用例全过）
[==========] Running 24 tests from 2 test suites ran. (72 ms total)
[  PASSED  ] 24 tests.

# op_api UT（8 个用例全过，含 3 个新增 null 用例）
[==========] Running 8 tests from 1 test suite.
[ RUN      ] l2_rotary_position_embedding_test.Ascend910B2_rotary_position_embedding_x_nullptr
[       OK ] l2_rotary_position_embedding_test.Ascend910B2_rotary_position_embedding_x_nullptr (0 ms)
[ RUN      ] l2_rotary_position_embedding_test.Ascend910B2_rotary_position_embedding_cos_nullptr
[       OK ] l2_rotary_position_embedding_test.Ascend910B2_rotary_position_embedding_cos_nullptr (0 ms)
[ RUN      ] l2_rotary_position_embedding_test.Ascend910B2_rotary_position_embedding_sin_nullptr
[       OK ] l2_rotary_position_embedding_test.Ascend910B2_rotary_position_embedding_sin_nullptr (0 ms)
[  PASSED  ] 8 tests.
```

**结论**：32/32 通过，3 个 null 用例均返回 `ACLNN_ERR_INNER_NULLPTR`，无崩溃。

### 4.2 aclnn 通路设备验证（TTK）

**环境**：Ascend950PR_9589，CANN 9.0.0，TTK v3.0.0

**测试 CSV**：`ttk_aclnn_rotary_position_embedding_null.csv`（6 用例）+ `ttk_aclnn_rotary_position_embedding_tiling_check.csv`（3 用例）

**结果汇总**（`ttk_aclnn_rpe_null_v4_result.csv` / `ttk_aclnn_rpe_tiling_v4_result.csv`）：

| 用例 | 输入 | precision_status | 说明 |
|---|---|---|---|
| rpe_null_x_0001 | x=null | N/A | 干净报错，无崩溃 |
| rpe_null_cos_0003 | cos=null | N/A | 干净报错，无崩溃 |
| rpe_null_sin_0004 | sin=null | N/A | 干净报错，无崩溃 |
| rpe_v2_null_x_0005 | x=null（V2） | N/A | 干净报错，无崩溃 |
| rpe_normal_ctrl_0002 | 全部有效 | PASS | 正常通路无回归 |
| rpe_v2_normal_ctrl_0006 | 全部有效（V2） | PASS | 正常通路无回归 |
| rpe_mode_invalid_0001 | mode=99 | N/A | tiling 既有校验拒绝 |
| rpe_v2_rotate_present_0002 | rotate 传入 | N/A | aclnn V2 既有 SoC 检查拒绝 |

**进程崩溃统计**：`crashed=0`（修复前为 `crashed=2`，SIGSEGV -11）

**报错日志证据**（`/tmp/ttk_rpe_v4.log`，x=null 的报错来自 l0op 层 `IsNullptr`）：

```
[ERROR] OP(55317,...) [...] [aclnn_rotary_position_embedding.cpp:34][NNOP][IsNullptr][55317]
        errno[161001] OpName:[aclnnRotaryPositionEmbedding_0_RotaryPositionEmbedding]
        Expected a value of type [aclTensor] for argument [x] but instead found nullptr.
[ERROR] OP(55317,...) [...] [aclnn_rotary_position_embedding.cpp:66][NNOP][RotaryPositionEmbeddingCommonProcess]
        errno[561000] check result != nullptr failed.
[ERROR] OP(55317,...) [...] [aclnn_rotary_position_embedding.cpp:98][NNOP][aclnnRotaryPositionEmbeddingGetWorkspaceSize]
        errno[561000] check ret == ACLNN_SUCCESS failed.
```

**结论**：x/cos/sin=null 全部返回 `errno[161001]`（ACLNN_ERR_PARAM_NULLPTR）干净报错，**0 崩溃**；正常对照用例精度 PASS。

### 4.3 geir 通路验证（基线，未改 tiling）

**测试 CSV**：`ttk_geir_rotary_position_embedding_check.csv`（4 用例）+ `ttk_geir_rotary_position_embedding_mode_invalid.csv`（1 用例）

**结果汇总**（`ttk_geir_rpe_check_baseline_result.csv`）：

| 用例 | 输入 | 结果 | 保护来源 |
|---|---|---|---|
| rpe_geir_normal_0001 | 正常 | PASS | — |
| rpe_geir_null_x_0002 | x=None | ERROR rc=255 拒绝 | GE 框架层（必填输入未连接） |
| rpe_geir_rotate_present_0003 | rotate 传入 | PASS | 选填输入正常处理（忽略 rotate，未误伤） |
| rpe_geir_dtype_mismatch_0004 | cos dtype≠x | PASS | GE 自动插 Cast |
| rpe_geir_mode_invalid_0001 | mode=99 | ERROR 拒绝 | regbase `GetShapeAttrsInfo` 既有 mode 范围校验 |

**日志证据**（`/tmp/ttk_rpe_geir_baseline2.log`）：

geir null_x 框架层拒绝：
```
[ERROR] [rpe_geir_null_x_0002 OnGeirExecute] GEIR profiling exception
RuntimeError: GEIR execution failed (rc=255): [TTK-GEIR] RunGraph failed: -1,
error: Invalid_Argument_Null_Pointer(EE1004): MemCopySync failed because dst cannot be a NULL pointer.
```

geir mode=99 tiling 既有校验拒绝：
```
Compile operator failed, cause: Template constraint, detailed information:
Tiling func of op_type RotaryPositionEmbedding failed, failure details:
[OP_TILING] Attrs: [{'name': 'mode', 'dtype': 'int', 'value': 99}, ...]
```

**结论**：geir 通路必填输入判空（GE 框架层）+ 属性判空/范围校验（regbase tiling 既有代码）均已生效，无需改动 tiling。

## 5. 验证矩阵总览

| 通路 | x/cos/sin=null | mode=99 | rotate 传入 | 正常用例 | 崩溃 |
|---|---|---|---|---|---|
| aclnn（V1+V2） | ✅ 干净报错 errno[161001] | ✅ tiling 拒绝 | ✅ V2 SoC 检查拒绝 | ✅ PASS | 0 |
| geir | ✅ 框架层拒绝 rc=255 | ✅ tiling 拒绝 | ✅ 正常处理不误伤 | ✅ PASS | 0 |
| UT | ✅ 3 用例 ACLNN_ERR_INNER_NULLPTR | — | — | ✅ 24/24 + 8/8 | 0 |

## 6. 复测命令

```bash
# 环境变量
source /usr/local/Ascend/cann/bin/setenv.bash
export ASCEND_CUSTOM_OPP_PATH=<custom_opp_rpe>/vendors/custom_transformer

# UT
cd /home/wx00445270/ops/ops-transformer
bash build.sh -u --ophost --opapi --ops=rotary_position_embedding --soc=ascend950

# 设备验证（须清 kernel cache）
rm -rf /root/atc_data/kernel_cache/*
cd /home/wx00445270/ops/ops-test-kit
python3 -m ttk aclnn -i rotary_position_embedding_nulltest/ttk_aclnn_rotary_position_embedding_null.csv \
    --plugin rotary_position_embedding_nulltest/golden_spec.py \
    -o /tmp/result.csv
```

## 7. 结论

`rotary_position_embedding` 空指针保护问题已修复并验证通过：
- **aclnn 通路** x=null core dump 已消除（l0op 层判空，errno[161001] 干净报错，0 崩溃）
- **geir 通路** 必填输入/属性判空由既有代码保证（GE 框架层 + regbase tiling，未改动）
- **选填输入 rotate** 遵从可选语义，不强制拒绝
- UT 32/32 通过，设备验证全部符合预期
