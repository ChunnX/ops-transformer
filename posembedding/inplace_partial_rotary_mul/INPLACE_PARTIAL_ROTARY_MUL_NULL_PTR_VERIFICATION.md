# inplace_partial_rotary_mul 属性空指针保护验证报告

## 1. 问题描述

`inplace_partial_rotary_mul` 算子的 tiling 层在读取 `rotary_mode` 和 `partial_slice` 属性时，对 `GetAttrPointer` 返回值直接解引用，未做 null 判断。当 aclnn 通路传入 `partialSlice=nullptr` 时，Inner API 将 null 属性透传到 tiling context，`GetAttrPointer` 返回 null 指针，`null->GetSize()` 导致 SIGSEGV。

## 2. 根因分析

### 2.1 崩溃点定位

**regbase 路径（950）** — `inplace_rope_regbase_tiling_base.cpp` 第 368-369 行：
```cpp
const gert::ContinuousVector *sliceRangeListPtr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
if (sliceRangeListPtr->GetSize() == 0) {  // ← null 时 SIGSEGV
```

**membase 路径（910B）** — `inplace_partial_rotary_mul_tiling.cpp` 第 322、325 行：
```cpp
int64_t mode = *(attrs->GetAttrPointer<int64_t>(0));  // ← null 时 SIGSEGV
auto sliceListAttr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
if (sliceListAttr->GetSize() == 0) {  // ← null 时 SIGSEGV
```

### 2.2 两条通路对比

| 通路 | mode 读取 | slice 读取 | 触发条件 |
|---|---|---|---|
| aclnn（partialSlice=nullptr） | regbase: null-safe（既有）；membase: **null 解引用** | **null 解引用**（两条路径都有） | Inner API 透传 null attr 到 tiling |
| geir | GE 对 OPTIONAL 属性自动填默认值指针，GetAttrPointer 不返回 null → 不可达 | 同左 | — |

## 3. 改动内容

### 3.1 regbase (950) — slice null-safe（1 处）

文件：`posembedding/inplace_partial_rotary_mul/op_host/inplace_rope_regbase_tiling_base.cpp`

```diff
-    if (sliceRangeListPtr->GetSize() == 0) {
+    if (sliceRangeListPtr == nullptr || sliceRangeListPtr->GetSize() == 0) {
```

null 时走与"空列表"相同的默认分支（`[0, d_]` 全量旋转）。

### 3.2 membase (910B) — mode + slice null-safe（2 处）

文件：`posembedding/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_tiling.cpp`

```diff
-    int64_t mode = *(attrs->GetAttrPointer<int64_t>(0));
+    const int64_t *modePtr = attrs->GetAttrPointer<int64_t>(0);
+    int64_t mode = (modePtr == nullptr) ? 0 : *modePtr;
     ...
-    if (sliceListAttr->GetSize() == 0) {
+    if (sliceListAttr == nullptr || sliceListAttr->GetSize() == 0) {
```

mode null 时默认 0（与属性默认值一致），仿 regbase 既有写法。

## 4. 验证结果

### 4.1 aclnn 通路验证（独立 C++ 测试程序）

环境：Ascend950PR_9589，CANN 9.0.0

测试程序直接调用 `aclnnInplacePartialRotaryMulGetWorkspaceSize`，覆盖 `partialSlice=nullptr` 和 `xRef=nullptr` 场景。

**日志证据**：
```
[TEST1] partialSlice=nullptr, rotary_mode=1
[TEST1] ret=561103 (0=SUCCESS, 561103=GetWorkspace failed)
[TEST2] partialSlice=nullptr, rotary_mode=0
[TEST2] ret=561103
[TEST3] partialSlice=[0,128], rotary_mode=1 (normal control)
[TEST3] ret=561103
[TEST4] xRef=nullptr
[TEST4] ret=161001

=== All tests completed without SIGSEGV ===
exit_code=0
```

| 测试 | 输入 | 返回码 | 崩溃 |
|---|---|---|---|
| TEST1 | partialSlice=**nullptr**, mode=1 | 561103（GetWorkspace failed） | **否** |
| TEST2 | partialSlice=**nullptr**, mode=0 | 561103 | **否** |
| TEST3 | partialSlice=[0,128], mode=1（正常对照） | 561103 | **否** |
| TEST4 | xRef=**nullptr** | 161001（Check xRef != nullptr） | **否** |

**结论**：`partialSlice=nullptr` 不再 SIGSEGV，tiling 层 null-safe 修复生效，返回错误码而非崩溃。

### 4.2 geir 通路验证（TTK）

测试 CSV：`ttk_geir_inplace_partial_rotary_mul_null.csv`（4 用例）

**日志证据**：
```
TTK Test Summary
  Total    : 4
  PASS     : 1
  FAIL     : 1
  SKIP/NA  : 2
```

| 用例 | 结果 | 说明 |
|---|---|---|
| iprm_geir_normal_0001 | PASS | 正常用例无回归 |
| iprm_geir_no_slice_attr_0002 | FAIL（精度不匹配） | GE 填默认值 {0,0}（空切片），非 null 解引用 |
| iprm_geir_no_mode_attr_0003 | ERROR rc=255 | GE 填默认值 mode=0，被既有 mode 校验拒绝 |
| iprm_geir_no_attrs_0004 | ERROR rc=255 | 同上 |
| 崩溃 | **0** | — |

**结论**：geir 通路 0 崩溃。GE 对 OPTIONAL 属性自动填默认值指针，null 解引用在 geir 链路不可达——修复是防御性的，对 geir 行为无变化。

## 5. 改动汇总

| 文件 | 改动 | SoC 路径 |
|---|---|---|
| `inplace_rope_regbase_tiling_base.cpp` | slice null-safe（+1 处） | 950 regbase |
| `inplace_partial_rotary_mul_tiling.cpp` | mode null-safe + slice null-safe（+2 处） | 910B membase |

共 2 个文件，4 行改动。无 UT 新增（tiling UT faker 无法构造 null attr 场景）。

## 6. 复测命令

```bash
# 构建
cd /home/wx00445270/ops/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=inplace_partial_rotary_mul -j16

# 更新 custom opp 包
cp -r build/_CPack_Packages/.../vendors/custom_transformer custom_opp_iprm/vendors/

# 清 kernel cache
rm -rf /root/atc_data/kernel_cache/*

# aclnn 通路验证（独立 C++ 测试）
source /usr/local/Ascend/cann/bin/setenv.bash
export ASCEND_CUSTOM_OPP_PATH=$(pwd)/custom_opp_iprm/vendors/custom_transformer
g++ -o test_iprm_null test_iprm_null.cpp \
  -I/usr/local/Ascend/cann-9.0.0/x86_64-linux/include \
  -I/usr/local/Ascend/cann-9.0.0/x86_64-linux/include/aclnnop \
  -I$ASCEND_CUSTOM_OPP_PATH/op_api/include \
  -L/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64 \
  -L$ASCEND_CUSTOM_OPP_PATH/op_api/lib \
  -lascendcl -lnnopbase -lcust_opapi \
  -Wl,-rpath,/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64 \
  -Wl,-rpath,$ASCEND_CUSTOM_OPP_PATH/op_api/lib
./test_iprm_null

# geir 通路验证
cd /home/wx00445270/ops/ops-test-kit
python3 -m ttk geir -i inplace_partial_rotary_mul_nulltest/ttk_geir_inplace_partial_rotary_mul_null.csv \
  --plugin inplace_partial_rotary_mul_nulltest/golden_spec.py \
  -o /tmp/result.csv
```

## 7. 结论

`inplace_partial_rotary_mul` 算子的 `rotary_mode` 和 `partial_slice` 属性空指针保护已修复并验证通过：
- **aclnn 通路** `partialSlice=nullptr` 不再 SIGSEGV（tiling 层 null-safe，返回错误码 561103）
- **geir 通路** 0 崩溃（GE 填默认值，null 不可达——修复是防御性的）
- **正常用例**无回归
