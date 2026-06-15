# 任务 7：Tiling 实现 + UT 验证

> 在 arch35 tiling 中为 A16W4 pergroup ND 场景生成正确的 tiling 参数，采用 TDD 方式。

## 1. 任务目标

在现有 `GroupedWeightQuantBatchMatmulTiling` 模板中新增 A16W4 PerGroup ND 数据流支持，使 tiling 能正确：
1. 识别 pergroup 场景（`IsA16W4NDPergroup()`）
2. 校验 pergroup 特有约束（groupSize ∈ {32,64,128,256}、K%gs==0、scale shape 匹配）
3. 推导 groupSize（从 antiquantScale shape）
4. 生成正确的 tiling key（`W_QUANT_TYPE=WQGMM_PER_GROUP`）
5. 填充 tiling data（`GMMWeightQuantParam.groupSize` 等字段）

## 2. 参考文档

| 文档 | 路径 | 用途 |
|------|------|------|
| Tiling 设计 | `.workflow/raw/design/03_tiling_design.md` | **核心参考**：拦截校验、切分方案、tiling key |
| 交付件清单 | `.workflow/raw/design/01_deliverables.md` | 修改文件列表 |
| 平台无关方案 | `.workflow/raw/design/02_platform_independent_design.md` | 地址偏移公式 |
| SRS 实现方案 | `.workflow/raw/development/gmm_a16w4_pergroup_nd_srs.md` | 总体方案 |
| 测试用例设计 | `.workflow/raw/design/05_test_case_design.md` | T-01~T-44, TW-01~TW-20 |

## 3. 现有代码结构分析

### 3.1 Tiling 类

- **头文件**: `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h`
  - 类 `GroupedWeightQuantBatchMatmulTiling`，已有 `IsA16W4ND()` 判断
  - 已有 `groupSize_` 成员变量（`uint32_t`）
  - 已有 `SetAntiquantGroupSize()` 和 `CheckGroupSize()` 函数

- **实现文件**: `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp`
  - `SetTiling()` 主流程：AnalyzeAttr → CalcResplitTiling → SetBaseTiling → SetMatMulTiling → SetTilingKey → SetCustomParam
  - `SetTilingKey()` 当前仅设 `PER_CHANNEL` 和 `MX`，无 `PER_GROUP`
  - `CheckGroupSize()` 当前仅支持 S8S4 的 128/192/256/512
  - `SetAntiquantGroupSize()` 当前仅处理 3 维（单单单 perchannel）和 4 维（MX）scale

### 3.2 Tiling Key 注册

- **文件**: `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h`
  - 使用 `ASCENDC_TPL_ARGS_SEL` 宏注册模板参数组合
  - 已有 `WQGMM_PER_CHANNEL`、`WQGMM_PER_TENSOR` 等枚举
  - 需确认 `WQGMM_PER_GROUP` 枚举值是否已定义

### 3.3 现有 UT 结构

- **A16W8/A8W4 weight quant UT**: `test_grouped_weight_quant_arch35_tiling.cpp`
  - 使用内联 `TEST_F` 方式，每个用例手动构造 `TilingContextPara`
  - 已有 10 个用例（A16W4 ND perchannel、A8W4 NZ 等）

- **CSV 驱动 UT 参考**: `test_grouped_quant_arch35_tiling.cpp`
  - 使用 `testing::TestWithParam` + CSV 解析
  - CSV 格式：24 列（socVersion, caseName, enable, prefix, coreNum, m, k, n, groupNum, ...）
  - 参数类 `GroupedQuantArch35TilingTestParam` 包含 `Prepare()`、`InvokeTilingFunc()`、`Test()`

## 4. 执行计划

### 阶段 7.1：UT 测试框架搭建 + 用例编写

#### 4.1.1 新建 CSV 文件

**文件**: `gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_pergroup_tiling.csv`

**CSV 列定义**（26 列，在现有 24 列基础上新增 2 列）：

| 列号 | 列名 | 类型 | 说明 |
|------|------|------|------|
| 0 | socVersion | string | Ascend950 |
| 1 | caseName | string | 用例名 |
| 2 | enable | bool | 是否启用 |
| 3 | prefix | string | 参数化名称前缀 |
| 4 | coreNum | int | AIC 核数（-1=默认32） |
| 5 | m | int | M 维度 |
| 6 | k | int | K 维度 |
| 7 | n | int | N 维度 |
| 8 | groupNum | int | E（expert 数）或 tensor 数 |
| 9 | transposeX | bool | x 转置（固定 false） |
| 10 | transposeWeight | bool | weight 转置 |
| 11 | groupType | int | -1（多多多）或 0（单单单） |
| 12 | weightFormat | string | ND |
| 13 | xDtype | string | FLOAT16/BFLOAT16 |
| 14 | weightDtype | string | INT4 |
| 15 | antiquantScaleDtype | string | FLOAT16/BFLOAT16 |
| 16 | antiquantOffsetDtype | string | FLOAT16/BFLOAT16/NULL |
| 17 | biasDtype | string | FLOAT16/BFLOAT16/FLOAT/NULL |
| 18 | yDtype | string | FLOAT16/BFLOAT16 |
| 19 | hasBias | bool | 是否有 bias |
| 20 | hasAntiquantOffset | bool | 是否有 offset |
| 21 | groupSize | int | 32/64/128/256 |
| 22 | result | bool | 预期结果（true=通过，false=拦截） |
| 23 | blockDim | int | 预期 blockDim |
| 24 | tilingKey | int | 预期 tiling key |
| 25 | tilingData | string | 预期 tiling data（空格分隔） |

#### 4.1.2 新建测试代码

**文件**: `gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_pergroup_tiling.cpp`

**关键设计**：
- 参考 `test_grouped_quant_arch35_tiling.cpp` 的 CSV 解析模式
- 参数类 `GroupedWeightQuantPergroupArch35TilingTestParam`
- `InvokeTilingFunc()` 中根据 `groupType` 和 `groupSize` 构造 pergroup 特有的 shape：
  - 单单单：antiquantScale shape = `(E, K/gs, N)`，antiquantOffset shape = `(E, K/gs, N)` 或空
  - 多多多：antiquantScale shape = `(K/gs, N)`，antiquantOffset shape = `(K/gs, N)` 或空
- 调用 `GroupedWeightQuantBatchMatmulTiling::SetTiling()` 并验证结果

#### 4.1.3 用例覆盖

| 类别 | 数量 | 覆盖内容 |
|------|------|----------|
| 正常通过-单单单 | ~12 | gs=32/64/128/256 × 转置/非转置 × 有/无 offset × 有/无 bias（部分组合） |
| 正常通过-多多多 | ~12 | gs=32/64/128/256 × 转置/非转置 × 有/无 offset × 有/无 bias（部分组合） |
| 正常通过-边界 | ~6 | M=1, K=gs, N=2, E=1, E=大 等 |
| 异常拦截-dtype | ~7 | T-01~T-07 |
| 异常拦截-dim | ~8 | T-08~T-15 |
| 异常拦截-shape | ~6 | T-16~T-21 |
| 异常拦截-其他 | ~9 | T-22~T-44（部分） |
| Tiling 白盒 | ~10 | TW-01~TW-20（部分，验证 groupSize 推导和 tiling key） |

### 阶段 7.2：Tiling 代码实现

#### 4.2.1 修改清单

| 序号 | 文件 | 修改内容 | 参考文档 |
|------|------|----------|----------|
| 1 | `.h` | 新增 `bool IsA16W4NDPergroup() const;` 声明 | 03 §2.1 |
| 2 | `.cpp` | 实现 `IsA16W4NDPergroup()`：`return IsA16W4ND() && groupSize_ > 0;` | 03 §2.1 |
| 3 | `.cpp` | 修改 `CheckGroupSize()`：A16W4 ND 场景支持 32/64/128/256 | 03 §2.2 |
| 4 | `.cpp` | 修改 `SetAntiquantGroupSize()`：处理多多多 2 维 scale `(G_i, N_i)` | 03 §1.2 |
| 5 | `.cpp` | 修改 `SetTilingKey()`：`IsA16W4NDPergroup()` 时设 `PER_GROUP` | 03 §5.1 |
| 6 | `.cpp` | 修改 `CheckTensorDimSingleXSingleWeightSingleY()`：pergroup 时 scale 期望 3 维 | 03 §2.3 |
| 7 | `.cpp` | 修改 `CheckTensorDimMultiXMultiWeightMultiY()`：pergroup 时 scale 期望 2 维 | 03 §2.3 |
| 8 | `.cpp` | 修改 `CheckTensorShape()`：pergroup scale/offset 的 G 轴和 N 轴校验 | 03 §2.4 |
| 9 | `weight_quant_tiling_key.h` | 新增 8 个 `ASCENDC_TPL_ARGS_SEL` 组合 | 03 §5.2 |

#### 4.2.2 关键代码修改点

**`CheckGroupSize()` 扩展**：
```cpp
bool GroupedWeightQuantBatchMatmulTiling::CheckGroupSize(const gert::TilingContext *context) const
{
    if (IsA16W4NDPergroup()) {
        // A16W4 pergroup: 支持 32/64/128/256
        OP_CHECK_IF(
            groupSize_ != 32u && groupSize_ != 64u && groupSize_ != 128u && groupSize_ != 256u,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupSize",
                std::to_string(groupSize_),
                "A16W4 pergroup: groupSize must be 32/64/128/256"),
            return false);
        return true;
    }
    if (xDType_ != ge::DT_INT8 || weightDtype_ != ge::DT_INT4) {
        return true;
    }
    // 原有 S8S4 逻辑...
}
```

**`SetAntiquantGroupSize()` 扩展**：
```cpp
bool GroupedWeightQuantBatchMatmulTiling::SetAntiquantGroupSize(const gert::TilingContext *context)
{
    auto antiquantScale = context->GetDynamicInputTensor(ANTIQUANT_SCALE_IDX, 0);
    auto antiquantScaleShape = antiquantScale->GetStorageShape();
    int64_t antiquantScaleDimNum = antiquantScaleShape.GetDimNum();

    if (antiquantScaleDimNum == 3) {
        // 单单单 pergroup/perchannel: (E, G, N) 或 (E, N, K)
        int64_t groupNum = transB_ ? antiquantScaleShape.GetDim(antiquantScaleDimNum - 1)
                                   : antiquantScaleShape.GetDim(antiquantScaleDimNum - 2);
        // ... 原有逻辑
        groupSize_ = groupNum > 0 ? kSize_ / static_cast<uint64_t>(groupNum) : 0;
    } else if (antiquantScaleDimNum == 2 && IsA16W4ND()) {
        // 多多多 pergroup: (G_i, N_i)
        uint64_t scaleDim0 = antiquantScaleShape.GetDim(0);  // G_i
        if (scaleDim0 > 1 && kSize_ % scaleDim0 == 0) {
            groupSize_ = kSize_ / scaleDim0;
        } else {
            groupSize_ = 0;  // perchannel
        }
    } else if (antiquantScaleDimNum == ANTIQUANT_SCALE_DIM_NUM) {
        // MX 4维: 原有逻辑
    }

    OP_CHECK_IF(!CheckGroupSize(context), ...);
    return true;
}
```

**`SetTilingKey()` 扩展**：
```cpp
void GroupedWeightQuantBatchMatmulTiling::SetTilingKey(gert::TilingContext *context)
{
    // ... 原有逻辑

    if (antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
        tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::MX);
    } else if (IsA16W4NDPergroup()) {
        tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::PER_GROUP);
    } else {
        tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::PER_CHANNEL);
    }

    // ... 原有逻辑
}
```

### 阶段 7.3：UT 验证 + 迭代

#### 4.3.1 验证步骤

1. **编译**：
   ```bash
   bash build.sh -u --ophost --soc=ascend950 --ops=grouped_matmul -j8 --noexec
   ```

2. **运行 UT**：
   ```bash
   ASCEND_SLOG_PRINT_TO_STDOUT=0 ASCEND_GLOBAL_LOG_LEVEL=0 BUILD_PATH=$(pwd)/build \
     ./build/tests/ut/framework_normal/op_host/transformer_op_host_ut
   ```

3. **指定用例运行**（调试时）：
   ```bash
   ASCEND_SLOG_PRINT_TO_STDOUT=1 BUILD_PATH=$(pwd)/build \
     ./build/tests/ut/framework_normal/op_host/transformer_op_host_ut \
     --gtest_filter="*pergroup*"
   ```

4. **回归验证**：确保原有 A16W8/A8W4 用例全部通过

#### 4.3.2 迭代策略

- 如果 tiling key 不匹配：先获取实际 key，更新 CSV 中的预期值
- 如果 shape 校验失败：检查 `CheckTensorDim*` 和 `CheckTensorShape` 的分支逻辑
- 如果 groupSize 推导错误：检查 `SetAntiquantGroupSize` 的维度解析

## 5. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| tiling key 编码值不确定 | 预期 tilingKey 无法预先确定 | 阶段 7.2 实现后先跑一个用例获取实际 key，再批量更新 CSV |
| CSV 格式扩展兼容性 | 新增列可能与现有解析不兼容 | 新建独立 CSV 文件和测试类，不影响现有测试 |
| 多多多 2 维 scale 与 perchannel 区分 | 可能误判 | 通过 `IsA16W4ND()` + scale dim + `scaleDim0 > 1` 组合判断 |
| 原有 A16W8 perchannel 用例回归 | 修改可能破坏现有功能 | 所有修改通过 `IsA16W4NDPergroup()` 隔离，阶段 7.3 回归验证 |

## 6. 交付件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `test_grouped_weight_quant_arch35_pergroup_tiling.csv` | 新增 | CSV 用例数据（~60 条） |
| `test_grouped_weight_quant_arch35_pergroup_tiling.cpp` | 新增 | CSV 驱动测试代码 |
| `grouped_weight_quant_batch_matmul_tiling.h` | 修改 | 新增 `IsA16W4NDPergroup()` 声明 |
| `grouped_weight_quant_batch_matmul_tiling.cpp` | 修改 | 6 个修改点 |
| `weight_quant_tiling_key.h` | 修改 | 8 个新 tiling key 组合 |

## 7. 验收标准

- [ ] 编译通过（`build.sh` 无错误）
- [ ] 新增 pergroup UT 全部通过（~60 个用例）
- [ ] 原有 A16W8/A8W4 weight quant UT 全部通过（回归验证）
- [ ] tiling key 正确生成 `WQGMM_PER_GROUP` 类型
- [ ] groupSize 正确推导（32/64/128/256）
- [ ] 异常输入正确拦截（非法 groupSize、shape 不匹配等）
