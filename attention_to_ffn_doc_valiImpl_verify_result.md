# aclnnAttentionToFFN 文档校验实现验证报告

**算子路径**: `mc2/attention_to_ffn`
**硬件版本**: A3 (Atlas A3 训练系列产品/Atlas A3 推理系列产品，对应 NpuArch::DAV_2201，SocVersion: Ascend910_93)
**aclnn入口**: `aclnnAttentionToFFN`

---

## 第一步：文档限制清单

从 [aclnnAttentionToFFN.md](mc2/attention_to_ffn/docs/aclnnAttentionToFFN.md) 中提取的 A3 相关限制：

| 序号 | 参数名 | 限制类型 | 限制内容描述 |
|------|--------|---------|-------------|
| 1 | x/sessionId/microBatchId/layerId/expertIds/expertRankTable | 空指针 | 必选输入不能为空指针 |
| 2 | x | dtype | 数据类型支持FLOAT16、BFLOAT16 |
| 3 | sessionId | dtype | 数据类型支持INT32 |
| 4 | microBatchId | dtype | 数据类型支持INT32 |
| 5 | layerId | dtype | 数据类型支持INT32 |
| 6 | expertIds | dtype | 数据类型支持INT32 |
| 7 | expertRankTable | dtype | 数据类型支持INT32 |
| 8 | scalesOptional | dtype | 数据类型支持FLOAT32 |
| 9 | activeMaskOptional | dtype | 数据类型支持BOOL |
| 10 | x | format | 数据格式支持ND |
| 11 | sessionId | format | 数据格式支持ND |
| 12 | microBatchId | format | 数据格式支持ND |
| 13 | layerId | format | 数据格式支持ND |
| 14 | expertIds | format | 数据格式支持ND |
| 15 | expertRankTable | format | 数据格式支持ND |
| 16 | scalesOptional | format | 数据格式支持ND |
| 17 | activeMaskOptional | format | 数据格式支持ND |
| 18 | x | shape | 3维，shape为(X, Bs, H) |
| 19 | sessionId | shape | 1维，shape为(X,) |
| 20 | microBatchId | shape | 1维，shape为(X,) |
| 21 | layerId | shape | 1维，shape为(X,) |
| 22 | expertIds | shape | 3维，shape为(X, Bs, K) |
| 23 | expertRankTable | shape | 3维，shape为(L, moeExpertNum + sharedExpertNum, M) |
| 24 | scalesOptional | shape | 3维，shape为(L, moeExpertNum + sharedExpertNum, H) |
| 25 | activeMaskOptional | shape | 2维，shape为(X, Bs) |
| 26 | group | 属性值 | 字符串长度范围为[1, 128) |
| 27 | worldSize | 属性值 | 取值区间[2, 768] |
| 28 | ffnTokenInfoTableShape | 属性值 | 长度为3 |
| 29 | ffnTokenDataShape | 属性值 | 长度为5 |
| 30 | attnTokenInfoTableShape | 属性值 | 长度为3 |
| 31 | moeExpertNum | 属性值 | 取值范围(0, 1024] |
| 32 | quantMode | 属性值 | 仅支持0（非量化）、2（动态量化） |
| 33 | syncFlag | 属性值 | 仅支持0（同步）、1（异步） |
| 34 | ffnStartRankId | 属性值 | 取值范围为[0, worldSize) |
| 35 | X | shape变量 | 当前版本只支持X = 1 |
| 36 | H | shape变量 | 取值范围为[1024, 8192] |
| 37 | Bs | shape变量 | 取值范围为0 < Bs ≤ 512 |
| 38 | K | shape变量 | 取值范围0 < K ≤ 16，且0 < K ≤ moeExpertNum |
| 39 | L | shape变量 | 当前版本只支持L = 1 |
| 40 | M | shape变量 | 取值范围1 < M ≤ FFNWorkerNum * 2 + 1 |
| 41 | sharedExpertNum | shape变量 | 取值范围[0, 4] |
| 42 | scalesOptional | 空指针 | 非量化场景下必须传空指针，动态量化可传有效数据或空指针 |
| 43 | sessionId | shape值 | 取值范围为[0, attentionWorkerNum) |
| 44 | expertIds | shape值 | 取值区间为[0, moeExpertNum) |
| 45 | ffnTokenInfoTableShape | shape约束 | ffnTokenInfoTableShape[0] == ffnTokenDataShape[0] |
| 46 | ffnTokenInfoTableShape | shape约束 | ffnTokenInfoTableShape[1] == ffnTokenDataShape[1] |
| 47 | ffnTokenDataShape | shape约束 | ffnTokenDataShape[2] == Bs |
| 48 | attnTokenInfoTableShape | shape约束 | attnTokenInfoTableShape[0] == ffnTokenInfoTableShape[1] |
| 49 | attnTokenInfoTableShape | shape约束 | attnTokenInfoTableShape[1] == Bs |
| 50 | attnTokenInfoTableShape | shape约束 | attnTokenInfoTableShape[2] == ffnTokenDataShape[3] |
| 51 | attentionWorkerNum | shape约束 | 0 < attentionWorkerNum < worldSize |
| 52 | quantMode与scales | 一致性 | 非量化场景scales必须为nullptr，动态量化scales可传有效数据或空指针 |
| 53 | activeMaskOptional | 空指针 | 可传有效数据或空指针，传空指针时默认所有token参与通信 |
| 54 | expertRankTable dim1 | shape约束 | expertRankTable dim1 ∈ [moeExpertNum, moeExpertNum + 4] |

---

## 第二步：代码校验实现核对

### 2.1 op_api层校验（[aclnn_attention_to_ffn.cpp](mc2/attention_to_ffn/op_api/aclnn_attention_to_ffn.cpp)）

op_api层仅做轻量校验，核心校验委托给`aclnnInnerAttentionToFFNGetWorkspaceSize`（预编译库，不可见）。

| 序号 | 文档限制内容 | 限制类型 | 代码是否实现 | 实现位置 | 实现是否合适 | UT是否覆盖 | 备注 |
|------|------------|---------|------------|---------|------------|-----------|------|
| 1 | x/sessionId/microBatchId/layerId/expertIds/expertRankTable不能为空指针 | 空指针 | 是 | `CheckNullStatus` → `OP_CHECK_NULL` | 合适 | 否 | op_api UT无空指针测试 |
| 2 | group不能为空指针 | 空指针 | 是 | `CheckNullStatus` → `if ((group == nullptr) \|\| (strnlen(group, HCCL_GROUP_NAME_MAX) == 0))` | 合适 | 否 | 无UT覆盖 |
| 3 | group字符串长度[1, 128) | 属性值 | 是 | `CheckParams` → `strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX` | 合适 | 否 | 无UT覆盖 |
| 4 | x dtype为FLOAT16/BFLOAT16 | dtype | 否 | op_api层未校验 | 不合适，依赖inner库校验 | 否 | 应在op_api层前置校验 |
| 5 | sessionId/microBatchId/layerId/expertIds/expertRankTable dtype为INT32 | dtype | 否 | op_api层未校验 | 不合适 | 否 | 应在op_api层前置校验 |
| 6 | scalesOptional dtype为FLOAT32 | dtype | 否 | op_api层未校验 | 不合适 | 否 | 应在op_api层前置校验 |
| 7 | activeMaskOptional dtype为BOOL | dtype | 否 | op_api层未校验 | 不合适 | 否 | 应在op_api层前置校验 |
| 8 | x format为ND | format | 否 | op_api层未校验 | 不合适 | 否 | 应在op_api层前置校验 |
| 9 | x shape为3维 | shape | 否 | op_api层未校验 | 不合适 | 否 | 应在op_api层前置校验 |
| 10 | 其他所有shape/dtype/format/属性值校验 | 多种 | 否 | op_api层未校验 | 不合适 | 否 | inner库不可见，无法确认是否实现 |

### 2.2 op_tiling层校验（[attention_to_ffn_tiling.cpp](mc2/attention_to_ffn/op_host/op_tiling/attention_to_ffn_tiling.cpp)）

| 序号 | 文档限制内容 | 限制类型 | 代码是否实现 | 实现位置 | 实现是否合适 | UT是否覆盖 | 备注 |
|------|------------|---------|------------|---------|------------|-----------|------|
| 1 | moeExpertNum取值范围(0, 1024] | 属性值 | 是 | `CheckAndSetAttrs` → `OP_TILING_CHECK((moeExpertNum <= 0) \|\| (moeExpertNum > MOE_EXPERT_MAX_NUM))` | 合适 | 否 | 无异常UT |
| 2 | quantMode仅支持0/2 | 属性值 | 是 | `CheckAndSetAttrs` → `OP_TILING_CHECK((*quantModePtr < 0) \|\| (*quantModePtr > 2))` | 合适，但校验范围是[0,2]而非仅0/2 | 部分 | 允许quantMode=1(静态量化)通过，后续在`CheckInputAndSetTilingData`中拒绝 |
| 3 | syncFlag仅支持0/1 | 属性值 | 是 | `CheckAndSetAttrs` → `OP_TILING_CHECK((*syncFlagPtr != 0) && (*syncFlagPtr != 1))` | 合适 | 否 | 无异常UT |
| 4 | ffnStartRankId取值[0, worldSize) | 属性值 | 是 | `CheckAndSetAttrs` → `OP_TILING_CHECK((*ffnStartRankIdPtr >= *worldSizePtr) \|\| (*ffnStartRankIdPtr < 0))` | 合适 | 否 | 无异常UT |
| 5 | worldSize取值[2, 768] | 属性值 | 是 | `CheckAndSetAttrs` → `OP_TILING_CHECK((worldSize < 2) \|\| (worldSize > 768))` | 合适 | 否 | 无异常UT |
| 6 | ffnTokenInfoTableShape长度为3 | 属性值 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(ffnTokenInfoTableDimNum != 3)` | 合适 | 否 | 无异常UT |
| 7 | ffnTokenDataShape长度为5 | 属性值 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(ffnTokenDataDimNum != 5)` | 合适 | 否 | 无异常UT |
| 8 | attnTokenInfoTableShape长度为3 | 属性值 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(attnTokenDataDimNum != 3)` | 合适 | 否 | 无异常UT |
| 9 | ffnTokenInfoTableShape[0] == ffnTokenDataShape[0] | shape约束 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(ffnTokenInfoTableShape[0] != ffnTokenDataShape[0])` | 合适 | 否 | 无异常UT |
| 10 | ffnTokenInfoTableShape[1] == ffnTokenDataShape[1] | shape约束 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(ffnTokenInfoTableShape[1] != ffnTokenDataShape[1])` | 合适 | 否 | 无异常UT |
| 11 | ffnTokenDataShape[2] == Bs | shape约束 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(ffnTokenDataShape[2] != bs)` | 合适 | 否 | 无异常UT |
| 12 | attnTokenInfoTableShape[0] == ffnTokenInfoTableShape[1] | shape约束 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(attnTokenDataShape[0] != ffnTokenInfoTableShape[1])` | 合适 | 否 | 无异常UT |
| 13 | attnTokenInfoTableShape[1] == Bs | shape约束 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(attnTokenDataShape[1] != bs)` | 合适 | 否 | 无异常UT |
| 14 | attnTokenInfoTableShape[2] == ffnTokenDataShape[3] | shape约束 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(attnTokenDataShape[2] != ffnTokenDataShape[3])` | 合适 | 否 | 无异常UT |
| 15 | attentionWorkerNum范围(0, worldSize) | shape约束 | 是 | `CheckAndSetShapeAttrs` → `OP_TILING_CHECK(attentionWorkerNum <= 0 \|\| attentionWorkerNum >= *worldSizePtr)` | 合适 | 否 | 无异常UT |
| 16 | x dtype为FLOAT16/BFLOAT16 | dtype | 是 | `CheckTensorDataType` → `OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16))` | 合适 | 否 | 无异常UT |
| 17 | sessionId/microBatchId/layerId/expertIds/expertRankTable dtype为INT32 | dtype | 是 | `CheckTensorDataType` → 逐个校验 `!= ge::DT_INT32` | 合适 | 否 | 无异常UT |
| 18 | scalesOptional dtype为FLOAT32 | dtype | 是 | `CheckTensorDataType` → `OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT)` | 合适 | 否 | 无异常UT |
| 19 | activeMaskOptional dtype为BOOL | dtype | 是 | `CheckTensorDataType` → `OP_TILING_CHECK(activeMaskDesc->GetDataType() != ge::DT_BOOL)` | 合适 | 否 | 无异常UT |
| 20 | x format不支持FRACTAL_NZ | format | 是 | `CheckTensorFormat` → `OP_TILING_CHECK(... == ge::FORMAT_FRACTAL_NZ)` | 合适，但仅校验不支持NZ，未校验必须为ND | 部分 | 文档说支持ND，代码仅拒绝NZ |
| 21 | sessionId/microBatchId/layerId/expertIds/expertRankTable format不支持FRACTAL_NZ | format | 是 | `CheckTensorFormat` → 逐个校验 | 合适，同上 | 部分 | 同上 |
| 22 | scalesOptional/activeMaskOptional format不支持FRACTAL_NZ | format | 是 | `CheckTensorFormat` → 逐个校验 | 合适，同上 | 部分 | 同上 |
| 23 | x shape为3维 | shape | 是 | `CheckTensorDim` → `OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != 3)` | 合适 | 是 | tiling UT有X=2异常测试 |
| 24 | sessionId/microBatchId/layerId shape为1维 | shape | 是 | `CheckTensorDim` → 逐个校验 `!= 1` | 合适 | 否 | 无异常UT |
| 25 | expertIds shape为3维 | shape | 是 | `CheckTensorDim` → `!= 3` | 合适 | 否 | 无异常UT |
| 26 | expertRankTable shape为3维 | shape | 是 | `CheckTensorDim` → `!= 3` | 合适 | 否 | 无异常UT |
| 27 | scalesOptional shape为3维 | shape | 是 | `CheckTensorDim` → `!= 3` | 合适 | 否 | 无异常UT |
| 28 | activeMaskOptional shape为2维 | shape | 是 | `CheckTensorDim` → `!= 2` | 合适 | 否 | 无异常UT |
| 29 | X只支持1 | shape变量 | 是 | `CheckTensorShapeAndSetTinglingData` → `OP_TILING_CHECK(xDim0 != 1)` | 合适 | 是 | tiling UT有MicroBatchNumInvalid测试 |
| 30 | Bs范围(0, 512] | shape变量 | 是 | `CheckTensorShapeAndSetTinglingData` → `OP_TILING_CHECK((xDim1 <= 0) \|\| (xDim1 > BS_MAX))` | 合适 | 是 | tiling UT有BsInvalid测试 |
| 31 | H范围[1024, 8192] | shape变量 | 是 | `CheckTensorShapeAndSetTinglingData` → `OP_TILING_CHECK((xDim2 < H_MIN) \|\| (xDim2 > H_MAX))` | 合适 | 否 | 无异常UT |
| 32 | K范围(0, min(16, moeExpertNum)] | shape变量 | 是 | `CheckTensorShapeAndSetTinglingData` → `OP_TILING_CHECK((expertIdsDim2 <= 0) \|\| (expertIdsDim2 > K_MAX) \|\| (expertIdsDim2 > moeExpertNum))` | 合适 | 是 | tiling UT有KInvalid测试 |
| 33 | L只支持1 | shape变量 | 是 | `CheckTensorShapeAndSetTinglingData` → `OP_TILING_CHECK(expertRankTableDim0 != 1)` | 合适 | 否 | 无异常UT |
| 34 | expertRankTable dim1 ∈ [moeExpertNum, moeExpertNum + 4] | shape约束 | 是 | `CheckTensorShapeAndSetTinglingData` → `OP_TILING_CHECK((expertRankTableDim1 < moeExpertNum) \|\| (expertRankTableDim1 > moeExpertNum + 4))` | 合适 | 否 | 无异常UT |
| 35 | sessionId/microBatchId/layerId/expertIds dim0 == X | shape约束 | 是 | `CheckTensorShapeAndSetTinglingData` → `OP_TILING_CHECK((xDim0 != sessionIdDim0) \|\| ...)` | 合适 | 否 | 无异常UT |
| 36 | expertIds dim1 == Bs | shape约束 | 是 | `CheckTensorShapeAndSetTinglingData` → `OP_TILING_CHECK(expertIdsDim1 != xDim1)` | 合适 | 否 | 无异常UT |
| 37 | scalesOptional shape为(L, moeExpertNum+sharedExpertNum, H) | shape | 是 | `CheckInputForScales` → 校验dim0/dim1/dim2 | 合适 | 否 | 无异常UT |
| 38 | activeMaskOptional shape为(X, Bs) | shape | 是 | `CheckInputForActiveMask` → 校验dim0/dim1 | 合适 | 否 | 无异常UT |
| 39 | 非量化场景scales必须为nullptr | 一致性 | 是 | `CheckInputAndSetTilingData` → `OP_TILING_CHECK((isScales && (quantMode == NO_SCALES)) \|\| ((!isScales) && (quantMode == STATIC_SCALES)))` | 合适 | 否 | 无异常UT |
| 40 | 静态量化不支持 | 属性值 | 是 | `CheckInputAndSetTilingData` → `OP_TILING_CHECK(quantMode == STATIC_SCALES)` | 合适 | 否 | 无异常UT |
| 41 | 通信窗口大小校验 | 环境约束 | 是 | `CalWinSize` + `OP_TILING_CHECK(neededSize > viableWindowSize)` | 合适 | 否 | 无异常UT |
| 42 | sessionId取值范围[0, attentionWorkerNum) | shape值 | 否 | 未校验sessionId的值域 | 不合适 | 否 | 值域校验应在kernel层或文档说明为外部保证 |
| 43 | expertIds取值范围[0, moeExpertNum) | shape值 | 否 | 未校验expertIds的值域 | 不合适 | 否 | 同上 |
| 44 | M取值范围(1, FFNWorkerNum*2+1] | shape变量 | 否 | 未校验expertRankTable dim2(M)的取值范围 | 不合适 | 否 | 缺失校验 |
| 45 | sharedExpertNum取值范围[0, 4] | shape变量 | 部分 | 通过expertRankTable dim1范围间接约束（dim1 ∈ [moeExpertNum, moeExpertNum+4]），但sharedExpertNum = dim1 - moeExpertNum，未显式校验 | 部分 | 否 | 间接约束，但未校验sharedExpertNum ≥ 0 |
| 46 | group有效性 | 通信域 | 是 | `CheckAndSetAttrs` → `OP_TILING_CHECK(groupPtr == nullptr)` | 合适 | 否 | 无异常UT |

---

## 第三步：UT覆盖情况

### 3.1 op_api层UT（[test_aclnn_attention_to_ffn.cpp](mc2/attention_to_ffn/tests/ut/op_api/test_aclnn_attention_to_ffn.cpp)）

**现有测试用例概览：**

| 测试类 | 用例名 | 覆盖场景 |
|--------|-------|---------|
| AclnnAttentionToFfnTest | TestAttentionToFfnNoQuant | 非量化+activeMask正向 |
| AclnnAttentionToFfnTest | TestAttentionToFfnQuant | 量化+scales+activeMask正向 |

**缺失的异常测试场景：**
- 无空指针测试（x/sessionId/microBatchId/layerId/expertIds/expertRankTable为nullptr）
- 无group空指针/空字符串/超长字符串测试
- 无dtype异常测试（x为INT32、sessionId为FLOAT16等）
- 无format异常测试
- 无shape维度异常测试
- 无属性值异常测试（worldSize/moeExpertNum/quantMode/syncFlag/ffnStartRankId越界）
- 无scalesOptional在非量化场景非空测试
- 无activeMaskOptional shape不匹配测试

### 3.2 op_host层UT（arch32: [test_attention_to_ffn_tiling.cpp](mc2/attention_to_ffn/tests/ut/op_host/arch32/test_attention_to_ffn_tiling.cpp)）

**现有测试用例概览：**

| 用例名 | 覆盖场景 | 类型 |
|--------|---------|------|
| NoQuantSyncNoActivateMask | 非量化+同步+无mask | 正向 |
| NoQuantAsyncNoActivateMask | 非量化+异步+无mask | 正向 |
| QuantSyncNoActivateMask | 量化+同步+无mask | 正向 |
| QuantAsyncNoActivateMask | 量化+异步+无mask | 正向 |
| NoQuantSyncActivateMask | 非量化+同步+有mask | 正向 |
| NoQuantAsyncActivateMask | 非量化+异步+有mask | 正向 |
| QuantSyncActivateMask | 量化+同步+有mask | 正向 |
| QuantAsyncActivateMask | 量化+异步+有mask | 正向 |
| MicroBatchNumInvalid | X != 1 | 异常 |
| BsInvalid | Bs > 512 | 异常 |
| KInvalid | K > 16 | 异常 |

**缺失的异常测试场景：**
- 无H范围越界测试（H < 1024 或 H > 8192）
- 无L != 1测试
- 无moeExpertNum越界测试（≤ 0 或 > 1024）
- 无quantMode异常测试（quantMode = 1 静态量化、quantMode > 2）
- 无syncFlag异常测试（syncFlag > 1）
- 无ffnStartRankId越界测试（≥ worldSize 或 < 0）
- 无worldSize越界测试（< 2 或 > 768）
- 无ffnTokenInfoTableShape/ffnTokenDataShape/attnTokenInfoTableShape长度异常测试
- 无shape约束不匹配测试（各shape间维度不一致）
- 无dtype异常测试
- 无format异常测试（FRACTAL_NZ）
- 无scalesOptional shape不匹配测试
- 无activeMaskOptional shape不匹配测试
- 无expertRankTable dim1越界测试
- 无M(expertRankTable dim2)越界测试
- 无attentionWorkerNum越界测试
- 无通信窗口大小越界测试

### 3.3 op_host层UT（infershape: [test_attention_to_ffn_infershape.cpp](mc2/attention_to_ffn/tests/ut/op_host/test_attention_to_ffn_infershape.cpp)）

仅有1个正向测试用例（Basic），无异常测试。

---

## 第四步：校验结果汇总表

| 序号 | 文档限制内容 | 限制类型 | 代码是否实现 | 实现位置 | 实现是否合适 | UT是否覆盖 | 备注 |
|------|------------|---------|------------|---------|------------|-----------|------|
| 1 | 必选输入不能为空指针 | 空指针 | 是 | op_api: `CheckNullStatus` | 合适 | 否 | 无空指针UT |
| 2 | group不能为空指针/空字符串 | 空指针 | 是 | op_api: `CheckNullStatus` | 合适 | 否 | 无UT |
| 3 | group字符串长度[1, 128) | 属性值 | 是 | op_api: `CheckParams` | 合适 | 否 | 无UT |
| 4 | x dtype为FLOAT16/BFLOAT16 | dtype | 是 | tiling: `CheckTensorDataType` | 合适，但op_api层未校验 | 否 | 应在op_api层前置校验 |
| 5 | sessionId/microBatchId/layerId/expertIds/expertRankTable dtype为INT32 | dtype | 是 | tiling: `CheckTensorDataType` | 合适，但op_api层未校验 | 否 | 应在op_api层前置校验 |
| 6 | scalesOptional dtype为FLOAT32 | dtype | 是 | tiling: `CheckTensorDataType` | 合适，但op_api层未校验 | 否 | 应在op_api层前置校验 |
| 7 | activeMaskOptional dtype为BOOL | dtype | 是 | tiling: `CheckTensorDataType` | 合适，但op_api层未校验 | 否 | 应在op_api层前置校验 |
| 8 | 所有输入format不支持FRACTAL_NZ | format | 是 | tiling: `CheckTensorFormat` | 部分，仅拒绝NZ未校验必须为ND | 否 | 文档说支持ND |
| 9 | x shape为3维 | shape | 是 | tiling: `CheckTensorDim` | 合适 | 是 | tiling UT覆盖 |
| 10 | sessionId/microBatchId/layerId shape为1维 | shape | 是 | tiling: `CheckTensorDim` | 合适 | 否 | 无异常UT |
| 11 | expertIds shape为3维 | shape | 是 | tiling: `CheckTensorDim` | 合适 | 否 | 无异常UT |
| 12 | expertRankTable shape为3维 | shape | 是 | tiling: `CheckTensorDim` | 合适 | 否 | 无异常UT |
| 13 | scalesOptional shape为3维 | shape | 是 | tiling: `CheckTensorDim` | 合适 | 否 | 无异常UT |
| 14 | activeMaskOptional shape为2维 | shape | 是 | tiling: `CheckTensorDim` | 合适 | 否 | 无异常UT |
| 15 | X只支持1 | shape变量 | 是 | tiling: `CheckTensorShapeAndSetTinglingData` | 合适 | 是 | tiling UT覆盖 |
| 16 | Bs范围(0, 512] | shape变量 | 是 | tiling: `CheckTensorShapeAndSetTinglingData` | 合适 | 是 | tiling UT覆盖 |
| 17 | H范围[1024, 8192] | shape变量 | 是 | tiling: `CheckTensorShapeAndSetTinglingData` | 合适 | 否 | 无异常UT |
| 18 | K范围(0, min(16, moeExpertNum)] | shape变量 | 是 | tiling: `CheckTensorShapeAndSetTinglingData` | 合适 | 是 | tiling UT覆盖 |
| 19 | L只支持1 | shape变量 | 是 | tiling: `CheckTensorShapeAndSetTinglingData` | 合适 | 否 | 无异常UT |
| 20 | expertRankTable dim1 ∈ [moeExpertNum, moeExpertNum+4] | shape约束 | 是 | tiling: `CheckTensorShapeAndSetTinglingData` | 合适 | 否 | 无异常UT |
| 21 | sessionId/microBatchId/layerId/expertIds dim0 == X | shape约束 | 是 | tiling: `CheckTensorShapeAndSetTinglingData` | 合适 | 否 | 无异常UT |
| 22 | expertIds dim1 == Bs | shape约束 | 是 | tiling: `CheckTensorShapeAndSetTinglingData` | 合适 | 否 | 无异常UT |
| 23 | scalesOptional shape为(L, moeExpertNum+sharedExpertNum, H) | shape | 是 | tiling: `CheckInputForScales` | 合适 | 否 | 无异常UT |
| 24 | activeMaskOptional shape为(X, Bs) | shape | 是 | tiling: `CheckInputForActiveMask` | 合适 | 否 | 无异常UT |
| 25 | moeExpertNum取值(0, 1024] | 属性值 | 是 | tiling: `CheckAndSetAttrs` | 合适 | 否 | 无异常UT |
| 26 | quantMode仅支持0/2 | 属性值 | 是 | tiling: `CheckAndSetAttrs` + `CheckInputAndSetTilingData` | 合适 | 否 | 无异常UT |
| 27 | syncFlag仅支持0/1 | 属性值 | 是 | tiling: `CheckAndSetAttrs` | 合适 | 否 | 无异常UT |
| 28 | ffnStartRankId取值[0, worldSize) | 属性值 | 是 | tiling: `CheckAndSetAttrs` | 合适 | 否 | 无异常UT |
| 29 | worldSize取值[2, 768] | 属性值 | 是 | tiling: `CheckAndSetAttrs` | 合适 | 否 | 无异常UT |
| 30 | ffnTokenInfoTableShape长度为3 | 属性值 | 是 | tiling: `CheckAndSetShapeAttrs` | 合适 | 否 | 无异常UT |
| 31 | ffnTokenDataShape长度为5 | 属性值 | 是 | tiling: `CheckAndSetShapeAttrs` | 合适 | 否 | 无异常UT |
| 32 | attnTokenInfoTableShape长度为3 | 属性值 | 是 | tiling: `CheckAndSetShapeAttrs` | 合适 | 否 | 无异常UT |
| 33 | ffnTokenInfoTableShape/ffnTokenDataShape/attnTokenInfoTableShape维度一致性 | shape约束 | 是 | tiling: `CheckAndSetShapeAttrs` | 合适 | 否 | 无异常UT |
| 34 | attentionWorkerNum范围(0, worldSize) | shape约束 | 是 | tiling: `CheckAndSetShapeAttrs` | 合适 | 否 | 无异常UT |
| 35 | 非量化场景scales必须为nullptr | 一致性 | 是 | tiling: `CheckInputAndSetTilingData` | 合适 | 否 | 无异常UT |
| 36 | 通信窗口大小校验 | 环境约束 | 是 | tiling: `CalWinSize` | 合适 | 否 | 无异常UT |
| 37 | sessionId取值范围[0, attentionWorkerNum) | shape值 | 否 | — | 不合适 | 否 | 缺失校验（值域校验） |
| 38 | expertIds取值范围[0, moeExpertNum) | shape值 | 否 | — | 不合适 | 否 | 缺失校验（值域校验） |
| 39 | M取值范围(1, FFNWorkerNum*2+1] | shape变量 | 否 | — | 不合适 | 否 | 缺失校验 |
| 40 | sharedExpertNum取值范围[0, 4] | shape变量 | 部分 | 通过expertRankTable dim1间接约束 | 部分 | 否 | 未显式校验 |

---

## 第五步：总结与建议

### 覆盖率统计

| 统计项 | 数量 |
|--------|------|
| 文档限制总条目数 | 40 |
| 已实现条目数 | 34 (85.0%) |
| 未实现条目数 | 3 (7.5%) |
| 部分实现条目数 | 3 (7.5%) |
| UT已覆盖条目数 | 4 (10.0%) |
| UT未覆盖条目数 | 36 (90.0%) |

### 关键问题清单

**高优先级（文档有描述但代码未实现）：**

1. **M(expertRankTable dim2)取值范围校验缺失**：文档明确M取值范围为(1, FFNWorkerNum * 2 + 1]，但tiling层未校验expertRankTable dim2的取值范围
2. **sessionId值域校验缺失**：文档明确sessionId取值范围为[0, attentionWorkerNum)，代码未校验（值域校验可能需在kernel层实现，但tiling层也应做范围检查）
3. **expertIds值域校验缺失**：文档明确expertIds取值范围为[0, moeExpertNum)，代码未校验

**高优先级（op_api层校验不足）：**

4. **op_api层仅校验空指针和group长度**：dtype、format、shape、属性值等校验全部委托给inner库（不可见），op_api层应前置校验关键参数（dtype、shape维度等），以便在入口处快速失败

**中优先级（代码有实现但UT未覆盖）：**

5. **op_api层UT仅有2个正向用例**：无任何异常测试，包括空指针、dtype异常、属性值越界等
6. **op_host tiling层UT异常用例仅3个**：仅覆盖X!=1、Bs>512、K>16，其余大量校验（H范围、L!=1、moeExpertNum越界、quantMode异常、syncFlag异常、ffnStartRankId越界、worldSize越界、shape约束不匹配、dtype异常、format异常等）均无UT覆盖
7. **infershape UT仅1个正向用例**：无异常测试

**低优先级（实现细节问题）：**

8. **format校验逻辑**：`CheckTensorFormat`仅校验不支持FRACTAL_NZ，未校验必须为ND。如果存在其他非ND格式（如FRACTAL_Z等），当前校验可能遗漏
9. **quantMode校验分两步**：`CheckAndSetAttrs`允许quantMode∈[0,2]通过，`CheckInputAndSetTilingData`再拒绝quantMode=1（静态量化），建议在`CheckAndSetAttrs`中直接校验quantMode仅支持0和2

### 改进建议

**需要新增的校验代码及建议实现位置：**

1. **op_api层**：在`CheckParams`或新增校验函数中增加以下校验：
   - x dtype必须为FLOAT16/BFLOAT16
   - sessionId/microBatchId/layerId/expertIds/expertRankTable dtype必须为INT32
   - scalesOptional dtype必须为FLOAT32（非空时）
   - activeMaskOptional dtype必须为BOOL（非空时）
   - x shape必须为3维
   - 关键属性值范围校验（worldSize/moeExpertNum/quantMode/syncFlag/ffnStartRankId）

2. **op_tiling层**：在`CheckTensorShapeAndSetTinglingData`中增加以下校验：
   - M(expertRankTable dim2)取值范围(1, FFNWorkerNum*2+1]
   - sharedExpertNum = expertRankTable dim1 - moeExpertNum，校验sharedExpertNum ∈ [0, 4]

**需要新增的UT测试用例：**

1. **op_api层异常UT**：
   - 空指针测试（各必选输入为nullptr）
   - group空指针/空字符串/超长字符串测试
   - x dtype异常测试（INT32/INT8等）
   - sessionId/microBatchId等dtype异常测试
   - x shape维度异常测试
   - 关键属性值越界测试

2. **op_host tiling层异常UT**：
   - H范围越界（< 1024 或 > 8192）
   - L != 1
   - moeExpertNum越界（≤ 0 或 > 1024）
   - quantMode异常（= 1 或 > 2）
   - syncFlag异常（> 1）
   - ffnStartRankId越界（≥ worldSize 或 < 0）
   - worldSize越界（< 2 或 > 768）
   - ffnTokenInfoTableShape/ffnTokenDataShape/attnTokenInfoTableShape长度异常
   - shape约束不匹配（各shape间维度不一致）
   - dtype异常
   - format异常（FRACTAL_NZ）
   - scalesOptional shape不匹配
   - activeMaskOptional shape不匹配
   - expertRankTable dim1越界
   - M(expertRankTable dim2)越界
   - attentionWorkerNum越界
   - 非量化场景scales非空
