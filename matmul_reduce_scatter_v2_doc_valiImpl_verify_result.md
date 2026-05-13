# aclnnMatmulReduceScatterV2 文档校验实现验证报告

**算子路径**: `mc2/matmul_reduce_scatter_v2`
**硬件版本**: A3 (Atlas A3 训练系列产品/Atlas A3 推理系列产品，对应 NpuArch::DAV_2201，SocVersion: Ascend910_93)
**aclnn入口**: `aclnnMatmulReduceScatterV2`

---

## 第一步：文档限制清单

从 [aclnnMatmulReduceScatterV2.md](mc2/matmul_reduce_scatter_v2/docs/aclnnMatmulReduceScatterV2.md) 中提取的 A3 相关限制：

| 序号 | 参数名 | 限制类型 | 限制内容描述 |
|------|--------|---------|-------------|
| 1 | x1/x2/output | 空指针 | x1、x2或output不能为空指针 |
| 2 | x1 | dtype | A3+aiv模式：数据类型支持FLOAT16、BFLOAT16、INT8 |
| 3 | x2 | dtype | A3+aiv模式：数据类型支持FLOAT16、BFLOAT16、INT8 |
| 4 | output | dtype | A3+aiv模式：数据类型支持FLOAT16、BFLOAT16。如果x1类型为FLOAT16/BFLOAT16，则output类型与x1保持一致 |
| 5 | x1 | format | A3+aiv模式：数据格式仅支持ND |
| 6 | x2 | format | A3+aiv模式：数据格式支持ND、FRACTAL_NZ |
| 7 | x1 | shape | 2维，shape为[m, k]，仅支持不转置场景 |
| 8 | x2 | shape | 2维，支持转置/不转置场景 |
| 9 | x1/x2 | dtype一致性 | A3：x1和x2的数据类型需要保持一致 |
| 10 | bias | 空指针 | A3+aiv模式：当前版本仅支持输入nullptr |
| 11 | bias | dtype | 文档通用：数据类型支持FLOAT16、BFLOAT16、FLOAT |
| 12 | x1Scale | 空指针 | A3+aiv模式：当x1和x2数据类型为FLOAT16/BFLOAT16时，仅支持输入为nullptr |
| 13 | x1Scale | dtype | A3+aiv模式：数据类型支持FLOAT |
| 14 | x1Scale | shape | A3+aiv模式：pertoken场景，shape为(m, 1) |
| 15 | x2Scale | 空指针 | A3+aiv模式：当x1和x2数据类型为FLOAT16/BFLOAT16时，仅支持输入为nullptr |
| 16 | x2Scale | dtype | A3+aiv模式：数据类型支持FLOAT、INT64。INT64数据类型仅在output数据类型为FLOAT16场景支持 |
| 17 | x2Scale | shape | A3+aiv模式：perchannel场景，shape为(1, n) |
| 18 | quantScale | 空指针 | 文档通用：当前仅支持传入空指针场景 |
| 19 | x1 | 转置 | 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景 |
| 20 | x1 | shape约束 | m须为卡数rank_size的整数倍 |
| 21 | x2 | shape约束 | k轴相等，且k轴取值范围为[256, 65535) |
| 22 | bias | shape | 1维，shape为(n,) |
| 23 | output | shape | 2维，shape为(m/rank_size, n) |
| 24 | 空tensor | 空tensor | A3：不支持空tensor |
| 25 | x2 | 非连续 | 仅支持两根轴转置情况下的非连续Tensor，其他场景不支持 |
| 26 | streamMode | 属性值 | 当前只支持枚举值1 |
| 27 | reduceOp | 属性值 | 当前版本仅支持"sum" |
| 28 | commTurn | 属性值 | 当前版本仅支持输入0 |
| 29 | groupSize | 属性值 | A3+aiv模式：当前版本仅支持输入为0 |
| 30 | commMode | 属性值 | A3+aiv模式：当前仅支持输入"aiv" |
| 31 | rankSize | 属性值 | A3：支持2、4、8卡 |
| 32 | blockSize | 属性值 | 当前版本只支持blockSizeM=blockSizeN=0（即blockSize=0） |
| 33 | amaxOutOptional | 空指针 | 当前版本仅支持nullptr或空tensor |
| 34 | output | 空tensor | 仅当输出类型为FLOAT16、BFLOAT16时支持空Tensor |
| 35 | group | 通信域 | 通信域名称需有效（非空） |

---

## 第二步：代码校验实现核对

### 2.1 op_api层校验（A3走aiv模式：`matmulReduceScatterV2GetWorkSpaceSizeAivMode`）

| 序号 | 文档限制内容 | 限制类型 | 代码是否实现 | 实现位置 | 实现是否合适 | UT是否覆盖 | 备注 |
|------|------------|---------|------------|---------|------------|-----------|------|
| 1 | x1/x2/output不能为空指针 | 空指针 | 是 | [aclnn_matmul_reduce_scatter_v2.cpp](mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp):`CheckNotNull` | 合适，在入口处校验 | 部分 | CCU模式有空指针UT，AIV模式无独立空指针UT |
| 2 | x1 dtype为FLOAT16/BFLOAT16/INT8 | dtype | 是 | `CheckAivModeDtypeValid` → `AIV_MODE_INPUT_SUPPORT_LIST` | 合适 | 部分 | CCU模式有InvalidX1Dtype UT，AIV模式无 |
| 3 | x2 dtype为FLOAT16/BFLOAT16/INT8 | dtype | 是 | `CheckAivModeDtypeValid` → `AIV_MODE_INPUT_SUPPORT_LIST` | 合适 | 部分 | 同上 |
| 4 | output dtype为FLOAT16/BFLOAT16 | dtype | 是 | `CheckAivModeDtypeValid` → `AIV_MODE_OUTPUT_SUPPORT_TYPE` | 合适 | 部分 | CCU模式有InvalidOutputDtype UT |
| 5 | x1 format仅支持ND | format | 否 | — | 不合适，op_api层未校验x1的format | 否 | 缺失校验 |
| 6 | x2 format支持ND/FRACTAL_NZ | format | 部分 | `MatmulReduceScatterV2IsWeightNZFormat`仅做日志打印，未做校验 | 不合适，未做format合法性校验 | 否 | 缺失校验 |
| 7 | x1 shape为2维 | shape | 是 | `CheckShape` → `OP_CHECK_WRONG_DIMENSION(x1, TWO_DIMS)` | 合适 | 部分 | CCU模式有覆盖 |
| 8 | x2 shape为2维 | shape | 是 | `CheckShape` → `OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS)` | 合适 | 部分 | 同上 |
| 9 | x1/x2 dtype一致性 | dtype | 是 | `CheckCase` → `CheckHighAccuracyCase` → `OP_CHECK_DTYPE_NOT_SAME(x1, x2)` | 合适，高精度场景校验 | 部分 | CCU模式有HighAccuracyX1X2DtypeMismatch UT |
| 10 | bias仅支持nullptr | 空指针 | 否 | AIV模式`CheckAivModeParams`未校验bias | 不合适，AIV模式入口未校验bias必须为nullptr | 否 | 缺失校验，bias非空时可能进入异常路径 |
| 11 | bias dtype支持FLOAT16/BFLOAT16/FLOAT | dtype | 是（CCU模式） | `CheckDtypeValid` → `BIAS_OUTPUT_SUPPORT_TYPE` | AIV模式未校验bias | 否 | AIV模式下bias本应仅支持nullptr，但未校验 |
| 12 | x1Scale在FP16/BF16时仅支持nullptr | 空指针 | 否 | AIV模式未校验 | 不合适 | 否 | 缺失校验 |
| 13 | x1Scale dtype支持FLOAT | dtype | 部分 | tiling层`CheckDtype_X1`校验x1Scale类型为FLOAT | 合适，但在tiling层而非op_api层 | 否 | UT未覆盖 |
| 14 | x1Scale shape为(m,1) | shape | 否 | 未校验 | 不合适 | 否 | 缺失校验 |
| 15 | x2Scale在FP16/BF16时仅支持nullptr | 空指针 | 否 | AIV模式未校验 | 不合适 | 否 | 缺失校验 |
| 16 | x2Scale dtype支持FLOAT/INT64 | dtype | 部分 | tiling层`CheckDtype_X2`校验 | 合适，但在tiling层 | 否 | UT未覆盖 |
| 17 | x2Scale shape为(1,n) | shape | 否 | 未校验 | 不合适 | 否 | 缺失校验 |
| 18 | quantScale仅支持nullptr | 空指针 | 否 | AIV模式未校验 | 不合适 | 否 | 缺失校验 |
| 19 | x1不支持转置 | 转置 | 是 | `CheckShape` → `if (isTransA) { OP_LOGE; return false; }` | 合适 | 部分 | CCU模式有覆盖 |
| 20 | m须为rank_size的整数倍 | shape | 否 | op_api层未校验 | 不合适，该约束在tiling层隐式处理（`args_.mValue = args_.orgMValue / args_.rankDim`） | 否 | 缺失显式校验 |
| 21 | k轴取值范围[256, 65535) | range | 是 | `CheckShape` → `OP_API_CHECK((kVal1 < KVALUE_MIN \|\| kVal1 >= KVALUE_MAX))` | 合适 | 部分 | CCU模式有覆盖 |
| 22 | k轴相等 | shape | 是 | `CheckShape` → `OP_API_CHECK((kVal1 != kVal2))` | 合适 | 部分 | 同上 |
| 23 | bias shape为1维 | shape | 否 | op_api层未校验 | 不合适 | 否 | 缺失校验 |
| 24 | output shape为(m/rank_size, n) | shape | 否 | op_api层未校验output shape | 不合适 | 否 | 缺失校验 |
| 25 | 不支持空tensor | 空tensor | 否 | AIV模式未校验空tensor（CCU模式有处理） | 不合适，AIV模式未对空tensor做校验 | 否 | 缺失校验 |
| 26 | x2非连续仅限转置场景 | 非连续 | 是 | `matmulReduceScatterV2GetWorkSpaceSizeAivMode` → `if (!Ops::Transformer::IsTransposeLastTwoDims(x2) && !MC2Aclnn::IsTensorContiguous(x2))` | 合适，仅DAV_2201分支校验 | 否 | UT未覆盖 |
| 27 | streamMode仅支持1 | 属性值 | 是 | `CheckAttr` → `if (streamMode != NUM_ACL_STOP_ON_FAILURE)` | 合适 | 是 | CCU模式有InvalidStreamMode UT |
| 28 | reduceOp仅支持"sum" | 属性值 | 否 | op_api层未校验 | 不合适，仅在tiling层校验 | 否 | 缺失op_api层校验 |
| 29 | commTurn仅支持0 | 属性值 | 否 | op_api层未校验 | 不合适，仅在tiling层校验 | 否 | 缺失op_api层校验 |
| 30 | groupSize仅支持0 | 属性值 | 否 | op_api层未校验 | 不合适，仅在tiling层校验 | 否 | 缺失op_api层校验 |
| 31 | commMode仅支持"aiv" | 属性值 | 否 | op_api层未校验 | 不合适，仅在tiling层校验 | 否 | 缺失op_api层校验 |
| 32 | blockSize仅支持0 | 属性值 | 否 | op_api层未校验 | 不合适，仅在tiling层校验 | 否 | 缺失op_api层校验 |
| 33 | amaxOutOptional仅支持nullptr | 空指针 | 否 | AIV模式未校验 | 不合适，CCU模式在`CheckCase`中校验了amax != nullptr | 否 | AIV模式缺失校验 |
| 34 | output空tensor | 空tensor | 否 | 未校验 | 不合适 | 否 | 缺失校验 |
| 35 | group有效性 | 通信域 | 否 | op_api层未校验 | 不合适，仅在tiling层校验 | 否 | 缺失op_api层校验 |

### 2.2 op_tiling层校验（A3走arch32 aiv模式：`MatmulReduceScatterTilingV2AivModeFunc`）

| 序号 | 文档限制内容 | 限制类型 | 代码是否实现 | 实现位置 | 实现是否合适 | UT是否覆盖 | 备注 |
|------|------------|---------|------------|---------|------------|-----------|------|
| 1 | group有效性 | 通信域 | 是 | `MatmulReduceScatterV2CheckAttrAndSetTiling` → `OP_TILING_CHECK(groupPtr == nullptr \|\| strlen(groupPtr) == 0)` | 合适 | 部分 | CSV中有group参数 |
| 2 | is_trans_a必须为false | 转置 | 是 | `MatmulReduceScatterV2CheckAttrAndSetTiling` → `info.isTransposeA = false` | 合适，强制设为false | 部分 | CSV中有is_trans_a参数 |
| 3 | x1/x2 shape维度 | shape | 部分 | `MatmulReduceScatterV2CheckShapeAndSetTiling`仅读取shape，未校验维度是否为2 | 不合适 | 否 | 缺失维度校验 |
| 4 | k轴相等 | shape | 部分 | `MatmulReduceScatterV2CheckShapeAndSetTiling`中通过条件判断处理k轴不等的情况（取N值），但未报错 | 不合适，应报错而非静默处理 | 否 | 缺失校验 |
| 5 | rankSize支持2/4/8 | 属性值 | 是 | `MatmulReduceScatterTilingV2AivModeFunc` → `OP_TILING_CHECK(rankSize != 2 && rankSize != 4 && rankSize != 8)` | 合适 | 部分 | CSV中有rank_size参数 |
| 6 | commMode为"aiv" | 属性值 | 是 | `matmul_reduce_scatter_v2_a2a3.cpp` → `OP_TILING_CHECK(std::strcmp(commModePtr, "aiv") != 0)` | 合适 | 部分 | CSV中有comm_mode参数 |
| 7 | reduceOp为"sum" | 属性值 | 否（CCU模式有） | CCU模式在`CheckAttrInfoValid`中校验，AIV模式未校验 | 不合适 | 否 | AIV tiling缺失校验 |
| 8 | commTurn为0 | 属性值 | 否（CCU模式有） | CCU模式在`CheckAttrInfoValid`中校验，AIV模式未校验 | 不合适 | 否 | AIV tiling缺失校验 |
| 9 | blockSize为0 | 属性值 | 否（CCU模式有） | CCU模式在`CheckAttrInfoValid`中校验，AIV模式未校验 | 不合适 | 否 | AIV tiling缺失校验 |
| 10 | groupSize为0 | 属性值 | 否（CCU模式有） | CCU模式在`CheckGroupSize`中校验，AIV模式未校验 | 不合适 | 否 | AIV tiling缺失校验 |
| 11 | quantScale为nullptr | 空指针 | 否（CCU模式有） | CCU模式在`CheckInputScale`中校验，AIV模式未校验 | 不合适 | 否 | AIV tiling缺失校验 |
| 12 | bias为nullptr | 空指针 | 否 | AIV tiling未校验 | 不合适 | 否 | 缺失校验 |
| 13 | x1Scale dtype为FLOAT | dtype | 是 | `CheckDtype_X1` → `if (x1Type != ge::DT_FLOAT) return false;` | 合适 | 否 | UT未覆盖 |
| 14 | x2Scale dtype为FLOAT/INT64 | dtype | 是 | `CheckDtype_X2` → 校验FLOAT和INT64（INT64仅FP16输出） | 合适 | 否 | UT未覆盖 |
| 15 | m须为rank_size整数倍 | shape | 部分 | `DoSplitMTiling`中 `args_.mValue = args_.orgMValue / args_.rankDim`，未做整除校验 | 不合适，应显式校验 | 否 | 缺失校验 |
| 16 | k轴范围[256, 65535) | range | 否 | AIV tiling未校验k轴范围 | 不合适 | 否 | 缺失校验 |
| 17 | x1/x2 dtype一致性 | dtype | 部分 | AIV tiling通过quantFlag判断是否为INT8量化场景，但未显式校验x1/x2 dtype一致性 | 不合适 | 否 | 缺失校验 |
| 18 | 不支持空tensor | 空tensor | 否 | AIV tiling未校验空tensor | 不合适 | 否 | 缺失校验 |
| 19 | x1Scale shape为(m,1) | shape | 否 | 未校验 | 不合适 | 否 | 缺失校验 |
| 20 | x2Scale shape为(1,n) | shape | 否 | 未校验 | 不合适 | 否 | 缺失校验 |
| 21 | bias shape为1维 | shape | 否（CCU模式有） | CCU模式在`CheckBias`中校验，AIV模式未校验 | 不合适 | 否 | AIV tiling缺失校验 |
| 22 | output shape为(m/rank_size, n) | shape | 否 | 未校验 | 不合适 | 否 | 缺失校验 |

---

## 第三步：UT覆盖情况

### 3.1 op_api层UT（[test_aclnn_matmul_reduce_scatter_v2.cpp](mc2/matmul_reduce_scatter_v2/tests/ut/op_api/test_aclnn_matmul_reduce_scatter_v2.cpp)）

**现有测试用例概览：**

| 测试类 | 用例名 | 覆盖场景 |
|--------|-------|---------|
| MatmulReduceScatterV2AclnnTest | Basic | 基本FP16 CCU模式 |
| MatmulReduceScatterV2AclnnTest | Basic2 | FP16带Scale CCU模式 |
| MatmulReduceScatterV2AclnnTest | 3Scale | 3维Scale CCU模式 |
| MatmulReduceScatterV2AclnnTest | EmptyTensor | 空tensor CCU模式 |
| MatmulReduceScatterV2AclnnTest | Fp16 | FP16无bias CCU模式 |
| MatmulReduceScatterV2AclnnTest | 16Bit | BF16 CCU模式 |
| MatmulReduceScatterV2AclnnTest | 16BitTrans | BF16转置 CCU模式 |
| MatmulReduceScatterV2AclnnTest | E4m3fnNotSupport | FP8不支持 CCU模式 |
| MatmulReduceScatterV2CcuModeTest | NullX1/NullX2/NullOutput | 空指针校验 |
| MatmulReduceScatterV2CcuModeTest | InvalidX1Dtype/InvalidOutputDtype/InvalidBiasDtype | dtype校验 |
| MatmulReduceScatterV2CcuModeTest | InvalidStreamMode | streamMode校验 |
| MatmulReduceScatterV2CcuModeTest | HighAccuracy* | 高精度场景 |
| MatmulReduceScatterV2CcuModeTest | HighAccuracyX1X2DtypeMismatch | x1/x2 dtype不一致 |
| MatmulReduceScatterV2CcuModeTest | HighAccuracyX1OutputDtypeMismatch | x1/output dtype不一致 |
| MatmulReduceScatterV2CcuModeTest | HighAccuracyBiasDtypeMismatch | bias dtype不一致 |
| MatmulReduceScatterV2CcuModeTest | EmptyX1/EmptyX2 | 空tensor（CCU模式） |
| MatmulReduceScatterV2CcuModeTest | LowAccuracyE4m3fnPerTensor | 低精度pertensor场景 |

**缺失的AIV模式UT场景：**
- 无AIV模式空指针校验UT
- 无AIV模式dtype校验UT
- 无AIV模式非连续tensor校验UT
- 无AIV模式bias非空校验UT
- 无AIV模式x1Scale/x2Scale校验UT
- 无AIV模式commMode校验UT
- 无AIV模式空tensor校验UT
- 无AIV模式amaxOutOptional非空校验UT

### 3.2 op_host层UT（arch32: [test_matmul_reduce_scatter_v2_tiling.csv](mc2/matmul_reduce_scatter_v2/tests/ut/op_host/arch32/test_matmul_reduce_scatter_v2_tiling.csv)）

**现有测试用例概览：**

| 用例名 | 覆盖场景 | 预期结果 |
|--------|---------|---------|
| FP16_8rank_LargeM | FP16 8卡大M | SUCCESS |
| FP16_4rank_LargeM | FP16 4卡大M | SUCCESS |
| FP16_8rank_LargeM_TransB | FP16 8卡转置B | SUCCESS |
| FP16_4rank_LargeM_TransB | FP16 4卡转置B | SUCCESS |
| FP16_A3_8rank_LargeM | A3 FP16 8卡 | SUCCESS |
| FP16_A3_4rank_LargeM | A3 FP16 4卡 | SUCCESS |
| FP16_A3_4rank_TransB | A3 FP16 4卡转置 | SUCCESS |
| BF16_8rank_LargeM | BF16 8卡 | SUCCESS |
| BF16_4rank_LargeM | BF16 4卡 | SUCCESS |
| BF16_8rank_LargeM_TransB | BF16 8卡转置 | SUCCESS |
| FP16_SmallM_Rank2/4/8 | 小M场景 | SUCCESS |
| FP16_SmallM_Rank4/8_TransB | 小M转置场景 | SUCCESS |
| BF16_SmallM_Rank8 | BF16小M | SUCCESS |
| FP16_UnalignedK | 非对齐K | SUCCESS |
| FP16_Rank2_LargeM | 2卡大M | SUCCESS |
| FP16_DiffShape_* | 不同shape | SUCCESS |
| A2_8rank_INT8_LargeM | A2 INT8 8卡 | FAILED |
| A2_4rank_INT8_LargeM | A2 INT8 4卡 | FAILED |
| A3_8rank_INT8_LargeM | A3 INT8 8卡 | FAILED |
| A3_4rank_INT8_LargeM | A3 INT8 4卡 | FAILED |

**缺失的异常测试场景：**
- 无reduceOp非"sum"的异常测试
- 无commTurn非0的异常测试
- 无blockSize非0的异常测试
- 无groupSize非0的异常测试
- 无k轴范围越界的异常测试
- 无m不能被rankSize整除的异常测试
- 无x1/x2 dtype不一致的异常测试
- 无bias非空的异常测试
- 无空tensor的异常测试
- 无x1Scale/x2Scale dtype错误的异常测试
- 无quantScale非空的异常测试
- 无x1转置的异常测试
- 无不支持rankSize的异常测试

---

## 第四步：校验结果汇总表

| 序号 | 文档限制内容 | 限制类型 | 代码是否实现 | 实现位置 | 实现是否合适 | UT是否覆盖 | 备注 |
|------|------------|---------|------------|---------|------------|-----------|------|
| 1 | x1/x2/output不能为空指针 | 空指针 | 是 | op_api: `CheckNotNull` | 合适 | 部分 | CCU有UT，AIV无UT |
| 2 | x1 dtype为FLOAT16/BFLOAT16/INT8 | dtype | 是 | op_api: `CheckAivModeDtypeValid` | 合适 | 部分 | CCU有UT |
| 3 | x2 dtype为FLOAT16/BFLOAT16/INT8 | dtype | 是 | op_api: `CheckAivModeDtypeValid` | 合适 | 部分 | 同上 |
| 4 | output dtype为FLOAT16/BFLOAT16 | dtype | 是 | op_api: `CheckAivModeDtypeValid` | 合适 | 部分 | 同上 |
| 5 | x1 format仅支持ND | format | 否 | — | 不合适，缺失校验 | 否 | op_api层未校验 |
| 6 | x2 format支持ND/FRACTAL_NZ | format | 部分 | op_api: `MatmulReduceScatterV2IsWeightNZFormat`仅打印日志 | 不合适，未做合法性校验 | 否 | 缺失校验 |
| 7 | x1 shape为2维 | shape | 是 | op_api: `CheckShape` | 合适 | 部分 | CCU有UT |
| 8 | x2 shape为2维 | shape | 是 | op_api: `CheckShape` | 合适 | 部分 | 同上 |
| 9 | x1/x2 dtype一致性 | dtype | 是 | op_api: `CheckHighAccuracyCase` | 合适 | 部分 | CCU有UT |
| 10 | bias仅支持nullptr(AIV) | 空指针 | 否 | — | 不合适，AIV模式缺失校验 | 否 | 高优先级 |
| 11 | x1Scale在FP16/BF16时仅支持nullptr | 空指针 | 否 | — | 不合适，缺失校验 | 否 | 高优先级 |
| 12 | x1Scale dtype支持FLOAT | dtype | 部分 | tiling: `CheckDtype_X1` | 合适，但在tiling层 | 否 | 应在op_api层前置校验 |
| 13 | x1Scale shape为(m,1) | shape | 否 | — | 不合适，缺失校验 | 否 | 中优先级 |
| 14 | x2Scale在FP16/BF16时仅支持nullptr | 空指针 | 否 | — | 不合适，缺失校验 | 否 | 高优先级 |
| 15 | x2Scale dtype支持FLOAT/INT64 | dtype | 部分 | tiling: `CheckDtype_X2` | 合适，但在tiling层 | 否 | 应在op_api层前置校验 |
| 16 | x2Scale shape为(1,n) | shape | 否 | — | 不合适，缺失校验 | 否 | 中优先级 |
| 17 | quantScale仅支持nullptr | 空指针 | 否 | — | 不合适，AIV模式缺失校验 | 否 | 高优先级 |
| 18 | x1不支持转置 | 转置 | 是 | op_api: `CheckShape` | 合适 | 部分 | CCU有UT |
| 19 | m须为rank_size整数倍 | shape | 否 | — | 不合适，缺失显式校验 | 否 | 高优先级 |
| 20 | k轴取值范围[256, 65535) | range | 是 | op_api: `CheckShape` | 合适 | 部分 | CCU有UT |
| 21 | k轴相等 | shape | 是 | op_api: `CheckShape` | 合适 | 部分 | 同上 |
| 22 | bias shape为1维 | shape | 否 | — | 不合适，缺失校验 | 否 | 中优先级 |
| 23 | output shape为(m/rank_size, n) | shape | 否 | — | 不合适，缺失校验 | 否 | 中优先级 |
| 24 | 不支持空tensor(A3) | 空tensor | 否 | — | 不合适，AIV模式缺失校验 | 否 | 高优先级 |
| 25 | x2非连续仅限转置场景 | 非连续 | 是 | op_api: `matmulReduceScatterV2GetWorkSpaceSizeAivMode` | 合适 | 否 | UT未覆盖 |
| 26 | streamMode仅支持1 | 属性值 | 是 | op_api: `CheckAttr` | 合适 | 是 | CCU有UT |
| 27 | reduceOp仅支持"sum" | 属性值 | 部分 | tiling(CCU): `CheckAttrInfoValid` | AIV tiling缺失校验 | 否 | 应在op_api层校验 |
| 28 | commTurn仅支持0 | 属性值 | 部分 | tiling(CCU): `CheckAttrInfoValid` | AIV tiling缺失校验 | 否 | 应在op_api层校验 |
| 29 | groupSize仅支持0 | 属性值 | 部分 | tiling(CCU): `CheckGroupSize` | AIV tiling缺失校验 | 否 | 应在op_api层校验 |
| 30 | commMode仅支持"aiv" | 属性值 | 部分 | tiling: `matmul_reduce_scatter_v2_a2a3.cpp` | 合适，在tiling入口校验 | 部分 | CSV中有覆盖 |
| 31 | rankSize支持2/4/8 | 属性值 | 是 | tiling: `MatmulReduceScatterTilingV2AivModeFunc` | 合适 | 部分 | CSV中有覆盖 |
| 32 | blockSize仅支持0 | 属性值 | 部分 | tiling(CCU): `CheckAttrInfoValid` | AIV tiling缺失校验 | 否 | 应在op_api层校验 |
| 33 | amaxOutOptional仅支持nullptr | 空指针 | 部分 | op_api(CCU): `CheckCase` | AIV模式缺失校验 | 否 | 高优先级 |
| 34 | output空tensor | 空tensor | 否 | — | 不合适，缺失校验 | 否 | 中优先级 |
| 35 | group有效性 | 通信域 | 部分 | tiling: `MatmulReduceScatterV2CheckAttrAndSetTiling` | 合适，但在tiling层 | 部分 | 应在op_api层前置校验 |

---

## 第五步：总结与建议

### 覆盖率统计

| 统计项 | 数量 |
|--------|------|
| 文档限制总条目数 | 35 |
| 已实现条目数 | 12 |
| 未实现条目数 | 14 |
| 部分实现条目数 | 9 |
| UT已覆盖条目数 | 2 |
| UT未覆盖条目数 | 24 |
| UT部分覆盖条目数 | 9 |

### 关键问题清单

**高优先级（文档有描述但代码未实现）：**

1. **AIV模式bias非空校验缺失**：文档明确A3+aiv模式下bias仅支持nullptr，但op_api层AIV模式入口`CheckAivModeParams`未校验bias参数
2. **AIV模式x1Scale/x2Scale空指针校验缺失**：文档明确FP16/BF16场景x1Scale/x2Scale仅支持nullptr，AIV模式未校验
3. **AIV模式quantScale空指针校验缺失**：文档明确quantScale仅支持nullptr，AIV模式未校验
4. **AIV模式amaxOutOptional校验缺失**：CCU模式在`CheckCase`中校验了amax != nullptr，但AIV模式未校验
5. **m须为rank_size整数倍校验缺失**：文档明确约束，op_api层和AIV tiling层均未显式校验
6. **A3不支持空tensor校验缺失**：文档明确A3不支持空tensor，但AIV模式未做校验（CCU模式有空tensor处理逻辑）
7. **x1 format校验缺失**：文档明确A3+aiv模式x1仅支持ND，op_api层未校验

**高优先级（代码有实现但逻辑不完整）：**

8. **AIV tiling层多项属性校验缺失**：reduceOp、commTurn、blockSize、groupSize等属性校验仅在CCU模式tiling中实现，AIV tiling未实现
9. **AIV tiling层k轴范围校验缺失**：CCU模式在`CheckAttrInfoValid`中校验k轴范围[256, 65535)，AIV tiling未校验
10. **x2 format校验不完整**：`MatmulReduceScatterV2IsWeightNZFormat`仅做日志打印，未做合法性校验和报错

**中优先级（代码有校验但UT未覆盖）：**

11. **x2非连续tensor校验UT缺失**：op_api层AIV模式已实现非连续校验，但无UT覆盖
12. **x1Scale/x2Scale dtype校验UT缺失**：tiling层已实现`CheckDtype_X1`/`CheckDtype_X2`，但无UT覆盖
13. **x1Scale shape(m,1)/x2Scale shape(1,n)校验缺失**：文档明确shape约束，代码未校验
14. **bias shape为1维校验缺失**：CCU模式tiling有校验，AIV tiling缺失
15. **output shape校验缺失**：文档明确output shape为(m/rank_size, n)，代码未校验

**低优先级（实现位置不合适）：**

16. **reduceOp/commTurn/blockSize/groupSize/commMode校验位置**：这些属性校验应在op_api入口层完成（快速失败），当前仅在tiling层校验
17. **x1Scale/x2Scale dtype校验位置**：应在op_api层前置校验，当前仅在tiling层校验

### 改进建议

**需要新增的校验代码及建议实现位置：**

1. **op_api层AIV模式**：在`CheckAivModeParams`或新增`CheckAivModeFullParams`函数中增加以下校验：
   - bias必须为nullptr
   - x1Scale/x2Scale在FP16/BF16场景必须为nullptr
   - quantScale必须为nullptr
   - amaxOutOptional必须为nullptr
   - x1 format必须为ND
   - x2 format必须为ND或FRACTAL_NZ
   - reduceOp必须为"sum"
   - commTurn必须为0
   - groupSize必须为0
   - commMode必须为"aiv"
   - blockSize必须为0
   - 空tensor不支持（A3场景）
   - m须为rank_size整数倍
   - output shape校验

2. **AIV tiling层**：在`MatmulReduceScatterTilingV2AivModeFunc`中增加以下校验：
   - k轴范围[256, 65535)校验
   - m须为rank_size整数倍校验
   - x1/x2 dtype一致性校验
   - reduceOp/commTurn/blockSize/groupSize属性校验
   - 空tensor校验

**需要新增的UT测试用例：**

1. **op_api层AIV模式异常UT**：
   - AIV模式空指针测试（x1/x2/output为nullptr）
   - AIV模式dtype异常测试（x1/x2/output不支持的数据类型）
   - AIV模式bias非空测试
   - AIV模式x1Scale/x2Scale在FP16/BF16场景非空测试
   - AIV模式quantScale非空测试
   - AIV模式amaxOutOptional非空测试
   - AIV模式非连续tensor异常测试（x2非转置但非连续）
   - AIV模式空tensor测试
   - AIV模式属性值异常测试（streamMode/commMode/reduceOp/commTurn/groupSize/blockSize）
   - AIV模式x1 format异常测试
   - AIV模式m不能被rankSize整除测试
   - AIV模式k轴范围越界测试
   - AIV模式x1/x2 dtype不一致测试

2. **op_host层AIV tiling异常UT**：
   - k轴范围越界
   - m不能被rankSize整除
   - x1/x2 dtype不一致
   - reduceOp非"sum"
   - commTurn非0
   - groupSize非0
   - 不支持的rankSize
   - 空tensor
