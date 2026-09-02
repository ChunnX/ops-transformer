# GroupedMatmul OP_LOGE 重构测试覆盖报告

针对重构提交 `refactor(gmm): migrate grouped_matmul OP_LOGE to structured macros`
（OP_LOGE → OP_LOGE_FOR_* / OP_LOGE_WITH_INVALID_INPUT）涉及的 **667 个日志站点**，
本文档说明每个站点的测试覆盖情况。

## 总览

| 分类 | 站点数 | 说明 |
|---|---|---|
| ATK 端到端覆盖（910B3 真机） | 9 | `aclnn_grouped_matmul.cpp`，30 条 ATK 用例 |
| ophost UT 覆盖（CSV 驱动） | 178 | `grouped_matmul_infershape.cpp`（168）+ `grouped_matmul_tiling.cpp`（10） |
| 950 专属，不测 | 450 | David/950 路径（按需求排除） |
| 不可达/防御性，文档说明 | 30 | 详见下表 |

## 1. ATK 端到端覆盖（9 站点）

文件：`gmm/grouped_matmul/op_api/aclnn_grouped_matmul.cpp`

测试仓：`gitcode.com/nobodyaaa/aclnn_grouped_matmul_case`（30 条用例：3 正向 + 27 负向，
覆盖该文件全部参数可达 OP_LOGE 站点；AllocTensor/CreateView 等 host OOM 站点在 DESIGN.md 中说明为不可达）。

## 2. ophost UT 覆盖（178 站点）

### 2.1 grouped_matmul_infershape.cpp（168 站点，80 条用例）

位置：`tests/ut/op_host/test_grouped_matmul_infershape.csv`（caseName 前缀 `gmm_`）

| 用例组 | 覆盖的检查函数/分支 |
|---|---|
| `gmm_attr_*`（5 条） | CheckAttrs/CheckGroupType：splitItem 越界、groupType=SPLIT_N、groupType 非法、actType≥6、actType=GELU_ERR_FUNC |
| `gmm_groupnum_over_1024_910b` | GetGroupSize：groupNum>1024 |
| `gmm_nonquant_*`（5 条） | CheckNonQuant/IsGmmQuantEmpty/IsGmmAntiQuantEmpty：scale/offset/perTokenScale/antiquantScale/antiquantOffset 非空 |
| `gmm_quant_*`（4 条） | CheckGroupedMatmulQuant：splitK 不支持、offset 非空、scale 为空、antiquant 非空 |
| `gmm_pertoken_*`（3 条） | CheckPerTokenScale：rank≠1、M 不等、多 tensor 场景 |
| `gmm_a8w4_*`（3 条） | CheckA8W4AsymQuantParams：bias shape、scale shape、antiquant 非空 |
| `gmm_antiquant_*`（12 条） | CheckGroupedMatmulAntiQuant*：splitK、offset 缺失、quant 混入、perGroupNum 不整除 K、rank 非法、pergroup 奇数（转置）、跨 list rank/pergroup 不一致 |
| `gmm_bias_*`（5 条） | CheckOptionalTensorList(bias)：数量、batch、N 维、rank、separated N 不匹配 |
| `gmm_nosplit_*`（13 条） | CheckCaseNoSplit 系列：x/y、x/weight 数量、groupList 长度/超界、weight rank、x rank、3D weight 首维、K 不等、K>65535、N>65535、int4 N 奇数（w0/w1）、transposeX |
| `gmm_splitm_*`（8 条） | SplitM 系列：groupList 缺失/数量/dtype、weight 内轴超界、transposeX、weight 数量 vs groupNum、K 不等、weight rank、无 groupList 的 m-m 场景、不支持组合 |
| `gmm_splitk_*`（5 条） | CheckCaseSplitK：x 未转置、x 多 tensor、bias 非空、groupList 缺失、weight rank |
| `gmm_transpose_both/grouplist_dim/310p/updatey_*`（4 条） | CheckParamDifferentGroupType：同时转置、groupList 维度、310P 场景限制；UpdateMultipleShapeY groupList 数据为空 |
| `gmm_dtype_*`（12 条，InferDataType） | CheckFunctionParamsForDtype 系列：无匹配 dtype 组合、950 bias dtype、quant bias/scale dtype、outputDtype 非法、scale undefined、per-channel/per-token scale dtype、antiquant dtype 不匹配、310P 非 fp16 |

### 2.2 grouped_matmul_tiling.cpp（10 站点，10 条用例）

位置：`tests/ut/op_host/test_grouped_matmul_tiling.csv`（caseName 前缀 `gmm_tiling_`）

| 用例 | 覆盖站点 |
|---|---|
| `gmm_tiling_grouptype_invalid` | 主分发器：groupType 非法（L309） |
| `gmm_tiling_splitm_ssm_unsupported` | SplitM 分发器：s-s-m 组合不支持（L334） |
| `gmm_tiling_splitk_ssm_unsupported` | SplitK 分发器：s-s-m 组合不支持（L350） |
| `gmm_tiling_splitk_no_grouplist` | SplitK s-s-s 缺 groupList（L465） |
| `gmm_tiling_tuningconfig_over_maxm` | Init：tuningConfig>maxM（L521） |
| `gmm_tiling_a4w4_scale_dim_invalid` | A4W4 scale 维度非 2/3（L552） |
| `gmm_tiling_a8w4_highperf_multi_weight` | A8W4 MSD 高性能路径多 tensor weight（L1940） |
| `gmm_tiling_a8w4_k_not_div_groupnum` | A8W4 K 不整除 quantGroupNum（L2147） |
| `gmm_tiling_a8w4_k_not_div_64` | A8W4 K 不整除 64（L2152） |
| `gmm_tiling_a8w4_k_too_large` | A8W4 K>18432（L2157） |

## 3. 950 专属路径（450 站点，按需求不测）

| 文件 | 站点数 | 说明 |
|---|---|---|
| `op_host/grouped_matmul_infershape_quant_checker.cpp` | 62 | David（950）quant 检查器，仅 soc=Ascend950 分发进入 |
| `op_host/grouped_matmul_infershape_weight_quant_checker.cpp` | 111 | David（950）weight-quant 检查器 |
| `op_api/grouped_matmul_weight_quant_950_checker.cpp` | 1 | 950 aclnn 检查器（编译期 DAV_3510） |
| `op_host/op_tiling/arch35/*`（5 个文件） | 272 | arch35 = Ascend950 tiling，910B 不经过 |
| `grouped_matmul_infershape.cpp` 中 David 分发函数 | 14 | InferShape4David*/InferDtype4David* 包装日志，仅 950 进入 |

## 4. 不可达/防御性站点（30 站点，文档说明）

| 文件/位置 | 站点数 | 不可达原因 |
|---|---|---|
| `op_api/grouped_matmul.cpp`（l0op::GroupedMatmul） | 5 | L0 图内算子包装，非公开 aclnn 接口；调用方（其他 L0 算子）保证输入合法；AllocTensor 失败需 host OOM |
| `op_api/grouped_matmul_util.cpp` | 3 | CreateView 失败需 host 内存耗尽 |
| `op_graph/fallback_grouped_matmul.cpp` | 15 | 其中 4 个为 MXFP（950）路径；其余为图模式执行路径（无 UT harness）及 attr/acl 函数空指针防御（框架保证非空） |
| `op_graph/fusion_pass/grouped_matmul_transpose_fusion_pass.cpp` | 3 | RelinkInput/SetTransposeAttr 内部图操作失败，外部输入不可触发 |
| `grouped_matmul_infershape.cpp` UpdateShapeY/MultiWeight 内 | 3 | 输出 shape 指针由框架保证非空，UpdateShapeY 不会失败 |

## 5. 已知限制

- UT CSV harness 中同一 tensor list 共享一个 dtype 列，无法构造"list 内 dtype 不一致"的输入
  （如 `CheckTensorListDataType` 的逐元素不匹配分支），此类分支与 4 中防御性站点合并说明。
- 950 路径如需后续补测，`quant_checker`/`weight_quant_checker` 可用 soc=Ascend950 的 CSV 行覆盖
  （基础设施已验证可行）。
