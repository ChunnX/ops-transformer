# GMM MxA8W4 单多单场景支持 STC

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档标题 | GMM MxA8W4 单多单场景支持 STC |
| 关联 SRS | `gmm/grouped_matmul/docs/GMM_MxA8W4_SMS_SRS.md` |
| 迭代/特性名称 | GMM MxA8W4 单多单支持 |
| 算子类型 | 单算子 |
| 主领域目录 | `gmm/grouped_matmul` |
| 责任人 | 待定 |
| 生成日期 | 2026-06-12 |
| 文档版本 | v0.1 |

## 2. 测试目标与范围

### 2.1 测试目标

验证 GMM MxA8W4 arch35 resplit controller 支持单多单组合：`x` 为单 tensor、`weight` 为 tensor list、`antiquantScaleOptional/weightScale` 为 tensor list、`biasOptional` 存在时为 tensor list、`y` 为单 tensor。

重点验证：

1. Host 能识别 MxA8W4 单多单场景，并允许 `x` 单、`weight` 多、`y` 单组合。
2. `antiquantScaleOptional`、`biasOptional`、`antiquantOffsetOptional` 在存在时与 `weight` tensor list 长度一致。
3. 每个 `weight[i]`、`antiquantScale[i]`、`bias[i]` 的 dtype、format、shape 与同下标 group 对齐。
4. MxA8W4 单多单场景生成 `IS_SINGLE_MULTI_SINGLE == 1` 的 TilingKey。
5. Kernel resplit controller 在单多单模板实例下可完成多 group 计算，输出与 golden 对齐。
6. 非单多单 MxA8W4 和其他 weight-quant 路径保持原行为。

### 2.2 测试范围

包含范围：

- aclnn grouped_matmul weight quant MxA8W4 arch35 路径。
- `groupType = SPLIT_M` 场景。
- `splitItem = 2/3` 且 `y` 为单 tensor 的场景。
- `weight` FRACTAL_NZ、`transB = true`、`transA = false` 的 MxA8W4 数据流。
- `biasOptional` 不存在和存在两类路径。
- `groupListType = 0/1` 两类 M 方向切分语义。
- Host checker、TilingKey、Kernel 运行结果、非单多单回归。

不包含范围：

- 非 MxA8W4 数据流的单多单扩展。
- 非 arch35 或非 resplit controller 路径改造。
- API 语义、数学定义、输出 dtype/format 变更。
- `groupListType`、`splitItem`、`groupType` 既有语义变更。
- 大范围 host shape 推导重构。

## 3. 环境、输入与限制

### 3.1 测试环境

| 类别 | 要求 |
| --- | --- |
| 硬件 | 支持 arch35 MxA8W4 的 Ascend 设备 |
| 软件包 | 最新主线包 + 本迭代代码 patch 包 |
| 调用通路 | aclnn grouped_matmul / ttk Aclnn 测试框架 |
| 校验方式 | golden 对比、host checker 失败日志、TilingKey 白盒日志/断点 |

### 3.2 输入输出规格

| 参数 | 必选/可选 | dtype | format | shape / 存储语义 | 关键约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | FLOAT8_E4M3FN | ND | 单 tensor，`[M, K]` | list size 为 1，dim 为 2，`K` 与 Mx group shape 匹配 |
| `weight` | 必选 | FLOAT4_E2M1 | FRACTAL_NZ | 多 tensor，`weight[g]` view `[Ng, K]`，storage `[K/32, Ng/16, 16, 32]` | list size 等于 `groupNum` 且大于 1，dim 为 4 |
| `antiquantScaleOptional` | 必选 | FLOAT8_E8M0 | ND | 多 tensor，`[Ng, K/64, 2, 2]` | list size 等于 `weight`，dim 为 4，N 轴与同下标 `weight` 对齐，`K/32` 分组关系满足 MX group size 32 |
| `antiquantOffsetOptional` | 可选 | 与既有 MxA8W4 支持保持一致 | ND | 多 tensor，按同下标 group 对齐 | 存在时 list size 等于 `weight`，dtype/shape 与同下标 `weight` 对齐 |
| `biasOptional` | 可选 | 与既有 MxA8W4 支持保持一致 | ND | 多 tensor，`[Ng]` | 存在时 list size 等于 `weight`，dim 为 1，N 轴与同下标 `weight` 对齐 |
| `perTokenScaleOptional` | 必选 | FLOAT8_E8M0 | ND | 单 tensor，`[M, K/64, 2]` | 与 `x` 的 M/K 对齐 |
| `groupList` | 必选 | INT64 | ND | `[groupNum]` | 长度等于 `weight` list size；`groupListType=0` 表示累计 offset，`groupListType=1` 表示每组 M 大小 |
| `y` | 输出 | 既有 MxA8W4 输出 dtype | ND | 单 tensor，`[M, N]` | `N` 与各 group 输出拼接语义一致 |

### 3.3 属性约束

| 属性 | 取值 / 约束 | 说明 |
| --- | --- | --- |
| `groupType` | SPLIT_M (`0`) | 单多单按 M 方向切分 |
| `splitItem` | `2` 或 `3` | 保障 `y` 为单 tensor |
| `groupListType` | `0` 或 `1` | 语义沿用既有实现 |
| `transA` | `false` | 本需求不扩展 x 转置 |
| `transB` | `true` | MxA8W4 当前 NZNK 路径 |

## 4. 测试策略

### 4.1 白盒 / 小组合

围绕本需求新增分支构造最小覆盖集：

- Host 识别 `IsMxA8W4() && isSingleX_ && !isSingleWeight_ && isSingleY_`。
- `CheckTensorListSize` 对 `x/weight/antiquantScale/bias/antiquantOffset` list size 的正负向覆盖。
- `CheckEveryTensorSingleXMultiWeightSingleY` 对 weight、antiquantScale、bias 的逐 tensor 校验。
- `SetShapeListSplitMSingleXMultiWeightSingleY` 从单 tensor `x` 和多 tensor `weight` 推导 `groupNum/kList/nList`。
- `SetTilingKey` 对 MxA8W4 单多单写入 `isSingleMultiSingle`。
- `UpdateGmAddr` 对单多单模板实例按 tensor list 切换 `weight/antiquantScale/bias`。

### 4.2 泛化 / 大空间

对 group 数、M/N/K、bias 存在性、groupListType 做组合扩展：

- `groupNum` 覆盖 2、3、4、8、边界接近上限的场景。
- `M` 覆盖单 basic block、跨 basic block、尾块、含空 group 或极小 group 的场景。
- `N` 覆盖 128、256、非整倍尾块、多 group 不同 `Ng` 场景。
- `K` 覆盖 64、128、256、512，保持 Mx group size 32 的约束。
- `biasOptional` 覆盖不存在、存在且多 tensor。
- `groupListType` 覆盖累计 offset 和每组大小。

### 4.3 异常 / 负向

异常用例必须与 host checker 条件或 SRS 约束一一对应，覆盖：

- tensor list 长度不满足单多单要求。
- `groupList` 长度与 groupNum 不一致。
- dtype 不一致或不是 MxA8W4 支持 dtype。
- format 不满足 MxA8W4 `weight` FRACTAL_NZ + `transB=true`。
- dim 不满足单多单专用校验。
- antiquantScale/bias N 轴与同下标 `weight` 不一致。
- antiquantScale MX group size 与 `K` 不匹配。
- 非本需求范围的 tensor 组合、转置组合、数据流组合不误入单多单路径。

### 4.4 性能 / 兼容 / 通路

- 单多单路径使用编译期 bool 生成 kernel 模板，不在 group 循环中新增运行时分支。
- 非单多单 MxA8W4 保持 `IS_SINGLE_MULTI_SINGLE == 0`。
- S8S4、A16W4ND、非量化等路径不受新增模板参数影响。
- aclnn / ttk 通路可完成构图、tiling、kernel 启动和结果校验。

## 5. 覆盖统计

| 类别 | 用例数 | 覆盖点 |
| --- | ---: | --- |
| ST-白盒 | 10 | 单多单 happy path、bias、groupListType、shape 推导、TilingKey、地址切换、尾块 |
| ST-泛化 | 8 | groupNum、M/N/K、不同 Ng、空/小 group、边界 group 数 |
| ST-回归 | 4 | 非单多单 MxA8W4、S8S4/A16W4ND/多多多路径 |
| NEG-异常 | 15 | list size、dtype、format、dim、shape、groupList、转置、范围外组合 |
| 合计 | 37 | 正向 22，负向 15 |

## 6. 用例命名规范

- 正向白盒用例：`ST001` ~ `ST099`。
- 泛化正向用例：`STG001` ~ `STG099`。
- 历史/兼容回归：`HIST001` ~ `HIST099`。
- 异常用例：`NEG001` ~ `NEG099`。
- 用例名称建议格式：`<编号>_<场景>_<关键参数>`。

## 7. 约束关系

| 约束编号 | 约束 | 正向覆盖 | 负向覆盖 |
| --- | --- | --- | --- |
| C01 | MxA8W4 数据流：x FLOAT8_E4M3FN，weight FLOAT4_E2M1，antiquantScale FLOAT8_E8M0 | ST001、ST002 | NEG006、NEG007 |
| C02 | `x` 单、`weight` 多、`y` 单 | ST001、ST003 | NEG001、NEG002 |
| C03 | `weight` list size > 1 且等于 groupNum | ST001、STG001 | NEG002、NEG005 |
| C04 | `antiquantScaleOptional` list size 等于 `weight` | ST001 | NEG003 |
| C05 | `biasOptional` 存在时 list size 等于 `weight` | ST002 | NEG004 |
| C06 | `groupList` 长度等于 groupNum | ST001、ST004 | NEG005 |
| C07 | `weight` FRACTAL_NZ 且 `transB=true` | ST001 | NEG008、NEG014 |
| C08 | `antiquantScale` dim 为 4 且 N/K 分组满足 MX 约束 | ST001、STG003 | NEG009、NEG010、NEG011 |
| C09 | `bias` dim 为 1 且 N 轴与 weight 对齐 | ST002 | NEG012、NEG013 |
| C10 | 非单多单路径 `IS_SINGLE_MULTI_SINGLE == 0` | HIST001、HIST002 | NEG015 |

## 8. 数据文件契约

建议每条 aclnn/ttk 数据文件最少描述以下字段：

| 字段 | 说明 |
| --- | --- |
| `case_id` | 用例编号，例如 `ST001` |
| `M` | `x` 总 M，等于各 group M 之和 |
| `K` | `x/weight` K 轴，满足 Mx group size 32 相关约束 |
| `N_list` | 每个 `weight[g]` 对应 `Ng` |
| `groupNum` | `weight` list size |
| `groupList` | 根据 `groupListType` 填累计 offset 或每组 M 大小 |
| `groupListType` | `0` 或 `1` |
| `hasBias` | 是否传入 `biasOptional` |
| `hasAntiquantOffset` | 是否传入 `antiquantOffsetOptional` |
| `dtype/format` | x、weight、antiquantScale、bias、perTokenScale、y 的 dtype/format |
| `expect` | `success` 或预期失败关键字 |

## 9. 脚本与自动化

推荐自动化流程：

1. 使用 ttk/Aclnn 测试框架构造 `x` 单 tensor、`weight/antiquantScale/bias` tensor list、`y` 单 tensor。
2. 正向用例生成 golden：按 group 计算 `Y_g = Matmul(X_g, Weight_g, WeightScale_g) + Bias_g`，再拼接到单 tensor `y`。
3. 异常用例只改变一个非法条件，检查 host checker 返回失败并匹配日志关键字。
4. 白盒 TilingKey 用例通过日志或断点检查 `tilingKeyConfigure.isSingleMultiSingle` 和实际 TilingKey 组合。
5. 回归用例复用既有 MxA8W4 单单单、多多多和其他 weight quant 用例，确认新增模板参数未改变原路径。

本次用户未提供 Excel 模板，因此不生成迭代测试用例 Excel；如后续提供模板，可按本 STC 的用例表生成 JSON 后填充 Excel。

## 10. 交付通路验证

| 通路 | 验证内容 | 关联用例 |
| --- | --- | --- |
| Host checker | list size、dtype、format、shape、groupList、属性组合 | ST001、NEG001~NEG015 |
| Host tiling | `groupNum/kList/nList` 推导，`SetBaseTiling` 写入 | ST001、STG001、STG004 |
| TilingKey | `IS_SINGLE_MULTI_SINGLE` 取 1/0 的组合合法性 | ST005、HIST001 |
| Kernel | resplit controller 执行多 group，输出与 golden 对齐 | ST001、ST002、ST006、STG002 |
| 回归 | 非单多单路径行为不变 | HIST001~HIST004 |

## 11. 实现/Tiling 对照

| 需求点 | 实现位置 | 测试关注 |
| --- | --- | --- |
| 单多单 shape list 分支 | `grouped_weight_quant_batch_matmul_tiling.cpp::SetShapeList` | ST001、NEG001 |
| list size 校验 | `CheckTensorListSize` | NEG002~NEG004 |
| 单多单逐 tensor 校验入口 | `CheckEveryTensorSingleXMultiWeightSingleY` | NEG006~NEG013 |
| groupList 长度按 weight list size 校验 | `CheckGroupList` | NEG005 |
| TilingKey 单多单 bool | `TilingKeyConfigure::isSingleMultiSingle`、`GenTilingKey` | ST005、HIST001 |
| 模板参数合法组合 | `weight_quant_tiling_key.h` | ST005、NEG015 |
| Kernel 入口透传 | `grouped_matmul_apt.cpp` | ST006 |
| resplit 地址更新 | `GMMWeightQuantResplitController::UpdateGmAddr` | ST006、ST007 |

## 12. 风险与未覆盖

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| `weight` 各 group 的 `Ng` 不一致时，tiling 统一 `nSize` 与实际 group N 关系可能存在限制 | 可能出现结果拼接或尾块切分异常 | STG004 覆盖不同 `Ng`；若实现实际只支持同 N，应在 checker 或 SRS 中补充限制 |
| bias dtype 支持范围未在 SRS 中完全展开 | 异常矩阵可能缺漏 dtype 组合 | 用既有 MxA8W4 checker 为准，ST002/STG005 覆盖主流合法 dtype |
| `antiquantOffsetOptional` SRS 主体未重点描述但代码存在校验 | 可能漏测可选输入长度/shape | STG006、NEG004 扩展覆盖 antiquantOffset |
| Excel 模板未提供 | 无法生成迭代测试用例表 | 待用户提供模板后补交 Excel |

## 附录 A：与 SRS 的一致性检查

| SRS 条目 | STC 覆盖 |
| --- | --- |
| MxA8W4 单多单 happy path | ST001、ST002、ST003 |
| bias 存在：bias 多 tensor 且长度等于 weight | ST002、NEG004 |
| bias 不存在：不访问 bias tensor list | ST001、ST006 |
| weightScale 长度与 weight 不一致 | NEG003 |
| bias 长度与 weight 不一致 | NEG004 |
| 非单多单 MxA8W4 回归 | HIST001、HIST002 |
| TilingKey 白盒检查 `IS_SINGLE_MULTI_SINGLE == 1` | ST005 |

## 附录 B：用例清单

### ST-白盒用例

| 编号 | 用例名称 | 输入要点 | 操作步骤 | 预期结果 |
| --- | --- | --- | --- | --- |
| ST001 | 单多单最小 happy path 无 bias | M=32, K=64, N_list=[128,128], groupNum=2, groupListType=1, splitItem=3, transA=false, transB=true | 构造 x 单 tensor、weight/antiquantScale 各 2 个 tensor、y 单 tensor；运行 aclnn；对比 golden | 执行成功，输出与 golden 对齐，TilingKey 单多单取 1 |
| ST002 | 单多单存在 bias | M=64, K=64, N_list=[128,128], groupNum=2, bias list size=2 | 在 ST001 基础上传入 bias[0..1]；运行 aclnn；对比 golden | 执行成功，bias 生效，输出与 golden 对齐 |
| ST003 | splitItem=2 的 y 单 tensor 场景 | M=64, K=128, groupNum=2, splitItem=2 | 构造 y 单 tensor 的 splitItem=2 调用 | Host 识别为单多单，执行成功 |
| ST004 | groupListType=0 累计 offset | groupList=[16,64,96], groupNum=3 | 按累计 offset 切分 x；运行并对比每组输出 | 每组 M 大小按差值解析，输出正确 |
| ST005 | TilingKey 白盒检查 | MxA8W4 单多单合法输入 | 运行 tiling，检查日志或断点中的 `isSingleMultiSingle` 和 TilingKey 参数 | `IS_SINGLE_MULTI_SINGLE == 1`，模板组合合法 |
| ST006 | Kernel 多 group 地址切换无 bias | groupNum=4, M/K/N 均合法，无 bias | 对每组 weight 填不同可识别数据；运行并对比 golden | 每组使用对应 `weight[g]`/`antiquantScale[g]`，无串组 |
| ST007 | Kernel 多 group 地址切换有 bias | groupNum=4, bias[g] 填不同偏置 | 运行并检查每组 bias 贡献 | 每组使用对应 `bias[g]` |
| ST008 | 最后一组不越界访问 | groupNum=2，最后一组 M 较小 | 运行启用内存检查或日志断点 | 最后一组后不访问 nextGroupIdx，执行成功 |
| ST009 | N 尾块 resplit | N=384 或非 core*256 整倍，groupNum=2 | 运行尾块场景并对比 golden | 尾块输出正确，resplit 参数合法 |
| ST010 | perTokenScale 单 tensor 推进 | groupNum=3，不同 group M | 设置 perTokenScale 不同区间值；运行并对比 golden | perTokenScale 按 x 的 M 方向切分使用 |

### ST-泛化用例

| 编号 | 用例名称 | 输入要点 | 操作步骤 | 预期结果 |
| --- | --- | --- | --- | --- |
| STG001 | groupNum=8 泛化 | M=256, K=128, groupNum=8 | 构造 8 个 weight/antiquantScale tensor | 执行成功，groupNum 推导为 8 |
| STG002 | 大 K 场景 | K=512, groupNum=4 | 构造满足 MX group size 的 scale shape | 输出与 golden 对齐 |
| STG003 | 小 K 边界 | K=64, groupNum=2 | 构造最小合法 MxA8W4 K 分组 | 执行成功 |
| STG004 | 不同 group N | N_list=[128,256,128] | 构造各组不同 Ng；若实现有限制则记录 checker 行为 | 支持时输出正确；不支持时需有明确约束日志 |
| STG005 | bias 存在且 groupNum=8 | groupNum=8, bias list size=8 | 运行并对比 golden | 执行成功 |
| STG006 | antiquantOffset 存在 | antiquantOffset list size=weight size | 运行包含 antiquantOffset 的合法场景 | 执行成功或按既有 MxA8W4 支持返回明确结果 |
| STG007 | 含 0 M 或极小 M group | groupListType=1, groupList 含 0/1 边界 | 运行并检查输出拼接 | 不崩溃，合法部分输出正确；若不支持需 checker 拦截 |
| STG008 | 接近最大 tensor list 数 | groupNum 接近框架支持上限 | 构造大 groupNum 输入 | Host 不越界，合法时执行成功 |

### 历史用例回归

| 编号 | 用例名称 | 输入要点 | 操作步骤 | 预期结果 |
| --- | --- | --- | --- | --- |
| HIST001 | MxA8W4 单单单回归 | x/weight/y 均单 tensor，weight 内含 group 维 | 运行既有 MxA8W4 单单单用例 | `IS_SINGLE_MULTI_SINGLE == 0`，结果不变 |
| HIST002 | MxA8W4 多多多回归 | x/weight/y 均多 tensor | 运行既有多多多用例 | 不误入单多单路径，结果不变 |
| HIST003 | S8S4 weight quant 回归 | int8/int4 路径 | 运行既有 S8S4 用例 | TilingKey 新参数为 0，结果不变 |
| HIST004 | A16W4ND 回归 | bf16/fp16 + int4 ND 路径 | 运行既有 A16W4ND 用例 | checker 与结果不变 |

### NEG-异常用例

| 编号 | 用例名称 | 异常注入 | 预期结果 |
| --- | --- | --- | --- |
| NEG001 | x 不是单 tensor | `numX > 1`，weight 多，y 单 | Host checker 失败，日志包含 single-multi-single 下 x size 必须为 1 |
| NEG002 | weight 不是多 tensor | `numWeight = 1`，x 单，y 单 | Host checker 失败，日志包含 weight must be greater than 1 |
| NEG003 | antiquantScale 长度不等于 weight | `numAntiquantScale != numWeight` | Host checker 失败，日志包含 antiquantScaleOptional size mismatch |
| NEG004 | bias 长度不等于 weight | hasBias=true，`numBias != numWeight` | Host checker 失败，日志包含 biasOptional size mismatch |
| NEG005 | groupList 长度不等于 groupNum | `groupList.shape[0] != numWeight` | Host checker 失败，日志包含 groupList length must match groupNum |
| NEG006 | x dtype 非 FLOAT8_E4M3FN | 将 x dtype 改为 INT8 或 FLOAT16 | 不进入 MxA8W4 单多单合法路径，checker 失败或走既有路径失败 |
| NEG007 | weight dtype 非 FLOAT4_E2M1 | 将 weight dtype 改为 INT4/BF16 | MxA8W4 dtype/dataflow 校验失败 |
| NEG008 | weight format 非 FRACTAL_NZ | weight 使用 ND format | Host checker 失败，日志包含 weight format 期望 FRACTAL_NZ |
| NEG009 | weight dim 非 4 | 单个 weight tensor 改成 3 维或 5 维 | Host checker 失败，日志包含 weight dimension |
| NEG010 | antiquantScale dim 非 4 | antiquantScale[g] 改为 3 维 | Host checker 失败，日志包含 antiquantScale dimension |
| NEG011 | antiquantScale K 分组非法 | `scaleGroupNum <= 0` 或 `K % scaleGroupNum != 0` | Host checker 失败，日志包含 groupNum/groupSize |
| NEG012 | antiquantScale N 与 weight 不一致 | antiquantScale[g] 的 N 轴不等于 weight[g] 的 N 轴 | Host checker 失败，日志包含 NDim of antiquantScale must be equal |
| NEG013 | bias dim 或 N 不匹配 | bias[g] 非 1 维或长度不等于 Ng | Host checker 失败，日志包含 bias dimension 或 NDim |
| NEG014 | transB=false | MxA8W4 单多单输入但 `transB=false` | Host checker 失败，日志包含 transposed and FRACTAL_NZ-format |
| NEG015 | 非 MxA8W4 单多单组合 | 使用 A16W4ND 或其他数据流做 x 单、weight 多、y 单 | 不启用 `IS_SINGLE_MULTI_SINGLE`，checker 按不支持组合失败 |
