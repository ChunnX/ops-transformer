# GMM MxA8W4 单多单场景支持 STC 脑图版

## 1. 所有输入参数

### x
- 必选 / 可选：必选
- tensor 形态：单 tensor
- dtype：FLOAT8_E4M3FN
- format：ND
- shape：`[M, K]`
- 约束：list size 为 1；dim 为 2；`K` 满足 Mx group size 32 相关 shape 约束；`M` 按 `groupList` 在 M 方向切分

### weight
- 必选 / 可选：必选
- tensor 形态：多 tensor，`weight[g]` 对应第 g 个 group
- dtype：FLOAT4_E2M1
- format：FRACTAL_NZ
- shape：单个 tensor shape：view shape `[Ng, K]`；storage shape `[K/32, Ng/16, 16, 32]`
- 约束：list size 等于 `groupNum` 且大于 1；dim 为 4；MxA8W4 场景要求 `transB=true`

### antiquantScaleOptional
- 必选 / 可选：必选
- tensor 形态：多 tensor，代码中对应 weightScale / `antiquantScaleGm_`
- dtype：FLOAT8_E8M0
- format：ND
- shape：单个 tensor shape：`[Ng, K/64, 2, 2]`
- 约束：list size 等于 `weight` list size；dim 为 4；N 轴与同下标 `weight[g]` 的 N 轴一致；MX group size 为 32

### biasOptional
- 必选 / 可选：可选
- tensor 形态：存在时为多 tensor，`bias[g]` 对应第 g 个 group
- dtype：与既有 MxA8W4 支持保持一致
- format：ND
- shape：单个 tensor shape：`[Ng]`
- 约束：存在时 list size 等于 `weight` list size；dim 为 1；N 轴与同下标 `weight[g]` 的 N 轴一致

### antiquantOffsetOptional
- 必选 / 可选：可选
- tensor 形态：存在时为多 tensor
- dtype：与既有 MxA8W4 支持保持一致
- format：ND
- shape：单个 tensor shape：按既有 MxA8W4 antiquantOffset 约束
- 约束：存在时 list size 等于 `weight` list size；dtype/shape 与同下标 `weight[g]` 对齐

### perTokenScaleOptional
- 必选 / 可选：必选
- tensor 形态：单 tensor
- dtype：FLOAT8_E8M0
- format：ND
- shape：`[M, K/64, 2]`
- 约束：与 `x` 的 M/K 对齐；按 M 方向 group 切分使用

### groupList
- 必选 / 可选：必选
- tensor 形态：单 tensor
- dtype：INT64
- format：ND
- shape：`[groupNum]`
- 约束：长度等于 `weight` list size；`groupListType=0` 表示累计 offset；`groupListType=1` 表示每组 M 大小

### y
- 必选 / 可选：输出
- tensor 形态：单 tensor
- dtype：既有 MxA8W4 输出 dtype
- format：ND
- shape：`[M, N]`
- 约束：按 group 输出连续写回；`splitItem=2/3` 时为单 tensor

### groupType
- 必选 / 可选：属性
- 类型：int64
- 取值：SPLIT_M (`0`)
- 约束：单多单只覆盖按 M 方向切分

### splitItem
- 必选 / 可选：属性
- 类型：int64
- 取值：`2` 或 `3`
- 约束：用于表达 `y` 为单 tensor 的场景

### groupListType
- 必选 / 可选：属性
- 类型：int64
- 取值：`0` 或 `1`
- 约束：语义沿用既有 GMM 实现

### transA
- 必选 / 可选：属性
- 类型：bool
- 取值：false
- 约束：本需求不扩展 x 转置

### transB
- 必选 / 可选：属性
- 类型：bool
- 取值：true
- 约束：MxA8W4 当前 NZNK 路径要求 B 转置

## 异常用例

### tensor list 长度异常
- NEG001：`x` 不是单 tensor，`numX > 1`；期望 host checker 失败，日志指向单多单下 x size 必须为 1
- NEG002：`weight` 不是多 tensor，`numWeight = 1`；期望 host checker 失败，日志指向 weight must be greater than 1
- NEG003：`antiquantScaleOptional` 长度不等于 `weight`；期望 host checker 失败，日志指向 antiquantScaleOptional size mismatch
- NEG004：`biasOptional` 存在但长度不等于 `weight`；期望 host checker 失败，日志指向 biasOptional size mismatch
- NEG005：`groupList` 长度不等于 `weight` list size；期望 host checker 失败，日志指向 groupList length must match groupNum

### dtype 异常
- NEG006：`x` dtype 非 FLOAT8_E4M3FN；期望不进入 MxA8W4 单多单合法路径，checker 失败或按既有路径失败
- NEG007：`weight` dtype 非 FLOAT4_E2M1；期望 MxA8W4 dtype/dataflow 校验失败

### format / 转置异常
- NEG008：`weight` format 非 FRACTAL_NZ；期望 host checker 失败，日志指向 weight format 期望 FRACTAL_NZ
- NEG014：`transB=false`；期望 host checker 失败，日志指向 MxA8W4 需要 transposed and FRACTAL_NZ-format

### shape / dim 异常
- NEG009：`weight[g]` dim 非 4；期望 host checker 失败，日志包含 weight dimension
- NEG010：`antiquantScale[g]` dim 非 4；期望 host checker 失败，日志包含 antiquantScale dimension
- NEG011：`antiquantScale[g]` 的 K 分组不满足 MX group size 32；期望 host checker 失败，日志包含 groupNum/groupSize
- NEG012：`antiquantScale[g]` 的 N 轴不等于同下标 `weight[g]` 的 N 轴；期望 host checker 失败，日志包含 NDim of antiquantScale must be equal
- NEG013：`bias[g]` 非 1 维或长度不等于 `Ng`；期望 host checker 失败，日志包含 bias dimension 或 NDim

### 范围外组合异常
- NEG015：非 MxA8W4 数据流构造 `x` 单、`weight` 多、`y` 单；期望不启用 `IS_SINGLE_MULTI_SINGLE`，checker 按不支持组合失败

## ST用例

### ST 正常用例设计原则
- 正向用例需覆盖 `x` 单、`weight` 多、`antiquantScale` 多、`y` 单的主路径
- bias 不存在和 bias 多 tensor 存在都需要覆盖
- `groupListType=0` 和 `groupListType=1` 都需要覆盖
- 至少一个用例检查 TilingKey 中 `IS_SINGLE_MULTI_SINGLE == 1`
- 至少一个用例通过可区分的 `weight[g]` / `bias[g]` 数据验证 kernel 按 tensor list 下标切换
- 回归用例需确认非单多单路径 `IS_SINGLE_MULTI_SINGLE == 0`

### 白盒 ST 用例
- ST001：单多单最小 happy path 无 bias；M=32, K=64, N_list=[128,128], groupNum=2；期望输出与 golden 对齐
- ST002：单多单存在 bias；bias list size=2；期望每组 bias 生效且输出正确
- ST003：`splitItem=2` 的 y 单 tensor 场景；期望 host 识别单多单并执行成功
- ST004：`groupListType=0` 累计 offset；groupList=[16,64,96]；期望每组 M 按 offset 差值解析
- ST005：TilingKey 白盒检查；期望 `isSingleMultiSingle=true` 且 `IS_SINGLE_MULTI_SINGLE == 1`
- ST006：Kernel 多 group 地址切换无 bias；不同 `weight[g]` 填不同模式；期望无串组
- ST007：Kernel 多 group 地址切换有 bias；不同 `bias[g]` 填不同偏置；期望每组使用对应 bias
- ST008：最后一组不越界访问；groupNum=2 且最后一组 M 较小；期望执行成功且无 nextGroupIdx 越界
- ST009：N 尾块 resplit；N=384 或非 core*256 整倍；期望尾块输出正确
- ST010：perTokenScale 单 tensor 推进；groupNum=3 且各组 M 不同；期望 perTokenScale 按 M 方向切分使用

### 历史用例回归
- HIST001：MxA8W4 单单单回归；x/weight/y 均单 tensor；期望 `IS_SINGLE_MULTI_SINGLE == 0`，结果不变
- HIST002：MxA8W4 多多多回归；x/weight/y 均多 tensor；期望不误入单多单路径，结果不变
- HIST003：S8S4 weight quant 回归；期望 TilingKey 新参数为 0，结果不变
- HIST004：A16W4ND 回归；期望 checker 与结果不变

### 泛化 ST 用例
- STG001：groupNum=8 泛化；构造 8 个 weight/antiquantScale tensor；期望执行成功
- STG002：大 K 场景；K=512；期望输出与 golden 对齐
- STG003：小 K 边界；K=64；期望最小合法 MxA8W4 K 分组执行成功
- STG004：不同 group N；N_list=[128,256,128]；期望支持时输出正确，不支持时有明确约束日志
- STG005：bias 存在且 groupNum=8；期望执行成功
- STG006：antiquantOffset 存在且 list size 等于 weight；期望执行成功或按既有 MxA8W4 支持返回明确结果
- STG007：含 0 M 或极小 M group；期望不崩溃，合法部分输出正确；若不支持需 checker 拦截
- STG008：groupNum 接近框架支持上限；期望 Host 不越界，合法时执行成功
