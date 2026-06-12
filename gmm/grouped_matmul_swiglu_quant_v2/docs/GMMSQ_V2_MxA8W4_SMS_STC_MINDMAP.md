# GMMSQ V2 MxA8W4 单多单 STC 脑图版

## 1. 所有输入参数

### x

- 必选 / 可选：必选
- tensor 形态：单 tensor
- dtype：`FLOAT8_E4M3FN`
- format：`ND`
- shape：`[M, K]`
- 约束：
  - `M >= 1`
  - `K >= 32`
  - `K % 32 == 0`
  - `x.shape` 维度数必须为 2
- 语义：
  - M 方向按 `groupList` 切分
  - group 间按单 tensor offset 推进

### xScale

- 必选 / 可选：必选
- tensor 形态：单 tensor
- dtype：`FLOAT8_E8M0`
- format：`ND` / `NCL`
- shape：`[M, ceil(K / 64), 2]`
- 约束：
  - shape 必须与 `M`、`K` 对齐
  - `ceil(K / 64)` 由 host tiling 使用 `GroupedMatmul::CeilDiv(K, 64)` 计算
- 语义：
  - 与 `x` 对应
  - group 间按单 tensor offset 推进

### groupList

- 必选 / 可选：必选
- tensor 形态：单 tensor
- dtype：`INT64`
- format：`ND`
- shape：`[groupNum]`
- 属性关联：`groupListType`
  - `0`：累积结束位置
  - `1`：每组 M 大小
- 约束：
  - shape 长度必须等于 `groupNum`
  - `groupListType` 只能为 0 或 1
- ST 数据要求：
  - 正常用例中 `groupList` 内容需与总 `M` 一致
  - 异常用例重点覆盖 shape / dtype / attr，不新增数值非法校验

### weight

- 必选 / 可选：必选
- tensor 形态：
  - 单多单：tensor list，多 tensor
  - 非单多单：单 tensor 回归
- 单多单 list 长度：`groupNum`
- dtype：`FLOAT4_E2M1`
- format：`FRACTAL_NZ` / `FRACTAL_NZ_C0_32`
- 单多单每个成员 `weight[i]`：
  - origin shape：`[N, K]`
  - storage shape：`[ceil(K / 32), ceil(N / 16), 16, 32]`
- 非单多单单 tensor：
  - origin / view shape：`[E, N, K]`
  - storage shape：`[E, ceil(K / 32), ceil(N / 16), 16, 32]`
- 约束：
  - `N >= 128`
  - `N % 128 == 0`
  - `K >= 32`
  - `K % 32 == 0`
  - `x.K == weight.K`
  - 单多单 `weight[i]` origin 维度数为 2
  - 非单多单 `weight` origin 维度数为 3

### weightScale

- 必选 / 可选：必选
- tensor 形态：
  - 单多单：tensor list，多 tensor
  - 非单多单：单 tensor 回归
- 单多单 list 长度：
  - 必须等于 `weight` tensor list 长度
  - `weightScaleListSize == weightListSize`
- dtype：`FLOAT8_E8M0`
- format：`ND` / `NCHW`
- 单多单每个成员 `weightScale[i]`：
  - shape：`[N, ceil(K / 64), 2]`
- 非单多单单 tensor：
  - shape：`[E, N, ceil(K / 64), 2]`
- 约束：
  - `weightScale[i]` shape 与 `N`、`K` 一致

### weightAssistMatrix

- 必选 / 可选：可选
- 当前支持状态：MxA8W4 weight quant 本路径不支持
- 输入要求：
  - 空指针
  - 空 list
- 异常输入：
  - 传入实际 tensor / list
- 期望报错关键字：`weightAssistMatrix is not supported`

### bias

- 必选 / 可选：可选
- 当前支持状态：MxA8W4 weight quant 本路径不支持
- 输入要求：空指针
- 异常输入：传入实际指针
- 期望报错关键字：`bias is not supported`

### smoothScale

- 必选 / 可选：可选
- 当前支持状态：MxA8W4 weight quant 本路径不支持
- 输入要求：空指针
- 异常输入：传入实际指针
- 期望报错关键字：`smoothScale is not supported`

### 标量属性

- `dequantMode`
  - 类型：`int64`
  - 取值：`2`
  - 语义：MX quant
- `dequantDtype`
  - 类型：`int64`
  - 取值：`DT_FLOAT`
- `quantMode`
  - 类型：`int64`
  - 取值：`2`
  - 语义：MX quant
- `quantDtype`
  - 类型：`int64`
  - 取值：`FLOAT8_E4M3FN`
- `transposeWeight`
  - 类型：`bool`
  - 取值：`true`
- `groupListType`
  - 类型：`int64`
  - 取值：`0` / `1`
- `tuningConfig`
  - 当前不新增语义
  - 沿用既有约束

### 输出参数

- `y` / `output`
  - dtype：`FLOAT8_E4M3FN`
  - format：`ND`
  - shape：`[M, N / 2]`
  - 语义：单 tensor，按 group 输出连续写回
- `yScale` / `outputScale`
  - dtype：`FLOAT8_E8M0`
  - format：`ND`
  - shape：`[M, ceil(N / 2 / 64), 2]`
  - 语义：单 tensor，按 group 输出 scale 连续写回

---

## 异常用例

### TensorList 长度异常

- NEG001：`weightScale` list 少于 `weight`
  - 构造：`weightListSize = 3`，`weightScaleListSize = 2`
  - 违反规则：`weightListSize == weightScaleListSize`
  - 期望：tiling 失败
  - 日志关键字：`weight and weightScale tensor list size should be equal`
- NEG002：`weightScale` list 多于 `weight`
  - 构造：`weightListSize = 2`，`weightScaleListSize = 3`
  - 违反规则：`weightListSize == weightScaleListSize`
  - 期望：tiling 失败
  - 日志关键字：`weight and weightScale tensor list size should be equal`
- NEG003：`weight` 单 tensor 但 `weightScale` 多 tensor
  - 构造：`weightListSize = 1`，`weightScaleListSize = 2`
  - 违反规则：任一侧多 tensor 时两侧 list 长度一致
  - 期望：tiling 失败
  - 日志关键字：`weight and weightScale tensor list size should be equal`

### TensorList 成员 desc 异常

- NEG004：`weight[i]` desc 为空
  - 构造：`weight[1]` desc 为 nullptr
  - 期望：tiling 失败
  - 日志关键字：`Dynamic input desc of weight[1] is nullptr`
- NEG005：`weightScale[i]` desc 为空
  - 构造：`weightScale[1]` desc 为 nullptr
  - 期望：tiling 失败
  - 日志关键字：`Dynamic input desc of weightScale[1] is nullptr`

### dtype 异常

- NEG006：`x` dtype 非法
  - 构造：`x.dtype != FLOAT8_E4M3FN`
  - 期望：tiling 失败
  - 日志关键字：`x dtype must be FLOAT8_E4M3FN`
- NEG007：`weight[i]` dtype 非法
  - 构造：`weight[1].dtype != FLOAT4_E2M1`
  - 期望：tiling 失败
  - 日志关键字：`weight[1] dtype must be`
- NEG008：`xScale` dtype 非法
  - 构造：`xScale.dtype != FLOAT8_E8M0`
  - 期望：tiling 失败
  - 日志关键字：`xScale dtype must be FLOAT8_E8M0`
- NEG009：`weightScale[i]` dtype 非法
  - 构造：`weightScale[1].dtype != FLOAT8_E8M0`
  - 期望：tiling 失败
  - 日志关键字：`weightScale[1] dtype must be`
- NEG010：`groupList` dtype 非法
  - 构造：`groupList.dtype != INT64`
  - 期望：tiling 失败
  - 日志关键字：`groupList dtype must be INT64`

### format 异常

- NEG011：`x` format 非法
  - 构造：`x.format != ND`
  - 期望：tiling 失败
  - 日志关键字：`x format must be ND`
- NEG012：`xScale` format 非法
  - 构造：`xScale.format` 非 `ND` / `NCL`
  - 期望：tiling 失败
  - 日志关键字：`xScale format must be ND or NCL`
- NEG013：`groupList` format 非法
  - 构造：`groupList.format != ND`
  - 期望：tiling 失败
  - 日志关键字：`groupList format must be ND`
- NEG014：`weight[i]` format 非法
  - 构造：`weight[1].format` 非 `FRACTAL_NZ` / `FRACTAL_NZ_C0_32`
  - 期望：tiling 失败
  - 日志关键字：`Weight[1] format is only support`
- NEG015：`weightScale[i]` format 非法
  - 构造：`weightScale[1].format` 非 `ND` / `NCHW`
  - 期望：tiling 失败
  - 日志关键字：`weightScale[1] format must be ND or NCHW`

### empty tensor / shape 空异常

- NEG016：`weight[i]` empty tensor
  - 构造：`weight[1].storageShape.GetShapeSize() == 0`
  - 期望：tiling 失败
  - 日志关键字：`Not yet support empty tensor for weight[1]`
- NEG017：`weightScale[i]` empty tensor
  - 构造：`weightScale[1].storageShape.GetShapeSize() == 0`
  - 期望：tiling 失败
  - 日志关键字：`Not yet support empty tensor for weightScale[1]`
- NEG018：`x` empty tensor
  - 构造：`x.shape.GetShapeSize() == 0`
  - 期望：tiling 失败
  - 日志关键字：`Not yet support empty tensor for x`

### shape / 维度异常

- NEG019：`x` 维度数非法
  - 构造：`x` 非 2 维
  - 期望：tiling 失败
  - 日志关键字：`input dimension of x should be 2`
- NEG020：单多单 `weight[0]` origin 维度数非法
  - 构造：`weight[0]` 非 2 维
  - 期望：tiling 失败
  - 日志关键字：`input dimension of weight should be 2`
- NEG021：`x.K != weight.K`
  - 构造：`x.shape[-1] != weight.shape[-1]`
  - 期望：tiling 失败
  - 日志关键字：`Inputs dimension is not match`
- NEG022：`xScale` shape 非法
  - 构造：非 `[M, ceil(K / 64), 2]`
  - 期望：tiling 失败
  - 日志关键字：`Check input xScale shape failed`
- NEG023：`groupList` shape 非法
  - 构造：非 `[groupNum]`
  - 期望：tiling 失败
  - 日志关键字：`Check input groupList shape failed`
- NEG024：`weight[i]` storage shape 非法
  - 构造：非 `[ceil(K / 32), ceil(N / 16), 16, 32]`
  - 期望：tiling 失败
  - 日志关键字：`Check input weight shape failed`
- NEG025：`weightScale[i]` shape 非法
  - 构造：非 `[N, ceil(K / 64), 2]`
  - 期望：tiling 失败
  - 日志关键字：`Check input weightScale shape failed`

### M / K / N / groupNum 边界异常

- NEG026：`M < 1`
  - 构造：`M = 0`
  - 期望：tiling 失败
  - 日志关键字：`M size` / `< min 1`
- NEG027：`K < 32`
  - 构造：`K = 16`
  - 期望：tiling 失败
  - 日志关键字：`K size` / `< min 32`
- NEG028：`K` 非 32 对齐
  - 构造：`K = 48`
  - 期望：tiling 失败
  - 日志关键字：`not 32 aligned`
- NEG029：`N < 128`
  - 构造：`N = 64`
  - 期望：tiling 失败
  - 日志关键字：`N size` / `< min 128`
- NEG030：`N` 非 128 对齐
  - 构造：`N = 192`
  - 期望：tiling 失败
  - 日志关键字：`not 128 aligned`
- NEG031：`groupNum > 1024`
  - 构造：`weightListSize = 1025`
  - 期望：tiling 失败
  - 日志关键字：`The input range for the group num is [1, 1024]`

### 属性异常

- NEG032：`dequantMode` 非法
  - 构造：`dequantMode != 2`
  - 期望：tiling 失败
  - 日志关键字：`dequantMode must be 2`
- NEG033：`dequantDtype` 非法
  - 构造：`dequantDtype != DT_FLOAT`
  - 期望：tiling 失败
  - 日志关键字：`dequantDtype must be`
- NEG034：`quantMode` 非法
  - 构造：`quantMode != 2`
  - 期望：tiling 失败
  - 日志关键字：`quantMode must be 2`
- NEG035：`quantDtype` 非法
  - 构造：`quantDtype != FLOAT8_E4M3FN`
  - 期望：tiling 失败
  - 日志关键字：`quantDtype must be`
- NEG036：`transposeWeight` 非法
  - 构造：`transposeWeight = false`
  - 期望：tiling 失败
  - 日志关键字：`transposeWeight must be true for MxA8W4`
- NEG037：`groupListType` 非法
  - 构造：`groupListType` 非 0 / 1
  - 期望：tiling 失败
  - 日志关键字：`groupListType must be 0 (cumsum) or 1 (count)`

### 可选输入异常

- NEG038：`weightAssistMatrix` 实际
  - 构造：实际指针或实际 list
  - 期望：tiling 失败
  - 日志关键字：`weightAssistMatrix is not supported`
- NEG039：`bias` 实际
  - 构造：`bias` 实际指针
  - 期望：tiling 失败
  - 日志关键字：`bias is not supported`
- NEG040：`smoothScale` 实际
  - 构造：`smoothScale` 实际指针
  - 期望：tiling 失败
  - 日志关键字：`smoothScale is not supported`

---

## ST用例

### ST 正常用例设计原则

- 覆盖单多单主路径
- 覆盖 `groupListType = 0 / 1`
- 覆盖 `weight` 两种 format
- 覆盖 `weightScale` 两种 format
- 覆盖 `xScale` 两种 format
- 覆盖 K / N 最小值和对齐边界
- 覆盖 `groupNum` 小值、中值、上界
- 覆盖 TilingKey 白盒检查
- 覆盖 kernel group 间 `weight` / `weightScale` 地址切换

### 白盒 ST 用例

- ST001：单多单最小 K/N，`groupListType=0`
  - 目标：单多单 happy path，TilingKey 第 3 参数为 1
  - `groupNum = 2`
  - `M = 16, K = 32, N = 128`
  - `xScale.format = ND`
  - `weight.format = FRACTAL_NZ`
  - `weightScale.format = ND`
  - `groupListType = 0`
  - 期望：aclnn 成功，结果与 golden 对齐
- ST002：单多单最小 K/N，`groupListType=1`
  - 目标：覆盖 `groupListType=1`
  - `groupNum = 2`
  - `M = 16, K = 32, N = 128`
  - `weight.format = FRACTAL_NZ`
  - `weightScale.format = ND`
  - `groupListType = 1`
  - 期望：成功
- ST003：`weight.format = FRACTAL_NZ_C0_32`
  - 目标：覆盖 `weight` C0_32 format
  - `groupNum = 3`
  - `M = 33, K = 64, N = 256`
  - `weight.format = FRACTAL_NZ_C0_32`
  - `weightScale.format = ND`
  - 期望：成功
- ST004：`weightScale.format = NCHW`
  - 目标：覆盖 `weightScale` NCHW format
  - `groupNum = 3`
  - `M = 33, K = 64, N = 256`
  - `weight.format = FRACTAL_NZ`
  - `weightScale.format = NCHW`
  - 期望：成功
- ST005：format 组合覆盖
  - 目标：`FRACTAL_NZ_C0_32 + NCHW`
  - `groupNum = 4`
  - `M = 64, K = 128, N = 256`
  - `groupListType = 1`
  - 期望：成功
- ST006：`xScale.format = NCL`
  - 目标：覆盖 xScale NCL
  - `groupNum = 2`
  - `M = 17, K = 64, N = 128`
  - `xScale.format = NCL`
  - 期望：成功
- ST007：`ceil(K / 64)` 边界 K=64
  - 目标：scaleKSize 为 1
  - `groupNum = 2`
  - `M = 8, K = 64, N = 128`
  - 期望：成功
- ST008：`ceil(K / 64)` 边界 K=96
  - 目标：K 满足 32 对齐但非 64 整数倍，scaleKSize 为 2
  - `groupNum = 2`
  - `M = 8, K = 96, N = 128`
  - 期望：成功
- ST009：SwiGLU 输出 N/2
  - 目标：覆盖 `y.shape = [M, N / 2]`
  - `groupNum = 2`
  - `M = 20, K = 128, N = 256`
  - `groupListType = 1`
  - 期望：成功，输出 shape 正确
- ST010：`groupNum = 1024` 上界
  - 目标：覆盖最大 tensor list 长度
  - `groupNum = 1024`
  - `M = 1024, K = 32, N = 128`
  - `groupListType = 1`
  - 期望：tiling 成功，运行可按资源情况选择冒烟 / 白盒
- ST013：单多单 TilingKey 白盒
  - 目标：检查 `IS_SINGLE_MULTI_SINGLE == 1`
  - 构造：正常单多单小 shape
  - 期望：TilingKey 第 3 参数为 1
- ST015：kernel `weight` 地址切换
  - 目标：验证 group 间从 tensor list 切换 `weight[i]`
  - `groupNum = 3`
  - 每组 `weight[i]` 填不同 pattern
  - 期望：输出能反映每组不同 weight
- ST016：kernel `weightScale` 地址切换
  - 目标：验证 group 间从 tensor list 切换 `weightScale[i]`
  - `groupNum = 3`
  - 每组 `weightScale[i]` 填不同 pattern
  - 期望：输出 scale / 最终结果与 golden 对齐

### 历史用例回归

- 复用既有 GMMSQ V2 MxA8W4 单 tensor weight 回归用例
- 覆盖非单多单路径 `IS_SINGLE_MULTI_SINGLE == 0`
- 覆盖既有 `FRACTAL_NZ` / `FRACTAL_NZ_C0_32` weight format
- 确认连续 offset 推进路径行为不变

### 泛化 ST 用例

- ST017：随机正常小 M
  - `groupNum = 2`
  - `M = 7, K = 32, N = 128`
  - format：`FRACTAL_NZ / ND / xScale ND`
- ST018：随机正常多 group
  - `groupNum = 4`
  - `M = 31, K = 64, N = 256`
  - format：`FRACTAL_NZ_C0_32 / ND / xScale NCL`
- ST019：随机正常奇数 group
  - `groupNum = 7`
  - `M = 113, K = 96, N = 384`
  - format：`FRACTAL_NZ / NCHW / xScale ND`
- ST020：随机正常中等规模
  - `groupNum = 16`
  - `M = 257, K = 128, N = 512`
  - format：`FRACTAL_NZ_C0_32 / NCHW / xScale NCL`
- ST021：随机正常较大 groupNum
  - `groupNum = 64`
  - `M = 1024, K = 256, N = 1024`
  - format：`FRACTAL_NZ / ND / xScale ND`
- ST022：大 K 小 N
  - `groupNum = 3`
  - `M = 65, K = 512, N = 128`
  - format：`FRACTAL_NZ_C0_32 / ND / xScale NCL`
- ST023：大 N
  - `groupNum = 5`
  - `M = 128, K = 64, N = 512`
  - format：`FRACTAL_NZ / NCHW / xScale ND`
- ST024：常见训练形态
  - `groupNum = 8`
  - `M = 512, K = 128, N = 256`
  - format：`FRACTAL_NZ_C0_32 / NCHW / xScale NCL`
- ST025：非整齐 M 与奇数 group
  - `groupNum = 11`
  - `M = 233, K = 96, N = 256`
  - format：`FRACTAL_NZ / ND / xScale ND`
- ST026：稳定性中大样本
  - `groupNum = 32`
  - `M = 768, K = 256, N = 384`
  - format：`FRACTAL_NZ_C0_32 / ND / xScale NCL`
- ST027：最小 K + 大 N
  - `groupNum = 6`
  - `M = 96, K = 32, N = 512`
  - format：`FRACTAL_NZ / NCHW / xScale ND`
- ST028：非整齐 M 大样本
  - `groupNum = 9`
  - `M = 999, K = 128, N = 128`
  - format：`FRACTAL_NZ_C0_32 / NCHW / xScale NCL`
