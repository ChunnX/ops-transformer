# aclnnGridSampler2D 参数说明

## 基本信息

- **算子名称**：aclnnGridSampler2D（两段式接口：aclnnGridSampler2DGetWorkspaceSize + aclnnGridSampler2D）
- **目标产品系列**：Ascend 950PR/Ascend 950DT（接口文档"产品支持情况"章节标注"支持"；文档中 910/A2/A3/310b/310p 的专属限制段不适用于目标产品，按产品隔离原则不提取）
- **功能说明**：提供一个输入 Tensor（input）以及一个对应的 grid 网格，根据 grid 中每个位置提供的归一化坐标 (x, y)，将 input 中对应位置的像素值填充到网格指定的位置，得到最终输出。input 形状为 (N, C, H_in, W_in)，grid 形状为 (N, H_out, W_out, 2)，输出 out 形状为 (N, C, H_out, W_out)。grid 最后一维依次存放 x 和 y 坐标，坐标通常归一化到 [-1, 1]（(-1,1) 表示左上角坐标，(1,-1) 表示右下角坐标）；实际输入坐标由 alignCorners 决定反归一化方式；越界坐标按照 paddingMode 指定的 zeros/border/reflection 方式处理，再按照 interpolationMode 指定的 bilinear/nearest/bicubic 方式计算输出。确定性计算：默认确定性实现。

## 参数列表

| 参数名 | 类型 | 输入/输出 | 必选/可选 | 维度/长度 | 数据格式 | 数据类型 | 取值范围 | 空tensor | 非连续 | 补充说明 |
|--------|------|-----------|-----------|-----------|----------|----------|----------|----------|--------|----------|
| input | aclTensor | 输入 | 必选 | 4维（固定） | ND | FLOAT32、FLOAT16、DOUBLE、BFLOAT16 | 依赖具体数据类型 | 支持 | 支持 | shape 为 (N, C, H_in, W_in)；H_in×W_in < INT32 最大值（R7）；shape 最后两维（H_in、W_in）的维度值不能为 0，即 H_in≥1、W_in≥1（R8）；N、C 可为 0（空 tensor 场景）；数据格式只支持按 (N, C, H, W) 处理，输入其他格式时默认按 NCHW 处理（来源：约束说明） |
| grid | aclTensor | 输入 | 必选 | 4维（固定） | ND | FLOAT32、FLOAT16、DOUBLE、BFLOAT16 | 归一化坐标，通常在 [-1, 1] 之间；元素功能语义值域：场景A: [-1.0, 1.0]（范围内正常采样）；场景B: 超出 [-1.0, 1.0]（越界坐标按 paddingMode 处理）（推断自：功能说明"会将x和y的取值范围归一化到[-1,1]之间"+"对于超出范围的坐标，会根据paddingMode进行不同处理"） | 支持 | 支持 | 数据类型与入参 input 的数据类型一致（R1）；shape 为 (N, H_out, W_out, 2)，N 与 input 的 shape 中的 N 一致（R3），最后一维固定为 2；各维均可为 0（空 tensor/空输出）；约束说明：当 grid 的输入值×图片（长或宽）大于 24 位二进制数（16777216）时采样点可能存在误差（精度提示，非硬约束） |
| interpolationMode | int64_t | 输入 | 必选 | - | - | INT64（固定） | 0（bilinear 双线性插值）、1（nearest 最邻近插值）、2（bicubic 双三次插值） | - | - | 值域 R5；interpolationMode 为 bicubic(2) 时，input、grid、out 的数据类型须为 FLOAT32 或 FLOAT16（R6，推断自错误码表 161002"interpolationMode为bicubic时，input、grid、out的数据类型不是FLOAT32或者FLOAT16"） |
| paddingMode | int64_t | 输入 | 必选 | - | - | INT64（固定） | 0（zeros）、1（border）、2（reflection） | - | - | 值域 R5；决定 grid 越界坐标的填充方式：0 用 0 填充、1 用边界值填充、2 用边界值的对称值填充（来源：功能说明） |
| alignCorners | bool | 输入 | 必选 | - | - | BOOL（固定） | true / false | - | - | true 表示特征值位于像素中心，false 表示特征值位于像素的角点；影响 grid 坐标反归一化公式（来源：参数表"使用说明"列） |
| out | aclTensor | 输出 | 必选 | 4维（固定） | ND | FLOAT32、FLOAT16、DOUBLE、BFLOAT16 | 不适用（输出参数） | 支持 | 支持 | 数据类型与 input 的数据类型一致（R2）；shape 为 (N, C, H_out, W_out)，其中 N、C 与 input 的 shape 中的 N、C 一致，H_out、W_out 与 grid 的 shape 中的 H_out、W_out 一致（R4） |

## 约束关系

### R1: grid 数据类型与 input 一致
涉及参数: input.dtype → grid.dtype
条件因子: 无
来源: 参数说明表 grid"使用说明"列 "数据类型与入参`input`的数据类型一致"；错误码表 161002 "input、grid、out的数据类型不在支持的范围之内或数据类型不一致"

### R2: out 数据类型与 input 一致
涉及参数: input.dtype → out.dtype
条件因子: 无
来源: 参数说明表 out"使用说明"列 "数据类型与input的数据类型一致"

### R3: input/grid 均为 4 维，grid.shape=(N, H_out, W_out, 2) 且 batch 维与 input 一致、末维固定为 2
涉及参数: input.shape, grid.dimensions → grid.shape
条件因子: 无
来源: 参数说明表 input/grid"使用说明"列 shape 约束；错误码表 161002 "input、grid、out的维度关系不匹配"

### R4: out.shape=(N, C, H_out, W_out)，N、C 取自 input，H_out、W_out 取自 grid
涉及参数: input.shape, grid.shape → out.shape
条件因子: 无
来源: 参数说明表 out"使用说明"列 "支持shape为(N, C, H_out, W_out)，且N、C与input的shape中的N、C一致，H_out、W_out与grid的shape中的H_out、W_out一致"

### R5: interpolationMode ∈ {0, 1, 2}；paddingMode ∈ {0, 1, 2}
涉及参数: interpolationMode.value, paddingMode.value
条件因子: 无
来源: 参数说明表"使用说明"列；错误码表 161002 "interpolationMode或paddingMode的值不在支持范围内"

### R6: interpolationMode 为 bicubic(2) 时，input、grid、out 的数据类型须为 FLOAT32 或 FLOAT16
涉及参数: interpolationMode.value → input.dtype
条件因子: interpolationMode.value
来源: 错误码表 161002 "interpolationMode为bicubic时，input、grid、out的数据类型不是FLOAT32或者FLOAT16"

### R7: input 的 H_in×W_in < INT32 最大值（2147483647）
涉及参数: input.shape
条件因子: 无
来源: 参数说明表 input"使用说明"列 "H*W < INT32的最大值"；约束说明 "输入`input`的（H轴的大小 * W轴的大小）< INT32的最大值"

### R8: input 的最后两维（H_in、W_in）≥ 1；N、C ≥ 0（可为 0，空 tensor）；grid/out 各维 ≥ 0
涉及参数: input.shape, grid.shape, out.shape
条件因子: 无
来源: 参数说明表 input"使用说明"列 "支持空Tensor"+"input的shape最后两维的维度值不能为0"；grid/out"使用说明"列 "支持空Tensor"

### R9: grid 元素为归一化坐标，基准区间 [-1, 1]；超出 [-1, 1] 的越界坐标按 paddingMode 语义处理（功能场景约束，通过 grid 值域分段实现场景A/B 覆盖）
涉及参数: grid.value_range
条件因子: paddingMode.value（越界处理方式由 paddingMode 决定）
来源: 功能说明 "会将x和y的取值范围归一化到[-1,1]之间" + "对于超出范围的坐标，会根据paddingMode进行不同处理"

## 约束提取反向校验表

| R{n} | 约束项 | 原文片段 | 符号化 | 转录结果 | 状态 |
|------|--------|---------|--------|---------|------|
| R1 | grid dtype 一致 | "数据类型与入参`input`的数据类型一致" | 一致 | grid.dtype == input.dtype | EXACT_MATCH |
| R2 | out dtype 一致 | "数据类型与input的数据类型一致" | 一致 | out.dtype == input.dtype | EXACT_MATCH |
| R3 | grid shape 关系 | "支持shape为(N, H_out, W_out, 2)，且N与入参`input`的shape中的N一致" | 长度=4、末维=2、N 一致 | grid.shape=[N,H_out,W_out,2]，N=input.shape[0] | EXACT_MATCH |
| R4 | out shape 公式 | "N、C与input的shape中的N、C一致，H_out、W_out与grid的shape中的H_out、W_out一致" | 匹配 | out.shape=[input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]] | EXACT_MATCH |
| R5 | 模式枚举值域 | "支持0：bilinear（双线性插值）、1：nearest（最邻近插值）、2：bicubic（双三次插值）" / "支持0：zeros、1：border、2：reflection" | ∈ {0,1,2} | interpolationMode ∈ {0,1,2}；paddingMode ∈ {0,1,2} | EXACT_MATCH |
| R6 | bicubic dtype 限制 | "interpolationMode为bicubic时，input、grid、out的数据类型不是FLOAT32或者FLOAT16" | bicubic → dtype ∈ {FLOAT32, FLOAT16} | mode=2 时 input.dtype ∈ {float32, float16} | EXACT_MATCH |
| R7 | H×W 上界 | "H*W < INT32的最大值" | < 2147483647 | input.shape[2] × input.shape[3] < 2147483647 | EXACT_MATCH |
| R8 | 最后两维非零 | "input的shape最后两维的维度值不能为0" | H_in ≥ 1 ∧ W_in ≥ 1 | input.shape[2] ≥ 1 ∧ input.shape[3] ≥ 1；N/C 可为 0 | EXACT_MATCH |
| R9 | grid 归一化坐标 | "会将x和y的取值范围归一化到[-1,1]之间" | ∈ [-1, 1]（基准区间，越界合法另处理） | grid 值域分段：[-1,1]（场景A）+ 越界段（场景B） | INFERENCE |

待确认项（不阻断流程，已在 spec.md 标注）：文档内部差异——错误码表声明 bicubic 时 dtype 须为 FLOAT32/FLOAT16（本设计按此编码），而 A2/A3 产品段落声明 BFLOAT16 亦支持 bicubic；目标产品 950PR/950DT 无专属说明，以第一段接口实际校验为准。
