# aclnnGridSampler3D 参数说明

## 基本信息

- **算子名称**：aclnnGridSampler3D（两段式接口：aclnnGridSampler3DGetWorkspaceSize + aclnnGridSampler3D）
- **目标产品系列**：Ascend 950PR/Ascend 950DT（接口文档"产品支持情况"章节标注"支持"；文档中 910 的 BFLOAT16 限制段不适用于目标产品，按产品隔离原则不提取）
- **功能说明**：提供一个输入张量 input，以及一个对应的 grid 网格，根据 grid 中每个位置提供的归一化坐标 (x, y, z)，将 input 中对应位置的像素值填充到 grid 指定的位置，得到最终的输出。input 形状为 (N, C, D_in, H_in, W_in)，grid 形状为 (N, D_out, H_out, W_out, 3)，输出 out 形状为 (N, C, D_out, H_out, W_out)。grid 最后一维依次存放 x、y、z 坐标，坐标归一化到 [-1, 1]；实际输入坐标由 alignCorners 决定反归一化方式；越界坐标按照 paddingMode 指定的 zeros/border/reflection 方式处理，再按照 interpolationMode 指定的 bilinear/nearest 方式计算输出（3D 不支持 bicubic）。确定性计算：默认确定性实现。

## 参数列表

| 参数名 | 类型 | 输入/输出 | 必选/可选 | 维度/长度 | 数据格式 | 数据类型 | 取值范围 | 空tensor | 非连续 | 补充说明 |
|--------|------|-----------|-----------|-----------|----------|----------|----------|----------|--------|----------|
| input | aclTensor | 输入 | 必选 | 5维（固定） | NCDHW、NDHWC、ND | FLOAT16、FLOAT32、DOUBLE、BFLOAT16 | 依赖具体数据类型 | 支持 | 支持 | shape 为 (N, C, D_in, H_in, W_in)；当数据格式为 ND 时按 NCDHW 格式处理；仅当满足条件（数据类型为 FLOAT32 且 interpolationMode 为 0 且 C 轴为 4 的倍数）时支持 NDHWC（R9）；D_in×H_in×W_in < INT32 最大值（R7）；空间三维（D_in、H_in、W_in）≥ 1（R8，推断自 README 约束"x和grid的shape，所有维度都必须大于0"+2D 同族接口"最后两维不能为0"，待确认）；N、C 可为 0（空 tensor 场景） |
| grid | aclTensor | 输入 | 必选 | 5维（固定） | ND | FLOAT16、FLOAT32、DOUBLE、BFLOAT16 | 归一化坐标，通常在 [-1, 1] 之间；元素功能语义值域：场景A: [-1.0, 1.0]（范围内正常采样）；场景B: 超出 [-1.0, 1.0]（越界坐标按 paddingMode 处理）（推断自：功能说明"会将x、y、z的取值范围归一化到[-1, 1]之间"+"对于超出范围的坐标，会根据paddingMode进行不同处理"） | 支持 | 支持 | 数据类型与入参 input 的数据类型一致（R1）；shape 为 (N, D_out, H_out, W_out, 3)，N 与 input 的 shape 中的 N 一致（R4），最后一维固定为 3；各维均可为 0（空 tensor/空输出） |
| interpolationMode | int64_t | 输入 | 必选 | - | - | INT64（固定） | 0（bilinear 双线性插值）、1（nearest 最邻近插值）。**3D 不支持 bicubic**（R6） | - | - | 值域 R6；NDHWC 格式时仅支持 0（bilinear）（R9，来源：使用说明"仅当满足条件（数据类型为FLOAT32 && interpolationMode为0 && C轴为4的倍数）时支持NDHWC"） |
| paddingMode | int64_t | 输入 | 必选 | - | - | INT64（固定） | 0（zeros）、1（border）、2（reflection） | - | - | 值域 R6；决定 grid 越界坐标的填充方式：0 用 0 填充、1 用边界值填充、2 用边界值的对称值填充（来源：功能说明） |
| alignCorners | bool | 输入 | 必选 | - | - | BOOL（固定） | true / false | - | - | true 表示极值 -1 和 1 视为参考输入的角像素点的中心点，false 表示视为角像素点的角点；影响 grid 坐标反归一化公式（来源：参数表"使用说明"列） |
| out | aclTensor | 输出 | 必选 | 5维（固定） | NCDHW、NDHWC、ND（与 input 保持一致） | FLOAT16、FLOAT32、DOUBLE、BFLOAT16 | 不适用（输出参数） | 支持 | 支持 | 数据格式和数据类型与 input 保持一致（R2/R3）；shape 的 N 轴、C 轴与 input 保持一致，D 轴、H 轴、W 轴与 grid 保持一致（R5） |

## 约束关系

### R1: grid 数据类型与 input 一致
涉及参数: input.dtype → grid.dtype
条件因子: 无
来源: 参数说明表 grid"使用说明"列 "数据类型与入参`input`的数据类型一致"；错误码表 161002 "input、grid、out的数据类型不在支持的范围之内或数据类型不一致"

### R2: out 数据类型与 input 保持一致
涉及参数: input.dtype → out.dtype
条件因子: 无
来源: 参数说明表 out"使用说明"列 "数据格式和数据类型与input保持一致"

### R3: out 数据格式与 input 保持一致
涉及参数: input.format → out.format
条件因子: 无
来源: 参数说明表 out"使用说明"列 "数据格式和数据类型与input保持一致"

### R4: input/grid 均为 5 维，grid.shape=(N, D_out, H_out, W_out, 3) 且 batch 维与 input 一致、末维固定为 3；ND 格式按 NCDHW 处理
涉及参数: input.shape, grid.dimensions → grid.shape
条件因子: 无
来源: 参数说明表 input/grid"使用说明"列 shape 约束（"当数据格式为ND时，会按照NCDHW格式进行处理"）；错误码表 161002 "input、grid和out的维度关系不匹配"

### R5: out.shape=(N, C, D_out, H_out, W_out)，N、C 取自 input，D、H、W 取自 grid
涉及参数: input.shape, grid.shape → out.shape
条件因子: 无
来源: 参数说明表 out"使用说明"列 "shape的N轴、C轴与input保持一致，shape的D轴、H轴、W轴与grid保持一致"

### R6: interpolationMode ∈ {0, 1}（3D 不支持 bicubic）；paddingMode ∈ {0, 1, 2}
涉及参数: interpolationMode.value, paddingMode.value
条件因子: 无
来源: 参数说明表 interpolationMode"使用说明"列 "支持0：bilinear（双线性插值）、1：nearest（最邻近插值）两种模式"；错误码表 161002 "interpolationMode或paddingMode的值不在支持范围内"

### R7: input 的 D_in×H_in×W_in < INT32 最大值（2147483647）
涉及参数: input.shape
条件因子: 无
来源: 约束说明 "输入`input`的（D轴的大小 * H轴的大小 * W轴的大小）< INT32的最大值"

### R8: input 空间三维（D_in、H_in、W_in）≥ 1；N、C ≥ 0（可为 0，空 tensor）；grid/out 各维 ≥ 0
涉及参数: input.shape, grid.shape, out.shape
条件因子: 无
来源: 参数说明表 input/grid/out"使用说明"列 "支持空Tensor"；空间三维 ≥1 为推断结论（推断自：README 约束说明"`x`和`grid`的shape，所有维度都必须大于0" + 2D 同族接口 aclnnGridSampler2D"input的shape最后两维的维度值不能为0"，待确认）

### R9: input.format=NDHWC 时须同时满足：数据类型为 FLOAT32、interpolationMode 为 0、C 轴为 4 的倍数
涉及参数: input.format → input.dtype, interpolationMode.value, input.shape
条件因子: input.format
来源: 参数说明表 input"使用说明"列 "仅当满足条件（数据类型为FLOAT32 && interpolationMode为0 && C轴为4的倍数）时支持NDHWC"

### R10: grid 元素为归一化坐标，基准区间 [-1, 1]；超出 [-1, 1] 的越界坐标按 paddingMode 语义处理（功能场景约束，通过 grid 值域分段实现场景A/B 覆盖）
涉及参数: grid.value_range
条件因子: paddingMode.value（越界处理方式由 paddingMode 决定）
来源: 功能说明 "会将x、y、z的取值范围归一化到[-1, 1]之间" + "对于超出范围的坐标，会根据paddingMode进行不同处理"

## 约束提取反向校验表

| R{n} | 约束项 | 原文片段 | 符号化 | 转录结果 | 状态 |
|------|--------|---------|--------|---------|------|
| R1 | grid dtype 一致 | "数据类型与入参`input`的数据类型一致" | 一致 | grid.dtype == input.dtype | EXACT_MATCH |
| R2 | out dtype 一致 | "数据格式和数据类型与input保持一致" | 一致 | out.dtype == input.dtype | EXACT_MATCH |
| R3 | out format 一致 | "数据格式和数据类型与input保持一致" | 一致 | out.format == input.format | EXACT_MATCH |
| R4 | grid shape 关系 | "支持shape为(N, D_out, H_out, W_out, 3)，且N与入参`input`的shape中的N一致" | 长度=5、末维=3、N 一致 | grid.shape=[N,D_out,H_out,W_out,3]，N=input.shape[0] | EXACT_MATCH |
| R5 | out shape 公式 | "shape的N轴、C轴与input保持一致，shape的D轴、H轴、W轴与grid保持一致" | 匹配 | out.shape=[input.shape[0..1], grid.shape[1..3]] | EXACT_MATCH |
| R6 | 模式枚举值域 | "支持0：bilinear（双线性插值）、1：nearest（最邻近插值）两种模式" | ∈ {0,1} | interpolationMode ∈ {0,1}；paddingMode ∈ {0,1,2} | EXACT_MATCH |
| R7 | D×H×W 上界 | "（D轴的大小 * H轴的大小 * W轴的大小）< INT32的最大值" | < 2147483647 | input.shape[2]×[3]×[4] < 2147483647 | EXACT_MATCH |
| R8 | 空间维非零 | "支持空Tensor"（input）+ README "所有维度都必须大于0" | D/H/W ≥ 1（推断） | input 空间三维 ≥1、N/C≥0；grid/out 各维 ≥0 | INFERENCE |
| R9 | NDHWC 条件 | "仅当满足条件（数据类型为FLOAT32 && interpolationMode为0 && C轴为4的倍数）时支持NDHWC" | NDHWC → (f32 ∧ mode=0 ∧ C%4==0) | input.format=NDHWC ⟹ dtype=float32 ∧ mode=0 ∧ input.shape[1]%4==0 | EXACT_MATCH |
| R10 | grid 归一化坐标 | "会将x、y、z的取值范围归一化到[-1, 1]之间" | ∈ [-1, 1]（基准区间，越界合法另处理） | grid 值域分段：[-1,1]（场景A）+ 越界段（场景B） | INFERENCE |

待确认项（不阻断流程）：R8 中 input 空间三维 ≥1 为推断约束（aclnnGridSampler3D.md 仅声明"支持空Tensor"未限定维度，README 声明"x和grid的shape所有维度都必须大于0"与之存在文档间差异）；本设计按保守取值（空间三维 ≥1、N/C/输出维可为 0）编码，若实测支持空间维为 0 可放宽 ET 场景。
